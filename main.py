import json
import pickle
import csv
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F

from cgcnn.data import collate_pool, pre_extract_structure_graphs
from cgcnn.model import CrystalGraphConvNet

from cgcnn.utils import initialize_kwargs
from moe.utils import normalizer_from_subsets, AverageMeter, \
    get_small_datasets_info_dict

parser = argparse.ArgumentParser()
parser.add_argument('--filename_prefix', type=str, default='')
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--head', type=str, default='sanity')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--layer_to_extract_from', type=str, default='conv')
parser.add_argument('--partition_seed', type=int, default=0)
parser.add_argument('--num_layers_to_unfreeze', type=int, default=0)
parser.add_argument('--backbone_name', type=str, default=None)
parser.add_argument('--option', type=str, default='single')
parser.add_argument('--test', action='store_true')
parser.add_argument('--use_all_backbones', action='store_true')
parser.add_argument('--k_backbone_gating', type=int, default=0)
parser.add_argument('--n_pseudo_attn_heads', type=int, default=0)
parser.add_argument('--optim', type=str, default='SGD')
args = parser.parse_args(sys.argv[1:])


def main(dataset_name='expt_eform', head='sanity', n_head_layers=1,
         layer_to_extract_from='conv', partition_seed=0,
         num_layers_to_unfreeze=0, backbone_name=None, option='single'):
    """

    :param dataset_name:
    :param head:
    :param n_head_layers:
    :param layer_to_extract_from: 'conv-2', 'conv', 'first_fc', or
        'penultimate_fc'
    :param partition_seed:
    :param num_layers_to_unfreeze:
    :param backbone_name:
    :param option: If 'single', train new layers on top of a single backbone.
        If 'concat', concatenate all the backbones, learn a scaling parameter
        for each one, and learn new layers on top. If 'add', learn a scaling
        parameter for each backbone, then add all the backbones into a single
        vector. If 'add_2', do the previous, but only keep the 2 scaling
        parameters with the largest absolute value (and set the rest to 0).
        If 'add_k', do the previous, but keep a user-defined k backbones.
        If 'add_attn', produce data-dependent scaling parameters for
        each backbone. If 'ensemble', train 3 separate heads on 3 separate
        backbones, then predict a weighted average of the outputs, where the
        weights are learned.
    :param use_all_backbones: If True, use all 18 backbones instead of
        3 manually chosen ones.
    :param k_backbone_gating: Only allow k backbones to be used in the learned
        representation.
    :param n_pseudo_attn_heads: This arg is only used if option=='add_k'. Sets the
        number of mixed backbones to learn and concatenate together.
    :return:
    """

    global args
    if args.option == 'add_k':
        assert args.k_backbone_gating != 0

    if args.dataset_name is not None:
        dataset_name = args.dataset_name
        head = args.head
        n_head_layers = args.num_layers
        layer_to_extract_from = args.layer_to_extract_from
        # partition_seed = args.partition_seed
        num_layers_to_unfreeze = args.num_layers_to_unfreeze
        backbone_name = args.backbone_name
        option = args.option
        test = args.test

        print('dataset: {}'.format(dataset_name))
        print('head: {}'.format(head))
        print('num layers learned from scratch: {}'.format(n_head_layers))
        print('layer to extract from: {}'.format(layer_to_extract_from))
        print('partition seed: {}'.format(partition_seed))
        print('num_layers_to_unfreeze: {}'.format(num_layers_to_unfreeze))
        print('backbone name: {}'.format(backbone_name))
        print('option: {}'.format(option))
        print('test: {}'.format(test))
        print('use_all_backbones: {}'.format(args.use_all_backbones))
        print('optim: {}'.format(args.optim))
        if option == 'add_k':
            print('k backbone gating: {}'.format(args.k_backbone_gating))
            print('n_pseudo_attn_heads: {}'.format(args.n_pseudo_attn_heads))

    if not args.filename_prefix:
        from datetime import date
        import os
        today = str(date.today())
        if not os.path.exists(today):
            os.makedirs(today)

        torch.manual_seed(partition_seed)

        filename_prefix = today + '/head_selection_sanity_check_'  # to pre-pend to filenames of results
    else:
        filename_prefix = args.filename_prefix

    cuda = torch.cuda.is_available()
    with open('cgcnn/data/sample-regression/atom_init.json') as atom_init_json:
        atom_init_dict = json.load(atom_init_json)
    _, _, dataset_kwargs, _, _, _ = initialize_kwargs()

    # All our feature extractor models have the same architecture
    hyperparameter_file = open('cgcnn/data/hyperparameters.json')
    hyperparameter_dict = json.load(hyperparameter_file)
    model_kwargs = hyperparameter_dict['model_kwargs']

    # ----------------------- Get pre-trained backbone -------------------------
    if args.use_all_backbones:
        model_paths = get_all_backbones()
    else:  # use hand-picked backbones
        if dataset_name == 'jarvis_2d_exfoliation':
            model_paths = {  # task_name: extractor_path
                'mp_eform':
                    'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar',
                'jarvis_eform':
                    'cgcnn/data/saved_extractors//2022-02-25-18:09_singletask_32hfl_jarvisEform_best_model.pth.tar',
                'jarvis_gvrh':
                    'cgcnn/data/saved_extractors/2022-02-25-19:03_singletask_32hfl_jarvisGvrh_best_model.pth.tar'
            }
        elif dataset_name == 'expt_eform':
            model_paths = {
                'mp_eform':
                    'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar',
                'jarvis_eform':
                    'cgcnn/data/saved_extractors/2022-02-25-18:09_singletask_32hfl_jarvisEform_best_model.pth.tar',
                'castelli_eform':
                    'cgcnn/data/saved_extractors/2022-03-03-05:32_singletask_castelliEform_32hfl_best_model.pth.tar'
            }
        elif dataset_name == 'piezoelectric_tensor':
            model_paths = {
                'mp_eform':
                    'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar',
                'mp_kvrh':
                    'cgcnn/data/saved_extractors/2022-03-27-15:50_singletask_32hfl_mpkvrh_best_model.pth.tar',
                'mp_bandgap':
                    'cgcnn/data/saved_extractors/2022-01-26-15:38_singletask_bandgap_32hfealen_best_model.pth.tar'
            }
        else:
            # transfer from the largest source task
            model_paths = {
                'mp_eform':
                    'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar'
            }

    backbones = []
    if option == 'single':
        backbone_path = model_paths.get(backbone_name, None)
        assert backbone_name is not None

        backbone = get_extractor(backbone_path, layer_to_extract_from,
                                 model_kwargs=model_kwargs,
                                 device='cuda' if cuda else 'cpu')
        backbones.append(backbone)
    elif option == 'add' or option == 'add_2' or option == 'concat' or \
            option == 'ensemble' or option == 'add_k':
        for backbone_name, backbone_path in model_paths.items():
            backbone = get_extractor(
                backbone_path, layer_to_extract_from, model_kwargs,
                device='cuda' if cuda else 'cpu')
            backbones.append(backbone)
    else:
        raise NotImplementedError

    # Load data
    small_dataset_info_dict = get_small_datasets_info_dict()
    small_dataset_path = small_dataset_info_dict[dataset_name]['path']
    target_name = small_dataset_info_dict[dataset_name]['target_name']
    df = pd.read_pickle(small_dataset_path)

    df['structure'].reset_index(drop=True, inplace=True)
    df[target_name].reset_index(drop=True, inplace=True)
    structure_graphs, targets, idxs = pre_extract_structure_graphs(
        X=df['structure'].tolist(),
        y=df[target_name].tolist(),
        atom_init_fea=atom_init_dict)

    task_dataset = saved_structure_graphs_dataset(structure_graphs, targets)

    # Get train/val/test indices from file
    with open(
            'data/matminer/saved_partition_indices/all_task_partition_indices_seed' +
            str(partition_seed) + '.pkl', 'rb') as f:
        dict_of_task_indices = pickle.load(f)

    train_indices, val_indices, test_indices = \
        dict_of_task_indices[dataset_name]

    train_subset = Subset(task_dataset, train_indices)
    val_subset = Subset(task_dataset, val_indices)
    test_subset = Subset(task_dataset, test_indices)

    # todo: try using test set too during normalization
    # from random import sample
    # import random
    # random.seed(partition_seed)
    # sample_data_list = [task_dataset[i] for i in
    #                     sample(range(len(task_dataset)), 500)]
    # _, labels_to_normalize_with, _ = collate_pool(sample_data_list)
    # normalizer = Normalizer(labels_to_normalize_with)

    # todo: normalize based only from train/val subsets
    normalizer = normalizer_from_subsets([train_subset, val_subset])

    file = open(
        'data/matminer/stl_baseline_val_maes/stl_small_task_val_maes_seed' +
        str(partition_seed) + '.json')
    dict_with_stl_val_maes = json.load(file)
    stl_val_mae = dict_with_stl_val_maes[dataset_name]

    train_dl = DataLoader(
        train_subset, batch_size=250, num_workers=0, shuffle=True,
        collate_fn=collate_pool)
    val_dl = DataLoader(
        val_subset, batch_size=250, num_workers=0, shuffle=True,
        collate_fn=collate_pool)
    test_dl = DataLoader(
        test_subset, batch_size=250, num_workers=0, shuffle=True,
        collate_fn=collate_pool)

    if layer_to_extract_from == 'conv' or layer_to_extract_from == 'conv-2':
        n_features = 64
    else:
        n_features = 32

    if option == 'concat':
        n_features = n_features * 3

    if head == 'sanity':
        if option == 'add_k' and args.n_pseudo_attn_heads > 0:
            assert args.k_backbone_gating > 0
            model = multihead_add_k_backbones(
                num_pseudo_attention_heads=args.n_pseudo_attn_heads,
                backbones=backbones, num_out_layers=n_head_layers,
                backbone_feature_dim=n_features,
                k_experts=args.k_backbone_gating)
            param_groups = [{'params': model.non_backbone_parameters()}]

        elif option != 'ensemble':
            model = Sanity_Check_Head(
                num_layers=n_head_layers, input_dim=n_features)
            ensembled_backbone = learned_backbone(backbones, option)
            param_groups = [
                {'params': model.parameters()},
                {'params': ensembled_backbone.non_backbone_parameters()}]

        elif option == 'ensemble':
            models = []
            for _ in backbones:
                model = Sanity_Check_Head(
                    num_layers=n_head_layers, input_dim=n_features)
                models.append(model)
            prediction_ensembler = ensemble_predictions(len(backbones))
            param_groups = [{'params': model.parameters()},
                            {'params': prediction_ensembler.parameters()}]
    else:
        raise NotImplementedError

    if num_layers_to_unfreeze > 0:
        for backbone in backbones:
            params_to_finetune = get_parameters_to_finetune(
                backbone, num_layers_to_unfreeze, layer_to_extract_from)
            param_groups.append({'params': params_to_finetune, 'lr': 0.005})

    if cuda:
        if option == 'add_k' and args.n_pseudo_attn_heads > 0:
            model.to('cuda')
        elif option != 'ensemble':
            model.to('cuda')
            ensembled_backbone.to('cuda')
        else:
            for i, backbone in enumerate(backbones):
                backbone.to('cuda')
                models[i].to('cuda')
            prediction_ensembler.to('cuda')

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            params=param_groups, lr=0.01, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            params=param_groups, lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=1000)

    # Create file for storing losses
    losses_filename = filename_prefix + option + '_transfer_to_' + \
        str(dataset_name) + '_' + layer_to_extract_from + '_' + str(n_head_layers)\
        + '_layers_' + 'seed' + str(partition_seed) + '_loss_data.csv'
    loss_file = open(losses_filename, 'w', encoding='utf-8')
    writer = csv.writer(loss_file)
    writer.writerow(['batch_num', 'avg training loss', 'avg validation loss',
                     'stl normed val mae'])
    loss_file.close()

    val_mae_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    train_loss_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    val_loss_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    best_val_mae = 1e10
    num_batches_since_improvement = 0
    best_model_filename = filename_prefix + '_' + dataset_name + '_seed' + \
        str(partition_seed) + '_best_model.pth'
    loss_regularizer = torch.tensor([0.], device='cuda' if cuda else 'cpu')

    if option == 'add_2' or option == 'add_k':
        early_stopping_n_epochs = 500
    else:
        early_stopping_n_epochs = 500  # 150

    # 1000 epochs
    for epoch in range(2):
        # print('epoch: {}'.format(epoch))

        for structures, labels, _ in train_dl:
            if cuda:
                structures = (structures[0].cuda(non_blocking=True),
                              structures[1].cuda(non_blocking=True),
                              structures[2].cuda(non_blocking=True),
                              [crys_idx.cuda(non_blocking=True)
                               for crys_idx in structures[3]])
                labels = labels.cuda(non_blocking=True)

            if option == 'add_k' and args.n_pseudo_attn_heads > 0:
                predictions, loss_regularizer = model(structures)
            elif option != 'ensemble':
                features = ensembled_backbone(structures)
                predictions = model(features)
            else:
                predictions_across_heads = []
                for i, backbone in enumerate(backbones):
                    features = backbone(*structures)
                    predictions = models[i](features)
                    if predictions.dim() == 2:
                        predictions = predictions.squeeze(1)
                    predictions_across_heads.append(predictions)

                predictions_across_heads = torch.stack(
                    predictions_across_heads, dim=1)
                predictions = prediction_ensembler(predictions_across_heads)

            if predictions.shape != labels.shape:
                predictions = predictions.squeeze(1)

            train_loss = F.mse_loss(predictions, normalizer.norm(labels))
            total_loss = train_loss + 0.01 * loss_regularizer

            train_loss_meter.update(train_loss.detach().clone(), labels.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for structures, labels, _ in val_dl:
            if cuda:
                structures = (structures[0].cuda(non_blocking=True),
                              structures[1].cuda(non_blocking=True),
                              structures[2].cuda(non_blocking=True),
                              [crys_idx.cuda(non_blocking=True)
                               for crys_idx in structures[3]])
                labels = labels.cuda(non_blocking=True)

            if option == 'add_k' and args.n_pseudo_attn_heads > 0:
                predictions, _ = model(structures)
            elif option != 'ensemble':
                features = ensembled_backbone(structures)
                predictions = model(features)
            else:
                predictions_across_heads = []
                for i, backbone in enumerate(backbones):
                    features = backbone(*structures)
                    predictions = models[i](features)
                    if predictions.dim() == 2:
                        predictions = predictions.squeeze(1)
                    predictions_across_heads.append(predictions)

                predictions_across_heads = torch.stack(
                    predictions_across_heads, dim=1)
                predictions = prediction_ensembler(predictions_across_heads)

            if predictions.shape != labels.shape:
                predictions = predictions.squeeze(1)

            val_loss = F.mse_loss(predictions, normalizer.norm(labels))
            val_mae = torch.mean(torch.abs(
                normalizer.denorm(predictions) - labels))

            val_loss_meter.update(val_loss.detach().clone(), labels.size(0))
            val_mae_meter.update(val_mae.detach().clone(), labels.size(0))

        lr_scheduler.step()
        val_mae = float(val_mae_meter.avg)
        if val_mae < best_val_mae:
            num_batches_since_improvement = 0
            best_val_mae = val_mae

            if option == 'add_k' and args.n_pseudo_attn_heads > 0:
                state_dicts = model.state_dict()
            elif option != 'ensemble':
                state_dicts = ensembled_backbone.state_dict()
            else:
                state_dicts = []
                for i in range(len(backbones)):
                    state_dicts.append(
                        (backbones[i].state_dict(), models[i].state_dict()))
            torch.save({
                'epoch': epoch,
                'option': option,
                'model_state_dict': state_dicts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_mae': best_val_mae}, best_model_filename)
        else:
            num_batches_since_improvement += 1

        loss_file = open(losses_filename, 'a', encoding='utf-8')
        writer = csv.writer(loss_file)
        writer.writerow([
            epoch, float(train_loss_meter.avg), float(val_loss_meter.avg),
            float(val_mae / stl_val_mae)])
        loss_file.close()

        if num_batches_since_improvement > early_stopping_n_epochs:  # early stopping
            print('best val mae / stl val mae: {}'.format(
                best_val_mae / stl_val_mae))
            break

        train_loss_meter.reset()
        val_loss_meter.reset()
        val_mae_meter.reset()

    print('best val mae / stl val mae: {}'.format(best_val_mae / stl_val_mae))

    # Get test error
    test_mae_meter = AverageMeter(tensor=True,
                                  device='cuda' if cuda else 'cpu')
    checkpoint = torch.load(best_model_filename)
    if option == 'add_k' and args.n_pseudo_attn_heads > 0:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    elif option != 'ensemble':
        state_dict = checkpoint['model_state_dict']
        ensembled_backbone.load_state_dict(state_dict)
    else:
        state_dicts = checkpoint['model_state_dict']
        for i, (backbone_dict, model_dict) in enumerate(state_dicts):
            backbones[i].load_state_dict(backbone_dict)
            models[i].load_state_dict(model_dict)

    for structures, labels, _ in test_dl:
        if cuda:
            structures = (structures[0].cuda(non_blocking=True),
                          structures[1].cuda(non_blocking=True),
                          structures[2].cuda(non_blocking=True),
                          [crys_idx.cuda(non_blocking=True) for crys_idx in
                           structures[3]])
            labels = labels.cuda(non_blocking=True)

        if option == 'add_k' and args.n_pseudo_attn_heads > 0:
            predictions, _ = model(structures)
        elif option != 'ensemble':
            features = ensembled_backbone(structures)
            predictions = model(features)
        else:
            predictions_across_heads = []
            for i, backbone in enumerate(backbones):
                features = backbone(*structures)
                predictions = models[i](features)
                if predictions.dim() == 2:
                    predictions = predictions.squeeze(1)
                predictions_across_heads.append(predictions)

            predictions_across_heads = torch.stack(
                predictions_across_heads, dim=1)
            predictions = prediction_ensembler(predictions_across_heads)

        if predictions.shape != labels.shape:
            predictions = predictions.squeeze(1)

        test_mae = torch.mean(torch.abs(
            normalizer.denorm(predictions) - labels))
        test_mae_meter.update(test_mae.detach().clone(), labels.size(0))

    print('Test MAE: {}'.format(test_mae_meter.avg))

    return float(test_mae_meter.avg), float(best_val_mae / stl_val_mae)


def get_extractor(checkpoint_path=None, layer_to_extract_from='conv',
                  model_kwargs=None, device='cpu'):
    """
    Args:
        checkpoint_path (str): Path to checkpoint containing the model state
            dict.
        layer_to_extract_from (str): If 'conv', extract the features from the
            last convolutional layer. If 'fc', extract the features from the
            penultimate fully connected layer. If 'conv-2', extract features
            from the second to last convolutional layer.
        task_name (str): Name of the task. Required if the model is multi-headed
            so we know which head to use as the feature extractor.
        model_kwargs (dict): CrystalGraphConvNet kwargs.

    Returns:
        Model with all layers after the feature extraction layer set to the
        identity. (torch.nn.module)
    """
    if model_kwargs is None:
        _, model_kwargs, _, _, _, _ = initialize_kwargs()
    model = CrystalGraphConvNet(**model_kwargs)

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device(device))
        except RuntimeError:
            raise RuntimeError(
                'Either your PyTorch version is too old, or the'
                'following path does not exist: ' + checkpoint_path)

        best_model_state_dict = checkpoint['state_dict']
        model.load_state_dict(best_model_state_dict)
    else:
        import warnings
        warnings.warn('checkpoint_path is None. Learning a backbone from '
                      'scratch.')

    # set layers after feature extractor layer to identity function
    if layer_to_extract_from == 'conv':
        model.conv_to_fc_softplus = Identity()
        model.conv_to_fc = Identity()
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = nn.Softplus()    # Softplus

    elif layer_to_extract_from == 'conv-2':
        # remove the last conv but still apply pooling
        model.convs[-1] = Identity()

        model.conv_to_fc_softplus = Identity()
        model.conv_to_fc = Identity()
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = nn.Softplus()    # Softplus

    elif layer_to_extract_from == 'first_fc':
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = Identity()

    elif layer_to_extract_from == 'penultimate_fc':
        model.fc_out = Identity()
    else:
        raise AttributeError(
            'layer_to_extract_from must be \'conv\', \'first_fc\', or '
            '\'penultimate_fc\'.')
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Sanity_Check_Head(nn.Module):
    def __init__(self, num_layers, input_dim, option='single', hidden_dim=None):
        super(Sanity_Check_Head, self).__init__()
        self.softplus = nn.Softplus()
        self.num_layers = num_layers

        if hidden_dim is None:
            if option == 'single' or option == 'add' or option == 'add_2' or \
                    option == 'add_attn':
                hidden_dim = 32
            elif option == 'concat':
                hidden_dim = 32 * 3

        layers = []
        if self.num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_dim))
        if self.num_layers > 2:
            for _ in range(self.num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        if num_layers == 1:
            self.out_layer = nn.Linear(input_dim, 1)
        else:
            self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, feature):
        out = feature
        for fc in self.layers:
            out = self.softplus(fc(out))
        out = self.out_layer(out)
        return out


class learned_backbone(nn.Module):
    def __init__(self, backbones, option='single', k_experts=0):
        """
        Args:
            backbones (list of nn.Module): Backbones which all have output
                of the same dimensionality.
            option (str):
        """
        super(learned_backbone, self).__init__()
        self.backbones = nn.ModuleList(backbones)
        self.option = option
        self.k_experts = k_experts

        if self.option == 'add_k':
            assert 0 < self.k_experts <= len(backbones)

        self.scaling_params = nn.Parameter(
            torch.ones(len(self.backbones)),
            requires_grad=option != 'single' and option != 'ensemble')

        # if self.option == 'add_2' or self.option == 'add_k':
        #     nn.init.normal_(
        #         self.scaling_params, mean=0., std=0.001)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, structure):
        out = self.scaling_params[0] * self.backbones[0](*structure)

        if self.option == 'single':
            return out
        elif self.option == 'add':
            for i in range(len(self.backbones)-1):
                scaled_backbone_to_add = self.scaling_params[i+1] * \
                    self.backbones[i+1](*structure)
                out = out + scaled_backbone_to_add
            return out
        elif self.option == 'add_2':
            top_2_values, top_2_indices = torch.topk(
                self.scaling_params, 2)
            top_2_probabilities = self.softmax(top_2_values)

            idx1, idx2 = top_2_indices[0], top_2_indices[1]
            prob1, prob2 = top_2_probabilities[0], top_2_probabilities[1]

            out = prob1 * self.backbones[idx1](*structure) + \
                prob2 * self.backbones[idx2](*structure)
            return out
        elif self.option == 'add_k':
            top_k_values, top_k_indices = torch.topk(
                self.scaling_params, self.k_experts)
            top_k_probabilities = self.softmax(top_k_values)

            top_prob, top_idx = top_k_probabilities[0], top_k_indices[0]
            out = top_prob * self.backbones[top_idx](*structure)

            for i in range(self.k_experts-1):

                prob, idx = top_k_probabilities[i+1], top_k_indices[i+1]
                out = out + prob * self.backbones[idx](*structure)

            # sparsified backbone weights of shape: (len(self.backbones),)
            backbone_scores = torch.zeros(
                len(self.backbones), device=self.scaling_params.device).scatter(
                0, top_k_indices, top_k_probabilities)

            return out, backbone_scores

        elif self.option == 'concat':
            for i in range(len(self.backbones) - 1):
                scaled_backbone_to_concat = \
                    self.scaling_params[i+1] * self.backbones[i+1](*structure)
                out = torch.cat((out, scaled_backbone_to_concat), dim=1)
            return out
        else:
            raise NotImplementedError

    def non_backbone_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad and 'backbones' not in n:
                yield p


class ensemble_predictions(nn.Module):
    def __init__(self, num_predictions_to_ensemble):
        super(ensemble_predictions, self).__init__()
        self.num_predictions_to_ensemble = num_predictions_to_ensemble
        self.weights = nn.Parameter(torch.ones(num_predictions_to_ensemble),
                                    requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, predictions):
        normalized_weights = self.softmax(self.weights)

        out = torch.sum(normalized_weights * predictions, dim=1)
        return out


class multihead_add_k_backbones(nn.Module):
    def __init__(self, num_pseudo_attention_heads, backbones,
                 num_out_layers, backbone_feature_dim, k_experts=1):
        super(multihead_add_k_backbones, self).__init__()
        self.num_pseudo_attention_heads = num_pseudo_attention_heads
        self.k_experts_per_head = k_experts
        self.backbone_feature_dim = backbone_feature_dim  # dimensionality of a single backbone's output

        self.pseudo_attention_heads = nn.ModuleList()
        for _ in range(num_pseudo_attention_heads):
            pseudo_attention_head = learned_backbone(
                backbones, option='add_k', k_experts=self.k_experts_per_head)
            self.pseudo_attention_heads.append(pseudo_attention_head)

        self.out_layer = Sanity_Check_Head(
            num_out_layers,
            input_dim=backbone_feature_dim * num_pseudo_attention_heads,
            option='single', hidden_dim=32*num_pseudo_attention_heads)

    def forward(self, structure):
        n_structures = len(structure[-1])
        head_outputs = []
        score_lst = []

        out, backbone_scores = self.pseudo_attention_heads[0](structure)
        head_outputs.append(out)
        score_lst.append(backbone_scores)

        for i in range(self.num_pseudo_attention_heads-1):
            out, backbone_scores = self.pseudo_attention_heads[i+1](structure)

            head_outputs.append(out)
            score_lst.append(backbone_scores)

        # batch_size, backbone_feature_size * num_pseudo_attention_heads
        multihead_feature = torch.stack(
            head_outputs, dim=-1).view(n_structures, -1)

        # shape: (n_backbones, num_pseudo_attention_heads)
        score_lst = torch.stack(score_lst, dim=-1)

        # Use loss regularizer to avoid collapse to a single set of backbones
        loss_regularizer = torch.pow(
            torch.norm(
                torch.transpose(score_lst, 0, 1) @ score_lst -
                torch.eye(self.num_pseudo_attention_heads,
                          device=score_lst.device)), 2)

        out = self.out_layer(multihead_feature)
        return out, loss_regularizer

    def non_backbone_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad and 'backbones' not in n:
                yield p


def get_parameters_to_finetune(
        backbone, num_layers_to_unfreeze, layer_to_extract_from='conv'):

    for n, p in backbone.named_parameters():
        p.requires_grad = False

    if layer_to_extract_from == 'conv':
        num_layers_to_unfreeze += backbone.n_h
    else:
        raise NotImplementedError

    layers_unfrozen = 0
    layers_to_unfreeze = []
    while layers_unfrozen < num_layers_to_unfreeze:
        if layers_unfrozen < backbone.n_h-1 and \
                layer_to_extract_from == 'penultimate_fc':
            # get the next layer, then continue:
            layers_to_unfreeze.extend(
                list(backbone.dense_heads.fcs[-(layers_unfrozen+1)].parameters()))
            layers_unfrozen += 1

        elif layers_unfrozen < backbone.n_h and \
                (layer_to_extract_from == 'first_fc' or
                 layer_to_extract_from == 'penultimate_fc'):

            if layer_to_extract_from == 'first_fc':
                layers_unfrozen += backbone.n_h-1

            # get the next layer
            layers_to_unfreeze.extend(
                list(backbone.dense_heads.conv_to_fc.parameters()))
            layers_unfrozen += 1

        elif layers_unfrozen < backbone.n_h + 1 and \
                (layer_to_extract_from == 'conv' or
                 layer_to_extract_from == 'first_fc' or
                 layer_to_extract_from == 'penultimate_fc'):
            # get the next layer
            print('UNFROZE backbone.convs[{}]'.format(str(-1)))

            if layer_to_extract_from == 'conv':
                layers_unfrozen += backbone.n_h

            layers_to_unfreeze.extend(
                list(backbone.convs[-1].fc_full.parameters()))
            layers_to_unfreeze.extend(
                list(backbone.convs[-1].bn1.parameters()))
            layers_to_unfreeze.extend(
                list(backbone.convs[-1].bn2.parameters()))

            layers_unfrozen += 1

        elif layers_unfrozen < backbone.n_h + backbone.n_conv and \
                (layer_to_extract_from == 'conv-2' or
                 layer_to_extract_from == 'conv' or
                 layer_to_extract_from == 'first_fc' or
                 layer_to_extract_from == 'penultimate_fc'):

            conv_to_grab = -(layers_unfrozen - backbone.n_h + 1)
            print('UNFROZE backbone.convs[{}]'.format(str(conv_to_grab)))

            if layer_to_extract_from == 'conv-2' and \
                    layers_unfrozen < backbone.n_h + 1:
                layers_unfrozen += backbone.n_h + 1

            # get the next layer
            layers_to_unfreeze.extend(
                list(backbone.convs[conv_to_grab].fc_full.parameters()))
            layers_to_unfreeze.extend(
                list(backbone.convs[conv_to_grab].bn1.parameters()))
            layers_to_unfreeze.extend(
                list(backbone.convs[conv_to_grab].bn2.parameters()))

            layers_unfrozen += 1

        else:
            print('UNFROZE EMBEDDING')
            layers_to_unfreeze.extend(
                list(backbone.embedding.parameters()))

            for p in layers_to_unfreeze:
                p.requires_grad = True

            return layers_to_unfreeze

    for p in layers_to_unfreeze:
        p.requires_grad = True

    return layers_to_unfreeze


class saved_structure_graphs_dataset(torch.utils.data.Dataset):
    def __init__(self, structure_graphs, targets):
        self.structure_graphs = structure_graphs
        self.labels = torch.tensor(targets)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.structure_graphs[idx], self.labels[idx], idx


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""

        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        normed_tensor = tensor
        return (normed_tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def get_all_backbones():
    model_paths = {
        'mp_eform':
            'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar',
        'mp_bandgap':
            'cgcnn/data/saved_extractors/2022-01-26-15:38_singletask_bandgap_32hfealen_best_model.pth.tar',
        'mp_gvrh':
            'cgcnn/data/saved_extractors/2022-01-26-19:08_singletask_gvrh_32hfealen_best_model.pth.tar',
        'mp_kvrh':
            'cgcnn/data/saved_extractors/2022-03-27-15:50_singletask_32hfl_mpkvrh_best_model.pth.tar',
        'weighted_n_e_cond':
            'cgcnn/data/saved_extractors/2022-02-03-21:12_singletask_necond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_p_e_cond':
            'cgcnn/data/saved_extractors/2022-02-04-00:05_singletask_pecond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_n_th_cond':
            'cgcnn/data/saved_extractors/2022-02-04-03:02_singletask_nthcond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_p_th_cond':
            'cgcnn/data/saved_extractors/2022-02-04-05:50_singletask_pthcond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'p_Seebeck':
            'cgcnn/data/saved_extractors/2022-02-26-04:44_singletask_32hfl_pSeebeck_best_model.pth.tar',
        'n_Seebeck':
            'cgcnn/data/saved_extractors/2022-02-26-02:00_singletask_32hfl_nSeebeck_best_model.pth.tar',
        'n_avg_eff_mass':
            'cgcnn/data/saved_extractors/2022-02-25-19:05_singletask_32hfl_nEffMass_best_model.pth.tar',
        'p_avg_eff_mass':
            'cgcnn/data/saved_extractors/2022-02-25-19:06_singletask_32hfl_pEffMass_best_model.pth.tar',
        'castelli_eform':
            'cgcnn/data/saved_extractors/2022-03-03-05:32_singletask_castelliEform_32hfl_best_model.pth.tar',
        'jarvis_eform':
            'cgcnn/data/saved_extractors/2022-02-25-18:09_singletask_32hfl_jarvisEform_best_model.pth.tar',
        'jarvis_dielectric_opt':
            'cgcnn/data/saved_extractors/2022-02-25-18:03_singletask_32hfl_jarvisDielectric_best_model.pth.tar',
        'jarvis_bandgap':
            'cgcnn/data/saved_extractors/2022-02-25-18:17_singletask_32hfl_jarvisEg_best_model.pth.tar',
        'jarvis_gvrh':
            'cgcnn/data/saved_extractors/2022-02-25-19:03_singletask_32hfl_jarvisGvrh_best_model.pth.tar',
        'jarvis_kvrh':
            'cgcnn/data/saved_extractors/2022-02-25-19:04_singletask_32hfl_jarvisKvrh_best_model.pth.tar'
    }
    return model_paths


if __name__ == '__main__':
    num_layers_to_unfreeze = 1  # number of backbone layers to unfreeze
    n_head_layers = 3  # number of head layers to train from scratch
    layer_to_extract_from = 'conv'

    small_dataset = 'jarvis_2d_exfoliation' #'piezoelectric_tensor' #'jarvis_2d_exfoliation'

    # 'backbone_name' is irrelevant if option != 'single'
    backbone_name = 'mp_eform' #'mp_kvrh' #'mp_eform' #'jarvis_gvrh'

    option = 'single'  #'concat'  #'concat'  #'add'  # 'single'  # 'ensemble'

    print('------ seed 0 -------')
    test1, val1 = main(
        dataset_name=small_dataset, n_head_layers=n_head_layers,
        head='sanity', layer_to_extract_from=layer_to_extract_from,
        partition_seed=0, num_layers_to_unfreeze=num_layers_to_unfreeze,
        backbone_name=backbone_name, option=option)

    print('------ seed 1 -------')
    test2, val2 = main(
        dataset_name=small_dataset, n_head_layers=n_head_layers,
        head='sanity', layer_to_extract_from=layer_to_extract_from,
        partition_seed=1, num_layers_to_unfreeze=num_layers_to_unfreeze,
        backbone_name=backbone_name, option=option)

    print('------ seed 2 --------')
    test3, val3 = main(
        dataset_name=small_dataset, n_head_layers=n_head_layers,
        head='sanity', layer_to_extract_from=layer_to_extract_from,
        partition_seed=2, num_layers_to_unfreeze=num_layers_to_unfreeze,
        backbone_name=backbone_name, option=option)

    print('------ seed 3 --------')
    test4, val4 = main(
        dataset_name=small_dataset, n_head_layers=n_head_layers,
        head='sanity', layer_to_extract_from=layer_to_extract_from,
        partition_seed=3, num_layers_to_unfreeze=num_layers_to_unfreeze,
        backbone_name=backbone_name, option=option)

    print('------ seed 4 --------')
    test5, val5 = main(
        dataset_name=small_dataset, n_head_layers=n_head_layers,
        head='sanity', layer_to_extract_from=layer_to_extract_from,
        partition_seed=4, num_layers_to_unfreeze=num_layers_to_unfreeze,
        backbone_name=backbone_name, option=option)

    avg = np.mean([val1, val2, val3, val4, val5])
    stdev = np.std([val1, val2, val3, val4, val5])
    print(avg)
    print(stdev)

    test_avg = np.mean([test1, test2, test3, test4, test5])
    test_std = np.std([test1, test2, test3, test4, test5])

    result_filename = args.filename_prefix + '_result.csv'
    result_file = open(result_filename, 'w', encoding='utf-8')
    writer = csv.writer(result_file)
    writer.writerow(['best val mae / stl val mae', '+/-', 'test_mae', '+/-'])
    writer.writerow([avg, stdev, test_avg, test_std])
    result_file.close()
