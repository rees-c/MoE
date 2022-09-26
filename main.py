import json
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

from moe.utils import normalizer_from_subsets, AverageMeter, \
    get_small_datasets_info_dict, get_parameters_to_finetune
from moe.model import *
from data.data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--filename_prefix', type=str, default='',
                    help='Path and filename prefix to prepend to output file '
                         'names.')
parser.add_argument('--dataset_name', type=str, default=None,
                    help='Downstream dataset name.')
parser.add_argument('--n_head_layers', type=int, default=3)
parser.add_argument('--layer_to_extract_from', type=str, default='conv',
                    help='conv-2, conv, first_fc, or penultimate_fc')
parser.add_argument('--num_layers_to_unfreeze', type=int, default=1,
                    help='Number of extractor layers to fine-tune.')
parser.add_argument(
    '--option', type=str, default='add_k',
    help='If \'pairwise_TL\', train new layers on top of a single extractor. '
         'If \'ensemble\', train 3 separate heads on 3 separate '
         'extractors, then predict a weighted average of the outputs, where '
         'the weights are learned.'
         'If \'concat\', concatenate all the extractor outputs, learn a '
         'scaling parameter to multiply against the output of each one, and '
         'learn new head layers on top.'
         'If \'add_k\', learn a scaling parameter for each extractor, then add '
         'the outputs of the k extractors with the largest scaling parameters '
         'into a single vector. ')
parser.add_argument('--extractor_name', type=str, default=None,
                    help='If --option is \'pairwise_TL\', then'
                    ' \'--extractor_name\' specifies the extractor. See '
                         'the get_all_extractors() function for possible names.')
parser.add_argument('--use_all_extractors', action='store_true',
                    help='If True, use all 18 backbones instead of 3 manually '
                         'chosen ones.')
parser.add_argument('--k_extractor_gating', type=int, default=2,
                    help='If --option is \'add_k\', determines the number'
                         ' of extractors to use in the combined '
                         'representation.')
parser.add_argument('--n_pseudo_attn_heads', type=int, default=1,
                    help='Number of combined feature vectors to produce from '
                         'the backbones.')
parser.add_argument('--optim', type=str, default='Adam')
args = parser.parse_args(sys.argv[1:])


def main(dataset_name='expt_eform', n_head_layers=3,
         layer_to_extract_from='conv', seed=0,
         num_layers_to_unfreeze=1, extractor_name='mp_eform',
         option='pairwise_TL'):

    global args
    if args.option == 'add_k':
        assert args.k_extractor_gating != 0

    if args.dataset_name is not None:
        dataset_name = args.dataset_name
        n_head_layers = args.n_head_layers
        layer_to_extract_from = args.layer_to_extract_from
        num_layers_to_unfreeze = args.num_layers_to_unfreeze
        extractor_name = args.extractor_name
        option = args.option

        print('dataset: {}'.format(dataset_name))
        print('num layers learned from scratch: {}'.format(n_head_layers))
        print('layer to extract from: {}'.format(layer_to_extract_from))
        print('seed: {}'.format(seed))
        print('num_layers_to_unfreeze: {}'.format(num_layers_to_unfreeze))
        print('extractor name: {}'.format(extractor_name))
        print('option: {}'.format(option))
        print('use_all_extractors: {}'.format(args.use_all_extractors))
        print('optim: {}'.format(args.optim))
        if option == 'add_k':
            print('k extractor gating: {}'.format(args.k_extractor_gating))
            print('n_pseudo_attn_heads: {}'.format(args.n_pseudo_attn_heads))

    if not args.filename_prefix:
        from datetime import date
        import os
        today = str(date.today())
        if not os.path.exists(today):
            os.makedirs(today)
        filename_prefix = today + '/'  # to prepend to filenames of results
    else:
        filename_prefix = args.filename_prefix

    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    with open('cgcnn/data/sample-regression/atom_init.json') as atom_init_json:
        atom_init_dict = json.load(atom_init_json)
    _, dataset_kwargs = initialize_kwargs()

    # All our feature extractor models have the same architecture
    hyperparameter_file = open('cgcnn/data/hyperparameters.json')
    hyperparameter_dict = json.load(hyperparameter_file)
    model_kwargs = hyperparameter_dict['model_kwargs']

    # ----------------------- Get pre-trained extractor ------------------------
    if args.use_all_extractors or option == 'pairwise_TL':
        model_paths = get_all_extractors()
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

    extractors = []
    if option == 'pairwise_TL':
        extractor_path = model_paths.get(extractor_name, None)
        assert extractor_name is not None

        extractor = get_extractor(extractor_path, layer_to_extract_from,
                                  model_kwargs=model_kwargs,
                                  device='cuda' if cuda else 'cpu')
        extractors.append(extractor)
    elif option == 'concat' or option == 'ensemble' or option == 'add_k':
        for extractor_name, extractor_path in model_paths.items():
            extractor = get_extractor(
                extractor_path, layer_to_extract_from, model_kwargs,
                device='cuda' if cuda else 'cpu')
            extractors.append(extractor)
    else:
        raise NotImplementedError

    # Load data
    small_dataset_info_dict = get_small_datasets_info_dict()
    small_dataset_path = small_dataset_info_dict[dataset_name]['path']
    target_name = small_dataset_info_dict[dataset_name]['target_name']
    df = pd.read_pickle(small_dataset_path)

    df['structure'].reset_index(drop=True, inplace=True)
    df[target_name].reset_index(drop=True, inplace=True)
    structure_graphs, targets, _ = pre_extract_structure_graphs(
        X=df['structure'].tolist(),
        y=df[target_name].tolist(),
        atom_init_fea=atom_init_dict)

    task_dataset = StructureGraphsDataset(structure_graphs, targets)

    # Get train/val/test indices from file
    with open('data/matminer/saved_partition_indices/'
              'task_partition_indices_seed' + str(seed) + '_correctedExptEf.json', 'rb') as f:
        dict_of_task_indices = json.load(f)

    train_indices, val_indices, test_indices = \
        dict_of_task_indices[dataset_name]

    train_subset = Subset(task_dataset, train_indices)
    val_subset = Subset(task_dataset, val_indices)
    test_subset = Subset(task_dataset, test_indices)

    # Normalize only from train/val subsets
    normalizer = normalizer_from_subsets([train_subset, val_subset])

    file = open(
        'data/matminer/stl_baseline_val_maes/stl_small_task_val_maes_seed' +
        str(seed) + '.json')
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

    ensembled_backbone = None
    prediction_ensembler = None
    model_heads = None
    model = None
    if option == 'add_k' and args.n_pseudo_attn_heads > 0:
        assert args.k_extractor_gating > 0
        model = MultiheadedMixtureOfExpertsModel(
            num_pseudo_attention_heads=args.n_pseudo_attn_heads,
            backbones=extractors, num_out_layers=n_head_layers,
            backbone_feature_dim=n_features,
            k_experts=args.k_extractor_gating)
        param_groups = [{'params': model.non_extractor_parameters()}]
    elif option == 'pairwise_TL' or option == 'concat':
        model = MultilayerPerceptronHead(
            num_layers=n_head_layers, input_dim=n_features)
        ensembled_backbone = MixtureOfExtractors(extractors, option)
        param_groups = [
            {'params': model.parameters()},
            {'params': ensembled_backbone.non_extractor_parameters()}]
    elif option == 'ensemble':
        model_heads = nn.ModuleList([])
        for _ in extractors:
            model_head = MultilayerPerceptronHead(
                num_layers=n_head_layers, input_dim=n_features)
            model_heads.append(model_head)
        prediction_ensembler = EnsemblePredictor(len(extractors))
        param_groups = [{'params': model_heads.parameters()},
                        {'params': prediction_ensembler.parameters()}]

    if num_layers_to_unfreeze > 0:
        for extractor in extractors:
            params_to_finetune = get_parameters_to_finetune(
                extractor, num_layers_to_unfreeze, layer_to_extract_from)
            param_groups.append({'params': params_to_finetune, 'lr': 0.005})

    if cuda:
        if option == 'add_k' and args.n_pseudo_attn_heads > 0:
            model.to('cuda')
        elif option != 'ensemble':
            model.to('cuda')
            ensembled_backbone.to('cuda')
        else:
            for i, extractor in enumerate(extractors):
                extractor.to('cuda')
                model_heads[i].to('cuda')
            prediction_ensembler.to('cuda')

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            params=param_groups, lr=0.01, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            params=param_groups, lr=0.01)
    else:
        raise AttributeError
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=1000)

    # Create file for storing losses
    losses_filename = filename_prefix + option + '_transfer_to_' + \
        str(dataset_name) + '_' + layer_to_extract_from + '_' + \
        str(n_head_layers) + '_layers_' + 'seed' + str(seed) + '_loss_data.csv'
    loss_file = open(losses_filename, 'w', encoding='utf-8')
    writer = csv.writer(loss_file)
    writer.writerow(['batch_num', 'avg training loss', 'avg validation loss',
                     'stl normed val mae'])
    loss_file.close()

    val_mae_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    train_loss_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    val_loss_meter = AverageMeter(tensor=True, device='cuda' if cuda else 'cpu')
    best_val_mae = 1e10
    num_epochs_since_improvement = 0
    best_model_filename = filename_prefix + '_' + dataset_name + '_seed' + \
        str(seed) + '_best_model.pth'

    early_stopping_n_epochs = 500
    for epoch in range(1000):  # 1000 epochs
        for structures, labels, _ in train_dl:
            train(structures, labels, model, cuda, option, normalizer,
                  train_loss_meter, optimizer, extractors, ensembled_backbone,
                  prediction_ensembler, model_heads)

        for structures, labels, _ in val_dl:
            val_loss, val_mae = evaluate(
                structures, labels, model, cuda, option, normalizer, extractors,
                ensembled_backbone, prediction_ensembler, model_heads,
                test=False)

            val_loss_meter.update(val_loss.detach().clone(), labels.size(0))
            val_mae_meter.update(val_mae.detach().clone(), labels.size(0))

        lr_scheduler.step()
        val_mae = float(val_mae_meter.avg)
        if val_mae < best_val_mae:
            num_epochs_since_improvement = 0
            best_val_mae = val_mae

            if option == 'add_k' and args.n_pseudo_attn_heads > 0:
                state_dicts = model.state_dict()
            elif option != 'ensemble':
                state_dicts = ensembled_backbone.state_dict()
            else:
                state_dicts = []
                for i in range(len(extractors)):
                    state_dicts.append(
                        (extractors[i].state_dict(), model_heads[i].state_dict()))
            torch.save({
                'epoch': epoch,
                'option': option,
                'model_state_dict': state_dicts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_mae': best_val_mae}, best_model_filename)
        else:
            num_epochs_since_improvement += 1

        loss_file = open(losses_filename, 'a', encoding='utf-8')
        writer = csv.writer(loss_file)
        writer.writerow([
            epoch, float(train_loss_meter.avg), float(val_loss_meter.avg),
            float(val_mae / stl_val_mae)])
        loss_file.close()

        if num_epochs_since_improvement > early_stopping_n_epochs:  # early stopping
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
            extractors[i].load_state_dict(backbone_dict)
            model_heads[i].load_state_dict(model_dict)

    for structures, labels, _ in test_dl:
        test_mae = evaluate(
            structures, labels, model, cuda, option, normalizer, extractors,
            ensembled_backbone, prediction_ensembler, model_heads, test=True)

        test_mae_meter.update(test_mae.detach().clone(), labels.size(0))

    print('Test MAE: {}'.format(test_mae_meter.avg))

    return float(test_mae_meter.avg), float(best_val_mae / stl_val_mae)


def train(structures, labels, model, cuda, option, normalizer, train_loss_meter,
          optimizer, extractors=None, ensembled_backbone=None,
          prediction_ensembler=None, model_heads=None):
    """

    :param structures:
    :param labels:
    :param model:
    :param cuda:
    :param option:
    :param extractors:
    :param normalizer:
    :param train_loss_meter:
    :param optimizer:
    :param ensembled_backbone: Only used if 'option' == 'pairwise_TL' or
        'concat'
    :param prediction_ensembler:  Only used if 'option' == 'ensemble'
    :param model_heads: Only used if 'option' == 'ensemble'
    :return:
    """

    if model is not None:
        model.train()
    if ensembled_backbone is not None:
        ensembled_backbone.train()
    if extractors is not None:
        for extractor in extractors:
            extractor.train()

    if cuda:
        structures, labels = move_batch_to_cuda(structures, labels)

    loss_regularizer = torch.tensor([0.], device='cuda' if cuda else 'cpu')

    # get predictions
    if option == 'add_k' and args.n_pseudo_attn_heads > 0:
        predictions, loss_regularizer = model(structures)
    elif option == 'pairwise_TL' or option == 'concat':
        assert ensembled_backbone is not None
        features = ensembled_backbone(structures)
        predictions = model(features)
    elif option == 'ensemble':
        assert model_heads is not None
        assert prediction_ensembler is not None
        predictions_across_heads = []
        for i, extractor in enumerate(extractors):
            features = extractor(*structures)
            predictions = model_heads[i](features)
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


def evaluate(structures, labels, model, cuda, option, normalizer, extractors=None,
             ensembled_backbone=None, prediction_ensembler=None, model_heads=None,
             test=False):
    """

    :param structures:
    :param labels:
    :param model:
    :param cuda:
    :param option:
    :param extractors:
    :param normalizer:
    :param ensembled_backbone: Only used if 'option' == 'pairwise_TL' or
        'concat'
    :param prediction_ensembler:  Only used if 'option' == 'ensemble'
    :param model_heads: Only used if 'option' == 'ensemble'
    :param test:
    :return:
    """

    if model is not None:
        model.eval()
    if ensembled_backbone is not None:
        ensembled_backbone.eval()
    if extractors is not None:
        for extractor in extractors:
            extractor.eval()

    if cuda:
        structures, labels = move_batch_to_cuda(structures, labels)

    with torch.no_grad():
        if option == 'add_k' and args.n_pseudo_attn_heads > 0:
            predictions, _ = model(structures)
        elif option != 'ensemble':  # concat or pairwise_TL
            features = ensembled_backbone(structures)
            predictions = model(features)
        else:  # ensemble final predictions of separate models
            predictions_across_heads = []
            for i, extractor in enumerate(extractors):
                features = extractor(*structures)
                predictions = model_heads[i](features)
                if predictions.dim() == 2:
                    predictions = predictions.squeeze(1)
                predictions_across_heads.append(predictions)

            predictions_across_heads = torch.stack(
                predictions_across_heads, dim=1)
            predictions = prediction_ensembler(predictions_across_heads)

        if predictions.shape != labels.shape:
            predictions = predictions.squeeze(1)

        loss = F.mse_loss(predictions, normalizer.norm(labels))
        mae = torch.mean(torch.abs(
            normalizer.denorm(predictions) - labels))

        if not test:
            return loss.detach().clone(), mae.detach().clone()
        else:
            return mae.detach().clone()


if __name__ == '__main__':
    num_layers_to_unfreeze = 1  # number of backbone layers to unfreeze
    n_head_layers = 3  # number of head layers to train from scratch
    layer_to_extract_from = 'conv'
    small_dataset = 'jarvis_2d_exfoliation'  #'piezoelectric_tensor' #'jarvis_2d_exfoliation'
    extractor_name = 'mp_eform'  #'mp_kvrh' #'mp_eform' #'jarvis_gvrh'
    option = 'add_k'  # 'pairwise_TL'  #'concat'  # 'single'  # 'ensemble'

    test_maes, normalized_val_maes = [], []
    for seed in list(range(5)):
        print('------ seed {} -------'.format(str(seed)))
        test_mae, normalized_val_mae = main(
            dataset_name=small_dataset, n_head_layers=n_head_layers,
            layer_to_extract_from=layer_to_extract_from,
            seed=seed, num_layers_to_unfreeze=num_layers_to_unfreeze,
            extractor_name=extractor_name, option=option)
        test_maes.append(test_mae)
        normalized_val_maes.append(normalized_val_mae)

    avg_normalized_val_mae = np.mean(normalized_val_maes)
    normalized_val_mae_stdev = np.std(normalized_val_maes)
    print(avg_normalized_val_mae)
    print(normalized_val_mae_stdev)

    test_avg = np.mean(test_maes)
    test_std = np.std(test_maes)

    if not args.filename_prefix:
        from datetime import date
        import os
        today = str(date.today())
        if not os.path.exists(today):
            os.makedirs(today)
        filename_prefix = today + '/' + today + '_'
    else:
        filename_prefix = args.filename_prefix

    result_filename = filename_prefix + '_results.csv'
    result_file = open(result_filename, 'w', encoding='utf-8')
    writer = csv.writer(result_file)
    writer.writerow(['best val mae / stl val mae', '+/-', 'test_mae', '+/-'])
    writer.writerow([avg_normalized_val_mae, normalized_val_mae_stdev,
                     test_avg, test_std])
    result_file.close()
