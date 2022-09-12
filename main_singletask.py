# python main_singletask.py --path_to_partition_indices data/matminer/saved_partition_indices/all_task_partition_indices_seed0.pkl --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --h_fea_len 32 --seed 0

import argparse
import os
import shutil
import sys
import time
import json
import random
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

from cgcnn.data import CIFDataWrapper_without_shuffling
from moe.utils import get_small_datasets_info_dict, normalizer_from_subsets

from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--property_to_train', type=str,
                    default='expt_eform',
                    help='See keys of moe.utils.get_small_dataset_info_dict '
                         'for possible property names.')
parser.add_argument('--filename_prefix', default='', type=str, metavar='F',
                    help='String to prepend to output files.')
parser.add_argument('--plot', action="store_true", help='If True, plot loss '
                                                        'curves. Default False.')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression',
                    help='complete a regression or classification task (default: regression)')
parser.add_argument('--num_classes', default=0, type=int,
                    help='Number of classes (if doing classification)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=250, type=int,
                    metavar='N', help='mini-batch size (default: 250)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[800], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--use_gradient_clipping', action='store_true',
                    help='Elementwise gradient clipping.')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--checkpoint_every_200_epochs', action='store_true',
                    help='Save a model checkpoint every 200 epochs.')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train_ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train_size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val_ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val_size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test_ratio', default=0, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0)')
test_group.add_argument('--test_size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h_fea_len', default=32, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n_conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--weighted_sampling', action='store_true')

parser.add_argument('--path_to_partition_indices', default='', type=str)

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_error = 1e20
else:
    best_error = 0.

from datetime import date
today = str(date.today())
if not os.path.exists(today):
    os.makedirs(today)

if not args.filename_prefix:
    filename_prefix = today + '/MRS_singletask'  # to pre-pend to filenames of results
else:
    filename_prefix = args.filename_prefix


def main():
    global args, best_error, filename_prefix

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)  # also sets the pandas random_state

    general_kwargs, model_kwargs, dataset_kwargs, \
    dataloader_kwargs, optimizer_kwargs, scheduler_kwargs = initialize_kwargs()
    optimizer_kwargs['weight_decay'] = args.weight_decay

    model_kwargs['h_fea_len'] = args.h_fea_len
    model_kwargs['n_conv'] = args.n_conv

    # Save hyperparameters to json
    hyperparameter_dict = {
        'property_to_train': args.property_to_train,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'general_kwargs': general_kwargs,
        'model_kwargs': model_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'optimizer_kwargs': optimizer_kwargs,
        'scheduler_kwargs': scheduler_kwargs}
    with open(filename_prefix + '_hyperparameters.json', 'w+') as file:
        json.dump(hyperparameter_dict, file)

    with open('cgcnn/data/sample-regression/atom_init.json') as atom_init_json:
        atom_init_dict = json.load(atom_init_json)

    # ----- Load pickled data -----
    learning_tasks = {}
    dataset_info = get_small_datasets_info_dict()
    dataset_info = dataset_info[args.property_to_train]
    filename = dataset_info['path']
    df = pd.read_pickle(filename)

    if args.path_to_partition_indices:
        with open(args.path_to_partition_indices, 'rb') as f:
            dict_of_task_indices = json.load(f)

        task_name = args.property_to_train
        train_indices, val_indices, test_indices = \
            dict_of_task_indices[task_name]

        test_size = len(test_indices)

        dataset = CIFDataWrapper_without_shuffling(
            X=df['structure'].tolist(),
            y=df[dataset_info['target_name']].tolist(),
            atom_init_fea=atom_init_dict, **dataset_kwargs)
    else:
        raise AttributeError

    learning_tasks[args.property_to_train] = {
        'dataset': dataset,
        'mean_absolute_deviation': df[dataset_info['target_name']].mad(),
        'train_dl': None, 'val_dl': None, 'test_dl': None,
        'weighted_sampling': args.weighted_sampling}

    # print('mad: {}'.format(df[dataset_info['target_name']].mad()))
    # raise NotImplementedError

    if args.weighted_sampling:
        try:
            learning_tasks[args.property_to_train]['sample_weights'] = \
                df[dataset_info['target_name'] + '_weights']
        except:
            raise AttributeError('You set --weighted_sampling to True, but '
                                 'did not provide sample weights.')

    # ----- Create train/val/test splits and normalize each dataset -----
    collate_fn = collate_pool

    for learning_task_name, learning_task_dict in learning_tasks.items():
        dataset = learning_task_dict['dataset']

        sample_weights = None if not args.weighted_sampling else \
            learning_task_dict['sample_weights']

        # Turn a torch.utils.data.Dataset into torch.utils.data.DataLoader train
        #   /validation/test splits
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            return_test=True if test_size > 0 else False,
            train_size=args.train_size,  # optional. Overrides train_ratio.
            test_size=args.test_size,  # optional. Overrides test_ratio.
            val_size=args.val_size,  # optional. Overrides val_ratio.
            weighted_sampling=args.weighted_sampling,
            sample_weights=sample_weights,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices
        )

        learning_tasks[learning_task_name]['train_dl'] = train_loader
        learning_tasks[learning_task_name]['val_dl'] = val_loader

        # obtain target variable normalizer
        if args.task == 'classification':
            normalizer = Normalizer(torch.zeros(2))  # normalizes mean and std of tensor
            normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            normalizer = normalizer_from_subsets([train_subset, val_subset])

    # initialize model
    model = CrystalGraphConvNet(
        classification=True if args.task == 'classification'
        else False, **model_kwargs)
    if args.cuda:
        model.cuda()
    if args.use_gradient_clipping:  # Apply gradient clipping
        clip_value = 1
        for p in model.parameters():
            if p.requires_grad:
                # modify parameter gradient in-place during backward passes
                p.register_hook(lambda grad:
                                torch.clamp(grad, -clip_value, clip_value))

    if optimizer_kwargs['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), optimizer_kwargs['lr'],
                              momentum=optimizer_kwargs['momentum'],
                              weight_decay=optimizer_kwargs['weight_decay'])
    elif optimizer_kwargs['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), optimizer_kwargs['lr'],
                               weight_decay=optimizer_kwargs['weight_decay'])
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Create file for storing losses
    losses_filename = filename_prefix
    if args.resume:
        # Don't overwrite the old file
        losses_filename += '_loss_data_continued.csv'
    else:
        losses_filename += '_loss_data.csv'
    loss_file = open(losses_filename, 'w', encoding='utf-8')
    writer = csv.writer(loss_file)
    writer.writerow(['epoch', 'avg training loss', 'avg validation loss',
                     'avg training mae', 'avg validation mae'])
    loss_file.close()

    # Decay learning rate by gamma when number of training epochs have passed
    #   specified numbers/"milestones" (i.e., decays lr stepwise)
    scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1.0e-5)

    # Initialize lists for recording loss data
    epoch_idxs = np.empty(0)  # Store the epoch number
    train_losses_per_epoch = np.empty(0)  # Store each epoch's average training loss
    val_loss_per_epoch = np.empty(0)
    train_mae_errors_per_epoch = np.empty(0)
    val_mae_errors_per_epoch = np.empty(0)

    # Train/validate
    epochs_without_improvement = 0
    criterion = nn.MSELoss() if args.task == 'regression' else nn.NLLLoss()
    for epoch in range(args.start_epoch, args.epochs):  # default: 1000 epochs
        print('---------- epoch {} ----------'.format(epoch))

        # train for one epoch
        train_loss_avg, train_mae_error = train(
            train_loader, model, criterion, optimizer, epoch, normalizer)

        print('train_loss_avg: {}'.format(train_loss_avg))

        # evaluate on validation set
        val_loss_avg, mae_error = validate( # average loss of all batches in the epoch
            val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_error
            best_error = min(mae_error, best_error)
            if is_best:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            is_best = mae_error > best_error
            best_error = max(mae_error, best_error)

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)},
                is_best, filename=filename_prefix + '_checkpoint.pth.tar',
                best_model_filename=filename_prefix + '_best_model.pth.tar')

        if args.checkpoint_every_200_epochs and epoch % 200 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_mae_error': best_error}, False,
                filename=filename_prefix + '_checkpoint_epoch' + str(epoch) +
                '.pth.tar')

        # Store epoch numbers and losses for plotting
        epoch_idxs = np.append(epoch_idxs, epoch)
        train_losses_per_epoch = np.append(
            train_losses_per_epoch, train_loss_avg)  # average loss all batches in the epoch
        val_loss_per_epoch = np.append(val_loss_per_epoch, val_loss_avg)
        train_mae_errors_per_epoch = np.append(
            train_mae_errors_per_epoch, train_mae_error)
        val_mae_errors_per_epoch = np.append(
            val_mae_errors_per_epoch, mae_error
        )

        if epochs_without_improvement == 500:
            print('Early stopping!')
            break

        # Write losses to csv
        loss_file = open(losses_filename, 'a', encoding='utf-8')
        writer = csv.writer(loss_file)
        writer.writerow([epoch, train_loss_avg, val_loss_avg,
                         train_mae_error, mae_error])
        loss_file.close()

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(filename_prefix + '_best_model.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    final_test_loss, final_test_mae = validate(
        test_loader, model, criterion, normalizer, test=True)
    losses_filename = filename_prefix
    if args.resume:
        # Don't overwrite the old file
        losses_filename += '_test_performance_continued.csv'
    else:
        losses_filename += '_test_performance.csv'
    loss_file = open(losses_filename, 'w', encoding='utf-8')
    writer = csv.writer(loss_file)
    writer.writerow(['avg test loss', 'avg test MAE'])
    writer.writerow([final_test_loss, final_test_mae])
    loss_file.close()


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    """
    Original CGCNN training function.

    Returns:
        (losses.avg, mae_errors.avg)
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('training partition size: {}'.format(len(train_loader.sampler)))
    for i, (input, target, _) in enumerate(train_loader):
        # 'i' here is batch index

        # measure data loading time
        data_time.update(time.time() - end)

        # print('num crystals: {}'.format(len(input[3])))

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy = class_eval(output.data.cpu(), target, accuracy_only=True)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            # precisions.update(precision, target.size(0))
            # recalls.update(recall, target.size(0))
            # fscores.update(fscore, target.size(0))
            # auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies)
                )
    if args.task == 'regression':
        return losses.avg, mae_errors.avg
    else:
        return losses.avg, accuracies.avg


def validate(val_loader, model, criterion, normalizer, test=False):
    # This method is used for the validation and test sets (simply pass in
    # 'test_loader' instead of 'val_loader' and set 'test' to True.
    """

    :param val_loader:
    :param model:
    :param criterion:
    :param normalizer:
    :param test:

    Returns:
        If regression: losses.avg, mae_errors.avg
        If classification: auc_errors.avg
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    print('val_loader length: {}'.format(len(val_loader.sampler)))
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        # 'i' is the batch number
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy = class_eval(output.data.cpu(), target, accuracy_only=True)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            # precisions.update(precision, target.size(0))
            # recalls.update(recall, target.size(0))
            # fscores.update(fscore, target.size(0))
            # auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies))

    if test:
        star_label = '**'
        import csv
        with open(filename_prefix + 'test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return losses.avg, mae_errors.avg
    else:
        # print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
        #                                          auc=auc_scores))
        # return auc_scores.avg

        print(' {star} ACC {acc.avg:.3f}'.format(star=star_label,
                                                 acc=accuracies))
        return losses.avg, accuracies.avg


def get_input_and_target_vars(input, target, normalizer, task_type):
    if args.cuda:
        input_var = (Variable(input[0].cuda(non_blocking=True)),
                     Variable(input[1].cuda(non_blocking=True)),
                     input[2].cuda(non_blocking=True),
                     [crys_idx.cuda(non_blocking=True) for crys_idx in
                      input[3]])
    else:
        input_var = (Variable(input[0]),
                     Variable(input[1]),
                     input[2],
                     input[3])
    # normalize target
    if task_type == 'regression':
        target_normed = normalizer.norm(target)
    else:
        target_normed = target.view(-1).long()
    if args.cuda:
        target_var = Variable(target_normed.cuda(non_blocking=True))
    else:
        target_var = Variable(target_normed)

    return input_var, target_var


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


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target, accuracy_only=False):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])

    # if not accuracy_only:
    #     if prediction.shape[1] == 2:
    #         precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
    #             target_label, pred_label, average='binary')
    #         auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
    #         accuracy = metrics.accuracy_score(target_label, pred_label)
    #     else:
    #         raise NotImplementedError
    #     return accuracy, precision, recall, fscore, auc_score
    # else:
    #     accuracy = metrics.accuracy_score(target_label, pred_label)
    #     return accuracy


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
                    best_model_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def initialize_kwargs(**cgcnn_kwargs):
    """
    **Modified from matminer.featurizers.structure's CGCNNFeaturizer class

    Process and group kwargs into model_kwargs, dataset_kwargs,
    dataloader_kwargs, etc.
    Args:
        cgcnn_kwargs (dict, optional): CGCNN kwargs.

    Returns:
        general_kwargs (dict): general CGCNN settings
        model_kwargs (dict):
        dataset_kwargs (dict): kwargs to initialize CIFDataWrapper
        dataloader_kwargs (dict):
        optimizer_kwargs (dict):
        scheduler_kwargs (dict):

    """
    general_kwargs = {}
    # Initialize some common-purpose kwargs
    general_kwargs['task'] = cgcnn_kwargs.get('task', 'regression')
    general_kwargs['test'] = cgcnn_kwargs.get('test', False)
    general_kwargs['num_epochs'] = cgcnn_kwargs.get("num_epochs", 1000)
    general_kwargs['print_freq'] = cgcnn_kwargs.get('print_freq', 10)
    general_kwargs['cuda'] = torch.cuda.is_available() and \
        not cgcnn_kwargs.get("disable_cuda", False)
    general_kwargs['early_stopping_crit'] = cgcnn_kwargs.get(
        'early_stopping_crit', 500)  # stops training if validation error (not loss) does not improve after 500 epochs

    # Initialize CrystalGraphConvNet model kwargs
    model_kwargs = \
        {"orig_atom_fea_len": cgcnn_kwargs.get("orig_atom_fea_len", 92),
         "nbr_fea_len": cgcnn_kwargs.get("nbr_fea_len", 41),
         "n_conv": cgcnn_kwargs.get("n_conv", 4),
         "h_fea_len": cgcnn_kwargs.get("h_fea_len", 32),
         "n_h": cgcnn_kwargs.get("n_h", 1),
         "atom_fea_len": cgcnn_kwargs.get("atom_fea_len", 64)}

    # Initialize CIFDataWrapper (pytorch dataset) kwargs
    dataset_kwargs = \
        {  # "atom_init_fea": atom_init_fea,
         "max_num_nbr": cgcnn_kwargs.get("max_num_nbr", 12),
         "radius": cgcnn_kwargs.get("radius", 8),
         "dmin": cgcnn_kwargs.get("dmin", 0),
         "step": cgcnn_kwargs.get("step", 0.2)}

    # Initialize dataloader kwargs
    dataloader_kwargs = \
        {"batch_size": cgcnn_kwargs.get("batch_size", 256),
         "num_workers": cgcnn_kwargs.get("num_workers", 0),
         "train_size": cgcnn_kwargs.get("train_size", None),
         "val_size": cgcnn_kwargs.get("val_size", None),
         "test_size": cgcnn_kwargs.get("test_size", None),
         "return_test": general_kwargs['test'],
         "collate_fn": collate_pool,
         "pin_memory": general_kwargs['cuda']}

    # Initialize optimizer kwargs
    optimizer_kwargs = \
        {"optim": cgcnn_kwargs.get("optim", 'SGD'),
         "lr": cgcnn_kwargs.get("lr", 0.02),
         "momentum": cgcnn_kwargs.get("momentum", 0.9),
         "weight_decay": cgcnn_kwargs.get("weight_decay", 0)}

    # Initialize scheduler kwargs
    scheduler_kwargs = \
        {"gamma": cgcnn_kwargs.get("gamma", 0.1),
         "milestones": cgcnn_kwargs.get("lr_milestones", [800])}

    return general_kwargs, model_kwargs, dataset_kwargs, dataloader_kwargs, \
        optimizer_kwargs, scheduler_kwargs


if __name__ == '__main__':
    main()
