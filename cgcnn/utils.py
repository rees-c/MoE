import torch

from cgcnn.data import collate_pool


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
    # Initialize some common-purpose kwargs
    general_kwargs = \
        {'resume': cgcnn_kwargs.get('resume', ''),  # path to checkpoint (str)
         'task': cgcnn_kwargs.get('task', 'regression'),
         'test_ratio': cgcnn_kwargs.get('test_ratio', 0),
         'num_epochs': cgcnn_kwargs.get("num_epochs", 1000),
         'print_freq': cgcnn_kwargs.get('print_freq', 10),
         'cuda': torch.cuda.is_available() and not
            cgcnn_kwargs.get("disable_cuda", False),
         'early_stopping_crit': cgcnn_kwargs.get(
             'early_stopping_crit', 5000),  # stops training if validation error (not loss) does not improve after 500 epochs
         'transfer': cgcnn_kwargs.get('transfer', False),  # whether we are in the adaptation step of transfer learning
         'plot': cgcnn_kwargs.get('plot', True),  # whether to save loss/error data
         'start_epoch': cgcnn_kwargs.get('start_epoch', 0),
         'num_batches_per_epoch': cgcnn_kwargs.get('num_batches_per_epoch',
                                                   None),  # for metalearning
         'rng_seed': cgcnn_kwargs.get('rng_seed', 0),
         'use_gradient_clipping': cgcnn_kwargs.get('use_gradient_clipping',
                                                   False),
         'use_fomaml': cgcnn_kwargs.get('use_fomaml', False),
         'num_iters_to_track_for_gradnorm': cgcnn_kwargs.get(
             'num_iters_to_track_for_gradnorm', 1),
         'support_size': cgcnn_kwargs.get('support_size', None),
         'query_size': cgcnn_kwargs.get('query_size', None)
         }

    # Initialize CrystalGraphConvNet model kwargs
    model_kwargs = \
        {"orig_atom_fea_len": cgcnn_kwargs.get("orig_atom_fea_len", 92),
         "nbr_fea_len": cgcnn_kwargs.get("nbr_fea_len", 41),
         "atom_fea_len": cgcnn_kwargs.get("atom_fea_len", 64),
         "n_conv": cgcnn_kwargs.get("n_conv", 4),
         "h_fea_len": cgcnn_kwargs.get("h_fea_len", 32),
         "n_h": cgcnn_kwargs.get("n_h", 1),
         "classification": cgcnn_kwargs.get("classification", False)
         }

    # Initialize CIFDataWrapper (pytorch dataset) kwargs
    dataset_kwargs = \
        {  # "atom_init_fea": atom_init_fea,
         "max_num_nbr": cgcnn_kwargs.get("max_num_nbr", 12),
         "radius": cgcnn_kwargs.get("radius", 8),
         "dmin": cgcnn_kwargs.get("dmin", 0),
         "step": cgcnn_kwargs.get("step", 0.2),
         "random_seed": cgcnn_kwargs.get("random_seed", 123)}

    # Initialize dataloader kwargs
    dataloader_kwargs = \
        {"batch_size": cgcnn_kwargs.get("batch_size", 256),
         "num_workers": cgcnn_kwargs.get("num_workers", 0),
         "train_ratio": cgcnn_kwargs.get("train_ratio", 0.6),
         "val_ratio": cgcnn_kwargs.get("val_ratio", 0.2),
         # "return_test": general_kwargs['test_ratio'] > 0,
         "collate_fn": collate_pool,
         "pin_memory": general_kwargs['cuda']}

    # Initialize optimizer kwargs
    optimizer_kwargs = \
        {"optim": cgcnn_kwargs.get("optim", 'SGD'),
         "lr": cgcnn_kwargs.get("lr", 0.02),
         "momentum": cgcnn_kwargs.get("momentum", 0.9),
         "weight_decay": cgcnn_kwargs.get("weight_decay", 0),
         "n_inner_iter": cgcnn_kwargs.get("n_inner_iter", 2)}  # for MAML

    # Initialize scheduler kwargs
    scheduler_kwargs = \
        {"scheduler": cgcnn_kwargs.get("lr_scheduler", 'MultiStepLR'),
         'MultiStepLR_kwargs':
             {"gamma": cgcnn_kwargs.get("gamma", 0.1),
              "milestones": cgcnn_kwargs.get("lr_milestones", [800])},
         'CosineAnnealingLR_kwargs':  # MAML++ recommends this scheduler
             {'eta_min': 1.0e-5}
         }

    return general_kwargs, model_kwargs, dataset_kwargs, dataloader_kwargs, \
           optimizer_kwargs, scheduler_kwargs
