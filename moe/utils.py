import copy

import torch


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
                list(backbone.fcs[-(layers_unfrozen+1)].parameters()))
            layers_unfrozen += 1

        elif layers_unfrozen < backbone.n_h and \
                (layer_to_extract_from == 'first_fc' or
                 layer_to_extract_from == 'penultimate_fc'):

            if layer_to_extract_from == 'first_fc':
                layers_unfrozen += backbone.n_h-1

            # get the next layer
            layers_to_unfreeze.extend(
                list(backbone.conv_to_fc.parameters()))
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


def normalizer_from_subsets(list_of_subsets):
    """
    Create a Normalizer object for a target variable, ignoring all labels
    corresponding to the test set.

    Args:
        list_of_subsets (list(torch.utils.data.dataset.Subset)): list of 2
            subsets (training and validation). Each Subset is assumed to be of
            the same cgcnn.head_selection_data.TaskDataset.

    Returns:
        Normalizer containing only the labels from the 2 subsets.
    """
    full_dataset_size = len(list_of_subsets[0].dataset)
    all_indices = set(range(full_dataset_size))

    assert \
        set(list_of_subsets[0].indices).isdisjoint(list_of_subsets[1].indices)
    assert all(index in all_indices for index in list_of_subsets[0].indices)
    assert all(index in all_indices for index in list_of_subsets[1].indices)

    train_and_val_indices = set()
    train_and_val_indices.update(set(list_of_subsets[0].indices))
    train_and_val_indices.update(set(list_of_subsets[1].indices))

    test_indices = torch.tensor(list(all_indices ^ train_and_val_indices))

    try:
        all_labels = copy.deepcopy(list_of_subsets[0].dataset.labels)
    except AttributeError:
        target_data = copy.deepcopy(list_of_subsets[0].dataset.target_data)
        all_labels = torch.tensor([float(y) for (_, y) in target_data])
    all_labels[test_indices] = 0

    return Normalizer(all_labels)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, tensor=False, device=None):
        self.tensor = tensor
        self.device = device
        self.reset()

    def reset(self):
        if self.tensor:
            self.val = torch.tensor(0., device=self.device)
            self.avg = torch.tensor(0., device=self.device)
            self.sum = torch.tensor(0., device=self.device)
            self.count = torch.tensor(0., device=self.device)
        else:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            n = torch.tensor(n, device=val.device)

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_small_datasets_info_dict():
    small_dataset_info = {
        'mp_phonons': {  # 1,265
            'path': 'data/matminer/matbench_log_phonons.pkl',
            'target_name': 'log(last phdos peak)', 'weighted_sampling': False},
        'mp_dielectric': {  # 4,764
            'path': 'data/matminer/matbench_log_dielectric.pkl',
            'target_name': 'log(n)', 'weighted_sampling': False},

        'expt_eform': {  # 1,709
            'path': 'data/matminer/expt_formation_enthalpy_combined_RC.pkl',
            'target_name': 'e_form expt', 'weighted_sampling': False},
        'expt_bandgap': {  # 2,481
            'path': 'data/matminer/expt_gap_kingsbury_RC.pkl',
            'target_name': 'expt_gap', 'weighted_sampling': False},

        'piezoelectric_tensor': {  # 941
            'path': 'data/matminer/log_piezoelectric_tensor_RC.pkl',
            'target_name': 'log(1 + eij_max)', 'weighted_sampling': False},

        'jarvis_3d_gap_tbmbj': {  # 7,348
            'path': 'data/matminer/jarvis_dft_3d_GapTbmbj_RC_2022-02-21_2022-04-11.pkl',
            'target_name': 'gap tbmbj', 'weighted_sampling': False},
        'jarvis_3d_eps_tbmbj': {  # 8,043
            'path': 'data/matminer/jarvis_dft_3d_EpsTbmbj_RC_2022-02-21.pkl',
            'target_name': 'epsilon_avg tbmbj', 'weighted_sampling': False},

        'jarvis_2d_exfoliation': {  # 636
            'path': 'data/matminer/jarvis_dft_2d_exfol_energy.pkl',
            'target_name': 'exfoliation_en', 'weighted_sampling': False},  # meV/at
        'jarvis_2d_eform': {  # 633
            'path': 'data/matminer/jarvis_dft_2d_eform_2022-04-11.pkl',
            'target_name': 'e_form', 'weighted_sampling': False},
        'jarvis_2d_gap_opt': {  # 522
            'path': 'data/matminer/jarvis_dft_2d_gap_opt_2022-04-11.pkl',
            'target_name': 'gap opt', 'weighted_sampling': False},
        'jarvis_2d_gap_tbmbj': {  # 120
            'path': 'data/matminer/jarvis_dft_2d_gap_tbmbj_2022-04-11.pkl',
            'target_name': 'gap tbmbj', 'weighted_sampling': False},
        'jarvis_2d_dielectric_opt': {  # 522
            'path': 'data/matminer/jarvis_dft_2d_dielectric_opt_2022-04-11.pkl',
            'target_name': 'dielectric opt', 'weighted_sampling': False},
        'jarvis_2d_dielectric_tbmbj': {  # 120
            'path': 'data/matminer/jarvis_dft_2d_dielectric_tbmbj_2022-04-11.pkl',
            'target_name': 'dielectric tbmbj', 'weighted_sampling': False},

        'mp_elastic_anisotropy': {  # 1,181
            'path': 'data/matminer/elastic_tensor_2015_elastic_anisotropy.pkl',
            'target_name': 'elastic_anisotropy', 'weighted_sampling': False},
        'mp_poisson_ratio': {  # 1,181
            'path': 'data/matminer/elastic_tensor_2015_poisson.pkl',
            'target_name': 'poisson_ratio', 'weighted_sampling': False},

        'mp_eps_electronic': {  # 1,296
            'path': 'data/matminer/log_phonon_dielectric_mp_eps_electronic.pkl',
            'target_name': 'log_eps_electronic', 'weighted_sampling': False},
        'mp_eps_total': {  # 1,296
            'path': 'data/matminer/log_phonon_dielectric_mp_eps_total.pkl',
            'target_name': 'log_eps_total', 'weighted_sampling': False},

        'mp_poly_electronic': {  # 1,056
            'path': 'data/matminer/dielectric_constant_poly_electronic.pkl',
            'target_name': 'poly_electronic', 'weighted_sampling': False},
        'mp_poly_total': {  # 1,056
            'path': 'data/matminer/dielectric_constant_poly_total.pkl',
            'target_name': 'poly_total', 'weighted_sampling': False}
    }

    return small_dataset_info
