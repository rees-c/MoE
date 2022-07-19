import torch

from cgcnn.data import collate_pool


def initialize_kwargs(**cgcnn_kwargs):
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

    return model_kwargs, dataset_kwargs


def get_small_datasets_info_dict():
    small_dataset_info = {
        'mp_phonons': {  # 1,265
            'path': 'data/matbench_tasks/matbench_log_phonons.pkl',
            'target_name': 'log(last phdos peak)', 'weighted_sampling': False},
        'mp_dielectric': {  # 4,764
            'path': 'data/matbench_tasks/matbench_log_dielectric.pkl',
            'target_name': 'log(n)', 'weighted_sampling': False},

        'expt_eform': {  # 1,709
            'path': 'data/matbench_tasks/expt_formation_enthalpy_combined_RC.pkl',
            'target_name': 'e_form expt', 'weighted_sampling': False},
        'expt_bandgap': {  # 2,481
            'path': 'data/matbench_tasks/expt_gap_kingsbury_RC.pkl',
            'target_name': 'expt_gap', 'weighted_sampling': False},

        'piezoelectric_tensor': {  # 941
            'path': 'data/matbench_tasks/log_piezoelectric_tensor_RC.pkl',
            'target_name': 'log(1 + eij_max)', 'weighted_sampling': False},

        'jarvis_3d_gap_tbmbj': {  # 7,348
            'path': 'data/matbench_tasks/jarvis_dft_3d_GapTbmbj_RC_2022-02-21_2022-04-11.pkl',
            'target_name': 'gap tbmbj', 'weighted_sampling': False},
        'jarvis_3d_eps_tbmbj': {  # 8,043
            'path': 'data/matbench_tasks/jarvis_dft_3d_EpsTbmbj_RC_2022-02-21.pkl',
            'target_name': 'epsilon_avg tbmbj', 'weighted_sampling': False},

        'jarvis_2d_exfoliation': {  # 636
            'path': 'data/matbench_tasks/jarvis_dft_2d_exfol_energy.pkl',
            'target_name': 'exfoliation_en', 'weighted_sampling': False},  # meV/at
        'jarvis_2d_eform': {  # 633
            'path': 'data/matbench_tasks/jarvis_dft_2d_eform_2022-04-11.pkl',
            'target_name': 'e_form', 'weighted_sampling': False},
        'jarvis_2d_gap_opt': {  # 522
            'path': 'data/matbench_tasks/jarvis_dft_2d_gap_opt_2022-04-11.pkl',
            'target_name': 'gap opt', 'weighted_sampling': False},
        'jarvis_2d_gap_tbmbj': {  # 120
            'path': 'data/matbench_tasks/jarvis_dft_2d_gap_tbmbj_2022-04-11.pkl',
            'target_name': 'gap tbmbj', 'weighted_sampling': False},
        'jarvis_2d_dielectric_opt': {  # 522
            'path': 'data/matbench_tasks/jarvis_dft_2d_dielectric_opt_2022-04-11.pkl',
            'target_name': 'dielectric opt', 'weighted_sampling': False},
        'jarvis_2d_dielectric_tbmbj': {  # 120
            'path': 'data/matbench_tasks/jarvis_dft_2d_dielectric_tbmbj_2022-04-11.pkl',
            'target_name': 'dielectric tbmbj', 'weighted_sampling': False},

        'mp_elastic_anisotropy': {  # 1,181
            'path': 'data/matbench_tasks/elastic_tensor_2015_elastic_anisotropy.pkl',
            'target_name': 'elastic_anisotropy', 'weighted_sampling': False},
        'mp_poisson_ratio': {  # 1,181
            'path': 'data/matbench_tasks/elastic_tensor_2015_poisson.pkl',
            'target_name': 'poisson_ratio', 'weighted_sampling': False},

        'mp_eps_electronic': {  # 1,296
            'path': 'data/matbench_tasks/log_phonon_dielectric_mp_eps_electronic.pkl',
            'target_name': 'log_eps_electronic', 'weighted_sampling': False},
        'mp_eps_total': {  # 1,296
            'path': 'data/matbench_tasks/log_phonon_dielectric_mp_eps_total.pkl',
            'target_name': 'log_eps_total', 'weighted_sampling': False},

        'mp_poly_electronic': {  # 1,056
            'path': 'data/matbench_tasks/dielectric_constant_poly_electronic.pkl',
            'target_name': 'poly_electronic', 'weighted_sampling': False},
        'mp_poly_total': {  # 1,056
            'path': 'data/matbench_tasks/dielectric_constant_poly_total.pkl',
            'target_name': 'poly_total', 'weighted_sampling': False}
    }

    return small_dataset_info
