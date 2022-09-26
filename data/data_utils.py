import torch


class StructureGraphsDataset(torch.utils.data.Dataset):
    def __init__(self, structure_graphs, targets):
        self.structure_graphs = structure_graphs
        self.labels = torch.tensor(targets)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.structure_graphs[idx], self.labels[idx], idx


def get_all_extractors():
    model_paths = {
        'mp_eform':
            'cgcnn/data/saved_extractors/2022-01-26-15:48_singletask_eform_32hfealen_best_model.pth.tar',
        'mp_bandgap':
            'cgcnn/data/saved_extractors/2022-01-26-15:38_singletask_bandgap_32hfealen_best_model.pth.tar',
        'mp_gvrh':
            'cgcnn/data/saved_extractors/2022-01-26-19:08_singletask_gvrh_32hfealen_best_model.pth.tar',
        'mp_kvrh':
            'cgcnn/data/saved_extractors/2022-03-27-15:50_singletask_32hfl_mpkvrh_best_model.pth.tar',
        'weighted_n_e_cond':  # n-type electronic conductivity
            'cgcnn/data/saved_extractors/2022-02-03-21:12_singletask_necond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_p_e_cond':  # p-type electronic conductivity
            'cgcnn/data/saved_extractors/2022-02-04-00:05_singletask_pecond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_n_th_cond':  # n-type electronic thermal conductivity
            'cgcnn/data/saved_extractors/2022-02-04-03:02_singletask_nthcond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'weighted_p_th_cond':  # p-type electronic thermal conductivity
            'cgcnn/data/saved_extractors/2022-02-04-05:50_singletask_pthcond_32hfealen_sgd_weightedsampling_best_model.pth.tar',
        'p_Seebeck':
            'cgcnn/data/saved_extractors/2022-02-26-04:44_singletask_32hfl_pSeebeck_best_model.pth.tar',
        'n_Seebeck':
            'cgcnn/data/saved_extractors/2022-02-26-02:00_singletask_32hfl_nSeebeck_best_model.pth.tar',
        'n_avg_eff_mass':  # n-type average electron effective mass
            'cgcnn/data/saved_extractors/2022-02-25-19:05_singletask_32hfl_nEffMass_best_model.pth.tar',
        'p_avg_eff_mass':  # p-type average electron effective mass
            'cgcnn/data/saved_extractors/2022-02-25-19:06_singletask_32hfl_pEffMass_best_model.pth.tar',
        'castelli_eform':  # perovskite formation energies
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


def move_batch_to_cuda(structures, labels):
    structures = (structures[0].cuda(non_blocking=True),
                  structures[1].cuda(non_blocking=True),
                  structures[2].cuda(non_blocking=True),
                  [crys_idx.cuda(non_blocking=True) for crys_idx in
                   structures[3]])
    labels = labels.cuda(non_blocking=True)
    return structures, labels
