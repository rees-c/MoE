# Code adapted from https://github.com/txie-93/cgcnn.git
import functools
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, \
    WeightedRandomSampler


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomArrayInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Args:
        elem_embedding_file (str): The path to the .json file
    """
    def __init__(self, elem_embedding):
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomArrayInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def pre_extract_structure_graphs(
        X, y, atom_init_fea, max_num_nbr=12, radius=8, dmin=0, step=0.2):
    """
    Args:
        X (Series/list): An iterable of pymatgen Structure objects.
        y (Series/list): target property that CGCNN is to predict.
        atom_init_fea (dict): A dict of {atom type: atom feature}.
        max_num_nbr (int): The max number of every atom's neighbors.
        radius (float): Cutoff radius for searching neighbors.
        dmin (int): The minimum distance for constructing GaussianDistance.
        step (float): The step size for constructing GaussianDistance.
    """
    max_num_nbr = max_num_nbr
    radius = radius
    structures = X
    non_tensor_targets = y
    ari = AtomCustomArrayInitializer(atom_init_fea)
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    structure_graphs, targets, idxs = [], [], []
    for idx in range(len(X)):
        target = non_tensor_targets[idx]
        atom_idx = idx
        crystal = structures[atom_idx]
        atom_fea = np.vstack(
            [ari.get_atom_fea(crystal[i].specie.number)
             for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(radius,
                                             include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                warnings.warn(
                    '{} not find enough neighbors to build graph. '
                    'If it happens frequently, consider increase '
                    'radius.'.format(atom_idx))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        structure_graphs.append((atom_fea, nbr_fea, nbr_fea_idx))
        targets.append(target)
        idxs.append(atom_idx)

    return structure_graphs, targets, idxs


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False,
                              weighted_sampling=False, sample_weights=None,
                              train_indices=None, val_indices=None,
                              test_indices=None, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    indices = list(range(total_size))

    if train_indices is None or val_indices is None or test_indices is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print('[Warning] train_ratio is None, using all training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

        if kwargs.get('train_size', False):
            train_size = kwargs['train_size']
        else:
            train_size = int(train_ratio * total_size)
        if kwargs.get('test_size', False):
            test_size = kwargs['test_size']
        else:
            test_size = int(test_ratio * total_size)
        if kwargs.get('val_size', False):
            valid_size = kwargs['val_size']
        else:
            valid_size = int(val_ratio * total_size)

        train_indices = indices[:train_size]
        if return_test:
            val_indices = indices[-(valid_size + test_size):-test_size]
            test_indices = indices[-test_size:]
        else:
            val_indices = indices[-valid_size:-1]

    if not weighted_sampling:
        train_sampler = SubsetRandomSampler(train_indices)
    else:
        assert sample_weights is not None
        for idx in indices:
            if idx not in train_indices:
                sample_weights[idx] = 0
        train_sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights), len(sample_weights),
            replacement=True)  # w/ replacement so you can oversample

    val_sampler = SubsetRandomSampler(val_indices)
    if return_test:
        test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)

    # Using custom samplers instead of the DataLoader's built-in 'shuffle()'
    # results in each DataLoader holding the entire dataset, but having length
    # equal to only a subset of the dataset (the length of the sampler)

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, None


class CIFDataWrapper_without_shuffling(Dataset):
    """
    Wrapper for a dataset containing pymatgen Structure objects.
    This is modified from CGCNN repo's CIFData for wrapping dataset where the
    structures are stored in CIF files.
    As we already have X as an iterable of pymatgen Structure objects, we can
    use this wrapper instead of CIFData.
    """
    def __init__(self, X, y, atom_init_fea, max_num_nbr=12, radius=8,
                 dmin=0, step=0.2, random_seed=123):
        """
        Args:
            X (Series/list): An iterable of pymatgen Structure objects.
            y (Series/list): target property that CGCNN is to predict.
            atom_init_fea (dict): A dict of {atom type: atom feature}.
            max_num_nbr (int): The max number of every atom's neighbors.
            radius (float): Cutoff radius for searching neighbors.
            dmin (int): The minimum distance for constructing GaussianDistance.
            step (float): The step size for constructing GaussianDistance.
            random_seed (int): Random seed for shuffling the dataset.
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.target_data = list(zip(range(len(y)), y))
        self.structures = X
        self.ari = AtomCustomArrayInitializer(atom_init_fea)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.target_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        atom_idx, target = self.target_data[idx]
        crystal = self.structures[atom_idx]
        atom_fea = np.vstack(
            [self.ari.get_atom_fea(crystal[i].specie.number)
             for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius,
                                             include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    '{} not find enough neighbors to build graph. '
                    'If it happens frequently, consider increase '
                    'radius.'.format(atom_idx))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, atom_idx


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Rees:
    n_i := # of atoms in crystal i
    M := # of neighbors considered for each atom
    N0 := # of crystals in the batch
    N := # of atoms in the batch

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of (torch.LongTensor)s of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N0, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids
