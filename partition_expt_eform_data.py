import random
import json

import pandas as pd
import numpy as np


def main(seed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Separate structures into those that the Materials Project fits its
    formation energies to experimental values and those that are not fitted.
    Return a train
    """
    random.seed(seed)
    np.random.seed(seed)  # also sets the pandas random_state

    path = '/Users/reeschang/PycharmProjects/MoE/data/matminer/expt_formation_enthalpy_combined_RC.pkl'
    df = pd.read_pickle(path)
    total_data_size = len(df.index)

    print(df.columns)
    structure = df['structure'][0]
    print(structure.species)
    print(type(structure.species[0]))
    print([element.symbol for element in structure.species])

    # See https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability
    fitted_elements = \
        ['S', 'F', 'Cl', 'Br', 'I', 'N', 'H', 'Se', 'Si', 'Sb', 'Te', 'O',
         'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'W', 'Mo']

    fitted_mpids = []
    non_fitted_mpids = []
    for idx, row in df.iterrows():
        structure = row['structure']
        structure_elements = [element.symbol for element in structure.species]
        if any(element in structure_elements for element in fitted_elements):
            fitted_mpids.append(row['mpid'])
        else:
            non_fitted_mpids.append(row['mpid'])

    print('Number of compounds possibly used for fitting: {}'.format(len(fitted_mpids)))
    print('Number of compounds definitely not used for fitting: {}'.format(len(non_fitted_mpids)))

    fitted_df = df[df['mpid'].isin(fitted_mpids)]
    non_fitted_df = df[df['mpid'].isin(non_fitted_mpids)]

    # print(list(range(1709)) == list(df.index))  # True

    fitted_idxs = df.index[df['mpid'].isin(fitted_mpids)].tolist()
    non_fitted_idxs = df.index[df['mpid'].isin(non_fitted_mpids)].tolist()

    random.shuffle(fitted_idxs)
    random.shuffle(non_fitted_idxs)

    # Sample a test set which did not leak information into the Materials
    # Project formation energy pre-training data
    test_size = int(total_data_size * test_ratio)
    train_size = int(total_data_size * train_ratio)

    test_indices = non_fitted_idxs[:test_size]

    fitted_idxs.extend(non_fitted_idxs[test_size:])
    train_and_val_indices = fitted_idxs

    train_indices = train_and_val_indices[:train_size]
    val_indices = train_and_val_indices[train_size:]

    partition_indices_path = 'data/matminer/saved_partition_indices/' \
                             'task_partition_indices_seed' + str(seed) + '.json'
    with open(partition_indices_path, 'r') as f:
        task_partition_indices = json.load(f)

    task_partition_indices['expt_eform'] = (
        train_indices, val_indices, test_indices)
    with open('data/matminer/saved_partition_indices/'
              'task_partition_indices_seed' + str(seed) +
              '_correctedExptEf.json', 'w') as f:
        json.dump(task_partition_indices, f)


if __name__ == '__main__':
    for i in range(5):
        main(seed=i)

    # import time
    # a = list(range(20))
    # b = list(range(100))
    #
    # start = time.time()
    # print(set(a).isdisjoint(set(b)))
    # end = time.time()
    # print(start-end)
    #
    # start = time.time()
    # print(any(x in a for x in b))
    # end = time.time()
    # print(start-end)
