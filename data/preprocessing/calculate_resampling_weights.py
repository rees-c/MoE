import pandas as pd
import numpy as np
import torch


def main():
    """
    SAMPLE SCRIPT FOR COMPUTING SAMPLE WEIGHTS.
    """
    path = 'matminer/ricci_boltztrap_mp_tabular_Cond_&_Seebeck_RC_2022-02-21.pkl'
    skew_right = True
    df = pd.read_pickle(path)

    labels_vars = [
        'log(1+σ.n) [log 1/Ω/m/s]', 'log(1+σ.p) [log 1/Ω/m/s]',
        'log(1+κₑ.n) [log W/K/m/s]', 'log(1+κₑ.p) [log W/K/m/s]']
    for labels_var in labels_vars:

        pred_labels = df[labels_var]
        labels_min = min(pred_labels)
        labels_max = max(pred_labels)
        num_bins = 30
        # bin_edges = np.arange(
        #     start=labels_min-1, stop=labels_max,
        #     step=(labels_max-labels_min+1)/num_bins)
        bin_edges = np.linspace(labels_min-1, stop=labels_max, num=num_bins+1)

        labels = np.arange(start=0, stop=int(num_bins))
        df[labels_var + '_bins'] = pd.cut(pred_labels, bin_edges, labels=labels)
        weights = df[labels_var + '_bins'].value_counts()
        weights = 1/weights
        weights = weights.replace(np.inf, 0)
        weights = weights/sum(weights)
        weights = weights.to_dict()

        weights_column = []
        for i in df.index:
            weights_column.append(weights[df[labels_var + '_bins'].loc[i]])
        df[labels_var + '_weights'] = weights_column

    df.to_pickle(path[:-4] + '_with_bins.pkl')


if __name__ == '__main__':
    main()

    path = 'data/matbench_tasks/ricci_boltztrap_mp_tabular_RC_with_bins.pkl'
    df = pd.read_pickle(path)
    print(df.head())
    print(df.columns)
