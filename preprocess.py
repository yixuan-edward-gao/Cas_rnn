"""
Preprocesses data.

Author: Edward Gao
"""

import pandas as pd
from os.path import exists


def preprocess():
    """
    Preprocesses (if necessary) and loads data

    :return: Pandas DataFrame containing data
    """
    if exists('data_filtered.csv'):
        return pd.read_csv('data_filtered.csv')
    else:
        data = pd.read_csv('Kim_2019_TrainingData-features.csv')
        # remove one-hot features... will re calculate
        data = data.iloc[:, :18]
        # remove lower counts
        data = data[data['TotalReads'] >= 1000]
        data = data[data['CutReads'] >= 100]
        # drop redundant columns
        data = data.drop(columns=data.columns[1:5])
        # normalize
        cols = data.iloc[:, 3:]
        data.iloc[:, 3:] = (cols - cols.min()) / (cols.max() - cols.min())
        data = data.reset_index()
        data.to_csv('data_filtered.csv', index=False)
        return data
