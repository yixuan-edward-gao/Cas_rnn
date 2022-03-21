"""
Utility functions and classes

Author: Edward Gao
"""
import torch
import torch.utils.data as data
import numpy as np


def generate_Nt_1hot(sequence):
    """
    Generates a one-hot encoding of a DNA sequence.

    :param sequence: a DNA sequence (containing ACGT)
    :return: one-hot torch Tensor of shape (4, len(sequence))
    """
    n2i = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    img = torch.zeros((4, len(sequence)))
    
    for i, nt in enumerate(sequence):
        img[n2i[nt], i] = 1
    
    return img


def generate_DiNt_1hot(sequence):
    """
    Generates a one-hot encoding of dinucleotides in a DNA sequence.

    :param sequence: a DNA sequence (containing ACGT)
    :return: one-hot torch Tensor of shape (16, len(sequence) - 1)
    """
    di_nt = [''.join([x, y]) for x in 'ACGT' for y in 'ACGT']
    di_nt2i = {n: i for i, n in enumerate(di_nt)}
    
    img = torch.zeros((16, len(sequence) - 1))
    for i in range(len(sequence) - 1):
        img[di_nt2i[sequence[i:i+2]], i] = 1
    
    return img


def generate_pam_1hot(pam):
    """
    Generates a one-hot encoding of a PAM sequence

    :param pam: a PAM sequence
    :return: one-hot torch Tensor of shape (4)
    """
    n2i = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    img = torch.zeros(4)
    img[n2i[pam[0]]] = 1
    return img


def generate_idx(sequence):
    """
    Converts a DNA sequence of a sequence of indices

    :param sequence: a DNA sequence
    :return: a torch Tensor of shape (len(sequence))
    """
    n2i = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return torch.Tensor([n2i[n] for n in sequence]).int()


class CasDataset(data.Dataset):
    """
    Dataset used for training Cas activity prediction models
    """
    def __init__(self, data, model):
        """
        Initializes dataset.

        :param data: Pandas DataFrame containing data
        :param model: name of model (either 'conv' or 'rnn')
        """
        self.data = data
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data['Sequence'].values[idx]
        y = self.data['BackgroundCorrectedCutRate'].values[idx]
        pam = self.data['TargetPam'].values[idx]
        vals = torch.Tensor(self.data.iloc[idx, 4:].values.astype(np.float))

        if self.model == 'conv':
            return (generate_Nt_1hot(sequence), generate_DiNt_1hot(sequence), generate_pam_1hot(pam), vals), y
        elif self.model == 'rnn':
            return (generate_idx(sequence), generate_pam_1hot(pam), vals), y
        else:
            raise ValueError(f'no model named {self.model}')
