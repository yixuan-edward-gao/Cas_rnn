"""
Defines models used for predicting Cas cut activity

Author: Edward Gao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    A convolutional layer for processing sequence one-hot maps.

    First applies an Inception layer with 3 different filter sizes.
    The outputs are concatenated, and each channel is max-pooled.
    The resulting vector is then processed by a linear layer.
    """
    def __init__(self, seq_len, height, out_c, out_size, drop_prob):
        """
        Initializes layer

        :param seq_len: length (width) of input one-hot map
        :param height: height of input one-hot map
        :param out_c: number of output channels
        :param out_size: dimension of final output vector
        :param drop_prob: drop out probability
        """
        super(Conv, self).__init__()
        self.convs = [nn.Conv2d(1, out_c, kernel_size=(height, 3)),
                      nn.Conv2d(1, out_c, kernel_size=(height, 5)),
                      nn.Conv2d(1, out_c, kernel_size=(height, 7))]
        self.maxpool = nn.MaxPool2d(kernel_size=(1, seq_len * 3 - 12))
        self.proj = nn.Linear(out_c, out_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # x is (batch_size, height, seq_len)
        x = x.unsqueeze(1)
        x = torch.cat([m(x) for m in self.convs], dim=3)
        x = self.maxpool(x).squeeze()     # (batch_size, out_c)
        x = F.relu(self.proj(x))    # (batch_size, out_size)
        return self.dropout(x)


class ConvModel(nn.Module):
    """
    Convolutional model for predicting Cas cut rates.
    """
    def __init__(self, conv_ch, conv_out, drop_prob):
        """
        Initializes model.

        :param conv_ch: output channel of convolutional layer
        :param conv_out: dimension of vectors output by convolutional layer
        :param drop_prob: drop out probability
        """
        super(ConvModel, self).__init__()
        self.nt_conv = Conv(30, 4, conv_ch, conv_out, drop_prob)
        self.di_nt_conv = Conv(29, 16, conv_ch, conv_out, drop_prob)
        self.proj = nn.Linear(conv_out * 2 + 4 + 10, 50)
        self.out = nn.Linear(50, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        nt, di_nt, pam, vals = x
        nt_vec = self.nt_conv(nt)
        di_nt_vec = self.di_nt_conv(di_nt)
        x = torch.cat([nt_vec, di_nt_vec, pam, vals], dim=1)
        x = self.proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        y = self.out(x)
        return y.squeeze()


class RNNModel(nn.Module):
    """
    RNN-based model for predicting Cas cut rates.
    """
    def __init__(self, emb_size, hidden_size, drop_prob):
        """
        Initializes model.

        :param emb_size: size of embedding vectors
        :param hidden_size: size of RNN hidden state
        :param drop_prob: drop out probability
        """
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(4, emb_size)
        self.rnn = nn.GRU(input_size=16, hidden_size=hidden_size,
                          batch_first=True, bidirectional=True, dropout=drop_prob)
        self.proj = nn.Linear(2 * hidden_size + 14, 50)
        self.out = nn.Linear(50, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        seq, pam, vals = x
        _, seq_vec = self.rnn(self.embed(seq))
        seq_vec = seq_vec.transpose(0, 1)
        seq_vec = seq_vec.reshape(seq.shape[0], -1)
        x = torch.cat([seq_vec, pam, vals], dim=1)
        x = self.proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        y = self.out(x)
        return y.squeeze()
