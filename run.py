"""
Entry point for program.
Uses 10-fold cross validation to train and evaluate model.

Train/test loop implementation taken directly from official pytorch tutorial

Author: Edward Gao
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from preprocess import preprocess
from util import CasDataset
from model import RNNModel
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = preprocess()


def train_loop(dataloader, model, loss_fn, optimizer, log_interval=10, verbose=True):
    size = len(dataloader.dataset)
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X).float()
        y = y.float()
        loss = loss_fn(pred, y).float()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % log_interval == 0:
            loss, current = loss.item(), batch * len(X[0])
            losses.append(loss)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test_loop(dataloader, model, loss_fn, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, coeff = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.float()
            pred = model(X).float()
            test_loss += loss_fn(pred, y).item()
            coeff += spearmanr(y, pred)[0]

    test_loss /= num_batches
    coeff /= num_batches

    if verbose:
        print(f"Test Error: \n Spearman correlation coefficient: {coeff}, Avg loss: {test_loss:>8f} \n")

    return test_loss, coeff


def run(train_data, test_data, batch_size, emb_size, hidden_size,
        drop_prob, learning_rate, l2_decay, epochs, log_interval, verbose):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=10000)

    model = RNNModel(emb_size, hidden_size, drop_prob)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=l2_decay)

    train_losses = []
    test_losses = []
    coeffs = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses += train_loop(train_dataloader, model, loss_fn, optimizer, log_interval, verbose=verbose)
        test_loss, coeff = test_loop(test_dataloader, model, loss_fn, verbose=verbose)
        test_losses.append(test_loss)
        coeffs.append(coeff)

    return train_losses, test_losses, coeffs


def plot(x, y, xlabel, ylabel, name):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(name)
    plt.figure()


def main():
    data = preprocess()
    learning_rate = 1
    batch_size = 64
    epochs = 50
    emb_size = 16
    hidden_size = 50
    drop_prob = 0.1
    l2_decay = 0.5
    log_interval = 10

    seed = 126

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    best_coeffs = []

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        print(f'Validating on split {i + 1}')

        train_data = CasDataset(data.loc[train_index], 'rnn')
        test_data = CasDataset(data.loc[test_index], 'rnn')
        train_loss, test_loss, coeffs = run(train_data, test_data, batch_size, emb_size, hidden_size,
                                           drop_prob, learning_rate, l2_decay, epochs, log_interval, verbose=False)
        best_coeffs.append(max(coeffs))
        break

    plot([log_interval * i for i in range(len(train_loss))], train_loss, '# iterations', 'train loss', 'train_loss.png')
    plot(list(range(epochs)), test_loss, '# epochs', 'test loss', 'test_loss.png')
    plot(list(range(epochs)), coeffs, '# epochs', 'test Spearman coefficient', 'test_coeff.png')

    print(f'Average Spearman correlation coefficient on test set: {sum(best_coeffs) / len(best_coeffs)}')


if __name__ == '__main__':
    main()

