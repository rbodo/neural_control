import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from src.antisymmetricRNN import AntisymmetricRNNCell, \
    AntisymmetricGatingRNNCell


def run_single(model, _x, h, _num_steps, is_lstm=False):
    data = []
    for _ in range(_num_steps):
        out = model(_x, (h, _x) if is_lstm else h)
        h = out[0] if is_lstm else out
        data.append(h.detach().numpy())
    return np.concatenate(data)


def run_multiple(model, _x, h_list, _num_steps, is_lstm=False):
    with torch.no_grad():
        out = []
        for h in h_list:
            out.append(run_single(model, _x, h, _num_steps, is_lstm))
    plot_state_space(out)


def plot_state_space(data):
    plt.figure(figsize=(10, 10))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Neuron 1")
    plt.ylabel("Neuron 2")
    for d in data:
        plt.scatter(d[:, 0], d[:, 1], s=1)
    plt.show()


def get_ev(model):
    M = model.W.detach().numpy() - model.W.detach().numpy().transpose() - \
        model.gamma_I.detach().numpy()

    return np.linalg.eigvals(M)


def main():
    num_steps = 15000
    as_cell = AntisymmetricRNNCell(2, 2, 0.01, 0.01, 1.0)
    as_cell_g = AntisymmetricGatingRNNCell(2, 2, 0.01, 0.01, 1.0)
    as_cell_g.Vz_bias = nn.Parameter(torch.Tensor([0, 0]), requires_grad=False)

    lstm_cell = nn.LSTMCell(2, 2, bias=False)
    gru_cell = nn.GRUCell(2, 2, bias=False)
    rnn_cell = nn.RNNCell(2, 2, bias=False)

    # init different hidden states and zero input
    h1 = torch.Tensor([[0, 0.5]])
    h2 = torch.Tensor([[-0.5, -0.5]])
    h3 = torch.Tensor([[0.5, -0.75]])
    starting_points = [h1, h2, h3]
    x = torch.zeros(1, 2)

    run_multiple(as_cell, x, starting_points, num_steps)
    run_multiple(as_cell_g, x, starting_points, num_steps)
    run_multiple(gru_cell, x, starting_points, num_steps)
    run_multiple(lstm_cell, x, starting_points, num_steps, is_lstm=True)
    run_multiple(rnn_cell, x, starting_points, num_steps)

    # Eigen values of hidden state transformation matrix in antisymmetric rnn
    # cells
    ev_as = get_ev(as_cell)
    print(ev_as)

    # If gamma is equal to zero
    as_cell_zero_gamma = AntisymmetricRNNCell(2, 2, 0.01, 0, 1.0)
    run_multiple(as_cell_zero_gamma, x, starting_points, num_steps)
    ev_as_zero_gamma = get_ev(as_cell_zero_gamma)
    print(ev_as_zero_gamma)

    # Hidden states still converge to (0, 0) point even when eigenvalues have
    # real parts equal to zero. This happens because of Euler discritization
    # scheme instability. If one will decrease eps and increase number of
    # steps, following hidden state evolution will be observed:
    as_cell_zero_gamma = AntisymmetricRNNCell(2, 2, 0.0001, 0, 1.0)
    run_multiple(as_cell_zero_gamma, x, starting_points[:1], num_steps)

    # Also, interesting evolution patterns can be obtained through (eps, gamma,
    # init_W_std) grid search. For example:
    as_cell = AntisymmetricRNNCell(2, 2, 0.1, 0.01, 1.0)
    run_multiple(as_cell, x, starting_points, num_steps)
    ev_as = get_ev(as_cell)
    print(ev_as)


if __name__ == '__main__':
    main()
