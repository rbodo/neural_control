import numpy as np
import control as ct
import torch
import matplotlib.pyplot as plt

from src.antisymmetricRNN import AntisymmetricRNNCell
from src.antisymmetricRNN_dynamics import run_multiple, get_ev

torch.manual_seed(42)

num_inputs = 2
num_cells = 2
step_size = 0.01
gamma = 0.01
init_W_std = 1

num_steps = 1000

kp = 0.1
ki = 0.1
kd = 0

as_cell = AntisymmetricRNNCell(num_inputs, num_cells, step_size, gamma,
                               init_W_std)

# init different hidden states and zero input
x1 = [0, 0.5]
x2 = [-0.5, -0.5]
x3 = [0.5, -0.75]
starting_points = [x1, x2, x3]
u0 = [0, 0]
T = np.arange(0, num_steps * step_size, step_size)

run_multiple(as_cell, torch.Tensor([u0]), [torch.Tensor([s]) for s in starting_points], num_steps)

# ev_as = get_ev(as_cell)
# print(ev_as)


def network_update(t, x, u, params=None):
    """Network dynamics of AntisymmetricRNN cells.

    Parameters
    ----------

    t : array
        Time steps.
    x : array
        System state.
    u : array
        System input.
    params : dict
        Parameters.

    Returns
    -------

    float
        Network output
    """

    if params is None:
        params = {}

    c = params.get('cell')
    dx = c.update(torch.Tensor([u]), torch.Tensor([x]))

    return dx.detach().numpy()[0]


cell = AntisymmetricRNNCell(num_inputs, num_cells, step_size, gamma,
                            init_W_std)

network_tf = ct.NonlinearIOSystem(
    network_update, None, inputs=2, outputs=2, states=2, name='network',
    params={'cell': cell}, dt=step_size)

# x_eq, u_eq = ct.find_eqpt(network_tf, [0, 0.1], u0, [0, 0])
t, y = ct.input_output_response(network_tf, T, np.zeros((2, num_steps)), x1)
plt.scatter(y[0], y[1])
plt.show()

control_tf = ct.tf2io(ct.TransferFunction([kp, ki, kd], [1, 0.01 * ki / kp]),
                      name='control', inputs='u', outputs='y')

system_tf = ct.InterconnectedSystem(
    (network_tf, control_tf), name='system',
    connections=[('control.u', 'network.u')],
    inplist=('control.u', 'network.'),
    inputs=(),
    outlist=(),
    outputs=())

x0, u0 = ct.find_eqpt(system_tf, [])

t, y = ct.input_output_response(system_tf, T)
