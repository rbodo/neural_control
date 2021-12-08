import control
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import mxnet as mx

from src.double_integrator.utils import get_additive_white_gaussian_noise, RNG

sns.set_theme(style='whitegrid')

# Map internal dimension identifiers to more descriptive labels.
DIMENSION_MAP = {'y0': 'control', 'x0': 'x', 'x1': 'v', 'x2': r'$\hat{x}$',
                 'x3': r'$\hat{v}$', 'c0': 'cost'}


# noinspection PyUnusedLocal
def process_dynamics(t, x, u, params):
    A = params['A']
    B = params['B']
    W = params['W']

    u = np.atleast_1d(u)  # Input u can be scalar 0 if not specified.

    dx = A.dot(x) + B.dot(u) + get_additive_white_gaussian_noise(W, rng=RNG)

    return dx


# noinspection PyUnusedLocal
def process_output(t, x, u, params):
    C = params['C']
    D = params['D']
    V = params.get('V', None)

    u = np.atleast_1d(u)  # Input u can be scalar 0 if not specified.

    # Compute output:
    y = C.dot(x) + D.dot(u)

    # Add observation noise:
    if V is not None:
        if '_v' in params.keys():
            v = params['_v']
        else:
            v = get_observation_noise(V)
        y += v

    return y


# noinspection PyUnusedLocal
def lqr_controller_output(t, x, u, params):
    # Receives as input either the perfectly observed or estimated process
    # states.
    return -params['K'].dot(u)


# noinspection PyUnusedLocal
def lqe_dynamics(t, x, u, params):
    """Continuous-time Kalman-Bucy Filter."""

    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']
    L = params['L']

    # Noisy and partial process state observation:
    y = u

    # Compute control input (note that we feed in estimated states x as u):
    _u = lqr_controller_output(t, x, x, params)

    # Updated estimate:
    dx = A.dot(x) + B.dot(_u) + L.dot(y - C.dot(x) - D.dot(_u))

    return dx


# noinspection PyUnusedLocal
def lqe_filter_output(t, x, u, params):
    # Direct passthrough of estimated process states.
    return x


# noinspection PyUnusedLocal
def rnn_controller_dynamics(t, x, u, params):
    rnn = params['rnn']
    # Add dummy dimensions for shape [num_timesteps, batch_size, num_states].
    _u = mx.nd.array(np.expand_dims(u, [0, 1]))
    # Add dummy dimensions for shape [num_layers, batch_size, num_states].
    _x = mx.nd.array(np.reshape(x, (-1, 1, rnn.num_hidden)))
    y, x_new = rnn(_u, _x)
    dxdt = (x_new[0].asnumpy().ravel() - x) / params['dt']
    return dxdt


# noinspection PyUnusedLocal
def rnn_controller_output(t, x, u, params):
    rnn = params['rnn']
    # Add dummy dimensions for shape [num_timesteps, batch_size, num_states].
    _u = mx.nd.array(np.expand_dims(u, [0, 1]))
    # Add dummy dimensions for shape [num_layers, batch_size, num_states].
    _x = mx.nd.array(np.reshape(x, (-1, 1, rnn.num_hidden)))
    y, x_new = rnn(_u, _x)
    return y.asnumpy()[0, 0]


# noinspection PyUnusedLocal
def mlp_controller_output(t, x, u, params):
    mlp = params['mlp']
    # Add dummy dimension for batch size.
    _u = mx.nd.array(np.expand_dims(u, 0))
    y = mlp(_u)
    return y.asnumpy()[0]


class StochasticInterconnectedSystem(control.InterconnectedSystem):
    """Stochastic version of an `InterconnectedSystem`.

    Main difference: When we make noisy observations of a system output as in
    LQE, the `_compute_static_io` method will fail with an "Algrebraic loop
    detection" error. This exception is raised because the method assumes
    constant output when injecting a constant input and stepping through the
    system until steady state is reached. To maintain this assumption, we fix
    the observation noise while computing the static response.
    """

    # noinspection PyProtectedMember
    def _compute_static_io(self, t, x, u):
        # Loop over all systems to find the ones with observation noise.
        for sys in self.syslist:
            V = sys.params.get('V', None)
            # Freeze the noise vector applied to the system output at the
            # current time step. Necessary because the solver might call the
            # dynamics function multiple times per step.
            if V is not None:
                if t != sys._current_params.get('_t', -1):
                    v = get_observation_noise(V)
                    sys._update_params({'_v': v, '_t': t})

        # Call super to compute the static response to the current input.
        out = super()._compute_static_io(t, x, u)

        return out


def plot_timeseries(t, u=None, y=None, x=None, c=None, dimension_map=None,
                    path=None):
    # Map internal variable names to more descriptive labels.
    title_map = {'u': 'Inputs', 'y': 'Outputs', 'x': 'States', 'c': 'Cost'}

    # Populate pandas data container.
    dfs = []
    titles = []
    for label, data in [('u', u), ('y', y), ('x', x), ('c', c)]:

        if data is None:
            continue

        data = np.atleast_2d(data)  # Shape [num_dimensions, num_timesteps]

        # Create a new column for each dimension of the vector variable.
        columns = [label+str(i) for i in range(len(data))]

        # Apply human-readable labels.
        if dimension_map is not None:
            columns = [dimension_map.get(ll, ll) for ll in columns]

        d = pd.DataFrame(data.T, columns=columns)

        # Add list of time points.
        d['t'] = t

        # Reshape to create a single column of measurements.
        d = d.melt(id_vars=['t'], value_vars=columns, value_name='value',
                   var_name='dimension')

        # Add a column with the name of the current variable.
        d['variable'] = label

        dfs.append(d)
        titles.append(title_map[label])

    df = pd.concat(dfs, ignore_index=True)

    # Create line plot with one subplot for each variable and one line for each
    # dimension.
    g = sns.relplot(data=df, x='t', y='value', hue='dimension', row='variable',
                    kind='line', aspect=2, height=10/len(titles))

    # Add borders.
    sns.despine(right=False, top=False)

    # noinspection PyProtectedMember
    g._legend.set_title(None)

    for title, ax in zip(titles, g.axes.ravel()):
        ax.set_title(title)

    if path is not None:
        g.savefig(path, bbox_inches='tight')
    plt.show()


def get_observation_noise(V):
    return get_additive_white_gaussian_noise(V, rng=RNG)