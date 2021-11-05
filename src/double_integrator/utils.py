import control
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_theme(style='whitegrid')

RNG = np.random.default_rng(42)
RNG.standard_normal()

# Map internal dimension identifiers to more descriptive labels.
DIMENSION_MAP = {'y0': 'control', 'x0': 'x', 'x1': 'v', 'x2': r'$\hat{x}$',
                 'x3': r'$\hat{v}$', 'c0': 'cost'}


def get_additive_white_gaussian_noise(cov, size=None, rng=None,
                                      method='cholesky'):
    if rng is None:
        rng = np.random.default_rng()

    return get_gaussian_noise(np.zeros(len(cov)), cov, size, rng, method)


def get_gaussian_noise(mean, cov, size=None, rng=None, method='cholesky'):
    if rng is None:
        rng = np.random.default_rng()

    # Check for off-diagonal terms. If components are independent, can use more
    # efficient computation.
    is_correlated = np.count_nonzero(cov - np.diag(np.diagonal(cov)))
    if not is_correlated:  # Expensive
        return rng.multivariate_normal(mean, cov, size, method=method)

    # Use one-dimensional standard normal distribution (cheaper). Have to
    # account for possible shape specifications.
    if size is None:
        return mean + rng.standard_normal() * np.diagonal(cov)
    elif isinstance(size, int):
        return np.expand_dims(mean, 0) + \
               np.outer(rng.standard_normal(size), np.diagonal(cov))
    else:
        return np.expand_dims(mean, 0) + \
               np.expand_dims(rng.standard_normal(size), -1) * np.diagonal(cov)


def get_initial_states(mean, cov, num_states, n=1):
    if np.isscalar(mean):
        mean = mean * np.ones(num_states)
    else:
        assert np.array(mean).shape == (num_states,)

    if np.isscalar(cov):
        cov = cov * np.eye(num_states)

    return get_gaussian_noise(mean, cov, n, RNG)


def get_observation_noise(V):
    return get_additive_white_gaussian_noise(V, rng=RNG)


def get_lqr_cost(x, u, Q, R):
    """"Vectorized version for computing cost of an LQR system."""

    # Apply sum-product instead of matmul because we are dealing with a stack
    # of x and u vectors (one for each time step).
    return - np.sum(x * (Q @ x), 0) - np.sum(u * (R @ u), 0)


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
    # Assume perfect observability / estimation of the process states.
    return -params['K'].dot(u)


def lqe_controller_dynamics(t, x, u, params):
    """Continuous-time Kalman-Bucy Filter."""

    y = u

    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']
    L = params['L']

    # Control:
    _u = lqe_controller_output(t, x, u, params)

    # Updated estimate:
    dx = A.dot(x) + B.dot(_u) + L.dot(y - C.dot(x) - D.dot(_u))

    return dx


# noinspection PyUnusedLocal
def lqe_controller_output(t, x, u, params):
    # The hidden states x of the controller represent an estimate of the
    # process states based on noisy observations y.
    return -params['K'].dot(x)


class StochasticInterconnectedSystem(control.InterconnectedSystem):
    """Stochastic version of an `InterconnectedSystem`.

    Main difference: When we make noisy observations of a system output as in
    LQE, the `_compute_static_io` method will fail with an "Algrebraic loop
    detection" error. This exception is raised because the method assumes
    constant output when injecting a constant input and stepping through the
    system until steady state is reached. To maintain this assumption, we fix
    the observation noise while computing the static response.
    """

    def _compute_static_io(self, t, x, u):
        # Loop over all systems to find the ones with observation noise.
        for sys in self.syslist:
            V = sys.params.get('V', None)
            # Temporarily freeze the noise vector applied to the system output.
            if V is not None:
                v = get_observation_noise(V)
                # noinspection PyProtectedMember
                sys._update_params({'_v': v})

        # Call super to compute the static response to the current input.
        out = super()._compute_static_io(t, x, u)

        # Remove the noise vector so it is sampled dynamically again.
        for sys in self.syslist:
            # noinspection PyProtectedMember
            sys._current_params.pop('_v', None)

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

        # Apply human readable labels.
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
        g.savefig(path, bbox_inches='tight', format='png')
    plt.show()


def plot_phase_diagram(state_dict, odefunc=None, W=None, start_points=None,
                       n=10, xt=None, path=None):
    assert len(state_dict) == 2, "Two dimensions required for phase plot."
    labels = list(state_dict.keys())
    states = list(state_dict.values())
    i, j = (0, 1)  # np.argsort(labels))

    plt.figure()

    # Draw trajectory.
    plt.plot(states[i], states[j])
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])

    # Draw target state.
    if xt is not None:
        plt.scatter(xt[0], xt[1], s=32, marker='o')

    # Draw vector field.
    if odefunc is not None:

        # Get grid coordinates.
        ax = plt.gca()
        x1_min, x1_max = ax.get_xlim()
        x0_min, x0_max = ax.get_ylim()
        grid = np.mgrid[x0_min:x0_max:complex(0, n),
                        x1_min:x1_max:complex(0, n)]
        grid = grid[::-1]
        shape2d = grid.shape[1:]

        # Initialize the state vectors at each location in grid. Possibly
        # augment by noisy state estimate.
        if W is None:
            x = grid
        else:
            # Precompute noise for efficiency
            noise = get_additive_white_gaussian_noise(W, shape2d, RNG)
            # Add noise to grid coordinates.
            x_hat = np.empty_like(grid)
            for i, j in np.ndindex(shape2d):
                x_hat[:, i, j] = grid[:, i, j] + noise[i, j, :]
            x = np.concatenate([grid, x_hat])

        # Compute derivatives at each grid node.
        dx = np.empty_like(x)
        for i, j in np.ndindex(shape2d):
            dx[:, i, j] = odefunc(0, x[:, i, j], [])

        # Draw streamlines and arrows.
        plt.streamplot(grid[0], grid[1], dx[0, :, :], dx[1, :, :],
                       start_points=start_points, linewidth=0.3)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', format='png')
    plt.show()
