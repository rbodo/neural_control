from collections import OrderedDict

import control
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_theme(style='whitegrid')

RNG = np.random.default_rng(42)


class TimeSeriesVariable:
    def __init__(self, name, label=None, ndim=None, column_labels=None):
        self.name = name
        self.label = label or name  # Human-readable version of `name`.
        self.ndim = ndim
        if column_labels is None and ndim is not None:
            column_labels = [self.name + str(i) for i in range(self.ndim)]
        self.column_labels = column_labels

        self.data = []
        self.times = []
        self.parameters = {}

    def append_measurement(self, t, d, parameters):
        self.times.append(t)
        self.data.append(d)
        for key, value in parameters.items():
            self.parameters.setdefault(key, [])
            self.parameters[key].append(value)


    def get_dataframe(self):
        data = pd.DataFrame(self.data, columns=self.column_labels)
        data['times'] = self.times
        for key, value in self.parameters.items():
            data[key] = value
        data = data.melt(id_vars=['times']+list(self.parameters.keys()),
                         value_vars=self.column_labels, value_name='value',
                         var_name='dimension')
        data['variable'] = self.label
        return data


class Monitor:
    def __init__(self):
        self.variables = {}
        self.parameters = {}
        self._is_variable_updated = False
        self._dataframe = None

    class Decorators:

        @classmethod
        def mark_modified(cls, f):
            def inner(self, *args, **kwargs):
                self._is_variable_updated = True
                return f(self, *args, **kwargs)
            return inner

        @classmethod
        def check_update_dataframe(cls, f):
            def inner(self, *args, **kwargs):
                if self._is_variable_updated:
                    self._update_dataframe()
                return f(self, *args, **kwargs)
            return inner

    @Decorators.mark_modified
    def add_variable(self, name, label=None, ndim=None, column_labels=None):
        self.variables[name] = TimeSeriesVariable(name, label, ndim,
                                                  column_labels)

    @Decorators.mark_modified
    def update_variables(self, t, **kwargs):
        for key, value in kwargs.items():
            self.variables[key].append_measurement(t, value, self.parameters)

    @Decorators.mark_modified
    def update_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def _update_dataframe(self):
        dfs = [data.get_dataframe() for data in self.variables.values()]
        self._dataframe = pd.concat(dfs, ignore_index=True)
        self._is_variable_updated = False

    @Decorators.check_update_dataframe
    def get_dataframe(self):
        return self._dataframe

    @Decorators.check_update_dataframe
    def get_last_experiment_id(self):
        return self._dataframe['experiment'].max()

    @Decorators.check_update_dataframe
    def get_last_trajectory(self):
        df = self._dataframe
        i = self.get_last_experiment_id()
        d = OrderedDict(
            {'x': df[(df['dimension'] == 'x') &
                     (df['experiment'] == i)]['value'],
             'v': df[(df['dimension'] == 'v') &
                     (df['experiment'] == i)]['value']})
        return d

    @Decorators.check_update_dataframe
    def get_last_experiment(self):
        i = self.get_last_experiment_id()
        return self._dataframe[self._dataframe['experiment'] == i]


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
    is_correlated = np.count_nonzero(cov - np.diag(np.diagonal(cov))) > 0
    if is_correlated:  # Expensive
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


def get_lqr_cost(x, u, Q, R, dt=1, sign=1):
    """Compute cost of an LQR system."""

    return sign * dt * (np.dot(x, np.dot(Q, x)) + np.dot(u, np.dot(R, u)))


def get_lqr_cost_vectorized(x, u, Q, R, dt=1, sign=1):
    """Vectorized version for computing cost of an LQR system."""

    # Apply sum-product instead of matmul because we are dealing with a stack
    # of x and u vectors (one for each time step).
    return sign * dt * (np.sum(x * (Q @ x), 0) + np.sum(u * (R @ u), 0))


class StochasticLinearIOSystem(control.LinearIOSystem):

    def __rdiv__(self, other):
        control.StateSpace.__rdiv__(self, other)

    def __div__(self, other):
        control.StateSpace.__div__(self, other)

    def __init__(self, linsys, W=None, V=None, rng=None, **kwargs):
        super().__init__(linsys, **kwargs)
        self.W = W
        self.V = V
        self.rng = rng

    def step(self, t, x, u, method='euler-maruyama', deterministic=False):
        dxdt = super().dynamics(t, x, u)

        if method == 'euler-maruyama':
            x_new = x + self.dt * dxdt
            if self.W is not None and not deterministic:
                dW = get_additive_white_gaussian_noise(
                    np.eye(len(x)) * np.sqrt(self.dt), rng=self.rng)
                x_new += np.dot(self.W, dW)
            return x_new
        else:
            raise NotImplementedError

    def output(self, t, x, u, deterministic=False):
        out = super().output(t, x, u)
        if self.V is not None and not deterministic:
            out += get_additive_white_gaussian_noise(self.V, rng=self.rng)
        return out


def plot_timeseries(df, path=None):

    n = df['variable'].nunique()
    # Create line plot with one subplot for each variable and one line for each
    # dimension.
    g = sns.relplot(data=df, x='times', y='value', hue='dimension',
                    row='variable', kind='line', aspect=2, height=10/n,
                    facet_kws={'sharey': False})
    if 'Cost' in g.axes_dict:
        g.axes_dict['Cost'].set(yscale='log')

    g.set_axis_labels('Time', '')

    # Add borders.
    sns.despine(right=False, top=False)

    # noinspection PyProtectedMember
    g._legend.set_title(None)

    for ax in g.axes.ravel():
        title = ax.get_title()
        title = title.split(' = ')[1]
        ax.set_title(title)

    if path is not None:
        g.savefig(path, bbox_inches='tight')
    plt.show()


def plot_phase_diagram(state_dict, num_states=None, odefunc=None, W=None,
                       start_points=None, n=10, xt=None, path=None):
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

        # Initialize the process state vectors at each location in grid.
        x = grid
        # If we have an LQE, add initial values for noisy state estimates.
        if W is not None:
            noise = get_additive_white_gaussian_noise(W, shape2d, RNG)
            # Add noise to grid coordinates.
            x_hat = grid + np.moveaxis(noise, -1, 0)
            x = np.concatenate([x, x_hat])
        # If we have a stateful controller (e.g. RNN), add initial values.
        if num_states is not None:
            x = np.concatenate([x, np.zeros((num_states,) + shape2d)])
        # Compute derivatives at each grid node.
        dx = np.empty_like(x)
        for i, j in np.ndindex(shape2d):
            dx[:, i, j] = odefunc(0, x[:, i, j], [0])

        # Draw streamlines and arrows.
        plt.streamplot(grid[0], grid[1], dx[0], dx[1],
                       start_points=start_points, linewidth=0.3)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
