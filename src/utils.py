import os
import time
from collections import OrderedDict
from itertools import product
from typing import Tuple
from urllib.parse import unquote, urlparse

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


class TimeSeriesVariable:
    def __init__(self, name, label=None, ndim=None, column_labels=None,
                 dtype=None):
        self.name = name
        self.label = label or name  # Human-readable version of `name`.
        self.ndim = ndim
        if column_labels is None and ndim is not None:
            column_labels = [self.name + str(i) for i in range(self.ndim)]
        self.column_labels = column_labels
        self.dtype = dtype

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
        data = pd.DataFrame(self.data, columns=self.column_labels,
                            dtype=self.dtype)
        data['times'] = self.times
        for key, value in self.parameters.items():
            data[key] = value
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
    def add_variable(self, name, label=None, ndim=None, column_labels=None,
                     dtype=None):
        self.variables[name] = TimeSeriesVariable(name, label, ndim,
                                                  column_labels, dtype)

    @Decorators.mark_modified
    def update_variables(self, t, **kwargs):
        for key, value in kwargs.items():
            self.variables[key].append_measurement(t, value, self.parameters)

    @Decorators.mark_modified
    def update_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def _update_dataframe(self):
        dfs = [data.get_dataframe() for data in self.variables.values()]
        df = dfs.pop(0)
        for df_ in dfs:
            df = pd.merge(df, df_)

        self._dataframe = self.optimize_dtypes(df)
        self._is_variable_updated = False

    @staticmethod
    def optimize_dtypes(df):
        for d in ['float', 'unsigned']:
            df = df.apply(pd.to_numeric, errors='ignore', downcast=d)
        return df.infer_objects()

    @Decorators.check_update_dataframe
    def get_dataframe(self):
        return self._dataframe

    @Decorators.check_update_dataframe
    def get_dataframe_melted(self):
        df = self._dataframe.melt(
            id_vars=['times'] + list(self.parameters.keys()),
            value_name='value', var_name='dimension')
        for variable in self.variables.values():
            for label in variable.column_labels:
                df.loc[df['dimension'] == label, 'variable'] = variable.label
        return df

    @Decorators.check_update_dataframe
    def get_last_experiment_id(self):
        return self._dataframe['experiment'].max()

    @Decorators.check_update_dataframe
    def get_last_trajectory(self):
        df = self._dataframe
        i = self.get_last_experiment_id()
        d = OrderedDict({'x': df.loc[df['experiment'] == i, 'x'].to_numpy(),
                         'v': df.loc[df['experiment'] == i, 'v'].to_numpy()})
        return d

    @Decorators.check_update_dataframe
    def get_last_experiment(self):
        i = self.get_last_experiment_id()
        df = self.get_dataframe_melted()
        return df[df['experiment'] == i]


def get_additive_white_gaussian_noise(cov, size=None, rng=None,
                                      method='cholesky'):
    return get_gaussian_noise(np.zeros(len(cov), cov.dtype), cov, size, rng,
                              method)


def get_gaussian_noise(mean, cov, size=None, rng=None, method='cholesky'):
    if rng is None:
        rng = np.random.default_rng()

    return rng.multivariate_normal(mean, cov, size, method=method)


def get_initial_states(mean, cov, num_states, n=1, rng=None, dtype='float32'):
    if np.isscalar(mean):
        mean *= np.ones(num_states, dtype)
    else:
        assert np.array(mean, dtype).shape == (num_states,)

    if np.isscalar(cov):
        cov = cov * np.eye(num_states, dtype=dtype)

    return get_gaussian_noise(mean, cov, n, rng)


def get_lqr_cost(x, u, Q, R, dt=1, sign=1, normalize=False):
    """Compute cost of an LQR system."""

    c = np.float32(sign) * dt * (np.dot(x, np.dot(Q, x)) +
                                 np.dot(u, np.dot(R, u)))
    if normalize:
        # Assumes Q, R diagonal.
        c /= np.sqrt(np.trace(np.square(Q)) + np.trace(np.square(R)))
    return c


def get_lqr_cost_vectorized(x, u, Q, R, dt=1, sign=1):
    """Vectorized version for computing cost of an LQR system."""

    # Apply sum-product instead of matmul because we are dealing with a stack
    # of x and u vectors (one for each time step).
    return sign * dt * (np.sum(x * (Q @ x), 0) + np.sum(u * (R @ u), 0))


def split_train_test(data: pd.DataFrame, f: float = 0.2) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_total = data['experiment'].max() + 1
    num_test = int(num_total * f)
    test_idxs = np.linspace(0, num_total, num_test, endpoint=False, dtype=int)
    mask_test = np.isin(data['experiment'], test_idxs)
    mask_train = np.logical_not(mask_test)
    return data[mask_train], data[mask_test]


def select_noise_subset(data, process_noises, observation_noises):
    print("Using (combinations of) the following noise levels:")
    print(f"Process noise: {process_noises}")
    print(f"Observation noise: {observation_noises}")

    mask = False
    for process_noise, observation_noise in product(process_noises,
                                                    observation_noises):
        mask |= ((data['process_noise'] == process_noise) &
                 (data['observation_noise'] == observation_noise))
    return data[mask]


def apply_config(config):
    config.freeze()
    create_paths(config)
    save_config(config)


def create_paths(config):
    for k, p in config.paths.items():
        if 'FILE' in k:
            if p is not None:
                p = os.path.dirname(p)
        if p:
            os.makedirs(p, exist_ok=True)


def save_config(config):
    if config.paths.FILEPATH_OUTPUT_DATA:
        path = os.path.join(os.path.dirname(
            config.paths.FILEPATH_OUTPUT_DATA), '.config.txt')
        with open(path, 'w') as f:
            f.write(config.dump())


def apply_timestamp(path, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(path, timestamp)


def get_artifact_path(relative_subpath=None):
    return unquote(urlparse(mlflow.get_artifact_uri(relative_subpath)).path)


def get_trajectories(data, num_steps, variable: str):
    if variable == 'estimates':
        print("Using Kalman-filtered state estimates.")
        x0 = data[r'$\hat{x}$']
        x1 = data[r'$\hat{v}$']
        x0 = np.reshape(x0.to_numpy(), (-1, num_steps))
        x1 = np.reshape(x1.to_numpy(), (-1, num_steps))
        x = np.stack([x0, x1], 1)
    elif variable == 'observations':
        print("Using noisy partial observations.")
        x = data['y']
        x = np.reshape(x.to_numpy(), (-1, 1, num_steps))
    elif variable == 'states':
        print("Using states.")
        x0 = data['x']
        x1 = data['v']
        x0 = np.reshape(x0.to_numpy(), (-1, num_steps))
        x1 = np.reshape(x1.to_numpy(), (-1, num_steps))
        x = np.stack([x0, x1], 1)
    else:
        raise NotImplementedError
    return x.astype(np.float32)


def get_control(data, num_steps):
    y = data['u']
    y = np.reshape(y.to_numpy(), (-1, 1, num_steps))
    return y.astype(np.float32)


def get_data_loaders(data, config, variable):
    print("\nPreparing data loaders:")
    num_cpus = max(os.cpu_count() // 2, 1)
    num_steps = config.simulation.NUM_STEPS
    batch_size = config.training.BATCH_SIZE
    validation_fraction = config.training.VALIDATION_FRACTION
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    data = select_noise_subset(data, process_noises, observation_noises)

    data_train, data_test = split_train_test(data, validation_fraction)

    x_train = get_trajectories(data_train, num_steps, variable)
    y_train = get_control(data_train, num_steps)
    x_test = get_trajectories(data_test, num_steps, variable)
    y_test = get_control(data_test, num_steps)

    train_dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
    train_data_loader = mx.gluon.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_cpus,
        last_batch='rollover')
    test_dataset = mx.gluon.data.dataset.ArrayDataset(x_test, y_test)
    test_data_loader = mx.gluon.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_cpus,
        last_batch='discard')

    return test_data_loader, train_data_loader


def get_grid(n, x_max=1):
    x1_min, x1_max = -x_max, x_max
    x0_min, x0_max = -0.2, 0.2
    grid = np.mgrid[x0_min:x0_max:complex(0, n), x1_min:x1_max:complex(0, n)]
    grid = grid[::-1]
    grid = np.reshape(grid, (-1, n * n))
    grid = np.transpose(grid)
    return grid


def jitter(x, Sigma, rng):
    return x + get_additive_white_gaussian_noise(Sigma, len(x), rng)
