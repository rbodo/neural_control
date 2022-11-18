import os
import time
from collections import OrderedDict
from itertools import product
from typing import Tuple, List, Union, Optional, Iterable, Callable
from urllib.parse import unquote, urlparse

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd
import torch
from yacs.config import CfgNode

RNG = np.random.default_rng(42)


class TimeSeriesVariable:
    """Custom container for time series data."""
    
    def __init__(self, name: str, label: str = None, ndim: int = None,
                 column_labels: List[str] = None,
                 dtype: Union[str, np.dtype] = None):
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

    def append_measurement(self, t: float, d: float, parameters: dict):
        self.times.append(t)
        self.data.append(d)
        for key, value in parameters.items():
            self.parameters.setdefault(key, [])
            self.parameters[key].append(value)

    def get_dataframe(self) -> pd.DataFrame:
        data = pd.DataFrame(self.data, columns=self.column_labels,
                            dtype=self.dtype)
        data['times'] = self.times
        for key, value in self.parameters.items():
            data[key] = value
        return data


class Monitor:
    """Custom container to keep track of experiment parameters and
    measurements."""

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
    def add_variable(self, name: str, label: str = None, ndim: int = None,
                     column_labels: List[str] = None,
                     dtype: Union[str, np.dtype] = None):
        self.variables[name] = TimeSeriesVariable(name, label, ndim,
                                                  column_labels, dtype)

    @Decorators.mark_modified
    def update_variables(self, t: float, **kwargs):
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
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        for d in ['float', 'unsigned']:
            df = df.apply(pd.to_numeric, errors='ignore', downcast=d)
        return df.infer_objects()

    @Decorators.check_update_dataframe
    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @Decorators.check_update_dataframe
    def get_dataframe_melted(self) -> pd.DataFrame:
        df = self._dataframe.melt(
            id_vars=['times'] + list(self.parameters.keys()),
            value_name='value', var_name='dimension')
        for variable in self.variables.values():
            for label in variable.column_labels:
                df.loc[df['dimension'] == label, 'variable'] = variable.label
        return df

    @Decorators.check_update_dataframe
    def get_last_experiment_id(self) -> int:
        return self._dataframe['experiment'].max()

    @Decorators.check_update_dataframe
    def get_last_trajectory(self) -> OrderedDict:
        df = self._dataframe
        i = self.get_last_experiment_id()
        d = OrderedDict({'x': df.loc[df['experiment'] == i, 'x'].to_numpy(),
                         'v': df.loc[df['experiment'] == i, 'v'].to_numpy()})
        return d

    @Decorators.check_update_dataframe
    def get_last_experiment(self) -> pd.DataFrame:
        i = self.get_last_experiment_id()
        df = self.get_dataframe_melted()
        return df[df['experiment'] == i]


def get_additive_white_gaussian_noise(
        cov: np.ndarray, size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: Optional[str] = 'cholesky') -> np.ndarray:
    return get_gaussian_noise(np.zeros(len(cov), cov.dtype), cov, size, rng,
                              method)


def get_gaussian_noise(mean: np.ndarray, cov: np.ndarray,
                       size: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None,
                       method: Optional[str] = 'cholesky') -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    return rng.multivariate_normal(mean, cov, size, method=method)


def get_initial_states(mean: np.ndarray, cov: np.ndarray, num_states: int,
                       n: Optional[int] = 1,
                       rng: Optional[np.random.Generator] = None,
                       dtype: Optional[Union[float, np.dtype]] = 'float32'
                       ) -> np.ndarray:
    """Return noisy initial states of an environment."""
    if np.isscalar(mean):
        mean *= np.ones(num_states, dtype)
    else:
        assert np.array(mean, dtype).shape == (num_states,)

    if np.isscalar(cov):
        cov = cov * np.eye(num_states, dtype=dtype)

    return get_gaussian_noise(mean, cov, n, rng)


def get_lqr_cost(x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt: Optional[float] = 1, normalize: Optional[bool] = False
                 ) -> float:
    """Compute cost of an LQR system."""

    c = dt * (np.dot(x, np.dot(Q, x)) + np.dot(u, np.dot(R, u)))
    if normalize:
        # Assumes Q, R diagonal.
        c /= np.sqrt(np.trace(np.square(Q)) + np.trace(np.square(R)))
    return c.item()


def get_lqr_cost_vectorized(x: np.ndarray, u: np.ndarray, Q: np.ndarray,
                            R: np.ndarray, dt: Optional[float] = 1
                            ) -> np.ndarray:
    """Vectorized version for computing cost of an LQR system."""

    # Apply sum-product instead of matmul because we are dealing with a stack
    # of x and u vectors (one for each time step).
    return dt * (np.sum(x * (Q @ x), 0) + np.sum(u * (R @ u), 0))


def split_train_test(data: pd.DataFrame, f: Optional[float] = 0.2
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_total = data['experiment'].max() + 1
    num_test = int(num_total * f)
    test_idxs = np.linspace(0, num_total, num_test, endpoint=False, dtype=int)
    mask_test = np.isin(data['experiment'], test_idxs)
    mask_train = np.logical_not(mask_test)
    return data[mask_train], data[mask_test]


def select_noise_subset(data: pd.DataFrame, process_noises: Iterable[float],
                        observation_noises: Iterable[float]) -> pd.DataFrame:
    print("Using (combinations of) the following noise levels:")
    print(f"Process noise: {process_noises}")
    print(f"Observation noise: {observation_noises}")

    mask = False
    for process_noise, observation_noise in product(process_noises,
                                                    observation_noises):
        mask |= ((data['process_noise'] == process_noise) &
                 (data['observation_noise'] == observation_noise))
    return data[mask]


def apply_config(config: CfgNode):
    """Make config immutable, create paths, and save config."""
    config.freeze()
    create_paths(config)
    save_config(config)


def create_paths(config: CfgNode):
    for k, p in config.paths.items():
        if 'FILE' in k:
            if p is not None:
                p = os.path.dirname(p)
        if p:
            os.makedirs(p, exist_ok=True)


def save_config(config: CfgNode):
    if config.paths.FILEPATH_OUTPUT_DATA:
        path = os.path.join(os.path.dirname(
            config.paths.FILEPATH_OUTPUT_DATA), '.config.txt')
        with open(path, 'w') as f:
            f.write(config.dump())


def apply_timestamp(path: str, timestamp: Optional[str] = None) -> str:
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(path, timestamp)


def get_artifact_path(relative_subpath: Optional[str] = None) -> str:
    """Get the local filesystem path where mlflow stores logged artifacts.

    If `relative_subpath` is specified, a subdirectory with this name will be
    created in the base directory.
    """
    return uri_to_path(mlflow.get_artifact_uri(relative_subpath))


def uri_to_path(uri: str) -> str:
    return unquote(urlparse(uri).path)


def get_trajectories(data: pd.DataFrame, num_steps: int,
                     variable: Optional[str]) -> np.ndarray:
    """Return the trajectories of a system in state space.

    Parameters
    ----------
    data
        Dataframe containing various measurements of the system.
    num_steps
        How many time steps each trajectory should have.
    variable
        What kind of measurement to extract from `data`. Possible values:

        - 'estimates': Kalman-filtered state estimates.
        - 'observations': Noisy partial observations.
        - 'states': Full noiseless system states.

    Returns
    -------
    trajectories
        Numpy array of shape (`num_steps`, num_states).

    Raises
    ------
    NotImplementedError
        If `variable` not one of ['estimates', 'observations', 'states'].
    KeyError
        If `data` does not contain requested measurements.
    """
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


def get_control(data: pd.DataFrame, num_steps: int) -> np.ndarray:
    y = data['u']
    y = np.reshape(y.to_numpy(), (-1, 1, num_steps))
    return y.astype(np.float32)


def get_data_loaders(data: pd.DataFrame, config: CfgNode, variable: str
                     ) -> Tuple[mx.gluon.data.DataLoader,
                                mx.gluon.data.DataLoader]:
    """Create mxnet train and test data loaders from a pandas data frame."""
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


def get_grid(n: int, x_max: Optional[float] = 1) -> np.ndarray:
    """Create a rectangular 2d grid.

    Parameters
    ----------
    n
        Number of grid nodes along each dimension.
    x_max
        Half the width of the grid (centered around zero).
        The height is currently hard-coded to 0.4.
    """

    x1_min, x1_max = -x_max, x_max
    x0_min, x0_max = -0.2, 0.2
    grid = np.mgrid[x0_min:x0_max:complex(0, n), x1_min:x1_max:complex(0, n)]
    grid = grid[::-1]
    grid = np.reshape(grid, (-1, n * n))
    grid = np.transpose(grid)
    return grid


def jitter(x: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator
           ) -> np.ndarray:
    """Add gaussian noise to an array."""
    return x + get_additive_white_gaussian_noise(Sigma, len(x), rng)


def add_batch_dim(x):
    return np.expand_dims(x, 1)


def atleast_3d(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray,
                                                            torch.Tensor]:
    # Can't use np.atleast_3d here because want dimensions inserted in front.
    if isinstance(x, np.ndarray):
        for _ in range(3 - x.ndim):
            x = np.expand_dims(x, 0)
    elif isinstance(x, torch.Tensor):
        for _ in range(3 - x.ndim):
            x = x.unsqueeze(0)
    else:
        raise NotImplementedError
    return x


def get_data(config: CfgNode, variable: str):
    """Return training and test data loader as dict.

    Contains trajectories of a classic LQR controller in the double integrator
    state space with initial values sampled from a jittered rectangular grid.
    """

    path_data = config.paths.FILEPATH_INPUT_DATA
    data = pd.read_pickle(path_data)
    data_test, data_train = get_data_loaders(data, config, variable)
    return dict(data_train=data_train, data_test=data_test)


def gramian2metric(gramian: np.ndarray, metric: Optional[Callable] = np.prod
                   ) -> float:
    """Compute a scalar metric from a Gramian matrix.

    Use mean or product of eigenvalue spectrum as scalar matric for
    controllability and observability. The larger the better. Product reflects
    better the fact that controllability goes to zero when one eigenvalue is
    zero, but may result in numerical instabilities if Gramian is
    high-dimensional.
    """
    w, v = np.linalg.eig(gramian)
    return metric(w).item()
