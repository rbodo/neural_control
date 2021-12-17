import sys
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.plotting import create_plots
from src.double_integrator.utils import (RNG, Monitor,
                                         get_additive_white_gaussian_noise)

if TYPE_CHECKING:
    from yacs.config import CfgNode


def add_variables(monitor: Monitor):
    dtype = 'float32'
    kwargs = [
        dict(name='states', label='States', column_labels=['x', 'v'],
             dtype=dtype),
        dict(name='state_estimates', label='States',
             column_labels=[r'$\hat{x}$', r'$\hat{v}$'], dtype=dtype),
        dict(name='outputs', label='Output', column_labels=[r'$y_x$'],
             dtype=dtype),
        dict(name='control', label='Control', column_labels=['u'],
             dtype=dtype),
        dict(name='cost', label='Cost', column_labels=['c'], dtype=dtype)
    ]
    for k in kwargs:
        monitor.add_variable(**k)


def run_single(system, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']

    for t in times:
        x, y, u, c, x_est, Sigma = system.step(t, x, x_est, Sigma)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                 state_estimates=x_est)


def get_grid(n):
    x1_min, x1_max = -1, 1
    x0_min, x0_max = -0.2, 0.2
    grid = np.mgrid[x0_min:x0_max:complex(0, n), x1_min:x1_max:complex(0, n)]
    grid = grid[::-1]
    grid = np.reshape(grid, (-1, n * n))
    grid = np.transpose(grid)
    return grid


def jitter(x, Sigma, rng):
    return x + get_additive_white_gaussian_noise(Sigma, len(x), rng)


def main(config: 'CfgNode', show_plots: bool = False):
    label = 'lqg'

    filepath_output_data = config.paths.FILEPATH_OUTPUT_DATA
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    grid_size = config.simulation.GRID_SIZE
    dt = T / num_steps
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R

    # Sample some initial states.
    grid = get_grid(grid_size)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    for w in tqdm(process_noises, 'Process noise', leave=False):

        monitor.update_parameters(process_noise=w)

        for v in tqdm(observation_noises, 'Observation noise', leave=False):
            monitor.update_parameters(observation_noise=v)

            system = DiLqg(w, v, dt, RNG, q, r)

            # Initialize the state estimate.
            X0_est = jitter(grid, system.process.W, RNG)

            for i, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):

                monitor.update_parameters(experiment=i)
                inits = {'x': x, 'x_est': x_est, 'Sigma': Sigma0}
                run_single(system, times, monitor, inits)

                if show_plots:
                    create_plots(monitor, config, system, label, i, RNG)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(filepath_output_data)
    print(f"Saved data to {filepath_output_data}.")


if __name__ == '__main__':
    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqg.py')

    main(_config, show_plots=True)

    sys.exit()
