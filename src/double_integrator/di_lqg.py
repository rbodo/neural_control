import os
import sys
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.utils import (RNG, Monitor,
                                         get_additive_white_gaussian_noise)
from src.double_integrator.plotting import plot_timeseries, plot_phase_diagram
from src.double_integrator.di_lqr import add_variables as add_variables_lqr

if TYPE_CHECKING:
    from yacs.config import CfgNode


def add_variables(monitor: Monitor):
    add_variables_lqr(monitor)
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'],
                         dtype='float32')


def run_single(system_open, system_closed, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']

    for t in times:
        u = system_closed.get_control(x_est)
        x = system_open.step(t, x, u)
        y = system_open.output(t, x, u)
        x_est, Sigma = system_closed.apply_filter(t, x_est, Sigma, u, y)
        c = system_closed.get_cost(x_est, u)

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

    path_out = config.paths.PATH_OUT
    path_training_data = config.paths.PATH_TRAINING_DATA
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
    state_target = config.controller.STATE_TARGET

    # Sample some initial states.
    grid = get_grid(grid_size)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    for w in tqdm(process_noises, 'Process noise'):

        monitor.update_parameters(process_noise=w)

        for v in tqdm(observation_noises, 'Observation noise'):
            monitor.update_parameters(observation_noise=v)

            system_closed = DiLqg(w, v, dt, RNG, q, r)
            system_open = system_closed.system

            # Initialize the state estimate.
            X0_est = jitter(grid, system_closed.W, RNG)

            for i, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):

                monitor.update_parameters(experiment=i)
                inits = {'x': x, 'x_est': x_est, 'Sigma': Sigma0}
                run_single(system_open, system_closed, times, monitor, inits)

                if show_plots:
                    path = os.path.join(path_out, f'timeseries_{label}_{i}')
                    plot_timeseries(monitor.get_last_experiment(), path=path)

                    path = os.path.join(path_out, f'phase_diagram_{label}_{i}')
                    plot_phase_diagram(monitor.get_last_trajectory(),
                                       odefunc=system_closed.step, rng=RNG,
                                       xt=state_target, path=path)

    # Store state trajectories and corresponding control signals.
    if os.path.isfile(path_training_data):
        print(f"Saved data to {path_training_data}.")
        df = monitor.get_dataframe()
        df.to_pickle(path_training_data)


if __name__ == '__main__':
    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqg.py')

    main(_config, show_plots=True)

    sys.exit()
