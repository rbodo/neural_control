import sys

import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.utils import (
    RNG, get_additive_white_gaussian_noise, Monitor)


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


def add_variables(monitor):
    monitor.add_variable('states', 'States', column_labels=['x', 'v'],
                         dtype='float32')
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'],
                         dtype='float32')
    monitor.add_variable('outputs', 'Output', column_labels=['y'],
                         dtype='float32')
    monitor.add_variable('control', 'Control', column_labels=['u'],
                         dtype='float32')
    monitor.add_variable('cost', 'Cost', column_labels=['c'], dtype='float32')


def  run_single(system_open, system_closed, times, monitor, inits):

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


def main(config):

    path_out = config.paths.PATH_TRAINING_DATA
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R

    # Sample some initial states.
    n = 100  # Number of grid lines along each dimension.
    grid = get_grid(n)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    process_noise = np.logspace(-2, -1, 5, dtype='float32')
    observation_noise = np.logspace(-1, 0, 5, dtype='float32')

    for w in tqdm(process_noise, 'Process noise'):

        monitor.update_parameters(process_noise=w)

        for v in tqdm(observation_noise, 'Observation noise'):
            monitor.update_parameters(observation_noise=v)

            system_closed = DiLqg(w, v, dt, RNG, q, r)
            system_open = system_closed.system

            # Initialize the state estimate.
            X0_est = jitter(grid, system_closed.W, RNG)

            for i, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):

                monitor.update_parameters(experiment=i)
                inits = {'x': x, 'x_est': x_est, 'Sigma': Sigma0}
                run_single(system_open, system_closed, times, monitor, inits)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(path_out)


if __name__ == '__main__':

    _config = get_config(
        '/home/bodrue/PycharmProjects/neural_control/src/double_integrator/'
        'configs/config_collect_training_data.py')

    main(_config)

    sys.exit()
