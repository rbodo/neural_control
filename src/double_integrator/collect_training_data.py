import os
import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_lqg import DiLqg
from src.double_integrator.utils import (
    RNG, get_additive_white_gaussian_noise, Monitor)


def main(config):

    path_out = config.paths.PATH_TRAINING_DATA
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))

    # Create double integrator with LQG feedback.
    system_closed = DiLqg(process_noise, observation_noise, dt, RNG,
                          config.controller.cost.lqr.Q,
                          config.controller.cost.lqr.R)
    system_open = system_closed.system

    # Sample some initial states.
    n = 100  # Number of grid lines along each dimension.
    num_samples = n * n
    x1_min, x1_max = -1, 1
    x0_min, x0_max = -0.2, 0.2
    grid = np.mgrid[x0_min:x0_max:complex(0, n), x1_min:x1_max:complex(0, n)]
    grid = grid[::-1]
    grid = np.reshape(grid, (-1, num_samples))
    grid = np.transpose(grid)

    # Initialize the state vectors at each jittered grid location.
    noise = get_additive_white_gaussian_noise(Sigma0, num_samples, RNG)
    X0 = grid + noise
    # Add noisy state estimate.
    noise = get_additive_white_gaussian_noise(system_closed.W, num_samples,
                                              RNG)
    X0_est = grid + noise

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with LQG control.
    for i, (x, x_est) in enumerate(zip(X0, X0_est)):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        Sigma = Sigma0
        for t in times:
            u = system_closed.get_control(x_est)
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            x_est, Sigma = system_closed.apply_filter(t, x_est, Sigma, u, y)
            c = system_closed.get_cost(x_est, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                     state_estimates=x_est)
        print("\r{:3.2%}".format((i + 1) / num_samples), end='', flush=True)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(os.path.join(path_out, 'lqg.pkl'))


if __name__ == '__main__':

    _config = get_config(
        '/home/bodrue/PycharmProjects/neural_control/src/double_integrator/'
        'configs/config_collect_training_data.py')

    main(_config)

    sys.exit()
