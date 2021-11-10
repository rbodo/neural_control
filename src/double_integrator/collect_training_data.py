import os
import sys

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.double_integrator_lqg import DiLqg
from src.double_integrator.utils import get_additive_white_gaussian_noise, RNG


def main(config):
    np.random.seed(42)

    # Create double integrator with LQR feedback.
    di_lqg = DiLqg(config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   config.process.PROCESS_NOISE,
                   config.process.OBSERVATION_NOISE)
    system_closed = di_lqg.get_system()

    # Sample some initial states.
    n = 100  # Number of grid lines along each dimension.
    x1_min, x1_max = -1, 1#
    x0_min, x0_max = -0.2, 0.2
    grid = np.mgrid[x0_min:x0_max:complex(0, n), x1_min:x1_max:complex(0, n)]
    grid = grid[::-1]
    shape2d = grid.shape[1:]

    # Initialize the state vectors at each jittered grid location.
    S = np.eye(di_lqg.n_x_process) * config.process.STATE_COVARIANCE
    noise = get_additive_white_gaussian_noise(S, shape2d, RNG)
    x0 = grid + np.moveaxis(noise, -1, 0)
    # Add noisy state estimate.
    noise = get_additive_white_gaussian_noise(di_lqg.W, shape2d, RNG)
    x0_hat = grid + np.moveaxis(noise, -1, 0)
    X0 = np.concatenate([x0, x0_hat])
    X0 = np.reshape(X0, (len(X0), -1))  # Flatten spatial dimensions.
    X0 = np.transpose(X0)  # Shape: [num_samples, num_states]
    num_samples, num_states = X0.shape

    num_steps = config.simulation.NUM_STEPS
    times = np.linspace(0, config.simulation.T, num_steps, endpoint=False)

    # Simulate the system with LQR control.
    X = np.empty((num_samples, di_lqg.n_x_control, num_steps), np.float32)
    Y = np.empty((num_samples, di_lqg.n_y_control, num_steps), np.float32)
    for i in range(num_samples):
        t, y, x = control.input_output_response(system_closed, times,
                                                X0=X0[i], return_x=True)
        X[i] = x[-di_lqg.n_x_control:]  # Get state estimate of Kalman filter
        Y[i] = y  # Get control signal
        print("\r{:3.2%}".format((i + 1) / num_samples), end='', flush=True)

    # Store state trajectories and corresponding control signals.
    np.savez_compressed(os.path.join(config.paths.PATH_TRAINING_DATA, 'lqg'),
                        X=X, Y=Y)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/'
                         'config_collect_training_data.py')

    main(_config)

    sys.exit()
