import sys
from typing import TYPE_CHECKING

import control
import numpy as np
from tqdm import trange
from tqdm.contrib import tzip

from src.double_integrator import configs
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.plotting import create_plots
from src.double_integrator.di_lqg import run_single, add_variables
from src.double_integrator.utils import (RNG, Monitor, apply_config, get_grid,
                                         jitter)


if TYPE_CHECKING:
    from yacs.config import CfgNode


def main(config: 'CfgNode', config_train: 'CfgNode', show_plots: bool = False):
    label = 'lqg'

    filepath_output_data = config.paths.FILEPATH_OUTPUT_DATA
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    grid_size = config.simulation.GRID_SIZE
    dt = T / num_steps
    process_noises_train = config_train.process.PROCESS_NOISES
    observation_noises_train = config_train.process.OBSERVATION_NOISES
    process_noises_test = config.process.PROCESS_NOISES
    observation_noises_test = config.process.OBSERVATION_NOISES
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R
    f = int(1 / config.training.VALIDATION_FRACTION)

    # Sample some initial states.
    grid = get_grid(grid_size)

    grid = grid[::f]

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    for i in trange(len(process_noises_train), desc='Process noise',
                    leave=False):
        w_train = process_noises_train[i]
        w_test = process_noises_test[i]
        monitor.update_parameters(process_noise=w_test)

        for j in trange(len(observation_noises_train),
                        desc='Observation noise', leave=False):
            v_train = observation_noises_train[j]
            v_test = observation_noises_test[j]
            monitor.update_parameters(observation_noise=v_test)

            system = DiLqg(w_train, v_train, dt, RNG, q, r)
            process = system.process

            # Recompute Kalman gain with slightly different noise levels.
            W = w_test * np.eye(process.num_states)
            V = v_test * np.eye(process.num_outputs)
            L = control.lqe(process.A, np.eye(len(process.A)), process.C,
                            W, V)[0]
            system.estimator.L = L

            # Initialize the state estimate.
            X0_est = jitter(grid, process.W, RNG)

            for k, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):

                monitor.update_parameters(experiment=k)
                inits = {'x': x, 'x_est': x_est, 'Sigma': Sigma0}
                run_single(system, times, monitor, inits)

                if show_plots:
                    create_plots(monitor, config, system, label, k, RNG)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(filepath_output_data)
    print(f"Saved data to {filepath_output_data}.")


if __name__ == '__main__':
    _config_test = configs.config_lqg_generalization.get_config()
    _config_train = configs.config_collect_training_data.get_config()

    apply_config(_config_test)

    print(_config_test)

    main(_config_test, _config_train)

    sys.exit()
