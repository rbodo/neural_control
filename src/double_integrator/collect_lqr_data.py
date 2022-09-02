import sys

import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
from yacs.config import CfgNode

from src.double_integrator import configs
from src.double_integrator.control_systems import DiLqr
from src.double_integrator.di_lqr import add_variables, run_single
from src.double_integrator.plotting import create_plots
from src.double_integrator.utils import apply_config, Monitor, get_grid, \
    jitter, RNG


def main(config: 'CfgNode', show_plots: bool = False):
    label = 'lqr'

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

            system = DiLqr(w, v, dt, RNG, q, r)

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

    _config = configs.config_collect_lqr_data.get_config()

    apply_config(_config)

    print(_config)

    main(_config)

    sys.exit()
