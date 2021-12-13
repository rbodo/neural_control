import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiMlpLqe
from src.double_integrator.di_lqg import add_variables, run_single
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import create_plots


def main(config):

    label = 'mlp_lqe'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))

    # Create double integrator with MLP feedback.
    system_closed = DiMlpLqe(process_noise, observation_noise, dt, RNG,
                             config.controller.cost.lqr.Q,
                             config.controller.cost.lqr.R,
                             config.model.NUM_HIDDEN,
                             config.paths.PATH_MODEL)
    system_open = system_closed.system

    # Sample some initial states.
    X0 = system_closed.get_initial_states(mu0, Sigma0)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with MLP control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        inits = {'x': x, 'x_est': mu0, 'Sigma': Sigma0}
        run_single(system_open, system_closed, times, monitor, inits)

        create_plots(monitor, config, system_closed, label, i, RNG)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_mlp_lqe.py')

    main(_config)

    sys.exit()
