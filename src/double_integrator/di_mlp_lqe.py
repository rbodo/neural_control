import sys

import numpy as np

from src.double_integrator import configs
from src.double_integrator.control_systems import DiLqeMlp
from src.double_integrator.di_lqg import add_variables, run_single
from src.double_integrator.utils import RNG, Monitor, apply_config
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
    system = DiLqeMlp(process_noise, observation_noise, dt, RNG,
                      config.controller.cost.lqr.Q,
                      config.controller.cost.lqr.R,
                      config.paths.FILEPATH_MODEL,
                      {'num_hidden': config.model.NUM_HIDDEN})

    # Sample some initial states.
    X0 = system.process.get_initial_states(mu0, Sigma0)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with MLP control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        inits = {'x': x, 'x_est': mu0, 'Sigma': Sigma0}
        run_single(system, times, monitor, inits)

        create_plots(monitor, config, system, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_mlp_lqe.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
