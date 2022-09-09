import sys

import numpy as np

from src.double_integrator import configs
from src.double_integrator.control_systems_mxnet import DiLqeRnn
from src.double_integrator.utils import RNG, Monitor, apply_config
from src.double_integrator.plotting import create_plots
from src.double_integrator.di_lqg import add_variables


def run_single(system, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']
    x_rnn = inits['x_rnn']

    for t in times:
        x, y, u, c, x_rnn, x_est, Sigma = system.step(t, x, x_rnn, x_est,
                                                      Sigma)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                 state_estimates=x_est)


def main(config):

    label = 'rnn_lqe'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'activation': config.model.ACTIVATION}

    # Create double integrator with RNN feedback.
    system = DiLqeRnn(process_noise, observation_noise, dt, RNG,
                      config.controller.cost.lqr.Q,
                      config.controller.cost.lqr.R,
                      config.paths.FILEPATH_MODEL, rnn_kwargs)

    # Sample some initial states.
    X0 = system.process.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with RNN control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)

        inits = {'x': x, 'x_est': mu0, 'Sigma': Sigma0,
                 'x_rnn': np.zeros((system.control.model.num_layers,
                                    system.control.model.num_hidden))}
        run_single(system, times, monitor, inits)

        create_plots(monitor, config, system, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_rnn_lqe.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
