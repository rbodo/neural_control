import os
import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiRnnLqe
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import plot_timeseries, plot_phase_diagram
from src.double_integrator.di_lqg import add_variables


def run_single(system_open, system_closed, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']
    x_rnn = inits['x_rnn']

    for t in times:
        u, x_rnn = system_closed.get_control(x_rnn, x_est)
        x = system_open.step(t, x, u)
        y = system_open.output(t, x, u)
        x_est, Sigma = system_closed.apply_filter(t, x_est, Sigma, u, y)
        c = system_closed.get_cost(x_est, u)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                 state_estimates=x_est)


def main(config):
    label = 'rnn_lqe'
    path_out = config.paths.PATH_OUT
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
    system_closed = DiRnnLqe(process_noise, observation_noise, dt, RNG,
                             config.controller.cost.lqr.Q,
                             config.controller.cost.lqr.R,
                             config.paths.PATH_MODEL, rnn_kwargs)
    system_open = system_closed.system

    # Sample some initial states.
    n = 1
    X0 = system_closed.get_initial_states(config.process.STATE_MEAN,
                                          config.process.STATE_COVARIANCE, n,
                                          RNG)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with RNN control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)

        inits = {'x': x, 'x_est': mu0, 'Sigma': Sigma0,
                 'x_rnn': np.zeros((system_closed.model.num_layers,
                                    system_closed.model.num_hidden))}
        run_single(system_open, system_closed, times, monitor, inits)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(), rng=RNG,
                           xt=config.controller.STATE_TARGET, path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_rnn_lqe.py')

    main(_config)

    sys.exit()
