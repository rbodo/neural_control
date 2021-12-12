import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.di_lqr import add_variables
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import create_plots


def run_single(system_open, system_closed, times, monitor, inits):
    x = inits['x']
    x_rnn = inits['x_rnn']
    y = inits['y']

    for t in times:
        u, x_rnn = system_closed.get_control(x_rnn, y)
        x = system_open.step(t, x, u)
        y = system_open.output(t, x, u)
        c = system_closed.get_cost(x, u)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)


def main(config):

    label = 'rnn'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'activation': config.model.ACTIVATION}

    # Create double integrator with RNN feedback.
    system_closed = DiRnn(process_noise, observation_noise, dt, RNG,
                          config.controller.cost.lqr.Q,
                          config.controller.cost.lqr.R,
                          config.paths.FILEPATH_MODEL, rnn_kwargs)
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
        x_rnn = np.zeros((system_closed.model.num_layers,
                          system_closed.model.num_hidden))
        y = system_open.output(0, x, 0)
        inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
        run_single(system_open, system_closed, times, monitor, inits)

        create_plots(monitor, config, system_closed, label, i, RNG)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_rnn.py')

    main(_config)

    sys.exit()
