import sys

import numpy as np
import mxnet as mx

from src.double_integrator import configs
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.di_rnn import add_variables, run_single
from src.double_integrator.utils import RNG, Monitor, apply_config
from src.double_integrator.plotting import create_plots


def main(config):

    label = 'rnn'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    activation = config.model.ACTIVATION
    num_hidden = config.model.NUM_HIDDEN
    if activation == 'linear':
        activation = mx.gluon.nn.LeakyReLU(1)
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': num_hidden,
                  'activation': activation,
                  'input_size': 1}

    # Create double integrator with RNN feedback.
    system = DiRnn(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   config.paths.FILEPATH_MODEL, rnn_kwargs)
    beta = 0.1
    alpha = 0.7
    gamma = (1 - alpha) / dt
    system.model.rnn.h2h_bias.set_data(np.zeros(num_hidden))
    system.model.rnn.h2h_weight.set_data(np.array([[beta, 0],
                                                   [-gamma, alpha]]))
    system.model.rnn.i2h_bias.set_data(np.zeros(num_hidden))
    system.model.rnn.i2h_weight.set_data(np.array([[1 - beta, gamma]]).T)
    system.model.decoder.weight.set_data(-np.array([[1, np.sqrt(2)]]))
    system.model.decoder.bias.set_data(np.zeros(system.process.num_outputs))

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
        x_rnn = np.zeros((system.model.num_layers,
                          system.model.num_hidden))
        y = system.process.output(0, x, 0)
        inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
        run_single(system, times, monitor, inits)

        create_plots(monitor, config, system, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_rnn_analytic.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
