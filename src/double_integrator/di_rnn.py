import sys

import numpy as np

from src.double_integrator import configs
from src.double_integrator.control_systems_mxnet import DiRnn
from src.double_integrator.utils import RNG, Monitor, apply_config
from src.double_integrator.plotting import create_plots


def add_variables(monitor: Monitor):
    dtype = 'float32'
    kwargs = [
        dict(name='states', label='States', column_labels=['x', 'v'],
             dtype=dtype),
        dict(name='outputs', label='Output', column_labels=[r'$y_x$'],
             dtype=dtype),
        dict(name='control', label='Control', column_labels=['u'],
             dtype=dtype),
        dict(name='cost', label='Cost', column_labels=['c'], dtype=dtype)
    ]
    for k in kwargs:
        monitor.add_variable(**k)


def run_single(system, times, monitor, inits):
    x = inits['x']
    x_rnn = inits['x_rnn']
    y = inits['y']

    for t in times:
        x, y, u, c, x_rnn = system.step(t, x, y, x_rnn)

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
    system = DiRnn(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
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
        x_rnn = np.zeros((system.model.num_layers,
                          system.model.num_hidden))
        y = system.process.output(0, x, 0)
        inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
        run_single(system, times, monitor, inits)

        create_plots(monitor, config, system, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_rnn.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
