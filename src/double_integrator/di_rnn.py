import os
import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import plot_timeseries, plot_phase_diagram


def main(config):

    label = 'rnn'
    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator with RNN feedback.
    system_closed = DiRnn(process_noise, observation_noise, dt, RNG,
                          config.controller.cost.lqr.Q,
                          config.controller.cost.lqr.R,
                          config.model.NUM_HIDDEN,
                          config.model.NUM_LAYERS,
                          config.paths.PATH_MODEL)
    system_open = system_closed.system

    # Sample some initial states.
    n = 1
    X0 = system_closed.get_initial_states(config.process.STATE_MEAN,
                                          config.process.STATE_COVARIANCE, n,
                                          RNG)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with RNN control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        x_rnn = np.zeros((system_closed.model.num_layers,
                          system_closed.model.num_hidden))
        y = system_open.output(0, x, 0)
        for t in times:
            u, x_rnn = system_closed.get_control(x_rnn, y)
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            c = system_closed.get_cost(x, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(), rng=RNG,
                           xt=config.controller.STATE_TARGET, path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_rnn.py')

    main(_config)

    sys.exit()
