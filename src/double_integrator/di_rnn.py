import os
import sys

import numpy as np
import mxnet as mx

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_open import DI
from src.double_integrator.train_rnn import RNNModel
from src.double_integrator.utils import (
    plot_phase_diagram, get_lqr_cost, RNG, Monitor, plot_timeseries)


class DiRnn(DI):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 num_hidden=1, num_layers=1, path_model=None):
        super().__init__(var_x, var_y, dt, rng)
        # Here we use the noisy state measurements as input to the RNN.

        self.n_y_control = self.n_u_process

        # State cost matrix:
        self.Q = q * np.eye(self.n_x_process)

        # Control cost matrix:
        self.R = r * np.eye(self.n_y_control)

        self.model = RNNModel(num_hidden, num_layers)
        # self.rnn.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def get_control(self, x, u):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = mx.nd.array(np.expand_dims(u, [0, 1]))
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x = mx.nd.array(np.reshape(x, (-1, 1, self.model.num_hidden)))
        y, x = self.model(u, x)
        return y.asnumpy().ravel(), x[0].asnumpy().ravel()

    def step(self, t, x, u):
        # Todo: The RNN hidden states need to be initialized better.
        x_rnn = np.zeros((self.model.num_layers, self.model.num_hidden))
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(x_rnn, y)[0] + u)


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
                                          config.process.STATE_COVARIANCE, n)

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
        plot_phase_diagram(monitor.get_last_trajectory(),
                           # odefunc=system_closed.step,
                           xt=config.controller.STATE_TARGET,
                           path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_rnn.py')

    main(_config)

    sys.exit()
