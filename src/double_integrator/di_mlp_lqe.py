import os
import sys

import numpy as np
import mxnet as mx

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_lqg import DiLqg
from src.double_integrator.train_mlp import MLPModel
from src.double_integrator.utils import (
    plot_phase_diagram, RNG, Monitor, plot_timeseries)


class DiMlpLqe(DiLqg):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 num_hidden=1, path_model=None):
        super().__init__(var_x, var_y, dt, rng, q, r)

        self.model = MLPModel(num_hidden)
        # self.mlp.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_control(self, x, u=None):
        # Add dummy dimension for batch size.
        x = mx.nd.array(np.expand_dims(x, 0))
        u = self.model(x)
        return u.asnumpy()[0]

    def step(self, t, x, u):
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(y) + u)


def main(config):

    label = 'mlp_lqe'
    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
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
    n = 1
    X0 = system_closed.get_initial_states(mu0, Sigma0, n)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with MLP control.
    for i, x in enumerate(X0):

        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        x_est = mu0
        Sigma = Sigma0
        for t in times:
            u = system_closed.get_control(x_est)
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            x_est, Sigma = system_closed.apply_filter(t, x_est, Sigma, u, y)
            c = system_closed.get_cost(x_est, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                     state_estimates=x_est)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=system_closed.step,
                           xt=config.controller.STATE_TARGET,
                           path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_mlp_lqe.py')

    main(_config)

    sys.exit()
