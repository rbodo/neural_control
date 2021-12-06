import os
import sys

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_open import DI
from src.double_integrator.utils import (
    get_lqr_cost, plot_timeseries2, plot_phase_diagram, RNG, Monitor)


class DiLqr(DI):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        super().__init__(var_x, var_y, dt, rng)

        self.n_y_process = self.n_x_process  # Number of process outputs
        self.n_y_control = self.n_u_process  # Number of control outputs
        self.n_u_control = self.n_y_process  # Number of control inputs

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Assume both states are perfectly observable.
        self.C[1, 1] = 1
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # State cost matrix:
        self.Q = q * np.eye(self.n_x_process)

        # Control cost matrix:
        self.R = r * np.eye(self.n_y_control)

        # Feedback gain matrix:
        self.K = self.get_feedback_gain()

    def get_feedback_gain(self):
        # Solve LQR. Returns state feedback gain K, solution S to Riccati
        # equation, and eigenvalues E of closed-loop system.
        K, S, E = control.lqr(self.A, self.B, self.Q, self.R)
        return K

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def get_control(self, x):
        return -self.K.dot(x)

    def get_closed_loop(self, t, x, u):
        return self.system.dynamics(t, x, self.get_control(x) + u)


def main(config):

    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator with LQR feedback.
    di_lqr = DiLqr(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R)
    system_open = di_lqr.system

    # Sample some initial states.
    X0 = di_lqr.get_initial_states(config.process.STATE_MEAN,
                                   config.process.STATE_COVARIANCE, 1)

    times = np.linspace(0, T, num_steps, endpoint=False)
    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with LQR control.
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        for t in times:
            u = di_lqr.get_control(x)
            x = system_open.integrate(t, x, u)
            y = system_open.output(t, x, u)
            c = di_lqr.get_cost(x, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        path = os.path.join(path_out, 'timeseries_lqr_' + str(i))
        plot_timeseries2(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_lqr_' + str(i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=di_lqr.get_closed_loop,
                           xt=config.controller.STATE_TARGET,
                           path=path)


if __name__ == '__main__':

    _config = get_config('configs/config_lqr.py')

    main(_config)

    sys.exit()
