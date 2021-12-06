import os
import sys

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_lqr import DiLqr
from src.double_integrator.utils import (
    plot_timeseries2, plot_phase_diagram, RNG, Monitor)


class DiLqg(DiLqr):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        super().__init__(var_x, var_y, dt, rng, q, r)

        self.n_y_process = 1  # Number of process outputs

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Only observe position.
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # Kalman gain matrix:
        self.L = self.get_Kalman_gain()

    def get_Kalman_gain(self):
        # Solve LQE. Returns Kalman estimator gain L, solution P to Riccati
        # equation, and eigenvalues F of estimator poles A-LC.
        L, P, F = control.lqe(self.A, np.eye(self.n_x_process), self.C, self.W,
                              self.V)
        return L

    def apply_filter(self, t, mu, Sigma, u, y, asymptotic=True):
        mu = self.system.integrate(t, mu, u, deterministic=True)

        if asymptotic:
            L = self.L
        else:
            Sigma = self.A @ Sigma @ self.A.T + self.W
            L = Sigma @ self.C.T @ np.linalg.inv(self.C @ Sigma @ self.C.T +
                                                 self.V)
            Sigma = (1 - L @ self.C) @ Sigma

        mu += self.dt * L @ (y - self.system.output(t, mu, u,
                                                    deterministic=True))

        return mu, Sigma


def main(config):

    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))

    # Create double integrator with LQG feedback.
    di_lqg = DiLqg(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R)
    system_open = di_lqg.system

    # Sample some initial states.
    n = 1
    X0 = di_lqg.get_initial_states(mu0, Sigma0, n)

    times = np.linspace(0, T, num_steps, endpoint=False)
    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with LQG control.
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        x_est = mu0
        Sigma = Sigma0
        for t in times:
            u = di_lqg.get_control(x_est)
            x = system_open.integrate(t, x, u)
            y = system_open.output(t, x, u)
            x_est, Sigma = di_lqg.apply_filter(t, x_est, Sigma, u, y)
            c = di_lqg.get_cost(x_est, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                     state_estimates=x_est)

        path = os.path.join(path_out, 'timeseries_lqg_' + str(i))
        plot_timeseries2(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_lqg_' + str(i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=di_lqg.get_closed_loop,
                           xt=config.controller.STATE_TARGET,
                           path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqg.py')

    main(_config)

    sys.exit()
