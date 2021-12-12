import os
import sys
from collections import OrderedDict

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.deprecated.di_lqr import DiLqr
from src.double_integrator.deprecated.utils import (
    process_dynamics, process_output, StochasticInterconnectedSystem,
    DIMENSION_MAP, plot_timeseries, lqe_dynamics, lqe_filter_output,
    lqr_controller_output)
from src.double_integrator.plotting import plot_phase_diagram


class DiLqg(DiLqr):
    def __init__(self, q=0.5, r=0.5, var_x=0, var_y=0):
        super().__init__(q, r, var_x)

        self.n_y_process = 1                 # Number of process outputs
        self.n_u_filter = self.n_y_process  # Filter sees part of process outp.
        self.n_x_filter = self.n_x_process  # Filter estimates process states
        self.n_y_filter = self.n_x_filter  # Filter outputs estimated states
        self.n_u_control = self.n_y_filter  # Controller receives filter output

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Only observe position.
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # Observation noise:
        self.V = var_y * np.eye(self.n_y_process)

        # Kalman gain matrix:
        self.L = self.get_Kalman_gain()

    def get_Kalman_gain(self):
        # Solve LQE. Returns Kalman estimator gain L, solution P to Riccati
        # equation, and eigenvalues F of estimator poles A-LC.
        L, P, F = control.lqe(self.A, np.eye(self.n_x_process), self.C, self.W,
                              self.V)
        return L

    def _get_system_connections(self):
        connections = [[(0, i), (2, i)] for i in range(self.n_u_process)] + \
                      [[(1, i), (0, i)] for i in range(self.n_y_process)] + \
                      [[(2, i), (1, i)] for i in range(self.n_u_control)]
        return connections

    def get_system(self):

        system_open = control.NonlinearIOSystem(
            process_dynamics, process_output,
            inputs=self.n_u_process,
            outputs=self.n_y_process,
            states=self.n_x_process,
            name='system_open',
            params={'A': self.A,
                    'B': self.B,
                    'C': self.C,
                    'D': self.D,
                    'W': self.W,
                    'V': self.V})

        kalman_filter = control.NonlinearIOSystem(
            lqe_dynamics, lqe_filter_output,
            inputs=self.n_u_filter,
            outputs=self.n_y_filter,
            states=self.n_x_filter,
            name='filter',
            params={'A': self.A,
                    'B': self.B,
                    'C': self.C,
                    'D': self.D,
                    'L': self.L,
                    'K': self.K})

        controller = control.NonlinearIOSystem(
            None, lqr_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            name='control',
            params={'K': self.K})

        connections = self._get_system_connections()

        system_closed = StochasticInterconnectedSystem(
            [system_open, kalman_filter, controller], connections,
            outlist=['control.y[0]', 'system_open.y[0]'])

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with LQG feedback.
    di_lqg = DiLqg(config.controller.cost.lqr.Q,
                   config.controller.cost.lqr.R,
                   config.process.PROCESS_NOISE,
                   config.process.OBSERVATION_NOISE)
    system_closed = di_lqg.get_system()

    # Sample some initial states.
    n = 1
    X0_process = di_lqg.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE, n)
    X0_control = np.tile(config.process.STATE_MEAN, (n, 1))
    X0 = np.concatenate([X0_process, X0_control], 1)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system with LQG control.
    for x0 in X0:
        t, y, x = control.input_output_response(system_closed, times, X0=x0,
                                                return_x=True,
                                                solve_ivp_method='RK45')

        # Keep only control signal from output.
        y = y[:di_lqg.n_y_control]

        # Compute cost, using only true, not observed, states.
        c = di_lqg.get_cost(x[:di_lqg.n_x_process], y)
        print("Total cost: {}.".format(np.sum(c)))

        path_figures = config.paths.PATH_FIGURES
        plot_timeseries(t, None, y, x, c, DIMENSION_MAP,
                        os.path.join(path_figures, 'timeseries_lqg'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           odefunc=system_closed.dynamics, W=di_lqg.W,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_figures, 'phase_diagram_lqg'))


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqg.py')

    main(_config)

    sys.exit()
