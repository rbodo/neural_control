import os
import sys
from collections import OrderedDict

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.double_integrator_lqr import DiLqr
from src.double_integrator.utils import (
    process_dynamics, process_output, StochasticInterconnectedSystem,
    DIMENSION_MAP, plot_timeseries, plot_phase_diagram,
    lqe_dynamics, lqe_controller_output)


class DiLqg(DiLqr):
    def __init__(self, q=0.5, r=0.5, var_x=0, var_y=0):
        super().__init__(q, r, var_x)

        self.n_y_process = 1                 # Number of process outputs
        self.n_x_control = self.n_x_process  # Number of control states
        self.n_u_control = self.n_y_process  # Number of control inputs

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

        controller = control.NonlinearIOSystem(
            lqe_dynamics, lqe_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            states=self.n_x_control,
            name='control',
            params={'A': self.A,
                    'B': self.B,
                    'C': self.C,
                    'D': self.D,
                    'K': self.K,
                    'L': self.L})

        connections = self._get_system_connections()

        system_closed = StochasticInterconnectedSystem(
            [system_open, controller], connections, outlist=['control.y[0]'])

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with LQR feedback.
    di_lqg = DiLqg(config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
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

    # Simulate the system with LQR control.
    for x0 in X0:
        t, y, x = control.input_output_response(system_closed, times, X0=x0,
                                                return_x=True)

        # Compute cost, using only true, not observed, states.
        c = di_lqg.get_cost(x[:di_lqg.n_x_process], y)
        print("Total cost: {}.".format(np.sum(c)))

        path_out = config.paths.PATH_OUT
        plot_timeseries(t, None, y, x, c, DIMENSION_MAP,
                        os.path.join(path_out, 'timeseries_lqg'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           odefunc=system_closed.dynamics, W=di_lqg.W,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_out, 'phase_diagram_lqg'))


if __name__ == '__main__':

    _config = get_config('configs/config_lqg.py')

    main(_config)

    sys.exit()
