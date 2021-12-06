import os
import sys
from collections import OrderedDict

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.double_integrator_open import DI
from src.double_integrator.utils import (
    process_dynamics, process_output, get_lqr_cost_vectorized, DIMENSION_MAP,
    plot_timeseries, plot_phase_diagram, lqr_controller_output)


class DiLqr(DI):
    def __init__(self, q=0.5, r=0.5, var_x=0):
        super().__init__(var_x)

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
        return get_lqr_cost_vectorized(x, u, self.Q, self.R)

    def _get_system_connections(self):
        # Connect input ports of process with output ports of controller.
        # Connect input ports of controller with output ports of process.
        # The first entry in the tuple specifies the system index
        # {0: process, 1: controller}. The second entry specifies the component
        # within the input or output vector. The first tuple in a sublist
        # specifies an input port, the second tuple an output port of the
        # corresponding system.
        connections = [[(0, i), (1, i)] for i in range(self.n_u_process)] + \
                      [[(1, i), (0, i)] for i in range(self.n_u_control)]
        # # Equivalent, but less scalable:
        # connections = [['system_open.u[0]', 'control.y[0]'],
        #                ['control.u[0]', 'system_open.y[0]']]
        return connections

    def get_system(self):
        dt = None
        system_open = control.NonlinearIOSystem(
            process_dynamics, process_output,
            inputs=self.n_u_process,
            outputs=self.n_y_process,
            states=self.n_x_process,
            name='system_open',
            dt=dt,
            params={'A': self.A,
                    'B': self.B,
                    'C': self.C,
                    'D': self.D,
                    'W': self.W})

        controller = control.NonlinearIOSystem(
            None, lqr_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            name='control',
            dt=dt,
            params={'K': self.K})

        connections = self._get_system_connections()

        system_closed = control.InterconnectedSystem(
            [system_open, controller], connections, outlist=['control.y[0]'],
        dt=dt)

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with LQR feedback.
    di_lqr = DiLqr(config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   config.process.PROCESS_NOISE)
    system_closed = di_lqr.get_system()

    # Sample some initial states.
    X0 = di_lqr.get_initial_states(config.process.STATE_MEAN,
                                   config.process.STATE_COVARIANCE)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system with LQR control.
    for x0 in X0:
        t, y, x = control.input_output_response(system_closed, times, X0=x0,
                                                return_x=True)

        c = di_lqr.get_cost(x, y)
        print("Total cost: {}.".format(np.sum(c)))

        path_out = config.paths.PATH_OUT
        plot_timeseries(t, None, y, x, c, DIMENSION_MAP,
                        os.path.join(path_out, 'timeseries_lqr'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           odefunc=system_closed.dynamics,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_out, 'phase_diagram_lqr'))


if __name__ == '__main__':

    _config = get_config('configs/config_lqr.py')

    main(_config)

    sys.exit()
