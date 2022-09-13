import os
import sys
from collections import OrderedDict

import control
import numpy as np

from examples.configs.config import get_config
from scratch._deprecated.utils import (
    process_dynamics, process_output, DIMENSION_MAP, plot_timeseries)
from src.utils import get_initial_states
from src.plotting import plot_phase_diagram


class DI:
    def __init__(self, var_x=0):
        self.n_x_process = 2  # Number of process states
        self.n_y_process = 1  # Number of process outputs
        self.n_u_process = 1  # Number of process inputs

        # Dynamics matrix:
        self.A = np.zeros((self.n_x_process, self.n_x_process))
        self.A[0, 1] = 1

        # Input matrix:
        self.B = np.zeros((self.n_x_process, self.n_u_process))
        self.B[1, 0] = 1  # Control only second state (acceleration).

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Only observe position.
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # Process noise:
        self.W = var_x * np.eye(self.n_x_process)

    def get_initial_states(self, mu, Sigma, n=1):
        return get_initial_states(mu, Sigma, self.n_x_process, n)

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
                    'W': self.W})

        return system_open


def main(config):
    np.random.seed(42)

    # Create double integrator in open loop configuration.
    di = DI(config.process.PROCESS_NOISE)
    system_open = di.get_system()

    # Sample some initial states.
    X0 = di.get_initial_states(config.process.STATE_MEAN,
                               config.process.STATE_COVARIANCE)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system without control.
    for x0 in X0:
        t, y, x = control.input_output_response(system_open, times, 0, x0,
                                                return_x=True)

        path_figures = config.paths.PATH_FIGURES
        plot_timeseries(t, None, y, x, dimension_map=DIMENSION_MAP,
                        path=os.path.join(path_figures, 'timeseries'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           odefunc=system_open.dynamics,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_figures, 'phase_diagram'))


if __name__ == '__main__':

    _config = get_config('configs/config_open.py')

    main(_config)

    sys.exit()
