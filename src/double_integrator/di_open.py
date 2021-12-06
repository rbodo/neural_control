import os
import sys

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.utils import (
    get_initial_states, RNG, plot_timeseries, plot_phase_diagram,
    StochasticLinearIOSystem, Monitor)


class DI:
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None):
        self.dt = dt

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
        self.W = var_x * np.eye(self.n_x_process) if var_x else None

        # Output noise:
        self.V = var_y * np.eye(self.n_y_process) if var_y else None

        ss = control.StateSpace(self.A, self.B, self.C, self.D, self.dt)
        self.system = StochasticLinearIOSystem(ss, self.W, self.V, rng=rng)

    def get_initial_states(self, mu, Sigma, n=1):
        return get_initial_states(mu, Sigma, self.n_x_process, n)

    def step(self, t, x, u):
        return self.system.dynamics(t, x, u)


def main(config):

    label = 'open'
    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator in open loop configuration.
    system_closed = DI(process_noise, observation_noise, dt, RNG)
    system_open = system_closed.system

    # Sample some initial states.
    X0 = system_closed.get_initial_states(config.process.STATE_MEAN,
                                          config.process.STATE_COVARIANCE, 1)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])

    # Simulate the system without control.
    u = [0]
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        for t in times:
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            monitor.update_variables(t, states=x, outputs=y)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=system_closed.step,
                           xt=config.controller.STATE_TARGET,
                           path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_open.py')

    main(_config)

    sys.exit()
