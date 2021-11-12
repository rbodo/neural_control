import os
import sys
from collections import OrderedDict

import control
import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.double_integrator_lqg import DiLqg
from src.double_integrator.train_mlp import MLPModel
from src.double_integrator.utils import (
    process_dynamics, process_output, StochasticInterconnectedSystem,
    DIMENSION_MAP, plot_timeseries, plot_phase_diagram, mlp_controller_output,
    lqe_dynamics, lqe_filter_output)


class DiMlpLqe(DiLqg):
    def __init__(self, q=0.5, r=0.5, var_x=0, var_y=0, num_hidden=1,
                 path_model=None):
        super().__init__(q, r, var_x, var_y)
        # In LQG, we used the estimated states for feedback control. Here we
        # use them as input to the MLP. The LQG becomes just a filter; the MLP
        # the controller.
        self.n_u_filter = self.n_y_process  # Filter sees part of process outp.
        self.n_x_filter = self.n_x_process  # Filter estimates process states
        self.n_y_filter = self.n_x_filter  # Filter outputs estimated states
        self.n_u_control = self.n_y_filter  # Controller receives filter output

        self.mlp = MLPModel(num_hidden)
        # self.mlp.hybridize()
        if path_model is None:
            self.mlp.initialize()
        else:
            self.mlp.load_parameters(path_model)

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
            None, mlp_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            name='control',
            params={'mlp': self.mlp})

        connections = self._get_system_connections()

        system_closed = StochasticInterconnectedSystem(
            [system_open, kalman_filter, controller], connections,
            outlist=['control.y[0]'])

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with LQR feedback.
    di_mlp = DiMlpLqe(config.controller.cost.lqr.Q,
                      config.controller.cost.lqr.R,
                      config.process.PROCESS_NOISE,
                      config.process.OBSERVATION_NOISE,
                      config.model.NUM_HIDDEN,
                      config.paths.PATH_MODEL)
    system_closed = di_mlp.get_system()

    # Sample some initial states.
    n = 1
    X0_process = di_mlp.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE, n)
    X0_filter = np.tile(config.process.STATE_MEAN, (n, 1))
    X0 = np.concatenate([X0_process, X0_filter], 1)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system with MLP control.
    for x0 in X0:

        t, y, x = control.input_output_response(system_closed, times, X0=x0,
                                                return_x=True)

        # Compute cost, using only true, not observed, states.
        c = di_mlp.get_cost(x[:di_mlp.n_x_process], y)
        print("Total cost: {}.".format(np.sum(c)))

        path_out = config.paths.PATH_OUT
        plot_timeseries(t, None, y, x, c, DIMENSION_MAP,
                        os.path.join(path_out, 'timeseries_mlp_lqe'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}), W=di_mlp.W,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_out,
                                             'phase_diagram_mlp_lqe'))


if __name__ == '__main__':

    _config = get_config('configs/config_mlp.py')

    main(_config)

    sys.exit()
