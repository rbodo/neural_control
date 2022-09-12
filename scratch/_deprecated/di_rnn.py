import os
import sys
from collections import OrderedDict

import control
import numpy as np
import mxnet as mx

from scratch.configs.config import get_config
from scratch._deprecated.di_open import DI
from src.control_systems_mxnet import RnnModel
from scratch._deprecated.utils import (
    process_dynamics, process_output, StochasticInterconnectedSystem,
    DIMENSION_MAP, plot_timeseries, rnn_controller_output,
    rnn_controller_dynamics)
from src.utils import get_lqr_cost_vectorized
from src.plotting import plot_phase_diagram


class DiRnn(DI):
    def __init__(self, var_x=0, var_y=0, num_hidden=1, num_layers=1,
                 path_model=None):
        super().__init__(var_x)
        # Here we use the noisy state measurements as input to the RNN.
        self.n_u_control = self.n_y_process
        self.n_x_control = num_hidden * num_layers
        self.n_y_control = self.n_u_process

        # Observation noise:
        self.V = var_y * np.eye(self.n_y_process)

        self.rnn = RnnModel(num_hidden, num_layers, self.n_y_control,
                            self.n_u_control)
        # self.rnn.hybridize()
        if path_model is None:
            self.rnn.initialize()
        else:
            self.rnn.load_parameters(path_model)

    def _get_system_connections(self):
        connections = [[(0, i), (1, i)] for i in range(self.n_u_process)] + \
                      [[(1, i), (0, i)] for i in range(self.n_u_control)]
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

        controller = control.NonlinearIOSystem(
            rnn_controller_dynamics, rnn_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            states=self.n_x_control,
            name='control',
            params={'rnn': self.rnn,
                    'dt': 1e-1})

        connections = self._get_system_connections()

        system_closed = StochasticInterconnectedSystem(
            [system_open, controller], connections, outlist=['control.y[0]'])

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with RNN feedback.
    di_rnn = DiRnn(config.process.PROCESS_NOISE,
                   config.process.OBSERVATION_NOISE,
                   config.model.NUM_HIDDEN,
                   config.model.NUM_LAYERS,
                   config.paths.PATH_MODEL)
    system_closed = di_rnn.get_system()

    # State cost matrix:
    q = config.controller.cost.lqr.Q
    Q = np.zeros((di_rnn.n_x_process, di_rnn.n_x_process))
    Q[0, 0] = q  # Only position contributes to cost.

    # Control cost matrix:
    r = config.controller.cost.lqr.R
    R = r * np.eye(di_rnn.n_y_control)

    # Sample some initial states.
    n = 1
    X0_process = di_rnn.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE, n)
    X0_control = np.zeros((n, di_rnn.n_x_control))
    X0 = np.concatenate([X0_process, X0_control], 1)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system with RNN control.
    show_rnn_states = True
    c = None
    for x0 in X0:
        # Bring RNN to steady-state on current static input.
        if config.simulation.DO_WARMUP:
            input_shape = (config.simulation.NUM_STEPS // 2, 1,
                           di_rnn.n_u_process)
            _u0 = np.tile(config.process.STATE_MEAN, input_shape)
            _x0 = np.expand_dims(x0[-di_rnn.n_x_control:], (0, 1))
            _, _x0 = di_rnn.rnn(mx.nd.array(_u0), mx.nd.array(_x0))
            x0[-di_rnn.n_x_control:] = _x0[0].asnumpy()

        t, y, x = control.input_output_response(system_closed, times, X0=x0,
                                                return_x=True,
                                                solve_ivp_method='RK45')

        # Keep only control signal from output.
        y = y[:di_rnn.n_y_control]

        if Q is not None and R is not None:
            # Compute cost, using only true, not observed, states.
            c = get_lqr_cost_vectorized(x[:di_rnn.n_x_process], y, Q, R)
            print("Total cost: {}.".format(np.sum(c)))

        path_figures = config.paths.PATH_FIGURES
        _x = x if show_rnn_states else x[:-di_rnn.n_x_control]
        plot_timeseries(t, None, y, _x, c, DIMENSION_MAP,
                        os.path.join(path_figures, 'timeseries_rnn'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           di_rnn.n_x_control, W=di_rnn.W,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_figures, 'phase_diagram_rnn'))


if __name__ == '__main__':

    _config = get_config('configs/config_rnn.py')

    main(_config)

    sys.exit()
