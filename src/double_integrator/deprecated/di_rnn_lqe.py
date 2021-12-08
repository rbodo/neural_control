import os
import sys
from collections import OrderedDict

import control
import numpy as np
import mxnet as mx

from src.double_integrator.configs.config import get_config
from src.double_integrator.deprecated.di_lqg import DiLqg
from src.double_integrator.train_rnn import RNNModel
from src.double_integrator.deprecated.utils import (
    process_dynamics, process_output, StochasticInterconnectedSystem,
    DIMENSION_MAP, plot_timeseries, rnn_controller_output, lqe_dynamics,
    lqe_filter_output, rnn_controller_dynamics)
from src.double_integrator.plotting import plot_phase_diagram


class DiRnnLqe(DiLqg):
    def __init__(self, q=0.5, r=0.5, var_x=0, var_y=0, num_hidden=1,
                 num_layers=1, path_model=None):
        super().__init__(q, r, var_x, var_y)
        # Here we use the estimated states as input to the RNN. The LQR
        # controller is replaced by the RNN.
        self.n_x_control = num_hidden * num_layers

        self.rnn = RNNModel(num_hidden, num_layers)
        # self.rnn.hybridize()
        if path_model is None:
            self.rnn.initialize()
        else:
            self.rnn.load_parameters(path_model)

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
            rnn_controller_dynamics, rnn_controller_output,
            inputs=self.n_u_control,
            outputs=self.n_y_control,
            states=self.n_x_control,
            name='control',
            params={'rnn': self.rnn,
                    'dt': 1e-1})

        connections = self._get_system_connections()

        system_closed = StochasticInterconnectedSystem(
            [system_open, kalman_filter, controller], connections,
            outlist=['control.y[0]'])

        return system_closed


def main(config):
    np.random.seed(42)

    # Create double integrator with RNN feedback.
    di_rnn = DiRnnLqe(config.controller.cost.lqr.Q,
                      config.controller.cost.lqr.R,
                      config.process.PROCESS_NOISE,
                      config.process.OBSERVATION_NOISE,
                      config.model.NUM_HIDDEN,
                      config.model.NUM_LAYERS,
                      config.paths.PATH_MODEL)
    system_closed = di_rnn.get_system()

    # Sample some initial states.
    n = 1
    X0_process = di_rnn.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE, n)
    X0_filter = np.tile(config.process.STATE_MEAN, (n, 1))
    X0_control = np.zeros((n, di_rnn.n_x_control))
    X0 = np.concatenate([X0_process, X0_filter, X0_control], 1)

    times = np.linspace(0, config.simulation.T, config.simulation.NUM_STEPS,
                        endpoint=False)

    # Simulate the system with RNN control.
    show_rnn_states = True
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
                                                return_x=True)

        # Compute cost, using only true, not observed, states.
        c = di_rnn.get_cost(x[:di_rnn.n_x_process], y)
        print("Total cost: {}.".format(np.sum(c)))

        path_out = config.paths.PATH_OUT
        _x = x if show_rnn_states else x[:-di_rnn.n_x_control]
        plot_timeseries(t, None, y, _x, c, DIMENSION_MAP,
                        os.path.join(path_out, 'timeseries_rnn'))

        plot_phase_diagram(OrderedDict({'x': x[0], 'v': x[1]}),
                           di_rnn.n_x_control, W=di_rnn.W,
                           xt=config.controller.STATE_TARGET,
                           path=os.path.join(path_out,
                                             'phase_diagram_rnn_lqe'))


if __name__ == '__main__':

    _config = get_config('configs/config_rnn_lqe.py')

    main(_config)

    sys.exit()
