import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.di_lqr import add_variables
from src.double_integrator.di_rnn import run_single
from src.double_integrator.train_rnn import get_model_name, get_trajectories
from src.double_integrator.utils import (RNG, Monitor, select_noise_subset,
                                         split_train_test)


def main(config):

    path_data = config.paths.FILEPATH_INPUT_DATA
    path_out = config.paths.FILEPATH_OUTPUT_DATA
    use_single_model = config.model.USE_SINGLE_MODEL_IN_SWEEP
    path_model = config.paths.FILEPATH_MODEL
    path, model_name = os.path.split(path_model)
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'activation': config.model.ACTIVATION}

    data = pd.read_pickle(path_data)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    for w in tqdm(process_noises, 'Process noise'):

        monitor.update_parameters(process_noise=w)

        for v in tqdm(observation_noises, 'Observation noise'):
            monitor.update_parameters(observation_noise=v)

            if not use_single_model:
                path_model = os.path.join(path,
                                          get_model_name(model_name, w, v))
            system_closed = DiRnn(w, v, dt, RNG, q, r, path_model, rnn_kwargs)
            system_open = system_closed.system

            _, test_data = split_train_test(select_noise_subset(data, w, v))

            X = get_trajectories(test_data, num_steps, 'states')
            X0 = X[:, :, 0]
            for i, x in enumerate(X0):

                monitor.update_parameters(experiment=i)
                x_rnn = np.zeros((system_closed.model.num_layers,
                                  system_closed.model.num_hidden))
                y = system_open.output(0, x, 0)
                inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
                run_single(system_open, system_closed, times, monitor, inits)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(path_out)


if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    # filename = 'config_test_rnn.py'
    filename = 'config_test_rnn_generalization.py'
    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()