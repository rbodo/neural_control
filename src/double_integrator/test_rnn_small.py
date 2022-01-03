import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.double_integrator import configs
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.di_rnn import add_variables as _add_variables
from src.double_integrator.train_rnn import get_model_name, get_trajectories
from src.double_integrator.utils import (RNG, Monitor, select_noise_subset,
                                         split_train_test, apply_config)


def add_variables(monitor: Monitor):
    _add_variables(monitor)

    dtype = 'float32'
    kwargs = [
        dict(name='rnn_states', label='States',
             column_labels=[r'$h_0$', r'$h_1$'], dtype=dtype),
        dict(name='state_estimates', label='States',
             column_labels=[r'$\hat{x}$', r'$\hat{v}$'], dtype=dtype),
    ]
    for k in kwargs:
        monitor.add_variable(**k)


def run_single(system, times, monitor, inits, x_est):
    x = inits['x']
    x_rnn = inits['x_rnn']
    y = inits['y']

    for i, t in enumerate(times):
        x, y, u, c, x_rnn = system.step(t, x, y, x_rnn)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                 rnn_states=x_rnn, state_estimates=x_est[:, i])


def main(config):

    gpu = 1
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
    validation_fraction = config.training.VALIDATION_FRACTION
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'activation': config.model.ACTIVATION}

    data = pd.read_pickle(path_data)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    for w in tqdm(process_noises, 'Process noise', leave=False):

        monitor.update_parameters(process_noise=w)

        for v in tqdm(observation_noises, 'Observation noise', leave=False):
            monitor.update_parameters(observation_noise=v)

            if not use_single_model:
                path_model = os.path.join(path,
                                          get_model_name(model_name, w, v))
            system = DiRnn(w, v, dt, RNG, q, r, path_model, rnn_kwargs, gpu)

            _, test_data = split_train_test(
                select_noise_subset(data, [w], [v]), validation_fraction)

            X = get_trajectories(test_data, num_steps, 'states')
            X0 = X[:, :, 0]
            X_est = get_trajectories(test_data, num_steps, 'estimates')
            for i, (x, x_est) in enumerate(zip(X0, X_est)):

                monitor.update_parameters(experiment=i)
                x_rnn = np.zeros((system.model.num_layers,
                                  system.model.num_hidden))
                y = system.process.output(0, x, 0)
                inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
                run_single(system, times, monitor, inits, x_est)

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(path_out)


if __name__ == '__main__':

    _config = configs.config_test_rnn_small.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
