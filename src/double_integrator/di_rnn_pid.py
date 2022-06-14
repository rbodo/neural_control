import sys

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from typing import TYPE_CHECKING

from src.double_integrator import configs
from src.double_integrator.control_systems import DiPidRnn, DiRnn, DiLqg
from src.double_integrator.di_rnn import (add_variables,
                                          run_single as run_uncontrolled)
from src.double_integrator.di_lqg import jitter
from src.double_integrator.plotting import plot_cost_vs_noise_control, \
    plot_trajectories_vs_noise_control
from src.double_integrator.train_rnn import get_model_name, get_trajectories
from src.double_integrator.utils import (apply_config, Monitor, RNG,
                                         split_train_test, select_noise_subset)
from src.ff_pid.brownian import brownian

if TYPE_CHECKING:
    from yacs.config import CfgNode


def run_pid(system, times, monitor, inits):
    x = inits['x']
    x_rnn_perturbed = inits['x_rnn']
    x_rnn_setpoint = x_rnn_perturbed.copy()
    y = inits['y']

    for t in times:
        x, y, u, c, x_rnn_perturbed, x_rnn_setpoint = system.step(
            t, x, y, x_rnn_perturbed, x_rnn_setpoint)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)


def run_lqg(system, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']

    for t in times:
        x, y, u, c, x_est, Sigma = system.step(t, x, x_est, Sigma)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)


def apply_threshold_drift(model, dt, delta, drift, rng):
    shape = model.decoder.weight.shape
    w = np.ravel(model.decoder.weight.data().asnumpy())
    w = brownian(w, 1, dt, delta, drift, None, rng)[0]
    model.decoder.weight.data()[:] = np.reshape(w, shape)


def main(config: 'CfgNode'):

    gpu = 1
    path_data = config.paths.FILEPATH_INPUT_DATA
    filepath_output_data = config.paths.FILEPATH_OUTPUT_DATA
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES
    drift_levels = config.perturbation.DRIFT_LEVELS
    delta = config.perturbation.DELTA
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    filepath_model = config.paths.FILEPATH_MODEL
    path, model_name = os.path.split(filepath_model)
    validation_fraction = config.training.VALIDATION_FRACTION
    num_layers = config.model.NUM_LAYERS
    num_hidden = config.model.NUM_HIDDEN
    rnn_kwargs = {'num_layers': num_layers,
                  'num_hidden': num_hidden,
                  'activation': config.model.ACTIVATION}
    k_p = config.controller.KP
    k_i = config.controller.KI
    k_d = config.controller.KD

    monitor = Monitor()
    add_variables(monitor)

    w = process_noises[0]
    v = observation_noises[0]
    monitor.update_parameters(process_noise=w, observation_noise=v)

    data = pd.read_pickle(path_data)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')
    _x_rnn = np.zeros((num_layers, num_hidden))

    _, test_data = split_train_test(
        select_noise_subset(data, [w], [v]), validation_fraction)
    X = get_trajectories(test_data, num_steps, 'states')
    X0 = X[:, :, 0]

    path_model = os.path.join(path, get_model_name(model_name, w, v))
    system = DiRnn(w, v, dt, RNG, q, r, path_model, rnn_kwargs, gpu)
    system_pid = DiPidRnn(w, v, dt, RNG, q, r, path_model, rnn_kwargs, gpu,
                          k_p, k_i, k_d)
    system_lqg = DiLqg(w, v, dt, RNG, q, r)

    # Initialize the state estimate.
    X0_est = jitter(X0, Sigma0, RNG)
    monitor.update_parameters(control_mode='LQG')
    monitor.update_parameters(perturbation_mode='Threshold drift')
    monitor.update_parameters(perturbation_level=drift_levels[0])
    for i, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):
        monitor.update_parameters(experiment=i)
        inits = {'x': x, 'x_est': x_est, 'Sigma': Sigma0}
        run_lqg(system_lqg, times, monitor, inits)
        break

    # Sweep the various perturbation types separately (process noise,
    # observation noise, W dropout, H drift). Measure the LQR loss for the
    # uncontrolled and controlled system. In case of process and observation
    # noise also compare against the LQG used as oracle.
    monitor.update_parameters(perturbation_mode='Threshold drift')
    for drift in tqdm(drift_levels, 'Threshold drift', leave=False):
        monitor.update_parameters(perturbation_level=drift)
        apply_threshold_drift(system.model, dt, delta, drift, RNG)
        apply_threshold_drift(system_pid.model, dt, delta, drift, RNG)

        for i, x in enumerate(X0):
            monitor.update_parameters(experiment=i)

            monitor.update_parameters(control_mode='pid')
            x_rnn = _x_rnn.copy()
            y = system_pid.process.output(0, x, 0)
            inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
            run_pid(system_pid, times, monitor, inits)
            system_pid.pid.reset()

            monitor.update_parameters(control_mode='none')
            x_rnn = _x_rnn.copy()
            y = system.process.output(0, x, 0)
            inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
            run_uncontrolled(system, times, monitor, inits)
            break#

    # Store state trajectories and corresponding control signals.
    df = monitor.get_dataframe()
    df.to_pickle(filepath_output_data)
    print(f"Saved data to {filepath_output_data}.")

    path_figures = config.paths.PATH_FIGURES
    path = os.path.join(path_figures, 'cost_vs_noise.png')
    plot_cost_vs_noise_control(df, path)
    path = os.path.join(path_figures, 'trajectories_vs_noise.png')
    plot_trajectories_vs_noise_control(df, path, [-1, 1], [-1, 1])


if __name__ == '__main__':
    _config = configs.config_di_rnn_pid.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
