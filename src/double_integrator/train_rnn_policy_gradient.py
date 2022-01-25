import os
import sys
import time
from itertools import product, count

import mxnet as mx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.double_integrator import configs
from src.double_integrator.control_systems import DiRnn
from src.double_integrator.di_rnn import add_variables
from src.double_integrator.plotting import plot_training_curve, float2str
from src.double_integrator.train_rnn import get_model_name
from src.double_integrator.utils import apply_config, RNG, Monitor


def train_single(config, verbose=True, plot_loss=True, save_model=True):

    batch_size = config.training.BATCH_SIZE
    lr = config.training.LEARNING_RATE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    beta = 0.05
    epsilon = np.finfo(np.float32).eps.item()
    cost_discount = 0.99
    reward_threshold = 1e-3
    max_num_episodes = 10000
    rnn_kwargs = {'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'activation': config.model.ACTIVATION}

    # Create double integrator with RNN feedback.
    system = DiRnn(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   model_kwargs=rnn_kwargs)

    model = system.model

    # Sample some initial states.
    X0 = system.process.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    training_costs = []
    validation_costs = []
    running_cost = 0
    for episode in count():
        if episode > max_num_episodes:
            break
        tic = time.time()
        monitor.update_parameters(experiment=episode,
                                  process_noise=process_noise,
                                  observation_noise=observation_noise)
        x = X0[np.random.choice(len(X0))]
        x_rnn = np.zeros((system.model.num_layers,
                          system.model.num_hidden))
        y = system.process.output(0, x, 0)
        episode_costs = []
        for t in times:
            x, y, u, c, x_rnn = system.step(t, x, y, x_rnn)
            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

            episode_costs.append(c)

        running_cost = beta * np.sum(episode_costs) + (1 - beta) * running_cost

        discounted_cost = 0
        expected_costs = []
        for c in reversed(episode_costs):
            discounted_cost = c + cost_discount * discounted_cost
            expected_costs.append(discounted_cost)

        expected_costs = (np.array(expected_costs) - np.mean(expected_costs)) \
                         / (np.std(expected_costs) + epsilon)
        expected_costs = mx.nd.array(expected_costs[::-1])


        training_costs.append(running_cost)

        if verbose:
            print("Episode {:3} ({:2.1f} s): loss {:.3e}, running loss {:.3e}."
                  "".format(episode, time.time() - tic, episode_costs[-1],
                            running_cost))
        if running_cost > reward_threshold:
            print("Done training.")
            break

    if plot_loss:
        path_figures = config.paths.PATH_FIGURES
        w = float2str(config.process.PROCESS_NOISES[0])
        v = float2str(config.process.OBSERVATION_NOISES[0])
        path_plot = os.path.join(path_figures, f'training_curve_{w}_{v}.png')
        plot_training_curve(training_costs, validation_costs, path_plot)

    if save_model:
        model.save_parameters(config.paths.FILEPATH_MODEL)
        print("Saved model to {}.".format(config.paths.FILEPATH_MODEL))

    return training_costs, validation_costs


def train_sweep(config):
    path, filename = os.path.split(config.paths.FILEPATH_MODEL)
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    dfs = []
    config.defrost()
    for w, v in tqdm(product(process_noises, observation_noises), leave=False):
        path_model = os.path.join(path, get_model_name(filename, w, v))
        config.paths.FILEPATH_MODEL = path_model
        config.process.PROCESS_NOISES = [w]
        config.process.OBSERVATION_NOISES = [v]

        t_loss, v_loss = train_single(config, verbose=True)

        dfs.append(pd.DataFrame({'process_noise': w,
                                 'observation_noise': v,
                                 'training_loss': t_loss,
                                 'validation_loss': v_loss}))

    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(config.paths.FILEPATH_OUTPUT_DATA)


if __name__ == '__main__':
    _config = configs.config_train_rnn_reinforce.get_config()

    apply_config(_config)

    print(_config)

    train_sweep(_config)

    sys.exit()
