import os
import sys
import time
from itertools import product

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna

from src.double_integrator import configs
from src.double_integrator.control_systems import DI
from src.double_integrator.di_lqg import get_grid, jitter
from src.double_integrator.di_rnn import add_variables
from src.double_integrator.lqr_rnn_reinforce import RNN
from src.double_integrator.plotting import plot_training_curve, float2str, \
    plot_phase_diagram
from src.double_integrator.train_rnn import get_model_name
from src.double_integrator.utils import apply_config, Monitor


def objective(trial: optuna.Trial):
    dt = 0.1

    max_rewards = []
    seeds = [42, 52, 103]
    for seed in seeds:
        config = configs.config_train_rnn_lqg_reinforce.get_config()

        config.defrost()

        num_layers = 1  # trial.suggest_int('num_layers', 1, 2)
        num_hidden = trial.suggest_int('num_hidden', 2, 64, log=True)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        neuron_model = trial.suggest_categorical('neuron_model',
                                                 ['rnn', 'gru'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1,
                                            log=True)
        num_steps = trial.suggest_int('num_steps', 100, 10000, log=True)
        reward_discount = trial.suggest_float('reward_discount', 0.6, 1,
                                              log=True)

        config.model.NUM_LAYERS = num_layers
        config.model.NUM_HIDDEN = num_hidden
        config.model.ACTIVATION = activation
        config.model.NEURON_MODEL = neuron_model
        config.training.LEARNING_RATE = learning_rate
        config.simulation.NUM_STEPS = num_steps
        config.training.REWARD_DISCOUNT = reward_discount
        config.simulation.T = num_steps * dt

        config.SEED = seed

        apply_config(config)

        rewards = train_single(config, verbose=True, plot_loss=False,
                               save_model=False)
        max_rewards.append(np.max(rewards))

        config.defrost()

    return np.mean(max_rewards)


class DiRnn(RNN):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 model_kwargs: dict = None, gpu: int = 0, dtype='float32'):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, model_kwargs, gpu, dtype)


def train_single(config, verbose=True, plot_loss=True, save_model=True):

    grid_size = config.simulation.GRID_SIZE
    lr = config.training.LEARNING_RATE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = np.float32(T / num_steps)
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    dtype = 'float32'
    dtype_torch = getattr(torch, dtype)
    torch.set_default_dtype(dtype_torch)
    epsilon = np.finfo(np.float32).eps.item()
    reward_discount = config.training.REWARD_DISCOUNT
    reward_threshold = -1e-2
    max_num_episodes = config.training.NUM_EPOCHS
    log_every_nth_episode = 10
    gpu = None
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
        gpu = 0  # Need to set relative ID
    torch.set_num_threads(1)
    seed = config.SEED
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(os.path.join(config.paths.PATH_BASE, 'tensorboard'))

    rnn_kwargs = {'num_inputs': 1,
                  'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'num_outputs': 2,
                  'activation': config.model.ACTIVATION,
                  'neuron_model': config.model.NEURON_MODEL}

    # Create double integrator with RNN feedback.
    system = DiRnn(process_noise, observation_noise, dt, rng,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   model_kwargs=rnn_kwargs, dtype=dtype, gpu=gpu)

    model = system.model
    device = system.device
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Sample some initial states.
    grid = get_grid(grid_size)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, rng).astype(dtype)

    write_graph_to_tensorboard = False
    if write_graph_to_tensorboard:
        y = system.process.output(0, X0[0], 0)
        writer.add_graph(model, [
            torch.tensor(np.expand_dims(y, [0, 1]), device=device),
            torch.tensor(np.zeros((model.num_layers, 1, model.num_hidden),
                                  dtype), device=device)])

    times = np.linspace(0, T, num_steps, endpoint=False, dtype=dtype)

    log_states = True
    if log_states:
        monitor = Monitor()
        add_variables(monitor)
    else:
        monitor = None

    training_rewards = []
    for episode in range(max_num_episodes):
        tic = time.time()
        if log_states:
            monitor.update_parameters(experiment=episode,
                                      process_noise=process_noise,
                                      observation_noise=observation_noise)
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x_rnn = torch.zeros(model.num_layers, 1, model.num_hidden,
                            dtype=dtype_torch, device=device,
                            requires_grad=True)
        x = X0[np.random.choice(len(X0))]
        y = system.process.output(0, x, 0)
        episode_rewards = []
        logprobs = []
        for t in times:
            x, y, u, c, x_rnn, logprob = system.step(t, x, y, x_rnn)
            episode_rewards.append(-c)
            logprobs.append(logprob)
            if log_states:
                monitor.update_variables(t, states=x, outputs=y, control=u,
                                         cost=c)

        n = 10
        mean_episode_reward = np.mean(episode_rewards[-n:])
        training_rewards.append(mean_episode_reward)
        max_training_reward = np.max(training_rewards)
        mean_training_reward = np.mean(training_rewards[-n:])
        writer.add_scalar('Mean episode reward', mean_episode_reward, episode)

        discounted_reward = 0
        expected_rewards = []
        for c in reversed(episode_rewards):
            discounted_reward = c + reward_discount * discounted_reward
            expected_rewards.append(discounted_reward)

        expected_rewards = expected_rewards[::-1]
        expected_rewards = torch.tensor(expected_rewards, device=device,
                                        dtype=dtype_torch)
        expected_rewards = (expected_rewards - expected_rewards.mean()) / (
            torch.std(expected_rewards) + epsilon)

        # Network outputs the means of a normal distribution that models
        # probability of continuous actions. Assume unit variance. Then
        # after applying the log, only the squared mean remains.
        loss = [-logprob * c for logprob, c in zip(logprobs, expected_rewards)]
        loss = torch.cat(loss).sum()

        optimizer.zero_grad(set_to_none=True)  # for performance reasons

        loss.backward()

        optimizer.step()

        if mean_training_reward > reward_threshold:
            print("Done training.")
            break

        toc = time.time()
        if episode % log_every_nth_episode == 0:
            if verbose:
                print("\nEpisode {:3} finished in {:2.3f} s "
                      "with final reward {:.3f}. "
                      "Current best reward: {:.3f}. "
                      "Rewards averaged over past {} episodes: {:.3f}."
                      "".format(episode, toc - tic, episode_rewards[-1],
                                max_training_reward, n, mean_training_reward))
            log_weights = False
            if log_weights:
                writer.add_histogram('Weights/hidden_hh',
                                     model.rnn.weight_hh_l0, episode)
                writer.add_histogram('Biases/hidden_hh', model.rnn.bias_hh_l0,
                                     episode)
                writer.add_histogram('Weights/hidden_ih',
                                     model.rnn.weight_ih_l0, episode)
                writer.add_histogram('Biases/hidden_ih', model.rnn.bias_ih_l0,
                                     episode)
                writer.add_histogram('Weights/decoder', model.decoder.weight,
                                     episode)
                writer.add_histogram('Biases/decoder', model.decoder.bias,
                                     episode)
            if log_states:
                fig = plot_phase_diagram(monitor.get_last_trajectory(),
                                         show=False,
                                         xt=config.controller.STATE_TARGET,
                                         xlim=[-1.1, 1.1], ylim=[-1.1, 1.1])
                writer.add_figure('Trajectory', fig, episode)
            writer.close()

    if plot_loss:
        path_figures = config.paths.PATH_FIGURES
        w = float2str(config.process.PROCESS_NOISES[0])
        v = float2str(config.process.OBSERVATION_NOISES[0])
        path_plot = os.path.join(path_figures, f'training_curve_{w}_{v}.png')
        plot_training_curve(training_rewards, path_plot)

    if save_model:
        torch.save(model.state_dict(), config.paths.FILEPATH_MODEL)
        print("Saved model to {}.".format(config.paths.FILEPATH_MODEL))

    return training_rewards


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
    optuna.logging.get_logger('optuna').addHandler(
        logging.StreamHandler(sys.stdout))

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'reinforce/lqg'
    study_name = 'rnn_lqg_reinforce'
    filepath_output = os.path.join(base_path, study_name + '.db')
    storage_name = f'sqlite:///{filepath_output}'
    study = optuna.create_study(storage_name, study_name=study_name,
                                direction='maximize', load_if_exists=True)

    # objective(study.trials[11])
    study.optimize(objective, n_trials=10000, timeout=None,
                   show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    sys.exit()
