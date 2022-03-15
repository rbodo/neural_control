import os
import sys
import time
from itertools import product, count

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.double_integrator import configs
from src.double_integrator.control_systems import DI
from src.double_integrator.di_lqg import get_grid, jitter
from src.double_integrator.di_rnn import add_variables
from src.double_integrator.lqr_rnn_reinforce import RNN
from src.double_integrator.plotting import plot_training_curve, float2str, \
    plot_phase_diagram
from src.double_integrator.train_rnn import get_model_name
from src.double_integrator.utils import apply_config, RNG, Monitor


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
    beta = 0.05
    epsilon = np.finfo(np.float32).eps.item()
    cost_discount = 0.99
    reward_threshold = -1e-1
    max_num_episodes = 1000
    gpu = 2
    torch.manual_seed(54)
    writer = SummaryWriter(config.paths.PATH_BASE)

    rnn_kwargs = {'num_inputs': 1,
                  'num_layers': config.model.NUM_LAYERS,
                  'num_hidden': config.model.NUM_HIDDEN,
                  'num_outputs': 2,
                  'activation': config.model.ACTIVATION}

    # Create double integrator with RNN feedback.
    system = DiRnn(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   model_kwargs=rnn_kwargs, dtype=dtype, gpu=gpu)

    model = system.model
    device = system.device
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Sample some initial states.
    grid = get_grid(grid_size)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG).astype(dtype)

    y = system.process.output(0, X0[0], 0)
    writer.add_graph(model, [
        torch.from_numpy(np.expand_dims(y, [0, 1])).to(device),
        torch.from_numpy(np.zeros((model.num_layers, 1, model.num_hidden),
                                  dtype)).to(device)])

    times = np.linspace(0, T, num_steps, endpoint=False, dtype=dtype)

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
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x_rnn = torch.zeros(model.num_layers, 1, model.num_hidden,
                            dtype=dtype_torch, device=device)
        # x = [1, -0.1]
        x = X0[np.random.choice(len(X0))]
        y = system.process.output(0, x, 0)
        episode_costs = []
        logprobs = []
        for t in times:
            x, y, u, c, x_rnn, logprob = system.step(t, x, y, x_rnn)
            episode_costs.append(-c)
            logprobs.append(logprob)
            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        running_cost = beta * np.sum(episode_costs) + (1 - beta) * running_cost
        writer.add_scalar('Total episode reward', np.sum(episode_costs),
                          episode)

        discounted_cost = 0
        expected_costs = []
        for c in reversed(episode_costs):
            discounted_cost = c + cost_discount * discounted_cost
            expected_costs.append(discounted_cost)

        expected_costs = expected_costs[::-1]
        expected_costs = torch.tensor(expected_costs, device=device,
                                      dtype=dtype_torch)
        expected_costs = (expected_costs - expected_costs.mean()) / (
            torch.std(expected_costs) + epsilon)

        # Network outputs the means of a normal distribution that models
        # probability of continuous actions. Assume unit variance. Then
        # after applying the log, only the squared mean remains.
        loss = [-logprob * c for logprob, c in zip(logprobs, expected_costs)]
        loss = torch.cat(loss).sum()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        training_costs.append(running_cost)

        if verbose:
            print("\nEpisode {:3} ({:2.1f} s): Total loss of episode {:.3f}, "
                  "running loss {:.3f}.".format(episode, time.time() - tic,
                                                np.sum(episode_costs),
                                                running_cost))
        if running_cost > reward_threshold:
            print("Done training.")
            break

        if episode % 10 == 0:
            writer.add_histogram('Weights/hidden_hh', model.rnn.weight_hh_l0,
                                 episode)
            writer.add_histogram('Biases/hidden_hh', model.rnn.bias_hh_l0,
                                 episode)
            writer.add_histogram('Weights/hidden_ih', model.rnn.weight_ih_l0,
                                 episode)
            writer.add_histogram('Biases/hidden_ih', model.rnn.bias_ih_l0,
                                 episode)
            writer.add_histogram('Weights/decoder', model.decoder.weight,
                                 episode)
            writer.add_histogram('Biases/decoder', model.decoder.bias, episode)
            fig = plot_phase_diagram(monitor.get_last_trajectory(), show=False,
                                     xt=config.controller.STATE_TARGET,
                                     xlim=[-1.1, 1.1], ylim=[-1.1, 1.1])
            writer.add_figure('Trajectory', fig, episode)
            writer.close()

    if plot_loss:
        path_figures = config.paths.PATH_FIGURES
        w = float2str(config.process.PROCESS_NOISES[0])
        v = float2str(config.process.OBSERVATION_NOISES[0])
        path_plot = os.path.join(path_figures, f'training_curve_{w}_{v}.png')
        plot_training_curve(training_costs, validation_costs, path_plot)

    if save_model:
        torch.save(model.state_dict(), config.paths.FILEPATH_MODEL)
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
    _config = configs.config_train_rnn_lqg_reinforce.get_config()

    apply_config(_config)

    print(_config)

    train_sweep(_config)

    sys.exit()
