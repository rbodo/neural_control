import os
import sys
import time
from itertools import product, count

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn

from src.double_integrator import configs
from src.double_integrator.control_systems import DI
from src.double_integrator.di_lqg import get_grid, jitter
from src.double_integrator.di_rnn import add_variables
from src.double_integrator.plotting import plot_training_curve, float2str
from src.double_integrator.train_rnn import get_model_name
from src.double_integrator.utils import apply_config, RNG, Monitor, \
    get_lqr_cost


class PolicyModel(nn.Module):

    def __init__(self, num_inputs=1, num_hidden=1, num_outputs=1):

        super().__init__()

        self.num_hidden = num_hidden

        self.hidden = nn.Linear(num_inputs, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_outputs)
        # self.decoder = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.decoder(x)
        return x


class Policy:
    def __init__(self, process, q=0.5, r=0.5, model_kwargs: dict = None,
                 gpu: int = 0, dtype='float32'):

        self.process = process
        self.dtype = dtype
        torch.set_default_dtype(getattr(torch, dtype))

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=dtype)

        self.model = PolicyModel(**model_kwargs)
        initialize_with_lqr = False
        if initialize_with_lqr:
            dtype_torch = getattr(torch, dtype)
            with torch.no_grad():
                self.model.decoder.weight = torch.nn.Parameter(
                    torch.tensor([[-1, -np.sqrt(2)], [0, 0]], dtype_torch))
                self.model.decoder.bias = torch.nn.Parameter(
                    torch.tensor([0, 1e-3], dtype_torch))
        self.device = torch.device(f'cuda:{gpu}')
        self.model.to(self.device)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt)

    def get_control(self, u):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = torch.from_numpy(np.expand_dims(u, 0)).to(self.device)
        y = self.model(u)
        return y.squeeze()

    def step(self, t, x, y):
        out = self.get_control(y)
        # mean = out
        # var = 1e-3
        mean = torch.tanh(out[0])
        # mean = out[0]
        var = torch.sigmoid(out[1])
        m = Normal(mean, var)
        u = m.sample()
        logprob = torch.unsqueeze(m.log_prob(u), 0)
        u = np.array(u.clone().detach().cpu(), self.dtype, ndmin=1)
        # u = -np.dot([[1, np.sqrt(2)]], y)
        # logprob = 1
        x = self.process.step(t, x, u)
        x[0] = np.clip(x[0], -1, 1)
        y = self.process.output(t, x, u)
        c = self.get_cost(x, u)

        return x, y, u, c, logprob

    def dynamics(self, t, x, u):
        y = self.process.output(t, x, u)
        u = self.get_control(y)

        return self.process.dynamics(t, x, u)


class DiMLP(Policy):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 model_kwargs: dict = None, gpu: int = 0, dtype='float32'):
        num_inputs = 1
        num_outputs = 2
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
    max_num_episodes = 10000
    gpu = 2
    torch.manual_seed(54)
    writer = SummaryWriter(config.paths.PATH_BASE)

    model_kwargs = {'num_inputs': 2,
                    'num_hidden': config.model.NUM_HIDDEN,
                    'num_outputs': 2}

    # Create double integrator with MLP feedback.
    system = DiMLP(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   model_kwargs=model_kwargs, dtype=dtype, gpu=gpu)

    model = system.model
    device = system.device
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Sample some initial states.
    grid = get_grid(grid_size)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG).astype(dtype)

    writer.add_graph(model, torch.from_numpy(np.expand_dims(X0[0], 0)).to(
        device))

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
        x = X0[np.random.choice(len(X0))]
        y = system.process.output(0, x, 0)
        episode_costs = []
        logprobs = []
        for t in times:
            x, y, u, c, logprob = system.step(t, x, y)
            episode_costs.append(-c)
            logprobs.append(logprob)
            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        running_cost = beta * np.sum(episode_costs) + (1 - beta) * running_cost
        writer.add_scalar('Total episode reward', np.sum(episode_costs),
                          episode)
        writer.add_histogram('Weights/hidden', model.hidden1.weight, episode)
        writer.add_histogram('Biases/hidden', model.hidden1.bias, episode)
        writer.add_histogram('Weights/decoder', model.decoder.weight, episode)
        writer.add_histogram('Biases/decoder', model.decoder.bias, episode)
        writer.close()

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
            print("\nEpisode {:3} ({:2.1f} s): Total loss of episode {:.3e}, "
                  "running loss {:.3e}.".format(episode, time.time() - tic,
                                                np.sum(episode_costs),
                                                running_cost))
        if running_cost < reward_threshold:
            print("Done training.")
            break

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
    _config = configs.config_train_mlp_lqr_reinforce.get_config()

    apply_config(_config)

    print(_config)

    train_sweep(_config)

    sys.exit()
