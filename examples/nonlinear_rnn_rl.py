import logging
import os
import sys
from typing import Union, Optional, Tuple

import gym
import mlflow
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

from examples import configs
from examples.linear_rnn_rl import LinearRlPipeline
from src.control_systems import DiGym
from src.plotting import plot_phase_diagram
from src.ppo_recurrent import RecurrentPPO
from src.utils import get_additive_white_gaussian_noise, Monitor

os.environ['LD_LIBRARY_PATH'] += ':/usr/lib/nvidia:~/.mujoco/mujoco210/bin'


class EvaluationCallback(BaseCallback):
    """
    Custom torch callback to evaluate an RL agent and log a phase diagram.
    """
    def _on_step(self) -> bool:
        n = self.n_calls
        env = self.locals['env'].envs[0]
        episode_reward, episode_length = evaluate(self.model, env, 100)
        states, rewards = run_single(env, self.model)
        figure = plot_phase_diagram({'x': states[:, 0, 0],
                                     'v': states[:, 0, 2]}, show=False,
                                    line_label='position, velocity',
                                    draw_endpoints=True)
        figure = plot_phase_diagram({'x': states[:, 0, 1],
                                     'v': states[:, 0, 3]}, fig=figure,
                                    show=False, draw_endpoints=True,
                                    line_label='angle, anglular velocity')
        plt.legend()
        self.logger.record('trajectory/eval', Figure(figure, close=True),
                           exclude=('stdout', 'log', 'json', 'csv'))
        self.logger.record('test/episode_reward', episode_reward)
        mlflow.log_figure(figure, f'figures/trajectory_{n}.png')
        mlflow.log_metric(key='test_reward', value=episode_reward, step=n)
        mlflow.log_metric(key='training_reward',
                          value=self.model.ep_info_buffer[-1]['r'], step=n)
        plt.close()
        return True


def evaluate(model: Union[RecurrentPPO, BaseAlgorithm], env: DiGym, n: int,
             filename: Optional[str] = None,
             deterministic: Optional[bool] = True) -> Tuple[float, float]:
    """Evaluate RL agent and optionally plot phase diagram.

    Parameters
    ----------
    model
        The policy to evaluate.
    env
        The environment to evaluate `model` in.
    n
        Number of evaluation runs.
    filename
        If specified, plot phase diagram and save under given name in the
        mlflow artifact directory.
    deterministic
        If `True`, the actions are selected deterministically.

    Returns
    -------
    results
        A tuple with the average reward and episode length.
    """
    reward_means, episode_lengths = run_n(n, env, model,
                                          deterministic=deterministic)
    if filename is not None:
        states, rewards = run_single(env, model)
        fig = plot_phase_diagram({'x': states[:, 0, 0], 'v': states[:, 0, 2]},
                                 show=False, draw_endpoints=True,
                                 line_label='position, velocity')
        fig = plot_phase_diagram({'x': states[:, 0, 1], 'v': states[:, 0, 3]},
                                 show=False, draw_endpoints=True, fig=fig,
                                 line_label='angle, anglular velocity')
        plt.legend()
        mlflow.log_figure(fig, os.path.join('figures', filename))
    return np.mean(reward_means).item(), np.mean(episode_lengths).item()


def run_n(n: int, *args, **kwargs) -> Tuple[list, list]:
    """Run an RL agent for `n` episodes and return reward and run lengths."""
    reward_means = []
    episode_lengths = []
    for i in range(n):
        states, rewards = run_single(*args, **kwargs)
        reward_means.append(np.sum(rewards).item())
        episode_lengths.append(len(rewards))
    return reward_means, episode_lengths


def run_single(env: DiGym, model: Union[RecurrentPPO, BaseAlgorithm],
               monitor: Optional[Monitor] = None,
               deterministic: Optional[bool] = True
               ) -> Tuple[np.ndarray, list]:
    """Run an RL agent for one episode and return states and rewards.

    Parameters
    ----------
    model
        The policy to evaluate.
    env
        The environment to evaluate `model` in.
    monitor
        A logging container.
    deterministic
        If `True`, the actions are selected deterministically.

    Returns
    -------
    results
        A tuple containing:

        - The environment states with shape (num_timesteps, 1, num_states). The
          second axis is a placeholder for the batch size.
        - Episode rewards with shape (num_timesteps,).
    """
    t = 0
    y = env.reset()
    x = env.states
    reward = 0

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    states = []
    rewards = []
    while True:
        u, lstm_states = model.predict(y, state=lstm_states,
                                       episode_start=episode_starts,
                                       deterministic=deterministic)
        states.append(x)
        if monitor is not None:
            monitor.update_variables(t, states=x, outputs=y, control=u,
                                     cost=-reward)

        y, reward, done, info = env.step(u)
        rewards.append(reward)
        x = env.states
        t += env.dt
        episode_starts = done
        if done:
            env.reset()
            break
    return np.expand_dims(states, 1), rewards


class POMDP(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, sigma: float, rng: np.random.Generator,
                 observation_indices: np.iterable, dt: float):
        """Custom gym ObservationWrapper to remove part of the observations and
        add noise to the others.

        Parameters
        ----------
        env
            Environment to wrap.
        sigma
            Standard deviation of gaussian noise distribution.
        rng
            Pseudo-random number generator.
        observation_indices
            Indices of states to observe.
        dt
            Time constant for stepping through the environment.
        """
        super().__init__(env)
        self.rng = rng
        self.observation_indexes = observation_indices
        self.dt = dt
        num_observations = len(self.observation_indexes)
        self.sigma = np.eye(num_observations) * sigma
        self._add_noise = sigma > 0
        # noinspection PyUnresolvedReferences
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low[observation_indices],
            high=env.observation_space.high[observation_indices],
            shape=(num_observations,), dtype=env.observation_space.dtype)
        self.states = None

    def observation(self, observation):
        self.states = observation
        noise = 0
        if self._add_noise:
            noise = get_additive_white_gaussian_noise(self.sigma, rng=self.rng)
        return observation[self.observation_indexes] + noise


class NonlinearRlPipeline(LinearRlPipeline):

    @staticmethod
    def evaluate(model, env, n, filename=None, deterministic=True):
        return evaluate(model, env, n, filename, deterministic)

    def get_environment(self, **kwargs) -> POMDP:
        """Create a partially observable inverted pendulum gym environment."""

        observation_noise = self.config.process.OBSERVATION_NOISES[0]
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps
        rng = kwargs.get('rng', None)

        environment = gym.make('InvertedPendulum-v2')
        environment = POMDP(environment, observation_noise, rng, [0, 1], dt)
        return environment


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.nonlinear_rnn_rl.get_config()

    pipeline = NonlinearRlPipeline(_config)
    pipeline.main()

    sys.exit()
