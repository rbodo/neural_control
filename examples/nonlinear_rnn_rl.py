import logging
import os
import sys
from typing import Optional, Tuple

import gymnasium as gym
import mlflow
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from examples import configs
from examples.linear_rnn_rl import LinearRlPipeline, run_single, run_n
from src.plotting import plot_phase_diagram
from src.utils import atleast_3d, get_additive_white_gaussian_noise

matplotlib.use('Agg')


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
        # We add a dummy batch and time dimension to the observations to make
        # the environment compatible with the torch RNN pipeline.
        shape = (1, 1, num_observations)
        # noinspection PyUnresolvedReferences
        self.observation_space = gym.spaces.Box(
            low=atleast_3d(env.observation_space.low[observation_indices]),
            high=atleast_3d(env.observation_space.high[observation_indices]),
            shape=shape, dtype=env.observation_space.dtype)
        self.states = None

    def observation(self, observation):
        self.states = atleast_3d(observation)
        noise = 0
        if self._add_noise:
            noise = get_additive_white_gaussian_noise(self.sigma, rng=self.rng)
        partial_observation = observation[self.observation_indexes]
        return atleast_3d(partial_observation + noise)


class NonlinearRlPipeline(LinearRlPipeline):
    def evaluate(self, env: POMDP, n: int, filename: Optional[str] = None,
                 deterministic: Optional[bool] = True
                 ) -> Tuple[float, float, Optional[plt.Figure]]:
        """Evaluate RL agent and optionally plot phase diagram.

        Parameters
        ----------
        env
            The environment to evaluate model in.
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
        reward_means, episode_lengths = run_n(n, env, self.model,
                                              deterministic=deterministic)
        f = None
        if filename is not None:
            states, rewards = run_single(env, self.model)
            f = plot_phase_diagram(
                {'x': states[:, 0, 0], 'v': states[:, 0, 2]}, show=False,
                draw_endpoints=True, line_label='position, velocity')
            f = plot_phase_diagram(
                {'x': states[:, 0, 1], 'v': states[:, 0, 3]}, show=False,
                draw_endpoints=True, fig=f,
                line_label='angle, angular velocity')
            plt.legend()
            mlflow.log_figure(f, os.path.join('figures', filename))
        return np.mean(reward_means).item(), np.mean(episode_lengths).item(), f

    def get_environment(self, **kwargs) -> POMDP:
        """Create a partially observable inverted pendulum gym environment."""

        observation_indices = self.config.process.OBSERVATION_INDICES
        observation_noise = self.config.process.OBSERVATION_NOISES[0]
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps
        rng = kwargs.get('rng', None)
        environment = gym.make('InvertedPendulum-v4')
        environment = POMDP(environment, observation_noise, rng,
                            observation_indices, dt)
        return environment


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.nonlinear_rnn_rl.get_config()

    pipeline = NonlinearRlPipeline(_config)
    pipeline.main()

    sys.exit()
