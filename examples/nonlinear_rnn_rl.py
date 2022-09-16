import logging
import os
import sys
from typing import Optional, Tuple

import gym
import mlflow
import numpy as np
from matplotlib import pyplot as plt

from examples import configs
from examples.linear_rnn_rl import LinearRlPipeline, run_single, POMDP, run_n
from src.plotting import plot_phase_diagram

os.environ['LD_LIBRARY_PATH'] += \
        ':/usr/lib/nvidia:' + os.path.expanduser('~/.mujoco/mujoco210/bin')


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
                line_label='angle, anglular velocity')
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

        environment = gym.make('InvertedPendulum-v2')
        environment = POMDP(environment, observation_noise, rng,
                            observation_indices, dt)
        return environment


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.nonlinear_rnn_rl.get_config()

    pipeline = NonlinearRlPipeline(_config)
    pipeline.main()

    sys.exit()
