import sys
import os

import numpy as np
import gym
from gym import spaces
from gym.utils.env_checker import check_env
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import Figure

from src.double_integrator.control_systems import DI
from src.double_integrator.utils import get_lqr_cost


class DoubleIntegrator(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render.modes': ['console']}

    def __init__(self, var_x=0., var_y=0., dt=0.1, rng=None,
                 cost_threshold=1e-3, state_threshold=None, q=0.5, r=0.5,
                 dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        self.process = DI(num_inputs, num_outputs, num_states, var_x, var_y,
                          dt, rng)

        self.min = -1
        self.max = 1
        self.action_space = spaces.Box(-1, 1, (1,), self.dtype)
        self.observation_space = spaces.Box(self.min, self.max, (num_outputs,),
                                            self.dtype)
        self.init_state_space = spaces.Box(self.min / 2, self.max / 2,
                                           (num_states,), self.dtype)
        self.cost_threshold = cost_threshold
        self.state_threshold = state_threshold or self.max

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)
        self.Q[0, 0] *= 10#

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.states = None
        self.cost = None
        self.t = None

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt).item()

    def step(self, action):

        self.states = self.process.step(self.t, self.states, action)

        self.cost = self.get_cost(self.states, action)

        reward = -self.cost

        done = self.cost < self.cost_threshold
        # or abs(self.states[0].item()) > self.state_threshold

        observation = self.process.output(self.t, self.states, action)
        np.clip(observation, self.min, self.max, observation)

        self.t += self.process.dt

        return observation, reward, done, {}

    def reset(self):

        self.states = self.init_state_space.sample()
        action = 0

        self.cost = self.get_cost(self.states, action)

        self.t = 0

        return self.process.output(self.t, self.states, action)

    def render(self, mode='human'):
        if mode != 'console':
            raise NotImplementedError()
        print("States: ", self.states, "\tCost: ", self.cost)


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_rollout_end(self):
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True


def test(env, model):
    obs = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        env.render('console')


def main():
    gpu = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'

    env = DoubleIntegrator(var_x=1e-2, var_y=1e-1, q=1, r=0.01)#
    env = TimeLimit(env, 500)
    check_env(env)

    log_dir = '/home/bodrue/Data/neural_control/double_integrator/rnn_pid/' \
              'tensorboard_log'
    model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, device='cuda',
                         tensorboard_log=log_dir)

    model.learn(int(1e5), callback=FigureRecorderCallback())

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20,
                                              warn=False)
    print(mean_reward)

    model.save("ppo_recurrent")

    test(env, model)


if __name__ == '__main__':

    main()

    sys.exit()
