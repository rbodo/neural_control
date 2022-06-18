import sys
import os

import numpy as np
import gym
from gym import spaces
from gym.utils.env_checker import check_env
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt
# from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
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
        # self.Q[0, 0] *= 10#

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.states = None
        self.cost = None
        self.t = None

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt).item()

    def step(self, action):

        self.states = self.process.step(self.t, self.states, action)
        np.clip(self.states, self.min, self.max, self.states)

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


def get_states(model, i_step, num_steps=1):
    k, r = divmod(i_step, num_steps)
    if r == 0:
        return model.replay_buffer.observations[num_steps * (k - 1):
                                                num_steps * k]


def plot_trajectory(states, path=None, show=False):
    plt.plot(states[:, 0, 0], states[:, 0, 1])
    plt.plot(0, 0, 'kx')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('v')
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path)
    return plt.gcf()


class FigureRecorderTrain(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if hasattr(self.model, 'rollout_buffer'):
            states = self.model.rollout_buffer.observations
        else:
            states = get_states(self.model, self.n_calls, 1000)
            if states is None:
                return
        figure = plot_trajectory(states)
        self.logger.record("trajectory/train", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        plt.close()


class FigureRecorderTest(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if hasattr(self.model, 'replay_buffer') and self.n_calls % 1000 > 0:
            return
        states = eval_rnn(self.locals['env'].envs[0], self.model)
        figure = plot_trajectory(states)
        self.logger.record("trajectory/eval", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        plt.close()


def eval_rnn(env, model):
    x = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    states = []
    while True:
        u, lstm_states = model.predict(x, state=lstm_states,
                                       episode_start=episode_starts,
                                       deterministic=True)
        x, reward, done, info = env.step(u)
        episode_starts = done
        states.append(env.states)
        if done:
            env.reset()
            break
    print(f"Final reward: {reward}")
    return np.expand_dims(states, 1)


def eval_mlp(env, model):
    x = env.reset()
    states = []
    while True:
        u, _ = model.predict(x, deterministic=True)
        x, reward, done, info = env.step(u)
        states.append(env.states)
        if done:
            env.reset()
            break
    print(f"Final reward: {reward}")
    return np.array(states)


def main(save_model_to=None, load_model_from=None):
    gpu = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'

    num_steps = 500
    env = DoubleIntegrator(var_x=1e-2, var_y=1e-1)
    # env = DoubleIntegrator(q=1, r=0.01)
    env = TimeLimit(env, num_steps)
    check_env(env)

    log_dir = os.path.join(_path, 'tensorboard_log')
    model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, device='cuda',
                         tensorboard_log=log_dir)
    # model = PPO('MlpPolicy', env, verbose=1, device='cuda',
    #             tensorboard_log=log_dir)
    if load_model_from is None:
        model.learn(int(2e5), callback=[  # FigureRecorderTrain(),
                                        FigureRecorderTest()])
        if save_model_to is not None:
            model.save(save_model_to)
    else:
        model = model.load(load_model_from)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20,
    #                                           warn=False)
    # print(mean_reward)
    states = eval_rnn(env, model)
    plot_trajectory(states, path_figure)


if __name__ == '__main__':
    _path = '/home/bodrue/Data/neural_control/double_integrator/rnn_pid'
    path_model = os.path.join(_path, 'models', 'lqg_rnn_ppo')
    path_figure = os.path.join(_path, 'figures', 'lqg_rnn_ppo_trajectory.png')
    main(save_model_to=path_model)

    sys.exit()
