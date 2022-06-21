import gc
import logging
import sys
import os

import numpy as np
import gym
import optuna
from gym import spaces
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt
# from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure
from tqdm import trange
from typing import Union

from src.double_integrator.control_systems import DI
from src.double_integrator.utils import get_lqr_cost


class DoubleIntegrator(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render.modes': ['console']}

    def __init__(self, var_x=0., var_y=0., dt=0.1, rng=None,
                 cost_threshold=1e-3, state_threshold=None,
                 q: Union[float, np.iterable] = 0.5, r=0.5, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        self.process = DI(num_inputs, num_outputs, num_states, var_x, var_y,
                          dt, rng)

        self.min = -1
        self.max = 1
        self.action_space = spaces.Box(-10, 10, (1,), self.dtype)
        self.observation_space = spaces.Box(self.min, self.max, (num_outputs,),
                                            self.dtype)
        self.init_state_space = spaces.Box(self.min / 2, self.max / 2,
                                           (num_states,), self.dtype)
        self.cost_threshold = cost_threshold
        self.state_threshold = state_threshold or self.max

        # State cost matrix:
        if np.isscalar(q):
            self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)
        else:
            self.Q = np.diag(q)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.states = None
        self.cost = None
        self.t = None

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt,
                            normalize=True).item()

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

    def reset(self, state_init=None):

        self.states = self.init_state_space.sample() if state_init is None\
            else state_init
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


def eval_rnn(env, model, x0=None, monitor=None):
    t = 0
    y = env.reset(state_init=x0)
    x = env.states
    reward = None

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    states = []
    while True:
        u, lstm_states = model.predict(y, state=lstm_states,
                                       episode_start=episode_starts,
                                       deterministic=True)
        states.append(x)
        if monitor is not None:
            if reward is None:
                reward = -env.get_cost(x, u)
            monitor.update_variables(t, states=x, outputs=y, control=u,
                                     cost=-reward)

        y, reward, done, info = env.step(u)
        x = env.states
        t += env.process.dt
        episode_starts = done
        if done:
            env.reset()
            break
    # print(f"Final reward: {reward}")
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


def main(study: optuna.Study, path, frozen_params=None, show_plots=False):

    if frozen_params is not None:
        study.enqueue_trial(frozen_params)

    trial = study.ask()

    cost_threshold = trial.suggest_categorical('cost_threshold', [1e-4, 1e-3])
    q_x = trial.suggest_float('q_x', 0.1, 10, log=True)
    q_y = trial.suggest_float('q_y', 0.1, 10, log=True)
    r = trial.suggest_float('r', 0.01, 1, log=True)

    num_steps = 500
    env = DoubleIntegrator(var_x=1e-2, var_y=1e-1,
                           cost_threshold=cost_threshold, q=[q_x, q_y], r=r)
    env = TimeLimit(env, num_steps)

    log_dir = os.path.join(path, 'tensorboard_log')
    model = RecurrentPPO('MlpLstmPolicy', env, verbose=0, device='cuda',
                         tensorboard_log=log_dir)
    # model = PPO('MlpPolicy', env, verbose=1, device='cuda',
    #             tensorboard_log=log_dir)
    model.learn(int(1e5), callback=[FigureRecorderTest(),
                                    # FigureRecorderTrain()
                                    ])
    path_model = os.path.join(path_base, 'models',
                              f'lqg_rnn_ppo_{trial.number}')
    model.save(path_model)

    if show_plots:
        states = eval_rnn(env, model)
        path_figure = os.path.join(path, 'figures',
                                   'lqg_rnn_ppo_trajectory.png')
        plot_trajectory(states, path_figure)

    rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=100,
                                               warn=False,
                                               return_episode_rewards=True)
    study.tell(trial, np.mean(episode_lengths))


if __name__ == '__main__':
    gpu = 3

    _label = 'lqg_rnn_ppo'

    # Make sure the keys are spelled exactly as the parameter names in
    # trial.suggest calls. Every parameter listed here will not be swept over.
    _frozen_params = {'cost_threshold': 1e-4}

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.get_logger('optuna').addHandler(
        logging.StreamHandler(sys.stdout))

    path_base = '/home/bodrue/Data/neural_control/double_integrator/rnn_ppo'
    os.makedirs(path_base, exist_ok=True)

    study_name = _label
    filepath_output = os.path.join(path_base, 'optuna', study_name + '.db')
    storage_name = f'sqlite:///{filepath_output}'
    _study = optuna.create_study(storage_name, study_name=study_name,
                                 direction='minimize', load_if_exists=True)

    num_trials = 100
    for _ in trange(num_trials, desc='Optuna'):
        main(_study, path_base, _frozen_params)
        gc.collect()

    print("Number of finished trials: ", len(_study.trials))

    print("Best trial:")
    best_trial = _study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for _key, value in best_trial.params.items():
        print("    {}: {}".format(_key, value))

    sys.exit()
