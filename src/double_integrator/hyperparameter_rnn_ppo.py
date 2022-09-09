import gc
import logging
import sys
import os
import warnings
from typing import Callable

import mlflow
import numpy as np
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt
# from stable_baselines3 import PPO, SAC
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.integration.mlflow import MLflowCallback, RUN_ID_ATTRIBUTE_KEY
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure
from tqdm import trange

from src.double_integrator.control_systems_torch import DiGym
from src.double_integrator.plotting import plot_phase_diagram
from src.double_integrator.ppo_recurrent import RecurrentPPO, MlpRnnPolicy
from src.double_integrator.utils import get_artifact_path


def get_states(model, i_step, num_steps=1):
    k, r = divmod(i_step, num_steps)
    if r == 0:
        return model.replay_buffer.observations[num_steps * (k - 1):
                                                num_steps * k]


class FigureRecorderTest(BaseCallback):
    def _on_step(self) -> bool:
        n = self.n_calls
        env = self.locals['env'].envs[0]
        episode_reward, episode_length = evaluate(self.model, env, 100)
        states, rewards = run_single(env, self.model)
        figure = plot_phase_diagram({'x': states[:, 0, 0],
                                     'v': states[:, 0, 1]}, xt=[0, 0],
                                    show=False, xlim=[-1, 1], ylim=[-1, 1])
        self.logger.record('trajectory/eval', Figure(figure, close=True),
                           exclude=('stdout', 'log', 'json', 'csv'))
        self.logger.record('test/episode_reward', episode_reward)
        self.logger.record('test/episode_length', episode_length)
        mlflow.log_figure(figure, f'figures/trajectory_{n}.png')
        mlflow.log_metric(key='test_reward', value=episode_reward, step=n)
        mlflow.log_metric(key='test_episode_length', value=episode_length,
                          step=n)
        mlflow.log_metric(key='training_reward',
                          value=self.model.ep_info_buffer[-1]['r'], step=n)
        plt.close()
        return True


def evaluate(model, env, n, filename=None, deterministic=True):
    reward_means, episode_lengths = run_n(n, env, model,
                                          deterministic=deterministic)
    if filename is not None:
        states, rewards = run_single(env, model)
        fig = plot_phase_diagram({'x': states[:, 0, 0], 'v': states[:, 0, 1]},
                                 xt=[0, 0], show=False, xlim=[-1, 1],
                                 ylim=[-1, 1])
        mlflow.log_figure(fig, os.path.join('figures', filename))
    return np.mean(reward_means).item(), np.mean(episode_lengths).item()


def run_n(n, *args, **kwargs):
    reward_means = []
    episode_lengths = []
    for i in range(n):
        states, rewards = run_single(*args, **kwargs)
        reward_means.append(np.sum(rewards).item())
        episode_lengths.append(len(rewards))
    return reward_means, episode_lengths


def run_single(env, model, x0=None, monitor=None, deterministic=True):
    t = 0
    y = env.reset(state_init=x0)
    x = env.states.cpu().numpy()

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
                                     cost=env.cost)

        y, reward, done, info = env.step(u)
        rewards.append(reward)
        x = env.states.cpu().numpy()
        t += env.process.dt
        episode_starts = done
        if done:
            env.reset()
            break
    # logging.info(f"Final reward: {reward}")
    return np.expand_dims(states, 1), rewards


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
    logging.info(f"Final reward: {reward}")
    return np.array(states)


def linear_schedule(initial_value: float,
                    q: float = 0.01) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :param q: Fraction of initial learning rate at end of progress.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * (progress_remaining * (1 - q) + q)

    return func


# noinspection PyProtectedMember
def main(study: optuna.Study, path, frozen_params=None,
         mlflow_callback: MLflowCallback = None, show_plots=False):

    device = 'cpu' if GPU is None else 'cuda:0'

    if frozen_params is not None:
        study.enqueue_trial(frozen_params)

    trial = study.ask()

    if mlflow_callback is not None:
        mlflow_callback._initialize_experiment(study)
        run = mlflow.start_run(run_name=str(trial.number),
                               nested=mlflow_callback._nest_trials)
        trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)
    else:
        run = None

    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_steps = trial.suggest_int('num_steps', 100, 500, step=50)
    # cost_threshold = trial.suggest_float('cost_threshold', 1e-4, 1e-3,
    #                                      log=True)
    # q_x = trial.suggest_float('q_x', 0.1, 10, log=True)
    # q_y = trial.suggest_float('q_y', 0.1, 10, log=True)
    # r = trial.suggest_float('r', 0.01, 1, log=True)

    num_inputs = 1
    num_outputs = 1
    num_states = 2
    process_noise = 1e-2
    observation_noise = 1e-1
    env = DiGym(num_inputs, num_outputs, num_states, device,
                process_noise, observation_noise, cost_threshold=1e-4,
                use_observations_in_cost=True)
    env = TimeLimit(env, num_steps)

    policy_kwargs = {'lstm_hidden_size': 50,
                     'net_arch': [],
                     # 'shared_lstm': True,
                     # 'enable_critic_lstm': False,
                     }
    log_dir = os.path.join(path, 'tensorboard_log')
    model = RecurrentPPO(MlpRnnPolicy, env, verbose=0, device=device,
                         tensorboard_log=log_dir, policy_kwargs=policy_kwargs,
                         learning_rate=linear_schedule(learning_rate, 0.005),
                         n_steps=num_steps, n_epochs=10)
    # model = PPO('MlpPolicy', env, verbose=1, device='cuda',
    #             tensorboard_log=log_dir)
    model.learn(int(5e5), callback=[
        FigureRecorderTest(),
        # FigureRecorderTrain()
    ])

    if show_plots:
        states, rewards = run_single(env, model)
        path_figure = os.path.join(path, 'figures',
                                   'lqg_rnn_ppo_trajectory.png')
        plot_phase_diagram({'x': states[:, 0, 0], 'v': states[:, 0, 1]},
                           xt=[0, 0], show=False, xlim=[-1, 1], ylim=[-1, 1],
                           path=path_figure)

    rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=100,
                                               warn=False,
                                               return_episode_rewards=True)
    study.tell(trial, np.mean(rewards))

    if mlflow_callback is None:
        path_model = os.path.join(path_base, 'models',
                                  f'lqg_rnn_ppo_{trial.number}')
    else:
        frozen_trial = study._storage.get_trial(trial._trial_id)
        mlflow_callback(study, frozen_trial)
        mlflow.log_artifact(model.logger.get_dir(), 'tensorboard_log')
        path_model = get_artifact_path(f'models/model_{trial.number}')
        # mlflow.pytorch.log_model(model, f'models/model_{trial.number}')
        run.__exit__(None, None, None)
    model.save(path_model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore', category=ExperimentalWarning)

    GPU = 9  # Faster on CPU

    study_name = 'lqg_rnn_ppo'

    # Make sure the keys are spelled exactly as the parameter names in
    # trial.suggest calls. Every parameter listed here will not be swept over.
    _frozen_params = {
        'learning_rate': 2e-4,
        'num_steps': 300,
    }

    if GPU is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.get_logger('optuna').addHandler(
        logging.StreamHandler(sys.stdout))

    path_base = '/home/bodrue/Data/neural_control/double_integrator/rnn_ppo/' \
                'rnn/maximize_rewards'
    os.makedirs(path_base, exist_ok=True)

    filepath_output = os.path.join(path_base, 'optuna', study_name + '.db')
    os.makedirs(os.path.dirname(filepath_output), exist_ok=True)
    storage_name = f'sqlite:///{filepath_output}'
    _study = optuna.create_study(storage_name, study_name=study_name,
                                 direction='maximize', load_if_exists=True)

    _mlflow_callback = MLflowCallback('file:' + path_base + '/mlruns',
                                      metric_name='reward', nest_trials=True)

    num_trials = 5
    for _ in trange(num_trials, desc='Optuna'):
        main(_study, path_base, _frozen_params, _mlflow_callback)
        gc.collect()

    logging.info("Number of finished trials: ", len(_study.trials))

    logging.info("Best trial:")
    best_trial = _study.best_trial

    logging.info("  Value: ", best_trial.value)

    logging.info("  Params: ")
    for _key, value in best_trial.params.items():
        logging.info("    {}: {}".format(_key, value))

    sys.exit()
