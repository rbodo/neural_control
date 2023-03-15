import gc
import logging
import sys
import os
import warnings

import mlflow
import numpy as np
from gymnasium.wrappers import TimeLimit
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.integration.mlflow import MLflowCallback, RUN_ID_ATTRIBUTE_KEY
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import trange

from examples.linear_rnn_rl import (EvaluationCallback, run_single,
                                    linear_schedule)
from src.control_systems_torch import DiGym
from src.plotting import plot_phase_diagram
from src.ppo_recurrent import RecurrentPPO, MlpRnnPolicy
from src.utils import get_artifact_path


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
        EvaluationCallback(),
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
