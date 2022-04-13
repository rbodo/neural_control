import os
import sys

import logging
import numpy as np
import optuna

from src.double_integrator import configs
from src.double_integrator.utils import apply_config
from src.double_integrator.lqg_rnn_reinforce import train_single


def objective_single(trial: optuna.Trial, base_path):

    config = \
        configs.config_train_rnn_lqg_reinforce.get_config(base_path=base_path)

    config.defrost()

    config.SEED = int(trial.suggest_uniform('seed', 1, 1e4))

    apply_config(config)

    rewards = train_single(config, verbose=True, plot_loss=False,
                           save_model=False)

    return np.max(rewards)


def objective(trial: optuna.Trial, base_path):
    dt = 0.1

    max_rewards = []
    seeds = [42, 52, 103]
    for seed in seeds:
        config = \
            configs.config_train_rnn_lqg_reinforce_hyperparameter.get_config(
                base_path=base_path)

        config.defrost()

        num_layers = 1  # trial.suggest_int('num_layers', 1, 2)
        num_hidden = trial.suggest_int('num_hidden', 2, 64, log=True)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        neuron_model = trial.suggest_categorical('neuron_model',
                                                 ['rnn', 'gru'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1,
                                            log=True)
        num_steps = trial.suggest_int('num_steps', 100, 10000, log=True)
        reward_discount = trial.suggest_float('reward_discount', 0.6, 1,
                                              log=True)

        config.model.NUM_LAYERS = num_layers
        config.model.NUM_HIDDEN = num_hidden
        config.model.ACTIVATION = activation
        config.model.NEURON_MODEL = neuron_model
        config.training.LEARNING_RATE = learning_rate
        config.simulation.NUM_STEPS = num_steps
        config.training.REWARD_DISCOUNT = reward_discount
        config.simulation.T = num_steps * dt

        config.SEED = seed

        apply_config(config)

        rewards = train_single(config, verbose=True, plot_loss=False,
                               save_model=False)
        max_rewards.append(np.max(rewards))

        config.defrost()

    return np.mean(max_rewards)


if __name__ == '__main__':
    optuna.logging.get_logger('optuna').addHandler(
        logging.StreamHandler(sys.stdout))

    basepath = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
               'reinforce/lqg_single_config'
    study_name = 'rnn_lqg_reinforce'
    filepath_output = os.path.join(basepath, study_name + '.db')
    storage_name = f'sqlite:///{filepath_output}'
    study = optuna.create_study(storage_name, study_name=study_name,
                                direction='maximize', load_if_exists=True)

    # objective(study.trials[11])
    study.optimize(lambda t: objective_single(t, basepath), n_trials=10,
                   timeout=None, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    sys.exit()
