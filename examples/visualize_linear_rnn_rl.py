import os
import sys

from gym.wrappers import TimeLimit
from typing import List, Union, Optional

import pandas as pd
import seaborn as sns

from examples import configs
from examples.linear_rnn_rl import LinearRlPipeline, run_single
from examples.visualize_linear_rnn_lqr import (
    add_scalars, get_runs_perturbed, get_runs_unperturbed,
    get_runs_all, get_log_path, plot_trajectories_unperturbed,
    plot_trajectories_perturbed, plot_metric_vs_dropout, add_states,
    get_training_data_perturbed, get_training_data_unperturbed,
    get_model_trained, get_metric_vs_dropout, get_model_unperturbed_untrained,
    plot_training_curve_unperturbed, plot_training_curves_perturbed,
    plot_controller_effect, get_num_electrodes, plot_metric_vs_dropout_average)
from src.control_systems import DiGym
from src.ppo_recurrent import RecurrentPPO


def main(experiment_id, experiment_name, tag_start_time):
    # Get path where experiment data has been saved.
    log_path = get_log_path(experiment_name)
    path = os.path.join(log_path, 'mlruns', experiment_id)

    # Get all training runs.
    runs = get_runs_all(log_path, experiment_id, tag_start_time)

    # Get training runs of unperturbed models (multiple random seeds).
    runs_unperturbed = get_runs_unperturbed(runs)

    # Get configuration.
    config = configs.linear_rnn_rl.get_config()

    # Get pipeline for RL experiment.
    pipeline = LinearRlPipeline(config)
    pipeline.device = pipeline.get_device()

    # Get environment to produce example trajectories through phase space.
    environment = pipeline.get_environment()

    # Get unperturbed model before training.
    model_untrained = get_model_unperturbed_untrained(pipeline, environment)

    # Get unperturbed model after training.
    model_trained = get_model_trained(pipeline, environment, runs_unperturbed,
                                      path)
    pipeline.model = model_trained

    title = 'RL: Particle stabilization'

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed = get_trajectories_unperturbed(
        model_trained, model_untrained, environment)
    plot_trajectories_unperturbed(trajectories_unperturbed, log_path,
                                  '(c) ' + title)

    # Show metric vs times of unperturbed model.
    eval_every_n = 5000
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path, 'reward', eval_every_n)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path,
                                    axis_labels=('Episode', 'Reward'),
                                    formatx=True)

    # Show metric vs times of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path, 'reward',
        eval_every_n)
    test_metric_unperturbed = runs_unperturbed['metrics.test_reward'].mean()
    plot_training_curves_perturbed(training_data_perturbed, log_path,
                                   test_metric_unperturbed,
                                   ('Episode', 'Reward'), formatx=True)
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, ylabel='Reward')

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        [0], environment, path, perturbations, pipeline, runs)
    plot_trajectories_perturbed(trajectories_perturbed, log_path)

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    sns.set_context('talk')
    metric_vs_dropout = get_metric_vs_dropout(
        runs, perturbations, training_data_perturbed, 'reward')
    n = config.model.NUM_HIDDEN_NEURALSYSTEM
    num_electrodes = get_num_electrodes(runs, perturbations, path, n)
    plot_metric_vs_dropout_average(
        metric_vs_dropout, log_path, test_metric_unperturbed, 'test_reward',
        num_electrodes=num_electrodes, set_xlabels=False, set_col_labels=True,
        title='(a) ' + title)
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed, 'test_reward')


def get_trajectories_unperturbed(
        model_trained: RecurrentPPO, model_untrained: RecurrentPPO,
        environment: Union[DiGym, TimeLimit]) -> pd.DataFrame:

    data = {'index': [], 'controller': [], 'x0': [], 'x1': []}
    for test_index in range(4):

        # Get trajectories of trained model.
        environment_states, _ = run_single(environment, model_trained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='RNN after training')

        # Get trajectories of untrained model.
        environment_states, _ = run_single(environment, model_untrained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='RNN before training')

    return pd.DataFrame(data)


def get_trajectories_perturbed(
        test_indexes: List[int], environment: Union[DiGym, TimeLimit],
        path: str, perturbations: dict, pipeline: LinearRlPipeline,
        runs: pd.DataFrame, use_relative_levels: Optional[bool] = True
) -> pd.DataFrame:
    data = {'index': [], 'controller': [], 'x0': [], 'x1': [],
            'perturbation_type': [], 'perturbation_level': []}
    for perturbation_type in perturbations.keys():
        for i, level in enumerate(perturbations[perturbation_type]):
            runs_perturbed = get_runs_perturbed(runs, perturbation_type, level)
            model_trained = get_model_trained(
                pipeline, environment, runs_perturbed, path)
            model_untrained = get_model_perturbed_untrained(
                pipeline, environment, runs_perturbed, path)
            if use_relative_levels:
                level = (i + 1) / len(perturbations[perturbation_type])
            for test_index in test_indexes:
                kwargs = dict(index=test_index, perturbation_level=level,
                              perturbation_type=perturbation_type)

                # Get trajectories of trained model.
                environment_states, _ = run_single(environment, model_trained)
                add_states(data, environment_states)
                add_scalars(data, len(environment_states),
                            controller='RNN after training', **kwargs)

                # Get trajectories of untrained model.
                environment_states, _ = run_single(environment,
                                                   model_untrained)
                add_states(data, environment_states)
                add_scalars(data, len(environment_states),
                            controller='RNN before training', **kwargs)

    return pd.DataFrame(data)


def get_model_perturbed_untrained(
        pipeline: LinearRlPipeline, environment: Union[DiGym, TimeLimit],
        runs: pd.DataFrame, path: str):
    model = get_model_trained(pipeline, environment, runs, path)
    # Disconnect controller.
    model.policy.lstm_actor.controller.init_zero()
    return model


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'linear_rnn_rl'
    # _tag_start_time = '2022-10-01'
    _tag_start_time = '2022-11-11'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
