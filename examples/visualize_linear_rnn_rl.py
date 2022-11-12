import os
import sys

from gym.wrappers import TimeLimit
from typing import List, Union

import pandas as pd

from examples import configs
from examples.linear_rnn_rl import LinearRlPipeline, run_single
from examples.visualize_linear_rnn_lqr import (
    add_scalars, get_runs_perturbed, get_runs_unperturbed,
    get_runs_all, get_log_path, plot_trajectories,
    plot_metric_vs_dropout, add_states, get_training_data_perturbed,
    get_training_data_unperturbed, get_model_trained, get_metric_vs_dropout,
    get_model_unperturbed_untrained, plot_training_curve_unperturbed,
    plot_training_curves_perturbed)
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

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed = get_trajectories_unperturbed(
        model_trained, model_untrained, environment)
    plot_trajectories(trajectories_unperturbed, 'index', 'Test sample',
                      log_path, 'trajectories_unperturbed.png',
                      show_legend=False)

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

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        [0], environment, path, perturbations, pipeline, runs)

    for perturbation in perturbations.keys():
        # Select one perturbation type.
        data_subsystem = trajectories_perturbed.loc[
            trajectories_perturbed['perturbation_type'] == perturbation]
        plot_trajectories(data_subsystem, 'perturbation_level',
                          'Perturbation level', log_path,
                          f'trajectories_perturbed_{perturbation}.png',
                          show_legend=False)

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    metric_vs_dropout = get_metric_vs_dropout(runs, perturbations,
                                              training_data_perturbed,
                                              'reward')
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed, 'test_reward')


def get_trajectories_unperturbed(
        model_trained: RecurrentPPO, model_untrained: RecurrentPPO,
        environment: Union[DiGym, TimeLimit]) -> pd.DataFrame:

    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': []}
    for test_index in range(4):

        # Get trajectories of untrained model.
        environment_states, _ = run_single(environment, model_untrained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='RNN', stage='untrained')

        # Get trajectories of trained model.
        environment_states, _ = run_single(environment, model_trained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='RNN', stage='trained')
    return pd.DataFrame(data)


def get_trajectories_perturbed(
        test_indexes: List[int], environment: Union[DiGym, TimeLimit],
        path: str, perturbations: dict, pipeline: LinearRlPipeline,
        runs: pd.DataFrame) -> pd.DataFrame:
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': [],
            'perturbation_type': [], 'perturbation_level': []}
    for perturbation_type in perturbations.keys():
        for perturbation_level in perturbations[perturbation_type]:
            runs_perturbed = get_runs_perturbed(runs, perturbation_type,
                                                perturbation_level)
            model_trained = get_model_trained(
                pipeline, environment, runs_perturbed, path)
            model_untrained = get_model_perturbed_untrained(
                pipeline, environment, runs_perturbed, path)
            for test_index in test_indexes:
                kwargs = dict(index=test_index,
                              perturbation_type=perturbation_type,
                              perturbation_level=perturbation_level)

                # Get trajectories of untrained model.
                environment_states, _ = run_single(environment,
                                                   model_untrained)
                add_states(data, environment_states)
                add_scalars(data, len(environment_states), controller='RNN',
                            stage='untrained', **kwargs)

                # Get trajectories of trained model.
                environment_states, _ = run_single(environment, model_trained)
                add_states(data, environment_states)
                add_scalars(data, len(environment_states), controller='RNN',
                            stage='trained', **kwargs)
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
    _tag_start_time = '2022-11-12'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
