import os
import sys

from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from examples import configs
from examples.linear_rnn_rl import POMDP
from examples.nonlinear_rnn_rl import NonlinearRlPipeline, run_single
from examples.visualize_linear_rnn_lqr import (
    add_scalars, get_runs_perturbed, get_runs_unperturbed, get_runs_all,
    get_log_path, get_model_trained, get_model_unperturbed_untrained,
    get_training_data_unperturbed, plot_training_curve_unperturbed,
    get_training_data_perturbed, plot_training_curves_perturbed,
    get_metric_vs_dropout, plot_metric_vs_dropout, PALETTE,
    plot_controller_effect)
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
    config = configs.nonlinear_rnn_rl.get_config()

    # Get pipeline for RL experiment.
    pipeline = NonlinearRlPipeline(config)
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
                      log_path, 'trajectories_unperturbed.png')

    # Show metric vs times of unperturbed model.
    eval_every_n = 5000
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path, 'reward', eval_every_n)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path,
                                    axis_labels=('Episode', 'Reward'),
                                    formatx=True)

    # Show metric vs times of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, path, 'reward', eval_every_n)
    test_metric_unperturbed = runs_unperturbed['metrics.test_reward'].mean()
    plot_training_curves_perturbed(training_data_perturbed, log_path,
                                   test_metric_unperturbed, formatx=True,
                                   axis_labels=('Episode', 'Reward'))
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, ylabel='Reward')
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, ylabel='Reward',
                           kind='line')

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        [0], environment, path, perturbations, pipeline, runs)

    for perturbation in perturbations.keys():
        # Select one perturbation type.
        data_subsystem = trajectories_perturbed.loc[
            trajectories_perturbed['perturbation_type'] == perturbation]
        plot_trajectories(data_subsystem, 'perturbation_level',
                          'Perturbation level', log_path,
                          f'trajectories_perturbed_{perturbation}.png')

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    electrode_selections = config.perturbation.ELECTRODE_SELECTIONS
    # runs['params.electrode_selection'] = 'random'
    metric_vs_dropout = get_metric_vs_dropout(
        runs, perturbations, electrode_selections, training_data_perturbed,
        'reward')
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed, 'test_reward')


def plot_trajectories(data: pd.DataFrame, col_key: str, col_label: str,
                      path: str, filename: str):
    g = sns.relplot(data=data, x='x0', y='x2', row='stage', style='controller',
                    hue='controller', col=col_key, kind='line', sort=False,
                    palette=PALETTE, row_order=['untrained', 'trained'],
                    legend=False, facet_kws={'sharex': True, 'sharey': True,
                                             'margin_titles': True})

    # Draw coordinate system as inlet in first panel.
    arrowprops = dict(arrowstyle='-', connectionstyle='arc3', fc='k', ec='k')
    g.axes[0, 0].annotate(
        'Position', xy=(0.11, 0.1), xycoords='axes fraction', xytext=(75, 0),
        textcoords='offset points', verticalalignment='center',
        arrowprops=arrowprops)
    g.axes[0, 0].annotate(
        'Velocity', xy=(0.11, 0.1), xycoords='axes fraction', xytext=(0, 75),
        textcoords='offset points', horizontalalignment='center',
        arrowprops=arrowprops)

    # Same for angle and angular velocity (can't do it like this because it
    # overwrites previous plot):
    # g = sns.relplot(data=data, x='x1', y='x3', col='stage', row=row_key,
    #                 kind='line', sort=False, linestyle='--',
    #                 col_order=['untrained', 'trained'],
    #                 facet_kws={'sharex': True, 'sharey': True,
    #                            'margin_titles': True})
    # arrowprops = dict(linestyle='dashed', connectionstyle='arc3', fc='k',
    #                   ec='k')
    # g.axes[0, 0].annotate(
    #     'Angle', xy=(-0.5, -0.5), xycoords='data', xytext=(5, 0),
    #     textcoords='offset points', verticalalignment='center',
    #     arrowprops=arrowprops)
    # g.axes[0, 0].annotate(
    #     'Angluar velocity', xy=(-0.5, -0.5), xycoords='data', xytext=(0, 5),
    #     textcoords='offset points', horizontalalignment='center',
    #     arrowprops=arrowprops)

    for ax in g.axes.flat:
        steps = len(ax.get_lines()[0].get_xdata())
        ax.text(0.1, 0.9, f'{steps} steps', transform=ax.transAxes)
    g.set(xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template=col_label + ' {col_name}', row_template='')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_ylabel('Before training')
    g.axes[1, 0].set_ylabel('After training')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    path_fig = os.path.join(path, filename)
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_trajectories_unperturbed(
        model_trained: RecurrentPPO, model_untrained: RecurrentPPO,
        environment: POMDP) -> pd.DataFrame:

    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': [],
            'x2': [], 'x3': []}
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
        test_indexes: List[int], environment: POMDP, path: str,
        perturbations: dict, pipeline: NonlinearRlPipeline,
        runs: pd.DataFrame) -> pd.DataFrame:
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': [],
            'x2': [], 'x3': [], 'perturbation_type': [],
            'perturbation_level': []}
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
        pipeline: NonlinearRlPipeline, environment: POMDP, runs: pd.DataFrame,
        path: str):
    model = get_model_trained(pipeline, environment, runs, path)
    # Disconnect controller.
    model.policy.lstm_actor.controller.init_zero()
    return model


def add_states(data: dict, states: np.ndarray):
    data['x0'] += states[:, 0, 0].tolist()
    data['x1'] += states[:, 0, 1].tolist()
    data['x2'] += states[:, 0, 2].tolist()
    data['x3'] += states[:, 0, 3].tolist()


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'nonlinear_rnn_rl'
    _tag_start_time = '2022-11-11'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
