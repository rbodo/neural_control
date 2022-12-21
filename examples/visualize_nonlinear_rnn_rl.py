import os
import sys

from typing import List, Optional

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
    get_training_data_perturbed, get_metric_vs_dropout, PALETTE,
    plot_controller_effect, plot_metric_vs_dropout_average, PERTURBATIONS,
    draw_coordinate_system, draw_title, draw_legend)
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

    title = 'RL: Pendulum balancing'

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed = get_trajectories_unperturbed(
        model_trained, model_untrained, environment)
    plot_trajectories_unperturbed(trajectories_unperturbed, log_path, title,
                                  show_legend=False, show_coordinates=False)

    # Show metric vs times of unperturbed model.
    eval_every_n = 5000
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path, 'reward', eval_every_n)
    plot_training_curve_unperturbed(
        training_data_unperturbed, log_path, axis_labels=('Episode', 'Reward'),
        formatx=True, show_legend=False, height=3.8)

    # Show metric vs times of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path, 'reward',
        eval_every_n)
    test_metric_unperturbed = runs_unperturbed['metrics.test_reward'].mean()
    # plot_training_curves_perturbed(training_data_perturbed, log_path,
    #                                test_metric_unperturbed,
    #                                ('Episode', 'Reward'), formatx=True)
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, ylabel='Reward',
                           aspect=1.5)

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        [0], environment, path, perturbations, pipeline, runs)
    plot_trajectories_perturbed(trajectories_perturbed, log_path,
                                show_coordinates=True)

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    sns.set_context('talk')
    metric_vs_dropout = get_metric_vs_dropout(
        runs, perturbations, training_data_perturbed, 'reward')
    plot_metric_vs_dropout_average(
        metric_vs_dropout, log_path, test_metric_unperturbed, 'test_reward',
        set_xlabels=True, set_col_labels=False, title=title, show_legend=False)
    # plot_metric_vs_dropout(metric_vs_dropout, log_path,
    #                        test_metric_unperturbed, 'test_reward')


def plot_trajectories_unperturbed(data: pd.DataFrame, path: str,
                                  title: Optional[str] = None,
                                  show_legend: Optional[bool] = True,
                                  show_coordinates: Optional[bool] = True):
    print("Trajectories unperturbed.")
    g = sns.relplot(data=data, x='x0', y='x2', kind='line', style='controller',
                    hue='controller', col='index', sort=False, palette=PALETTE,
                    legend=show_legend, aspect=0.8, facet_kws={
                        'sharex': True, 'sharey': True, 'margin_titles': True})

    if show_coordinates:
        draw_coordinate_system(g)

    if title is not None:
        draw_title(g.axes[0, 0], title)

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

    if show_legend:
        draw_legend(g, 2)
    draw_rewards(g)

    g.set(xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template='',  # 'Test sample {col_name}'
                 row_template='')
    g.despine(left=False, bottom=False, top=False, right=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    path_fig = os.path.join(path, 'trajectories_unperturbed')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_trajectories_perturbed(data: pd.DataFrame, path: str,
                                show_coordinates: Optional[bool] = True):
    print("Trajectories perturbed.")
    g = sns.relplot(data=data, x='x0', y='x2', kind='line', style='controller',
                    hue='controller', col='perturbation_level', sort=False,
                    palette=PALETTE, row='perturbation_type', height=3.8,
                    legend=True, aspect=0.8,
                    facet_kws={'sharex': False, 'sharey': True,
                               'margin_titles': True})

    if show_coordinates:
        draw_coordinate_system(g, axis=(0, 4))

    legend = g.axes[0, 0].legend()
    lines = legend.get_lines()
    labels = [t.get_text().capitalize() for t in legend.texts]
    legend.remove()
    sns.move_legend(obj=g, loc='lower left', handles=lines, labels=labels,
                    frameon=False, ncol=1, title=None,
                    bbox_to_anchor=(0.35, 0.85))
    draw_rewards(g)

    g.set(xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template='',  # Perturbation level {col_name:.0%}',
                 row_template='')
    g.axes[2, 2].set_xlabel('Perturbation level')
    g.axes[2, 0].set_xticks([g.axes[2, 0].get_xlim()[0] * 1.2])
    g.axes[2, 0].set_xticklabels(['low'])
    g.axes[2, 4].set_xticks([g.axes[2, 4].get_xlim()[-1] * 0.9])
    g.axes[2, 4].set_xticklabels(['high'])
    for i, ylabel in enumerate(PERTURBATIONS.values()):
        draw_title(g.axes[i, 0], ylabel)
    g.despine(left=False, bottom=False, top=False, right=False)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    path_fig = os.path.join(path, 'trajectories_perturbed')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_trajectories_unperturbed(
        model_trained: RecurrentPPO, model_untrained: RecurrentPPO,
        environment: POMDP) -> pd.DataFrame:

    data = {'index': [], 'controller': [], 'x0': [], 'x1': [], 'x2': [],
            'x3': []}
    for test_index in range(4):

        # Get trajectories of trained model.
        environment_states, _ = run_single(environment, model_trained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='Neural system')

        # Get trajectories of untrained model.
        environment_states, _ = run_single(environment, model_untrained)
        add_states(data, environment_states)
        add_scalars(data, len(environment_states), index=test_index,
                    controller='No control')

    return pd.DataFrame(data)


def get_trajectories_perturbed(
        test_indexes: List[int], environment: POMDP, path: str,
        perturbations: dict, pipeline: NonlinearRlPipeline,
        runs: pd.DataFrame, use_relative_levels: Optional[bool] = True
) -> pd.DataFrame:
    data = {'index': [], 'controller': [], 'x0': [], 'x1': [], 'x2': [],
            'x3': [], 'perturbation_type': [], 'perturbation_level': []}
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
                            controller='Prosthesis on', **kwargs)

                # Get trajectories of untrained model.
                environment_states, _ = run_single(environment,
                                                   model_untrained)
                add_states(data, environment_states)
                add_scalars(data, len(environment_states),
                            controller='Prosthesis off', **kwargs)

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


def draw_rewards(g: sns.FacetGrid):
    for i, ax in enumerate(g.axes.flat):
        lines = []
        labels = []
        for j, line in enumerate(ax.get_lines()[:2]):
            reward = len(line.get_xdata())
            labels.append(str(reward))
            lines.append(line)
        ax.legend(lines, labels, loc='best', frameon=False, ncol=1, title=None)


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'nonlinear_rnn_rl'
    _tag_start_time = '2022-11-11'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
