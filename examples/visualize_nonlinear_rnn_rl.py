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
    add_scalars, get_runs_perturbed, draw_colorbar, get_runs_unperturbed,
    get_runs_all, get_log_path, PALETTE)
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
        runs_unperturbed, path, eval_every_n)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path)

    # Show metric vs times of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path, eval_every_n)
    test_metric_unperturbed = runs_unperturbed['metrics.test_reward'].mean()
    plot_training_curves_perturbed(training_data_perturbed, log_path,
                                   test_metric_unperturbed)

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        [0], environment, path, perturbations, pipeline, runs)

    # Select one perturbation type.
    data_subsystem = trajectories_perturbed.loc[
        trajectories_perturbed['perturbation_type'] == 'sensor']
    plot_trajectories(data_subsystem, 'perturbation_level',
                      'Perturbation level', log_path,
                      'trajectories_perturbed.png')

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    metric_vs_dropout = get_metric_vs_dropout(runs, perturbations,
                                              training_data_perturbed)
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed)


def plot_trajectories(data: pd.DataFrame, row_key: str, row_label: str,
                      path: str, filename: str):
    g = sns.relplot(data=data, x='x0', y='x2', col='stage', row=row_key,
                    kind='line', sort=False,
                    col_order=['untrained', 'trained'],
                    facet_kws={'sharex': True, 'sharey': True,
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
    g.set_titles(row_template=row_label+' {row_name}')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_title('Before training')
    g.axes[0, 1].set_title('After training')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    path_fig = os.path.join(path, filename)
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curve_unperturbed(data: pd.DataFrame, path: str):
    g = sns.relplot(data=data, x='time', y='metric', kind='line', legend=False)

    g.set_axis_labels('Episode', 'Reward')
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(lambda x, p: f'{int(x/1e3)}K')
    plt.tight_layout()
    path_fig = os.path.join(path, 'neuralsystem_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curves_perturbed(data: pd.DataFrame, path: str,
                                   test_metric_unperturbed: float):
    # Get test curves corresponding to full controllability and observability.
    data_full_control = data.loc[(data.dropout_probability == 0) &
                                 (data.phase == 'test')]
    g = sns.relplot(data=data_full_control, x='time',
                    y='metric', col='perturbation_type',
                    col_order=['sensor', 'processor', 'actuator'],
                    hue='perturbation_level', kind='line', palette=PALETTE,
                    legend=False, facet_kws={'sharex': False, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_metric_unperturbed, color='k', linestyle=':')

    g.set_axis_labels('Episode', 'Reward')
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(lambda x, p: f'{int(x/1e3)}K')
    g.set_titles('{col_name}')
    g.axes[0, 0].set(yticklabels=[])
    g.despine(left=True)
    draw_colorbar()
    path_fig = os.path.join(path, 'controller_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_metric_vs_dropout(data, path, test_metric_unperturbed):
    g = sns.relplot(data=data, x='gramian_value', y='metrics.test_reward',
                    row='gramian_type', col='params.perturbation_type',
                    hue='params.perturbation_level', palette=PALETTE,
                    col_order=['sensor', 'processor', 'actuator'],
                    kind='line', marker='o', markersize=10, legend=False,
                    facet_kws={'sharex': False, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_metric_unperturbed, color='k', linestyle=':')

    g.set(xlim=[-0.05, 1.05])
    g.set_axis_labels('', 'Reward')
    g.set_titles('{col_name}', '')
    g.axes[0, 0].set(yticklabels=[])
    g.axes[1, 0].set(yticklabels=[])
    g.despine(left=True)
    for ax in g.axes[0]:
        ax.set_xlabel('Controllability')
    for ax in g.axes[1]:
        ax.set_xlabel('Observability')
        ax.set_title('')
    draw_colorbar()
    path_fig = os.path.join(path, 'metric_vs_dropout.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_training_data_unperturbed(runs: pd.DataFrame, path: str,
                                  eval_every_n: int) -> pd.DataFrame:
    data = {'time': [], 'metric': [], 'phase': [], 'neuralsystem': 'RNN'}
    for run_id in runs['run_id']:
        add_training_curve(data, path, run_id, 'test', eval_every_n)
        add_training_curve(data, path, run_id, 'training', eval_every_n)
    return pd.DataFrame(data)


def get_training_data_perturbed(runs: pd.DataFrame, perturbations: dict,
                                dropout_probabilities: List[float], path: str,
                                eval_every_n: int) -> pd.DataFrame:
    data = {'time': [], 'metric': [], 'phase': [], 'neuralsystem': 'RNN',
            'perturbation_type': [], 'perturbation_level': [],
            'dropout_probability': []}
    for perturbation_type in perturbations.keys():
        for perturbation_level in perturbations[perturbation_type]:
            for dropout_probability in dropout_probabilities:
                runs_perturbed = get_runs_perturbed(runs, perturbation_type,
                                                    perturbation_level)
                for run_id in runs_perturbed['run_id']:
                    n = add_training_curve(data, path, run_id, 'test',
                                           eval_every_n)
                    add_scalars(data, n, perturbation_type=perturbation_type,
                                perturbation_level=perturbation_level,
                                dropout_probability=dropout_probability)
                    n = add_training_curve(data, path, run_id, 'training',
                                           eval_every_n)
                    add_scalars(data, n, perturbation_type=perturbation_type,
                                perturbation_level=perturbation_level,
                                dropout_probability=dropout_probability)
    return pd.DataFrame(data)


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


def get_metric_vs_dropout(runs: pd.DataFrame, perturbations: dict,
                          training_data_perturbed: pd.DataFrame
                          ) -> pd.DataFrame:
    # Get metric of uncontrolled perturbed model before training controller.
    t = training_data_perturbed
    data = {'metrics.test_reward': [], 'metrics.training_reward': [],
            'params.perturbation_type': [], 'params.perturbation_level': [],
            'gramian_type': [], 'gramian_value': []}
    for gramian_type in ['metrics.controllability', 'metrics.observability']:
        for perturbation_type in perturbations.keys():
            for perturbation_level in perturbations[perturbation_type]:
                # r are the uncontrolled perturbed runs at begin of training.
                r = t.loc[(t['perturbation_type'] == perturbation_type) &
                          (t['perturbation_level'] == perturbation_level) &
                          (t['time'] == 0) & (t['dropout_probability'] == 0)]
                # Add training and test metric.
                data['metrics.test_reward'] += list(
                    r[r.phase == 'test']['metric'])
                data['metrics.training_reward'] += list(
                    r[r.phase == 'training']['metric'])
                n = len(r) // 2
                add_scalars(data, n, **{
                    'params.perturbation_type': perturbation_type,
                    'params.perturbation_level': str(perturbation_level),
                    'gramian_type': gramian_type, 'gramian_value': 0})
    # Melt the controllability and observability columns into a "gramian_type"
    # and "gramian_value" column.
    runs = runs.melt(
        var_name='gramian_type', value_name='gramian_value',
        value_vars=['metrics.controllability', 'metrics.observability'],
        id_vars=['metrics.test_reward', 'metrics.training_reward',
                 'params.perturbation_type', 'params.perturbation_level'])
    # Remove empty rows corresponding to the unperturbed baseline models.
    runs = runs.loc[runs['params.perturbation_type'] != '']
    # Concatenate training curves with baseline results.
    return pd.concat([runs, pd.DataFrame(data)], ignore_index=True)


def get_model_trained(
        pipeline: NonlinearRlPipeline, environment: POMDP, runs: pd.DataFrame,
        path: str):
    run_id = runs['run_id'].iloc[0]  # First random seed.
    path_model = os.path.join(path, run_id, 'artifacts', 'models',
                              'rnn.params')
    return pipeline.get_model(True, True, environment, path_model)


def get_model_unperturbed_untrained(pipeline: NonlinearRlPipeline,
                                    environment: POMDP):
    return pipeline.get_model(True, True, environment)


def get_model_perturbed_untrained(
        pipeline: NonlinearRlPipeline, environment: POMDP, runs: pd.DataFrame,
        path: str):
    model = get_model_trained(pipeline, environment, runs, path)
    # Disconnect controller.
    model.policy.lstm_actor.controller.init_zero()
    return model


def add_training_curve(data: dict, path: str, run_id: str, phase: str,
                       eval_every_n: int) -> int:
    assert phase in {'training', 'test'}
    filepath = os.path.join(path, run_id, 'metrics', f'{phase}_reward')
    metric, times = np.loadtxt(filepath, usecols=[1, 2], unpack=True)
    num_steps = len(metric)
    data['time'] += list((times + 1) * eval_every_n)
    data['metric'] += list(metric)
    data['phase'] += [phase] * num_steps
    return num_steps


def add_states(data: dict, states: np.ndarray):
    data['x0'] += states[:, 0, 0].tolist()
    data['x1'] += states[:, 0, 1].tolist()
    data['x2'] += states[:, 0, 2].tolist()
    data['x3'] += states[:, 0, 3].tolist()


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'nonlinear_rnn_rl'
    _tag_start_time = '2022-09-26_17:58:31'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
