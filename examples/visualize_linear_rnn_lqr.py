import os
import sys
from collections import OrderedDict

from typing import Tuple, List, Optional

import mlflow
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from examples import configs
from examples.linear_rnn_lqr import LqrPipeline
from src.control_systems_mxnet import (StochasticLinearIOSystem,
                                       ClosedControlledNeuralSystem)
from src.utils import get_data

sns.set_style('white')
sns.set_context('talk')
PALETTE = 'copper'
PERTURBATIONS = OrderedDict({'sensor': 'Sensor', 'processor': 'Association',
                             'actuator': 'Motor'})


def main(experiment_id, experiment_name, tag_start_time):
    # Get path where experiment data has been saved.
    log_path = get_log_path(experiment_name)
    path = os.path.join(log_path, 'mlruns', experiment_id)

    # Get all training runs.
    runs = get_runs_all(log_path, experiment_id, tag_start_time)

    # Get training runs of unperturbed models (multiple random seeds).
    runs_unperturbed = get_runs_unperturbed(runs)

    # Get configuration.
    config = configs.linear_rnn_lqr.get_config()

    # Get data to produce example trajectories through phase space.
    data_dict = get_data(config, 'states')
    data_test = data_dict['data_test']

    # Get pipeline for LQR experiment.
    pipeline = LqrPipeline(config, data_dict)
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
    trajectories_unperturbed, lqr_loss = get_trajectories_unperturbed(
        data_test, model_trained, model_untrained, pipeline)
    plot_trajectories(trajectories_unperturbed, 'index', 'Test sample',
                      log_path, 'trajectories_unperturbed.png')

    # Show loss vs epochs of unperturbed model.
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path,
                                    lqr_loss, logy=True)

    # Show loss vs epochs of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path)
    test_metric_unperturbed = runs_unperturbed['metrics.test_loss'].mean()
    plot_training_curves_perturbed(training_data_perturbed, log_path,
                                   test_metric_unperturbed, logy=True)
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, logy=True)
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, logy=True, kind='line')

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        data_test, [5], environment, path, perturbations, pipeline, runs)

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
    metric_vs_dropout = get_metric_vs_dropout(
        runs, perturbations, electrode_selections, training_data_perturbed)
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed, logy=True)
    plot_gramian_vs_random(metric_vs_dropout, log_path,
                           test_metric_unperturbed, logy=True)


def plot_trajectories(data: pd.DataFrame, col_key: str, col_label: str,
                      path: str, filename: str,
                      show_legend: Optional[bool] = True):
    g = sns.relplot(data=data, x='x0', y='x1', row='stage', style='controller',
                    hue='controller', col=col_key, kind='line', sort=False,
                    palette=PALETTE, row_order=['untrained', 'trained'],
                    legend=show_legend,
                    facet_kws={'sharex': True, 'sharey': True,
                               'margin_titles': True})

    # Draw coordinate system as inlet in first panel.
    arrowprops = dict(arrowstyle='-', connectionstyle='arc3', fc='k', ec='k')
    g.axes[0, 0].annotate(
        'Position', xy=(-0.75, -0.75), xycoords='data', xytext=(75, 0),
        textcoords='offset points', verticalalignment='center',
        arrowprops=arrowprops)
    g.axes[0, 0].annotate(
        'Velocity', xy=(-0.75, -0.75), xycoords='data', xytext=(0, 75),
        textcoords='offset points', horizontalalignment='center',
        arrowprops=arrowprops)

    # Draw target state.
    xt = [0, 0]
    for ax in g.axes.ravel():
        ax.scatter(xt[0], xt[1], s=64, marker='x', c='k')

    lim = 0.99
    g.set(xlim=[-lim, lim], ylim=[-lim, lim], xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template=col_label + ' {col_name}', row_template='')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_ylabel('Before training')
    g.axes[1, 0].set_ylabel('After training')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if show_legend:
        sns.move_legend(g, 'upper center', bbox_to_anchor=(0.15, 0.92), ncol=2,
                        title=None)
    path_fig = os.path.join(path, filename)
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curve_unperturbed(
        data: pd.DataFrame, path: str, lqr_loss: Optional[float] = None,
        axis_labels: Optional[Tuple[str, str]] = ('Epoch', 'Loss'),
        show_legend: Optional[bool] = True, logy: Optional[bool] = False,
        formatx: Optional[bool] = False):
    g = sns.relplot(data=data, x='time', y='metric', style='phase',
                    style_order=['training', 'test'], hue='phase', kind='line',
                    legend=show_legend, palette=PALETTE)

    # Draw LQR baseline.
    if lqr_loss is not None:
        g.refline(y=lqr_loss, color='k', linestyle=':', label='LQR')
    if logy:
        g.set(yscale='log')
    g.set_axis_labels(*axis_labels)
    if show_legend:
        sns.move_legend(g, 'upper center', ncol=2, title=None)
    if formatx:
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: f'{int(x / 1e3)}K')
    plt.tight_layout()
    path_fig = os.path.join(path, 'neuralsystem_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curves_perturbed(
        data: pd.DataFrame, path: str, test_metric_unperturbed: float,
        axis_labels: Optional[Tuple[str, str]] = ('Epoch', 'Loss'),
        logy: Optional[bool] = False, formatx: Optional[bool] = False,
        sharey: Optional[bool] = True):
    # Get test curves corresponding to full controllability and observability.
    data_full_control = data.loc[(data.dropout_probability == 0) &
                                 (data.phase == 'test')]
    g = sns.relplot(data=data_full_control, x='time',
                    y='metric', col='perturbation_type',
                    col_order=PERTURBATIONS.keys(),
                    hue='perturbation_level', kind='line', palette=PALETTE,
                    legend=False, facet_kws={'sharex': False,
                                             'sharey': sharey})

    # Draw unperturbed baseline.
    g.refline(y=test_metric_unperturbed, color='k', linestyle=':')

    g.set_axis_labels(*axis_labels)
    if logy:
        g.set(yscale='log')
    if formatx:
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: f'{int(x/1e3)}K')
    for i, title in enumerate(PERTURBATIONS.values()):
        g.axes[0, i].set_title(title)
    g.axes[0, 0].set(yticklabels=[])
    g.despine(left=True)
    draw_colorbar()
    path_fig = os.path.join(path, 'controller_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_controller_effect(
        data: pd.DataFrame, path: str, test_metric_unperturbed: float,
        ylabel: Optional[str] = 'Loss', logy: Optional[bool] = False,
        formatx: Optional[bool] = False, sharey: Optional[bool] = True,
        remove_ticks: Optional[bool] = True, kind: Optional[str] = 'bar'):
    # Get test curves corresponding to full controllability and observability.
    data_full_control = data.loc[(data.dropout_probability == 0) &
                                 (data.phase == 'test')]
    # Extract first and last timestep and label it as False / True in new
    # "trained" column.
    t_max = data_full_control['time'].max()
    a = data_full_control.loc[data_full_control['time'] == 0].copy()
    b = data_full_control.loc[data_full_control['time'] == t_max].copy()
    a['trained'] = 'before training'
    b['trained'] = 'after training'
    c = pd.concat([a, b])

    if kind == 'bar':
        g = sns.FacetGrid(data=c, hue='trained', col='perturbation_type',
                          col_order=PERTURBATIONS.keys(), legend_out=False,
                          palette=PALETTE, sharex=False, sharey=sharey,
                          height=5)
        g.map(sns.barplot, 'perturbation_level', 'metric')
    elif kind == 'line':
        g = sns.relplot(data=c, x='perturbation_level',
                        y='metric', col='perturbation_type', hue='trained',
                        col_order=PERTURBATIONS.keys(), style='trained',
                        kind='line', palette=PALETTE, legend=True,
                        facet_kws={'sharex': False, 'sharey': sharey})
    else:
        raise NotImplementedError

    g.set_axis_labels('Perturbation level', ylabel)
    if logy:
        g.set(yscale='log')
    if formatx:
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: f'{int(x / 1e3)}K')
    for i, title in enumerate(PERTURBATIONS.values()):
        ax = g.axes[0, i]
        # Draw unperturbed baseline.
        ax.hlines(test_metric_unperturbed, *ax.get_xlim(), color='k',
                  linestyle=':', label='unperturbed')
        ax.set_title(title)
        if remove_ticks:
            ax.set_xticks(np.array(ax.get_xlim()) * [1.2, 0.9])
            ax.set_xticklabels(['low', 'high'])
            ax.set_yticks(np.array(ax.get_ylim()) * [1.2, 0.9])
            ax.set_yticklabels(['low', 'high'])
    g.axes[0, 0].legend()
    g.despine(left=True)
    path_fig = os.path.join(path, f'controller_effect_{kind}.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_gramian_vs_random(
        data: pd.DataFrame, path: str, test_metric_unperturbed: float,
        ylabel: Optional[str] = 'Loss', logy: Optional[bool] = False,
        formatx: Optional[bool] = False, sharey: Optional[bool] = True,
        remove_ticks: Optional[bool] = True,
        metric: Optional[str] = 'test_loss'):
    g = sns.relplot(data=data, x='gramian_value', y='metrics.' + metric,
                    col='params.perturbation_type', row='gramian_type',
                    kind='line', legend=True, hue='params.electrode_selection',
                    style='params.electrode_selection', palette=PALETTE,
                    col_order=PERTURBATIONS.keys(),
                    facet_kws={'sharex': False, 'sharey': sharey})
    g.set_axis_labels('', ylabel)
    g.set_titles('')
    if logy:
        g.set(yscale='log')
    if formatx:
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: f'{int(x / 1e3)}K')
    for i, title in enumerate(PERTURBATIONS.values()):
        ax = g.axes[0, i]
        # Draw unperturbed baseline.
        ax.hlines(test_metric_unperturbed, *ax.get_xlim(), color='k',
                  linestyle=':', label='unperturbed')
        ax.set_title(title)
        g.axes[0, i].set_xlabel('# stimulation electrodes')
        g.axes[1, i].set_xlabel('# recording electrodes')
        if remove_ticks:
            ax.set_xticks(np.array(ax.get_xlim()) * [1.2, 0.9])
            ax.set_xticklabels(['low', 'high'])
            ax.set_yticks(np.array(ax.get_ylim()) * [1.2, 0.9])
            ax.set_yticklabels(['low', 'high'])
    g.axes[0, 0].legend(title=None, frameon=False)
    g.despine(left=True)
    g.legend.remove()
    path_fig = os.path.join(path, f'gramian_vs_random.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_metric_vs_dropout(data: pd.DataFrame, path: str,
                           test_metric_unperturbed: float,
                           metric: Optional[str] = 'test_loss',
                           logy: Optional[bool] = False):
    ylabel = 'Reward' if metric == 'test_reward' else 'Loss'
    g = sns.relplot(data=data, x='gramian_value', y='metrics.' + metric,
                    row='gramian_type', col='params.perturbation_type',
                    hue='params.perturbation_level', palette=PALETTE,
                    col_order=PERTURBATIONS.keys(),
                    style='params.electrode_selection',
                    kind='line', marker='o', markersize=10, legend=False,
                    facet_kws={'sharex': False, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_metric_unperturbed, color='k', linestyle=':')

    if logy:
        g.set(yscale='log')
    g.set(xlim=[-0.05, 1.05])
    g.set_axis_labels('', ylabel)
    for i, title in enumerate(PERTURBATIONS.values()):
        g.axes[0, i].set_title(title)
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
                                  metric: Optional[str] = 'loss',
                                  eval_every_n: Optional[int] = 1
                                  ) -> pd.DataFrame:
    data = {'time': [], 'metric': [], 'phase': [], 'neuralsystem': 'RNN'}
    for run_id in runs['run_id']:
        add_training_curve(data, path, run_id, 'test', metric, eval_every_n)
        add_training_curve(data, path, run_id, 'training', metric,
                           eval_every_n)
    return pd.DataFrame(data)


def get_training_data_perturbed(runs: pd.DataFrame, perturbations: dict,
                                dropout_probabilities: List[float], path: str,
                                metric: Optional[str] = 'loss',
                                eval_every_n: Optional[int] = 1
                                ) -> pd.DataFrame:
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
                                           metric, eval_every_n)
                    add_scalars(data, n, perturbation_type=perturbation_type,
                                perturbation_level=perturbation_level,
                                dropout_probability=dropout_probability)
                    n = add_training_curve(data, path, run_id, 'training',
                                           metric, eval_every_n)
                    add_scalars(data, n, perturbation_type=perturbation_type,
                                perturbation_level=perturbation_level,
                                dropout_probability=dropout_probability)
    return pd.DataFrame(data)


def get_trajectories_unperturbed(
        data_test: mx.gluon.data.DataLoader,
        model_trained: ClosedControlledNeuralSystem,
        model_untrained: ClosedControlledNeuralSystem, pipeline: LqrPipeline
) -> Tuple[pd.DataFrame, float]:
    # Get LQR loss function to compute baseline.
    dt = pipeline.model.environment.dt.data().asnumpy().item()
    loss_function = pipeline.get_loss_function(dt=dt)

    num_batches = len(data_test)
    test_indexes = np.arange(0, num_batches, 16)
    num_steps = model_trained.num_steps
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': []}
    lqr_loss = 0
    for test_index, (lqr_states, lqr_control) in enumerate(data_test):
        # Compute LQR baseline loss.
        lqr_states = lqr_states.as_in_context(pipeline.device)
        lqr_control = lqr_control.as_in_context(pipeline.device)
        lqr_loss += loss_function(lqr_states, lqr_control).mean().asscalar()

        if test_index not in test_indexes:
            continue

        # Get initial state.
        lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
        x_init = lqr_states[:1]

        # Get trajectories of untrained model and LQR.
        _, environment_states = model_untrained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='untrained')
        add_states_mx(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQR',
                    stage='untrained')

        # Get trajectories of trained model and LQR.
        _, environment_states = model_trained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='trained')
        add_states_mx(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQR',
                    stage='trained')
    lqr_loss /= num_batches
    return pd.DataFrame(data), lqr_loss


def get_trajectories_perturbed(
        data_test: mx.gluon.data.DataLoader, test_indexes: List[int],
        environment: StochasticLinearIOSystem, path: str, perturbations: dict,
        pipeline: LqrPipeline, runs: pd.DataFrame) -> pd.DataFrame:
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
            num_steps = model_trained.num_steps
            for test_index, (lqr_states, _) in enumerate(data_test):
                if test_index not in test_indexes:
                    continue

                kwargs = dict(index=test_index,
                              perturbation_type=perturbation_type,
                              perturbation_level=perturbation_level)

                # Get initial state.
                lqr_states = lqr_states.as_in_context(pipeline.device)
                lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
                x_init = lqr_states[:1]

                # Get trajectories of untrained model and LQR.
                _, environment_states = model_untrained(x_init)
                add_states_mx(data, environment_states)
                add_scalars(data, num_steps, controller='RNN',
                            stage='untrained', **kwargs)
                add_states_mx(data, lqr_states)
                add_scalars(data, num_steps, controller='LQR',
                            stage='untrained', **kwargs)

                # Get trajectories of trained model and LQR.
                _, environment_states = model_trained(x_init)
                add_states_mx(data, environment_states)
                add_scalars(data, num_steps, controller='RNN',
                            stage='trained', **kwargs)
                add_states_mx(data, lqr_states)
                add_scalars(data, num_steps, controller='LQR',
                            stage='trained', **kwargs)
    return pd.DataFrame(data)


def get_metric_vs_dropout(runs: pd.DataFrame, perturbations: dict,
                          electrode_selections: list,
                          training_data_perturbed: pd.DataFrame,
                          metric: Optional[str] = 'loss') -> pd.DataFrame:
    # Get metric of uncontrolled perturbed model before training controller.
    t = training_data_perturbed
    data = {f'metrics.test_{metric}': [], f'metrics.training_{metric}': [],
            'params.perturbation_type': [], 'params.perturbation_level': [],
            'params.electrode_selection': [], 'gramian_type': [],
            'gramian_value': []}
    for gramian_type in ['metrics.controllability', 'metrics.observability']:
        for perturbation_type in perturbations.keys():
            for perturbation_level in perturbations[perturbation_type]:
                for electrode_selection in electrode_selections:
                    # r are the uncontrolled perturbed runs at training begin.
                    r = t.loc[(t['perturbation_type'] == perturbation_type) &
                              (t['perturbation_level'] == perturbation_level) &
                              (t['time'] == 0) &
                              (t['dropout_probability'] == 0)]
                    # Add training and test metric.
                    data[f'metrics.test_{metric}'] += list(
                        r[r.phase == 'test']['metric'])
                    data[f'metrics.training_{metric}'] += list(
                        r[r.phase == 'training']['metric'])
                    n = len(r) // 2
                    add_scalars(data, n, **{
                        'params.perturbation_type': perturbation_type,
                        'params.perturbation_level': str(perturbation_level),
                        'params.electrode_selection': electrode_selection,
                        'gramian_type': gramian_type, 'gramian_value': 0})
    # Melt the controllability and observability columns into a "gramian_type"
    # and "gramian_value" column.
    runs = runs.melt(
        var_name='gramian_type', value_name='gramian_value',
        value_vars=['metrics.controllability', 'metrics.observability'],
        id_vars=[f'metrics.test_{metric}', f'metrics.training_{metric}',
                 'params.perturbation_type', 'params.perturbation_level',
                 'params.electrode_selection'])
    # Remove empty rows corresponding to the unperturbed baseline models.
    runs = runs.loc[runs['params.perturbation_type'] != '']
    # Concatenate training curves with baseline results.
    return pd.concat([runs, pd.DataFrame(data)], ignore_index=True)


def get_model(perturbed: bool, trained: bool, *args, **kwargs):
    if trained:
        return get_model_trained(*args, **kwargs)
    else:
        if perturbed:
            return get_model_perturbed_untrained(*args, **kwargs)
        else:
            return get_model_unperturbed_untrained(*args, **kwargs)


def get_model_trained(pipeline, environment, runs: pd.DataFrame, path: str):
    run_id = runs['run_id'].iloc[0]  # First random seed.
    path_model = os.path.join(path, run_id, 'artifacts', 'models',
                              'rnn.params')
    return pipeline.get_model(True, True, environment, path_model)


def get_model_unperturbed_untrained(pipeline, environment):
    return pipeline.get_model(True, True, environment)


def get_model_perturbed_untrained(
        pipeline: LqrPipeline, environment: StochasticLinearIOSystem,
        runs: pd.DataFrame, path: str):
    model = get_model_trained(pipeline, environment, runs, path)
    # Disconnect controller.
    model.controller.initialize(mx.init.Zero(), pipeline.device,
                                force_reinit=True)
    return model


def get_runs_all(path: str, experiment_id: str, tag_start_time: str
                 ) -> pd.DataFrame:
    os.chdir(path)
    runs = mlflow.search_runs([experiment_id],  # f'tags.resume_experiment'
                              f'tags.resume_experiment = "{tag_start_time}"')
    assert len(runs) > 0
    runs.dropna(inplace=True, subset=['metrics.controllability'])
    return runs


def get_runs_perturbed(runs: pd.DataFrame, perturbation_type: str,
                       perturbation_level: float) -> pd.DataFrame:
    return runs.loc[
        (runs['params.perturbation_type'] == perturbation_type) &
        (runs['params.perturbation_level'] == str(perturbation_level)) &
        (runs['params.dropout_probability'] == '0')]


def get_runs_unperturbed(runs: pd.DataFrame) -> pd.DataFrame:
    return get_runs_perturbed(runs, '', 0)


def get_log_path(experiment_name: str) -> str:
    return os.path.expanduser(f'~/Data/neural_control/{experiment_name}')


def add_training_curve(data: dict, path: str, run_id: str, phase: str,
                       metric: Optional[str] = 'loss',
                       eval_every_n: Optional[int] = 1) -> int:
    assert phase in {'training', 'test'}
    filepath = os.path.join(path, run_id, 'metrics', f'{phase}_{metric}')
    values, times = np.loadtxt(filepath, usecols=[1, 2], unpack=True)
    num_steps = len(values)
    data['time'] += list((times + 1) * eval_every_n)
    data['metric'] += list(values)
    data['phase'] += [phase] * num_steps
    return num_steps


def add_scalars(data: dict, n: int, **kwargs):
    for key, value in kwargs.items():
        data[key] += [value] * n


def add_states_mx(data: dict, states: mx.nd.NDArray):
    add_states(data, states.asnumpy())


def add_states(data: dict, states: np.ndarray):
    data['x0'] += states[:, 0, 0].tolist()
    data['x1'] += states[:, 0, 1].tolist()


def draw_colorbar():
    fig = plt.gcf()
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    fig.colorbar(ScalarMappable(cmap=PALETTE), cax=cbar_ax,
                 label='Perturbation')


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'linear_rnn_lqr'
    _tag_start_time = '2022-11-11'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
