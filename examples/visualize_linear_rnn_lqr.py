import os
import sys
from collections import OrderedDict

import matplotlib
from matplotlib.ticker import PercentFormatter
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
sns.set_context('poster')
matplotlib.rc('axes', edgecolor='lightgrey')
matplotlib.rc('savefig', format='png')  # svg or png
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

    title = 'LQR direct: Particle stabilization'

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed, lqr_loss = get_trajectories_unperturbed(
        data_test, model_trained, model_untrained, pipeline)
    plot_trajectories_unperturbed(trajectories_unperturbed, log_path,
                                  '(a) ' + title)

    # Show metric vs times of unperturbed model.
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path,
                                    lqr_loss, logy=True)

    # Show metric vs times of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path)
    test_metric_unperturbed = runs_unperturbed['metrics.test_loss'].mean()
    plot_training_curves_perturbed(training_data_perturbed, log_path,
                                   test_metric_unperturbed, logy=True)
    plot_controller_effect(training_data_perturbed, log_path,
                           test_metric_unperturbed, logy=True, aspect=0.9)

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        data_test, [5], environment, path, perturbations, pipeline, runs)
    plot_trajectories_perturbed(trajectories_perturbed, log_path)

    # Show final test metric of perturbed controlled system for varying degrees
    # of controllability and observability.
    sns.set_context('talk')
    metric_vs_dropout = get_metric_vs_dropout(runs, perturbations,
                                              training_data_perturbed)
    n = config.model.NUM_HIDDEN_NEURALSYSTEM
    num_electrodes = get_num_electrodes(runs, perturbations, path, n)
    plot_metric_vs_dropout_average(
        metric_vs_dropout, log_path, test_metric_unperturbed, logy=True,
        num_electrodes=num_electrodes, set_xlabels=True, set_col_labels=True,
        title=None)
    plot_metric_vs_dropout(metric_vs_dropout, log_path,
                           test_metric_unperturbed, logy=True)


def plot_trajectories_unperturbed(data: pd.DataFrame, path: str,
                                  title: Optional[str] = None):
    g = sns.relplot(data=data, x='x0', y='x1', kind='line', style='controller',
                    hue='controller', col='index', sort=False, palette=PALETTE,
                    aspect=0.8, facet_kws={'sharex': True, 'sharey': True,
                                           'margin_titles': True})

    draw_coordinate_system(g, (-0.65, -0.65), axis=(0, 1))

    if title is not None:
        draw_title(g.axes[0, 0], title)

    # Draw target state.
    xt = [0, 0]
    for ax in g.axes.ravel():
        ax.scatter(xt[0], xt[1], s=64, marker='x', c='k')

    lim = 0.99
    g.set(xlim=[-lim, lim], ylim=[-lim, lim], xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template='',  # 'Test sample {col_name}'
                 row_template='')
    g.despine(left=False, bottom=False, top=False, right=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    g.axes[0, 0].legend(loc='best', frameon=False, ncol=1, title=None)
    g.axes[0, 0].set_zorder(1)  # So the legend is on top of other panels.
    g.legend.remove()
    path_fig = os.path.join(path, 'trajectories_unperturbed')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_trajectories_perturbed(data: pd.DataFrame, path: str):
    g = sns.relplot(data=data, x='x0', y='x1', kind='line', style='controller',
                    hue='controller', col='perturbation_level', sort=False,
                    palette=PALETTE, row='perturbation_type', height=3.8,
                    aspect=0.8, facet_kws={'sharex': False, 'sharey': True,
                                           'margin_titles': True})

    draw_coordinate_system(g, (-0.65, -0.65), axis=(2, 0))

    # Draw target state.
    xt = [0, 0]
    for ax in g.axes.ravel():
        ax.scatter(xt[0], xt[1], s=64, marker='x', c='k')

    lim = 0.99
    g.set(xlim=[-lim, lim], ylim=[-lim, lim], xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(col_template='',  # Perturbation level {col_name:.0%}',
                 row_template='')
    g.axes[2, 2].set_xlabel('Perturbation level')
    g.axes[2, 0].set_xticks([g.axes[2, 0].get_xlim()[0] * 1.2])
    g.axes[2, 0].set_xticklabels(['low'])
    g.axes[2, 4].set_xticks([g.axes[2, 4].get_xlim()[-1] * 0.9])
    g.axes[2, 4].set_xticklabels(['high'])
    enums = ['(a) ', '(b) ', '(c) ']
    for i, ylabel in enumerate(PERTURBATIONS.values()):
        draw_title(g.axes[i, 0], enums[i] + ylabel)
    g.despine(left=False, bottom=False, top=False, right=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
    g.legend.remove()
    g.axes[0, 0].legend(loc='best', ncol=1, title=None, frameon=False)
    g.axes[0, 0].set_zorder(1)  # So the legend is on top of other panels.
    path_fig = os.path.join(path, 'trajectories_perturbed')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curve_unperturbed(
        data: pd.DataFrame, path: str, lqr_loss: Optional[float] = None,
        axis_labels: Optional[Tuple[str, str]] = ('Epoch', 'Loss'),
        show_legend: Optional[bool] = True, logy: Optional[bool] = False,
        formatx: Optional[bool] = False, **plt_kwargs):
    g = sns.relplot(data=data, x='time', y='metric', style='phase',
                    style_order=['training', 'test'], hue='phase', kind='line',
                    legend=show_legend, palette=PALETTE, **plt_kwargs)

    # Draw LQR baseline.
    if lqr_loss is not None:
        g.refline(y=lqr_loss, color='k', linestyle=':', label='LQR')
    if logy:
        g.set(yscale='log')
    g.set_axis_labels(*axis_labels)
    if show_legend:
        g.axes[0, 0].legend(loc='best', frameon=False, ncol=1, title=None)
        if g.legend is not None:
            g.legend.remove()
    if formatx:
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(lambda x, p: f'{int(x / 1e3)}K')
    plt.tight_layout()
    g.despine(left=False, bottom=False, top=False, right=False)
    path_fig = os.path.join(path, 'neuralsystem_training')
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
                    col_order=PERTURBATIONS.keys(), legend=False,
                    hue='perturbation_level', kind='line', palette=PALETTE,
                    facet_kws={'sharex': False, 'sharey': sharey})

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
    g.despine(left=False, bottom=False, top=False, right=False)
    draw_colorbar()
    path_fig = os.path.join(path, 'controller_training')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_controller_effect(
        data: pd.DataFrame, path: str, test_metric_unperturbed: float,
        ylabel: Optional[str] = 'Loss', logy: Optional[bool] = False,
        formatx: Optional[bool] = False, sharey: Optional[bool] = True,
        remove_ticks: Optional[bool] = True, kind: Optional[str] = 'line',
        aspect: Optional[float] = 1):
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
    c = pd.concat([b, a])

    if kind == 'bar':
        g = sns.FacetGrid(data=c, hue='trained', row='perturbation_type',
                          row_order=PERTURBATIONS.keys(), legend_out=False,
                          palette=PALETTE, sharex=False, sharey=sharey,
                          height=5)
        g.map(sns.barplot, 'perturbation_level', 'metric')
    elif kind == 'line':
        g = sns.relplot(data=c, x='perturbation_level', y='metric',
                        row='perturbation_type', hue='trained', kind='line',
                        row_order=PERTURBATIONS.keys(), style='trained',
                        palette=PALETTE, markers=True, height=4, aspect=aspect,
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
        ax = g.axes[i, 0]
        # Draw unperturbed baseline.
        ax.hlines(test_metric_unperturbed, *ax.get_xlim(), color='k',
                  linestyle=':', label='unperturbed')
        ax.set_title('')
        if remove_ticks:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks(np.array(ax.get_ylim()) * [1.2, 0.9])
            ax.set_yticklabels(['low', 'high'])
    if remove_ticks:
        ax = g.axes[-1, 0]
        ax.set_xticks(np.array(ax.get_xlim()) * [1.2, 0.9])
        ax.set_xticklabels(['low', 'high'])
    g.axes[0, 0].legend(frameon=False)
    if g.legend is not None:
        g.legend.remove()
    g.despine(left=False, bottom=False, top=False, right=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
    path_fig = os.path.join(path, f'controller_effect_{kind}')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_metric_vs_dropout_average(
        data: pd.DataFrame, path: str, test_metric_unperturbed: float,
        metric: Optional[str] = 'test_loss', logy: Optional[bool] = False,
        title: Optional[str] = None, set_xlabels: Optional[bool] = True,
        set_col_labels: Optional[bool] = True,
        num_electrodes: Optional[pd.DataFrame] = None):
    ylabel = 'Reward' if metric == 'test_reward' else 'Loss'
    g = sns.relplot(data=data, x='gramian_value', y='metrics.' + metric,
                    style='gramian_type', col='params.perturbation_type',
                    hue='gramian_type', palette=PALETTE,
                    col_order=PERTURBATIONS.keys(), kind='line',
                    facet_kws={'sharex': True, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_metric_unperturbed, color='k', linestyle=':',
              label='unperturbed')

    # Draw electrode number to achieve a certain energy in eigenspectrum.
    if num_electrodes is not None:
        d = num_electrodes.groupby(['perturbation_type']).mean()
        for i, perturbation in enumerate(PERTURBATIONS.keys()):
            n_c = d.loc[perturbation]['num_actuators']
            n_o = d.loc[perturbation]['num_sensors']
            ax = g.axes[0, i]
            ax.annotate('', xy=(n_c, 0), xytext=(n_c, -0.1),
                        xycoords='axes fraction',
                        arrowprops=dict(color=ax.lines[0].get_color(),
                                        linestyle=ax.lines[0].get_linestyle()))
            ax.annotate('', xy=(n_o, 0), xytext=(n_o, -0.1),
                        xycoords='axes fraction',
                        arrowprops=dict(color=ax.lines[1].get_color(),
                                        linestyle=ax.lines[1].get_linestyle()))

    if logy:
        g.set(yscale='log')
    g.set(xlim=[-0.05, 1.05])
    g.set_axis_labels('', ylabel)
    g.set(xticklabels=[])
    if set_xlabels:
        for ax in g.axes[0]:
            ax.set_xlabel('Electrode coverage [%]')
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0,
                                                          symbol=''))
    for i, col_label in enumerate(PERTURBATIONS.values()):
        g.axes[0, i].set_title(col_label if set_col_labels else None)
    if title is not None:
        draw_title(g.axes[0, 0], title, 0.1 if set_col_labels else 0)
    g.axes[0, 0].set(yticklabels=[])
    g.axes[0, 0].legend(g.axes[0, 0].lines[2:], ['stimulation', 'recording',
                                                 'unperturbed'], frameon=False)
    if g.legend is not None:
        g.legend.remove()
    g.despine(left=False, bottom=False, top=False, right=False)
    path_fig = os.path.join(path, 'metric_vs_dropout_average')
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
                    col_order=PERTURBATIONS.keys(), legend=False,
                    kind='line', marker='o', markersize=10,
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
    g.despine(left=False, bottom=False, top=False, right=False)
    for ax in g.axes[0]:
        ax.set_xlabel('Controllability')
    for ax in g.axes[1]:
        ax.set_xlabel('Observability')
        ax.set_title('')
    draw_colorbar()
    path_fig = os.path.join(path, 'metric_vs_dropout')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_dim_at_energy(gramian: np.ndarray,
                      threshold: Optional[float] = 0.9) -> int:
    # Compute eigenvalues and eigenvectors of Gramian.
    w, v = np.linalg.eigh(gramian)
    # Sort ascending.
    w = w[::-1]
    # Determine how many eigenvectors to use.
    fraction_explained = np.cumsum(w) / np.sum(w)
    n = np.min(np.flatnonzero(fraction_explained > threshold))
    return n


def get_num_electrodes(runs: pd.DataFrame, perturbations: dict, path: str,
                       num_neurons: int, thr: Optional[float] = 0.9
                       ) -> pd.DataFrame:
    data = {'perturbation_type': [], 'perturbation_level': [],
            'num_sensors': [], 'num_actuators': []}
    for perturbation in perturbations.keys():
        for level in perturbations[perturbation]:
            run_ids = runs.loc[
                (runs['params.perturbation_type'] == perturbation) &
                (runs['params.perturbation_level'] == str(level)) &
                (runs['params.dropout_probability'] == '0')]['run_id']
            for run_id in run_ids:
                g = np.load(os.path.join(path, run_id, 'artifacts',
                                         'gramians.npz'))
                n_o = get_dim_at_energy(g['observability_gramian'], thr)
                n_c = get_dim_at_energy(g['controllability_gramian'], thr)
                data['perturbation_type'].append(perturbation)
                data['perturbation_level'].append(level)
                data['num_sensors'].append(n_o / num_neurons)
                data['num_actuators'].append(n_c / num_neurons)
    return pd.DataFrame(data)


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
    data = {'index': [], 'controller': [], 'x0': [], 'x1': []}
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

        # Get trajectories of trained model.
        _, environment_states = model_trained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index,
                    controller='RNN after training')

        # Get trajectories of untrained model.
        _, environment_states = model_untrained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index,
                    controller='RNN before training')

        # Store LQR trajectory.
        add_states_mx(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQR')

    lqr_loss /= num_batches
    return pd.DataFrame(data), lqr_loss


def get_trajectories_perturbed(
        data_test: mx.gluon.data.DataLoader, test_indexes: List[int],
        environment: StochasticLinearIOSystem, path: str, perturbations: dict,
        pipeline: LqrPipeline, runs: pd.DataFrame,
        use_relative_levels: Optional[bool] = True) -> pd.DataFrame:
    data = {'index': [], 'controller': [], 'x0': [], 'x1': [],
            'perturbation_type': [], 'perturbation_level': []}
    for perturbation_type in perturbations.keys():
        for i, level in enumerate(perturbations[perturbation_type]):
            runs_perturbed = get_runs_perturbed(runs, perturbation_type, level)
            model_trained = get_model_trained(
                pipeline, environment, runs_perturbed, path)
            model_untrained = get_model_perturbed_untrained(
                pipeline, environment, runs_perturbed, path)
            num_steps = model_trained.num_steps
            if use_relative_levels:
                level = (i + 1) / len(perturbations[perturbation_type])
            for test_index, (lqr_states, _) in enumerate(data_test):
                if test_index not in test_indexes:
                    continue

                kwargs = dict(index=test_index,
                              perturbation_type=perturbation_type,
                              perturbation_level=level)

                # Get initial state.
                lqr_states = lqr_states.as_in_context(pipeline.device)
                lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
                x_init = lqr_states[:1]

                # Get trajectories of trained model.
                _, environment_states = model_trained(x_init)
                add_states_mx(data, environment_states)
                add_scalars(data, num_steps, controller='RNN after training',
                            **kwargs)

                # Get trajectories of untrained model.
                _, environment_states = model_untrained(x_init)
                add_states_mx(data, environment_states)
                add_scalars(data, num_steps, controller='RNN before training',
                            **kwargs)

                # Store LQR trajectory.
                add_states_mx(data, lqr_states)
                add_scalars(data, num_steps, controller='LQR', **kwargs)
    return pd.DataFrame(data)


def get_metric_vs_dropout(runs: pd.DataFrame, perturbations: dict,
                          training_data_perturbed: pd.DataFrame,
                          metric: Optional[str] = 'loss') -> pd.DataFrame:
    # Get metric of uncontrolled perturbed model before training controller.
    t = training_data_perturbed
    data = {f'metrics.test_{metric}': [], f'metrics.training_{metric}': [],
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
                data[f'metrics.test_{metric}'] += list(
                    r[r.phase == 'test']['metric'])
                data[f'metrics.training_{metric}'] += list(
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
        id_vars=[f'metrics.test_{metric}', f'metrics.training_{metric}',
                 'params.perturbation_type', 'params.perturbation_level'])
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
    runs = mlflow.search_runs([experiment_id],
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


def get_log_path(experiment_name: str, path: Optional[str] = None) -> str:
    if path is None:
        path = '~/Data/neural_control_snellius'
    return os.path.expanduser(os.path.join(path, experiment_name))


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


def draw_coordinate_system(g: sns.FacetGrid, xy: Tuple[float, float],
                           xycoords: Optional[str] = 'data',
                           axis: Optional[Tuple[int, int]] = (0, 0)):
    """Draw coordinate system as inlet in first panel."""
    arrowprops = dict(arrowstyle='-', connectionstyle='arc3', fc='k', ec='k')
    ax = g.axes[axis[0], axis[1]]
    ax.annotate('Position', xy=xy, xycoords=xycoords, xytext=(50, 0),
                textcoords='offset points', verticalalignment='center',
                arrowprops=arrowprops)
    ax.annotate('Velocity', xy=xy, xycoords=xycoords, xytext=(0, 50),
                textcoords='offset points', horizontalalignment='center',
                arrowprops=arrowprops)


def draw_title(axis: plt.axis, title: str, yoffset: Optional[float] = 0):
    axis.annotate(title, xy=(0.01, 1.01 + yoffset), xycoords='axes fraction',
                  weight='bold')


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
    _tag_start_time = '2022-11-18'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
