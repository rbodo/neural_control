import os
import sys
from typing import Tuple, List

import mlflow
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from examples import configs
from examples.linear_rnn_lqr import LqrPipeline
from src.control_systems_mxnet import StochasticLinearIOSystem, \
    ClosedControlledNeuralSystem
from src.utils import get_data

sns.set_style('white')
sns.set_context('talk')
PALETTE = 'copper'


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
    plot_trajectories(trajectories_unperturbed, 'index', 'Test sample', path,
                      'trajectories_unperturbed.png')

    # Show loss vs epochs of unperturbed model.
    training_data_unperturbed = get_training_data_unperturbed(
        runs_unperturbed, path)
    plot_training_curve_unperturbed(training_data_unperturbed, lqr_loss, path)

    # Show loss vs epochs of perturbed models.
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    training_data_perturbed = get_training_data_perturbed(
        runs, perturbations, dropout_probabilities, path)
    test_loss_unperturbed = runs_unperturbed['metrics.test_loss'].mean()
    plot_training_curves_perturbed(training_data_perturbed, path,
                                   test_loss_unperturbed)

    # Show example trajectories of perturbed model before and after training.
    trajectories_perturbed = get_trajectories_perturbed(
        data_test, [5], environment, path, perturbations, pipeline, runs)

    # Select one perturbation type.
    data_subsystem = trajectories_perturbed.loc[
        trajectories_perturbed['perturbation_type'] == 'sensor']
    plot_trajectories(data_subsystem, 'perturbation_level',
                      'Perturbation level', path, 'trajectories_perturbed.png')

    # Show final test loss of perturbed controlled system for varying degrees
    # of controllability and observability.
    loss_vs_dropout = get_loss_vs_dropout(runs, perturbations,
                                          training_data_perturbed)
    plot_loss_vs_dropout(loss_vs_dropout, path, test_loss_unperturbed)


def plot_trajectories(data: pd.DataFrame, row_key: str, row_label: str,
                      path: str, filename: str):
    g = sns.relplot(data=data, x='x0', y='x1', col='stage', style='controller',
                    hue='controller', row=row_key, kind='line', sort=False,
                    palette=PALETTE, col_order=['untrained', 'trained'],
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
    g.set_titles(row_template=row_label+' {row_name}')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_title('Before training')
    g.axes[0, 1].set_title('After training')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    sns.move_legend(g, 'upper center', bbox_to_anchor=(0.2, 0.97), ncol=2,
                    title=None)
    path_fig = os.path.join(path, filename)
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curve_unperturbed(data: pd.DataFrame, lqr_loss: float,
                                    path: str):
    g = sns.relplot(data=data, x='epoch', y='loss', style='phase',
                    style_order=['training', 'test'], hue='phase', kind='line',
                    legend=True, palette=PALETTE)

    # Draw LQR baseline.
    g.refline(y=lqr_loss, color='k', linestyle=':')

    g.set(yscale='log')
    g.set_axis_labels('Epoch', 'Loss')
    sns.move_legend(g, 'upper center', ncol=2, title=None)
    plt.tight_layout()
    path_fig = os.path.join(path, 'neuralsystem_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_training_curves_perturbed(data: pd.DataFrame, path: str,
                                   test_loss_unperturbed: float):
    # Get test curves corresponding to full controllability and observability.
    data_full_control = data.loc[(data.dropout_probability == 0) &
                                 (data.phase == 'test')]
    g = sns.relplot(data=data_full_control, x='epoch',
                    y='loss', col='perturbation_type',
                    hue='perturbation_level', kind='line', palette=PALETTE,
                    legend=False, facet_kws={'sharex': False, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_loss_unperturbed, color='k', linestyle=':')

    g.set_axis_labels('Epoch', 'Loss')
    g.set(yscale='log')
    g.set_titles('{col_name}')
    g.axes[0, 0].set(yticklabels=[])
    g.despine(left=True)
    draw_colorbar()
    path_fig = os.path.join(path, 'controller_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def plot_loss_vs_dropout(data, path, test_loss_unperturbed):
    g = sns.relplot(data=data, x='gramian_value', y='metrics.test_loss',
                    row='gramian_type', col='params.perturbation_type',
                    hue='params.perturbation_level', palette=PALETTE,
                    kind='line', marker='o', markersize=10, legend=False,
                    facet_kws={'sharex': False, 'sharey': True})

    # Draw unperturbed baseline.
    g.refline(y=test_loss_unperturbed, color='k', linestyle=':')

    g.set(yscale='log', xlim=[-0.05, 1.05])
    g.set_axis_labels('', 'Loss')
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
    path_fig = os.path.join(path, 'loss_vs_dropout.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_training_data_unperturbed(runs: pd.DataFrame,
                                  path: str) -> pd.DataFrame:
    data = {'epoch': [], 'loss': [], 'phase': [], 'neuralsystem': 'RNN'}
    for run_id in runs['run_id']:
        add_training_curve(data, path, run_id, 'test')
        add_training_curve(data, path, run_id, 'training')
    return pd.DataFrame(data)


def get_training_data_perturbed(runs: pd.DataFrame, perturbations: dict,
                                dropout_probabilities: List[float], path: str
                                ) -> pd.DataFrame:
    data = {'epoch': [], 'loss': [], 'phase': [], 'neuralsystem': 'RNN',
            'perturbation_type': [], 'perturbation_level': [],
            'dropout_probability': []}
    for perturbation_type in perturbations.keys():
        for perturbation_level in perturbations[perturbation_type]:
            for dropout_probability in dropout_probabilities:
                runs_perturbed = get_runs_perturbed(runs, perturbation_type,
                                                    perturbation_level)
                for run_id in runs_perturbed['run_id']:
                    n = add_training_curve(data, path, run_id, 'test')
                    add_scalars(data, n, perturbation_type=perturbation_type,
                                perturbation_level=perturbation_level,
                                dropout_probability=dropout_probability)
                    n = add_training_curve(data, path, run_id, 'training')
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
        add_states(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='untrained')
        add_states(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQR',
                    stage='untrained')

        # Get trajectories of trained model and LQR.
        _, environment_states = model_trained(x_init)
        add_states(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='trained')
        add_states(data, lqr_states)
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
                add_states(data, environment_states)
                add_scalars(data, num_steps, controller='RNN',
                            stage='untrained', **kwargs)
                add_states(data, lqr_states)
                add_scalars(data, num_steps, controller='LQR',
                            stage='untrained', **kwargs)

                # Get trajectories of trained model and LQR.
                _, environment_states = model_trained(x_init)
                add_states(data, environment_states)
                add_scalars(data, num_steps, controller='RNN',
                            stage='trained', **kwargs)
                add_states(data, lqr_states)
                add_scalars(data, num_steps, controller='LQR',
                            stage='trained', **kwargs)
    return pd.DataFrame(data)


def get_loss_vs_dropout(runs: pd.DataFrame, perturbations: dict,
                        training_data_perturbed: pd.DataFrame) -> pd.DataFrame:
    # Get loss of uncontrolled perturbed model before training controller.
    t = training_data_perturbed
    data = {'metrics.test_loss': [], 'metrics.training_loss': [],
            'params.perturbation_type': [], 'params.perturbation_level': [],
            'gramian_type': [], 'gramian_value': []}
    for gramian_type in ['metrics.controllability', 'metrics.observability']:
        for perturbation_type in perturbations.keys():
            for perturbation_level in perturbations[perturbation_type]:
                # r are the uncontrolled perturbed runs at begin of training.
                r = t.loc[(t['perturbation_type'] == perturbation_type) &
                          (t['perturbation_level'] == perturbation_level) &
                          (t['epoch'] == 0) & (t['dropout_probability'] == 0)]
                # Add training and test loss.
                data['metrics.test_loss'] += list(r[r.phase == 'test']['loss'])
                data['metrics.training_loss'] += list(
                    r[r.phase == 'training']['loss'])
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
        id_vars=['metrics.test_loss', 'metrics.training_loss',
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


def get_model_trained(
        pipeline: LqrPipeline, environment: StochasticLinearIOSystem,
        runs: pd.DataFrame, path: str):
    run_id = runs['run_id'].iloc[0]  # First random seed.
    path_model = os.path.join(path, run_id, 'artifacts', 'models',
                              'rnn.params')
    return pipeline.get_model(True, True, environment, path_model)


def get_model_unperturbed_untrained(
        pipeline: LqrPipeline, environment: StochasticLinearIOSystem):
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
                              f'tags.main_start_time = "{tag_start_time}"')
    runs.dropna(inplace=True)
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


def add_training_curve(data: dict, path: str, run_id: str, phase: str) -> int:
    assert phase in {'training', 'test'}
    filepath = os.path.join(path, run_id, 'metrics', f'{phase}_loss')
    loss, epochs = np.loadtxt(filepath, usecols=[1, 2], unpack=True)
    num_steps = len(loss)
    data['epoch'] += list(epochs + 1)
    data['loss'] += list(loss)
    data['phase'] += [phase] * num_steps
    return num_steps


def add_scalars(data: dict, n: int, **kwargs):
    for key, value in kwargs.items():
        data[key] += [value] * n


def add_states(data: dict, states: mx.nd.NDArray):
    data['x0'] += states[:, 0, 0].asnumpy().tolist()
    data['x1'] += states[:, 0, 1].asnumpy().tolist()


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
    _tag_start_time = '2022-09-21_17:29:07'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
