import os
import sys

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from examples import configs
from examples.linear_rnn_lqg import LqgPipeline
from examples.visualize_linear_rnn_lqr import PALETTE
from src.control_systems_mxnet import StochasticLinearIOSystem, \
    ClosedControlledNeuralSystem
from src.utils import get_data


def main(experiment_id, experiment_name, tag_start_time):
    # Get path where experiment data has been saved.
    log_path = get_log_path(experiment_name)
    path = os.path.join(log_path, 'mlruns', experiment_id)

    # Get all training runs.
    runs = get_runs_all(log_path, experiment_id, tag_start_time)

    # Get configuration.
    config = configs.linear_rnn_lqg.get_config()

    # Get data to produce example trajectories through phase space.
    data_dict = get_data(config, 'states')
    data_test = data_dict['data_test']

    # Get pipeline for LQG experiment.
    pipeline = LqgPipeline(config, data_dict)
    pipeline.device = pipeline.get_device()

    # Get environment to produce example trajectories through phase space.
    environment = pipeline.get_environment()

    # Get unperturbed model before training.
    model_untrained = get_model_unperturbed_untrained(pipeline, environment)

    # Get unperturbed model after training.
    model_trained = get_model_trained(pipeline, environment, runs, path)
    pipeline.model = model_trained

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed = get_trajectories_unperturbed(
        data_test, model_trained, model_untrained, pipeline)
    plot_trajectories(trajectories_unperturbed, 'index', 'Test sample', path,
                      'trajectories_unperturbed.png')

    # Show loss vs epochs of unperturbed model.
    training_data_unperturbed = get_training_data_unperturbed(runs, path)
    plot_training_curve_unperturbed(training_data_unperturbed, path)


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


def plot_training_curve_unperturbed(data: pd.DataFrame, path: str):
    g = sns.relplot(data=data, x='epoch', y='loss', style='phase',
                    style_order=['training', 'test'], hue='phase', kind='line',
                    legend=True, palette=PALETTE)

    g.set(yscale='log')
    g.set_axis_labels('Epoch', 'Loss')
    sns.move_legend(g, 'upper center', ncol=2, title=None)
    plt.tight_layout()
    path_fig = os.path.join(path, 'neuralsystem_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


def get_training_data_unperturbed(runs: pd.DataFrame,
                                  path: str) -> pd.DataFrame:
    data = {'epoch': [], 'loss': [], 'phase': [], 'neuralsystem': 'RNN'}
    for run_id in runs['run_id']:
        add_training_curve(data, path, run_id, 'test')
        add_training_curve(data, path, run_id, 'training')
    return pd.DataFrame(data)


def get_trajectories_unperturbed(
        data_test: mx.gluon.data.DataLoader,
        model_trained: ClosedControlledNeuralSystem,
        model_untrained: ClosedControlledNeuralSystem, pipeline: LqgPipeline
) -> pd.DataFrame:
    num_batches = len(data_test)
    test_indexes = np.arange(0, num_batches, 16)
    num_steps = model_trained.num_steps
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': []}
    for test_index, (lqr_states, lqr_control) in enumerate(data_test):
        # Compute baseline loss.
        lqr_states = lqr_states.as_in_context(pipeline.device)

        if test_index not in test_indexes:
            continue

        # Get initial state.
        lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
        x_init = lqr_states[:1]

        # Get trajectories of untrained model and LQG.
        _, environment_states = model_untrained(x_init)
        add_states(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='untrained')
        add_states(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQG',
                    stage='untrained')

        # Get trajectories of trained model and LQG.
        _, environment_states = model_trained(x_init)
        add_states(data, environment_states)
        add_scalars(data, num_steps, index=test_index, controller='RNN',
                    stage='trained')
        add_states(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQG',
                    stage='trained')
    return pd.DataFrame(data)


def get_model_trained(
        pipeline: LqgPipeline, environment: StochasticLinearIOSystem,
        runs: pd.DataFrame, path: str):
    run_id = runs['run_id'].iloc[0]  # First random seed.
    path_model = os.path.join(path, run_id, 'artifacts', 'models',
                              'rnn.params')
    pipeline.model = pipeline.get_model(True, True, environment, path_model)
    return ClosedControlledNeuralSystem(
        environment, pipeline.model.neuralsystem, pipeline.model.controller,
        pipeline.device, pipeline.model.batch_size,
        pipeline.config.simulation.NUM_STEPS)


def get_model_unperturbed_untrained(
        pipeline: LqgPipeline, environment: StochasticLinearIOSystem):
    pipeline.model = pipeline.get_model(True, True, environment)
    return ClosedControlledNeuralSystem(
        environment, pipeline.model.neuralsystem, pipeline.model.controller,
        pipeline.device, pipeline.model.batch_size,
        pipeline.config.simulation.NUM_STEPS)


def get_runs_all(path: str, experiment_id: str, tag_start_time: str
                 ) -> pd.DataFrame:
    os.chdir(path)
    runs = mlflow.search_runs([experiment_id],
                              f'tags.resume_experiment = "{tag_start_time}"')
    return runs


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


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'linear_rnn_lqg'
    _tag_start_time = '2022-10-18_14:21:00'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
