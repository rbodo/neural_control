# Try how easy it is to iterate over mlflow directories and extract results
# into pandas DF.

import os
import sys

import mlflow
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from examples import configs
from examples.linear_rnn_lqr import LqrPipeline
from src.utils import get_data

sns.set_style('white')
sns.set_context('talk')


def main():
    experiment_id = '1'
    label = 'linear_rnn_lqr'
    tag = '2022-09-21_17:29:07'
    path = os.path.expanduser(f'~/Data/neural_control/{label}')
    os.chdir(path)
    runs = mlflow.search_runs([experiment_id],
                              f'tags.main_start_time = "{tag}"')
    runs.dropna(inplace=True)

    perturbation_type = ''
    perturbation_level = 0
    dropout_probability = 0
    runs_unperturbed = runs.loc[
        (runs['params.perturbation_type'] == str(perturbation_type)) &
        (runs['params.perturbation_level'] == str(perturbation_level)) &
        (runs['params.dropout_probability'] == str(dropout_probability))]

    config = configs.linear_rnn_lqr.get_config()

    # Get trajectories of a classic LQR controller in the double integrator
    # state space. In this study we use only the initial values for training
    # the RNN, and plot the LQR trajectories as comparison.
    data_dict = get_data(config, 'states')

    pipeline = LqrPipeline(config, data_dict)
    pipeline.device = pipeline.get_device()
    run_id = runs_unperturbed['run_id'].iloc[0]
    path_model = os.path.join(path, 'mlruns', experiment_id, run_id,
                              'artifacts', 'models', 'rnn.params')
    environment = pipeline.get_environment()
    model_untrained = pipeline.get_model(True, True, environment)
    model_trained = pipeline.get_model(True, True, environment, path_model)
    pipeline.model = model_trained

    dt = environment.dt.data().asnumpy().item()
    loss_function = pipeline.get_loss_function(dt=dt)
    data_test = data_dict['data_test']
    test_indexes = np.arange(0, len(data_test), 16)
    n = model_trained.num_steps
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': []}
    lqr_loss = 0
    for test_index, (lqr_states, label) in enumerate(data_test):
        lqr_states = lqr_states.as_in_context(pipeline.device)
        label = label.as_in_context(pipeline.device)
        lqr_loss += loss_function(lqr_states, label).mean().asscalar()
        if test_index not in test_indexes:
            continue
        lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
        x_init = lqr_states[:1]
        _, environment_states = model_trained(x_init)
        data['index'] += [test_index] * n
        data['controller'] += ['RNN'] * n
        data['stage'] += ['trained'] * n
        data['x0'] += environment_states[:, 0, 0].asnumpy().tolist()
        data['x1'] += environment_states[:, 0, 1].asnumpy().tolist()
        data['index'] += [test_index] * n
        data['controller'] += ['LQR'] * n
        data['stage'] += ['trained'] * n
        data['x0'] += lqr_states[:, 0, 0].asnumpy().tolist()
        data['x1'] += lqr_states[:, 0, 1].asnumpy().tolist()
        _, environment_states = model_untrained(x_init)
        data['index'] += [test_index] * n
        data['controller'] += ['RNN'] * n
        data['stage'] += ['untrained'] * n
        data['x0'] += environment_states[:, 0, 0].asnumpy().tolist()
        data['x1'] += environment_states[:, 0, 1].asnumpy().tolist()
        data['index'] += [test_index] * n
        data['controller'] += ['LQR'] * n
        data['stage'] += ['untrained'] * n
        data['x0'] += lqr_states[:, 0, 0].asnumpy().tolist()
        data['x1'] += lqr_states[:, 0, 1].asnumpy().tolist()
    lqr_loss /= len(data_test)
    df0 = pd.DataFrame(data)
    g = sns.relplot(data=df0, x='x0', y='x1', col='stage', style='controller',
                    hue='controller', row='index', kind='line', sort=False,
                    palette='copper', col_order=['untrained', 'trained'],
                    facet_kws={'sharex': True, 'sharey': True,
                               'margin_titles': True})
    lim = 0.99
    g.set(xlim=[-lim, lim], ylim=[-lim, lim], xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(row_template='Test sample {row_name}')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_title('Before training')
    g.axes[0, 1].set_title('After training')
    arrowprops = dict(arrowstyle='-', connectionstyle='arc3', fc='k', ec='k')
    g.axes[0, 0].annotate(
        'Position', xy=(-0.75, -0.75), xycoords='data', xytext=(75, 0),
        textcoords='offset points', verticalalignment='center',
        arrowprops=arrowprops)
    g.axes[0, 0].annotate(
        'Velocity', xy=(-0.75, -0.75), xycoords='data', xytext=(0, 75),
        textcoords='offset points', horizontalalignment='center',
        arrowprops=arrowprops)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    sns.move_legend(g, 'upper center', bbox_to_anchor=(0.2, 0.97), ncol=2,
                    title=None)
    # Draw target state.
    xt = [0, 0]
    for ax in g.axes.ravel():
        ax.scatter(xt[0], xt[1], s=64, marker='x', c='k')
    path_fig = os.path.join(path, 'mlruns', experiment_id,
                            f'trajectories_unperturbed.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()

    data = {'epoch': [], 'loss': [], 'phase': [], 'neuralsystem': 'RNN'}
    for run_id in runs_unperturbed['run_id']:
        filepath = os.path.join(path, 'mlruns', experiment_id, run_id,
                                'metrics', 'test_loss')
        test_loss, epochs = np.loadtxt(filepath, usecols=[1, 2], unpack=True)
        data['epoch'] += list(epochs + 1)
        data['loss'] += list(test_loss)
        data['phase'] += ['test'] * len(test_loss)
        filepath = os.path.join(path, 'mlruns', experiment_id, run_id,
                                'metrics', 'training_loss')
        training_loss, epochs = np.loadtxt(filepath, usecols=[1, 2],
                                           unpack=True)
        data['epoch'] += list(epochs + 1)
        data['loss'] += list(training_loss)
        data['phase'] += ['train'] * len(training_loss)
    df1 = pd.DataFrame(data)

    g = sns.relplot(data=df1, x='epoch', y='loss', style='phase',
                    style_order=['train', 'test'], hue='phase', kind='line',
                    legend=True, palette='copper')
    g.refline(y=lqr_loss, color='k', linestyle=':')
    g.set(yscale='log')
    g.set_axis_labels('Epoch', 'Loss')
    sns.move_legend(g, 'upper center', ncol=2, title=None)
    plt.tight_layout()

    path_fig = os.path.join(path, 'mlruns', experiment_id,
                            'neuralsystem_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()

    data = {'epoch': [], 'loss': [], 'phase': [], 'neuralsystem': 'RNN',
            'perturbation_type': [], 'perturbation_level': [],
            'dropout_probability': []}
    config = configs.linear_rnn_lqr.get_config()
    perturbations = dict(config.perturbation.PERTURBATIONS)
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    for perturbation_type in perturbations.keys():
        for perturbation_level in perturbations[perturbation_type]:
            for dropout_probability in dropout_probabilities:
                runs_perturbed = runs.loc[
                    (runs['params.perturbation_type'] == str(
                        perturbation_type)) &
                    (runs['params.perturbation_level'] == str(
                        perturbation_level)) &
                    (runs['params.dropout_probability'] == str(
                        dropout_probability))]
                for run_id in runs_perturbed['run_id']:
                    filepath = os.path.join(path, 'mlruns', experiment_id,
                                            run_id, 'metrics', 'test_loss')
                    test_loss, epochs = np.loadtxt(filepath, usecols=[1, 2],
                                                   unpack=True)
                    n = len(epochs)
                    data['epoch'] += list(epochs + 1)
                    data['loss'] += list(test_loss)
                    data['phase'] += ['test'] * n
                    data['perturbation_type'] += [perturbation_type] * n
                    data['perturbation_level'] += [perturbation_level] * n
                    data['dropout_probability'] += [dropout_probability] * n

                    filepath = os.path.join(path, 'mlruns', experiment_id,
                                            run_id, 'metrics', 'training_loss')
                    training_loss, epochs = np.loadtxt(
                        filepath, usecols=[1, 2], unpack=True)
                    data['epoch'] += list(epochs + 1)
                    data['loss'] += list(training_loss)
                    data['phase'] += ['training'] * n
                    data['perturbation_type'] += [perturbation_type] * n
                    data['perturbation_level'] += [perturbation_level] * n
                    data['dropout_probability'] += [dropout_probability] * n
    df2 = pd.DataFrame(data)

    g = sns.relplot(data=df2.loc[(df2.dropout_probability == 0) &
                                 (df2.phase == 'test')], x='epoch',
                    y='loss', col='perturbation_type',
                    hue='perturbation_level', kind='line', palette='copper',
                    legend=False, facet_kws={'sharex': False, 'sharey': True})

    test_loss_unperturbed = runs_unperturbed['metrics.test_loss'].mean()
    g.refline(y=test_loss_unperturbed, color='k', linestyle=':')

    g.set_axis_labels('Epoch', 'Loss')
    g.set(yscale='log')
    g.set_titles('{col_name}')
    g.axes[0, 0].set(yticklabels=[])
    g.despine(left=True)

    fig = plt.gcf()
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    fig.colorbar(ScalarMappable(cmap='copper'), cax=cbar_ax,
                 label='Perturbation')

    path_fig = os.path.join(path, 'mlruns', experiment_id,
                            'controller_training.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()

    test_indexes = [5]
    data = {'index': [], 'controller': [], 'stage': [], 'x0': [], 'x1': [],
            'perturbation_type': [], 'perturbation_level': []}
    for perturbation_type in perturbations.keys():
        for perturbation_level in perturbations[perturbation_type]:
            runs_perturbed = runs.loc[
                (runs['params.perturbation_type'] == str(
                    perturbation_type)) &
                (runs['params.perturbation_level'] == str(
                    perturbation_level)) &
                (runs['params.dropout_probability'] == '0')]
            run_id = runs_perturbed['run_id'].iloc[0]  # First random seed
            path_model = os.path.join(path, 'mlruns', experiment_id, run_id,
                                      'artifacts', 'models', 'rnn.params')
            model_trained = pipeline.get_model(True, True, environment,
                                               path_model)
            model_untrained = pipeline.get_model(True, True, environment,
                                                 path_model)
            model_untrained.controller.initialize(
                mx.init.Zero(), pipeline.device, force_reinit=True)
            n = model_trained.num_steps
            for test_index, (lqr_states, label) in enumerate(data_test):
                if test_index not in test_indexes:
                    continue
                lqr_states = lqr_states.as_in_context(pipeline.device)
                lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
                x_init = lqr_states[:1]
                _, environment_states = model_trained(x_init)
                data['index'] += [test_index] * n
                data['controller'] += ['RNN'] * n
                data['stage'] += ['trained'] * n
                data['x0'] += environment_states[:, 0, 0].asnumpy().tolist()
                data['x1'] += environment_states[:, 0, 1].asnumpy().tolist()
                data['perturbation_type'] += [perturbation_type] * n
                data['perturbation_level'] += [perturbation_level] * n
                data['index'] += [test_index] * n
                data['controller'] += ['LQR'] * n
                data['stage'] += ['trained'] * n
                data['x0'] += lqr_states[:, 0, 0].asnumpy().tolist()
                data['x1'] += lqr_states[:, 0, 1].asnumpy().tolist()
                data['perturbation_type'] += [perturbation_type] * n
                data['perturbation_level'] += [perturbation_level] * n
                _, environment_states = model_untrained(x_init)
                data['index'] += [test_index] * n
                data['controller'] += ['RNN'] * n
                data['stage'] += ['untrained'] * n
                data['x0'] += environment_states[:, 0, 0].asnumpy().tolist()
                data['x1'] += environment_states[:, 0, 1].asnumpy().tolist()
                data['perturbation_type'] += [perturbation_type] * n
                data['perturbation_level'] += [perturbation_level] * n
                data['index'] += [test_index] * n
                data['controller'] += ['LQR'] * n
                data['stage'] += ['untrained'] * n
                data['x0'] += lqr_states[:, 0, 0].asnumpy().tolist()
                data['x1'] += lqr_states[:, 0, 1].asnumpy().tolist()
                data['perturbation_type'] += [perturbation_type] * n
                data['perturbation_level'] += [perturbation_level] * n
    df4 = pd.DataFrame(data)
    g = sns.relplot(data=df4.loc[df4['perturbation_type'] == 'sensor'], x='x0',
                    y='x1', col='stage', style='controller', hue='controller',
                    row='perturbation_level', kind='line', sort=False,
                    palette='copper', col_order=['untrained', 'trained'],
                    facet_kws={'sharex': True, 'sharey': True,
                               'margin_titles': True})
    lim = 0.99
    g.set(xlim=[-lim, lim], ylim=[-lim, lim], xticklabels=[], yticklabels=[])
    g.set_axis_labels('', '')
    g.set_titles(row_template='Perturbation level {row_name}')
    g.despine(left=False, bottom=False, top=False, right=False)
    g.axes[0, 0].set_title('Before training')
    g.axes[0, 1].set_title('After training')
    g.axes[0, 0].annotate(
        'Position', xy=(-0.75, -0.75), xycoords='data', xytext=(75, 0),
        textcoords='offset points', verticalalignment='center',
        arrowprops=arrowprops)
    g.axes[0, 0].annotate(
        'Velocity', xy=(-0.75, -0.75), xycoords='data', xytext=(0, 75),
        textcoords='offset points', horizontalalignment='center',
        arrowprops=arrowprops)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    sns.move_legend(g, 'upper center', bbox_to_anchor=(0.2, 0.97), ncol=2,
                    title=None)

    # Draw target state.
    xt = [0, 0]
    for ax in g.axes.ravel():
        ax.scatter(xt[0], xt[1], s=64, marker='x', c='k')

    path_fig = os.path.join(path, 'mlruns', experiment_id,
                            f'trajectories_perturbed.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()

    df3 = runs.melt(
        var_name='gramian_type', value_name='gramian_value',
        value_vars=['metrics.controllability', 'metrics.observability'],
        id_vars=['metrics.test_loss', 'metrics.training_loss',
                 'params.perturbation_type', 'params.perturbation_level'])
    df3 = df3.loc[df3['params.perturbation_type'] != '']
    data = {'metrics.test_loss': [], 'metrics.training_loss': [],
            'params.perturbation_type': [], 'params.perturbation_level': [],
            'gramian_type': [], 'gramian_value': []}
    for gramian_type in ['metrics.controllability', 'metrics.observability']:
        for perturbation_type in perturbations.keys():
            for perturbation_level in perturbations[perturbation_type]:
                runs_uncontrolled = df2.loc[
                    (df2['perturbation_type'] == perturbation_type) &
                    (df2['perturbation_level'] == perturbation_level) &
                    (df2['epoch'] == 0) & (df2['dropout_probability'] == 0)]
                n = len(runs_uncontrolled) // 2
                data['metrics.test_loss'] += list(runs_uncontrolled[runs_uncontrolled.phase == 'test']['loss'])
                data['metrics.training_loss'] += list(runs_uncontrolled[runs_uncontrolled.phase == 'training']['loss'])
                data['params.perturbation_type'] += [perturbation_type] * n
                data['params.perturbation_level'] += [str(perturbation_level)] * n
                data['gramian_type'] += [gramian_type] * n
                data['gramian_value'] += [0] * n
    df3 = pd.concat([df3, pd.DataFrame(data)], ignore_index=True)
    g = sns.relplot(data=df3, x='gramian_value', y='metrics.test_loss',
                    row='gramian_type', col='params.perturbation_type',
                    hue='params.perturbation_level', palette='copper',
                    kind='line', marker='o', markersize=10, legend=False,
                    facet_kws={'sharex': False, 'sharey': True})

    g.set(yscale='log', xlim=[-0.05, 1.05])
    g.set_axis_labels('', 'Loss')
    g.set_titles('{col_name}', '')
    g.axes[0, 0].set(yticklabels=[])
    g.axes[1, 0].set(yticklabels=[])
    g.despine(left=True)
    g.refline(y=test_loss_unperturbed, color='k', linestyle=':')
    for ax in g.axes[0]:
        ax.set_xlabel('Controllability')
    for ax in g.axes[1]:
        ax.set_xlabel('Observability')
        ax.set_title('')
    fig = plt.gcf()
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    fig.colorbar(ScalarMappable(cmap='copper'), cax=cbar_ax,
                 label='Perturbation')

    path_fig = os.path.join(path, 'mlruns', experiment_id,
                            'loss_vs_dropout.png')
    plt.savefig(path_fig, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
    sys.exit()
