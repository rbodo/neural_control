import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from src.double_integrator.control_systems import DiLqg
from src.double_integrator.utils import get_additive_white_gaussian_noise

sns.set_theme(style='whitegrid')


def plot_timeseries(df, path=None):

    n = df['variable'].nunique()
    # Create line plot with one subplot for each variable and one line for each
    # dimension.
    g = sns.relplot(data=df, x='times', y='value', hue='dimension',
                    row='variable', kind='line', aspect=2, height=10/n,
                    facet_kws={'sharey': False})
    if 'Cost' in g.axes_dict:
        g.axes_dict['Cost'].set(yscale='log')

    g.set_axis_labels('Time', '')

    # Add borders.
    sns.despine(right=False, top=False)

    # noinspection PyProtectedMember
    g._legend.set_title(None)

    for ax in g.axes.ravel():
        title = ax.get_title()
        title = title.split(' = ')[1]
        ax.set_title(title)

    if path is not None:
        g.savefig(path, bbox_inches='tight')
    plt.show()


def plot_phase_diagram(state_dict, num_states=None, odefunc=None, W=None,
                       start_points=None, n=10, xt=None, rng=None, path=None):
    assert len(state_dict) == 2, "Two dimensions required for phase plot."
    labels = list(state_dict.keys())
    states = list(state_dict.values())
    i, j = (0, 1)  # np.argsort(labels))

    plt.figure()

    # Draw trajectory.
    plt.plot(states[i], states[j])
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])

    # Draw target state.
    if xt is not None:
        plt.scatter(xt[0], xt[1], s=32, marker='o')

    # Draw vector field.
    if odefunc is not None:

        # Get grid coordinates.
        ax = plt.gca()
        x1_min, x1_max = ax.get_xlim()
        x0_min, x0_max = ax.get_ylim()
        grid = np.mgrid[x0_min:x0_max:complex(0, n),
                        x1_min:x1_max:complex(0, n)]
        grid = grid[::-1]
        shape2d = grid.shape[1:]

        # Initialize the process state vectors at each location in grid.
        x = grid
        # If we have an LQE, add initial values for noisy state estimates.
        if W is not None:
            noise = get_additive_white_gaussian_noise(W, shape2d, rng)
            # Add noise to grid coordinates.
            x_hat = grid + np.moveaxis(noise, -1, 0)
            x = np.concatenate([x, x_hat])
        # If we have a stateful controller (e.g. RNN), add initial values.
        if num_states is not None:
            x = np.concatenate([x, np.zeros((num_states,) + shape2d)])
        # Compute derivatives at each grid node.
        dx = np.empty_like(x)
        for i, j in np.ndindex(shape2d):
            dx[:, i, j] = odefunc(0, x[:, i, j], [0])

        # Draw streamlines and arrows.
        plt.streamplot(grid[0], grid[1], dx[0], dx[1],
                       start_points=start_points, linewidth=0.3)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_trajectories_vs_noise(df, path=None):
    df = df[df['experiment'] == 0]
    col_wrap = int(np.sqrt(df['process_noise'].nunique()))
    g = sns.relplot(data=df, x='x', y='v', col='process_noise',
                    hue='observation_noise', col_wrap=col_wrap, kind='line',
                    palette=sns.color_palette('coolwarm', as_cmap=True),
                    hue_norm=LogNorm())
    g.set_titles("Process noise: {col_name:.2}")
    g.legend.set_title('Observation noise')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_cost_vs_noise(df, path=None):
    df = df[(df['experiment'] == 0) | (df['experiment'] == 1)]
    col_wrap = int(np.sqrt(df['process_noise'].nunique()))
    g = sns.relplot(data=df, x='times', y='c', col='process_noise',
                    hue='observation_noise', col_wrap=col_wrap, kind='line',
                    palette=sns.color_palette('coolwarm', as_cmap=True),
                    hue_norm=LogNorm())
    g.set(yscale='log')
    g.set_titles("Process noise: {col_name:.2}")
    g.legend.set_title('Observation noise')
    g.set_axis_labels('Time', 'Cost')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_cost_heatmap(df, path=None, tail=None):
    df = df[['experiment', 'process_noise', 'observation_noise', 'c']]
    if tail is not None:
        assert isinstance(tail, int)
        df = df.groupby(['experiment', 'process_noise',
                         'observation_noise']).tail(tail)
    df = df.groupby(['experiment', 'process_noise', 'observation_noise']).sum()
    df = df.reset_index().drop(columns='experiment')
    df = df.groupby(['process_noise', 'observation_noise']).agg(['mean',
                                                                 'std'])
    df = df.reset_index()
    mean = df.drop(columns='std', level=1).pivot(index='process_noise',
                                                 columns='observation_noise',
                                                 values=('c', 'mean'))
    std = df.drop(columns='mean', level=1).pivot(index='process_noise',
                                                 columns='observation_noise',
                                                 values=('c', 'std'))
    def float2str(x):
        return f'{x:g}'

    mean.rename(index=float2str, columns=float2str, inplace=True)
    std.rename(index=float2str, columns=float2str, inplace=True)

    f, ax = plt.subplots(2, 1, figsize=(8, 14), sharex='all', sharey='all')
    sns.heatmap(mean, annot=True, fmt='.2', linewidths=.5, square=True,
                robust=True, ax=ax[0])
    sns.heatmap(std, annot=True, fmt='.2', linewidths=.5, square=True,
                robust=True, ax=ax[1])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Process noise')
    ax[1].set_xlabel('Observation noise')
    ax[1].set_ylabel('Process noise')
    ax[0].set_title('Cost (mean)')
    ax[1].set_title('Cost (std)')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_kalman_gain_vs_noise_levels(process_noise, observation_noise,
                                     path=None):

    out = np.empty((len(process_noise), len(observation_noise)))
    for i, w in enumerate(process_noise):
        for j, v in enumerate(observation_noise):
            system_closed = DiLqg(w, v)
            out[i, j] = np.linalg.norm(system_closed.L, ord=np.inf)
            # out[i, j] = np.max(system_closed.L)
    df = pd.DataFrame(out,
                      [f'{p:.2g}' for p in process_noise],
                      [f'{p:.2g}' for p in observation_noise])
    g = sns.heatmap(df, annot=True, fmt='.2f', linewidths=.5, square=True,)
                    # norm=LogNorm())
    g.set_xlabel('Observation noise')
    g.set_ylabel('Process noise')
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
