import sys

import os
import pandas as pd

from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import (plot_cost_vs_noise,
                                            plot_cost_scatter)
from src.double_integrator.utils import split_train_test


def main(config1, config2=None):
    validation_fraction = config2.training.VALIDATION_FRACTION

    # Use output data file generated during testing.
    path_data1 = config1.paths.FILEPATH_OUTPUT_DATA
    df1 = pd.read_pickle(path_data1)
    df1['controller'] = 'rnn'

    path_data2 = config2.paths.FILEPATH_OUTPUT_DATA
    df2 = pd.read_pickle(path_data2)
    _, df2 = split_train_test(df2, validation_fraction)
    df2['controller'] = 'lqg'

    df = pd.concat([df1, df2], join='inner')

    path_figures = config1.paths.PATH_FIGURES
    path = os.path.join(path_figures, 'cost_vs_noise_combined.png')
    plot_cost_vs_noise(df, path)

    df1 = df1.groupby(
        ['process_noise', 'observation_noise', 'times'])['c'].mean()
    df1 = df1.reset_index().rename(columns={'c': 'c_rnn'}).drop(
        columns='times')

    df2 = df2.groupby(
        ['process_noise', 'observation_noise', 'times'])['c'].mean()
    df2 = df2.reset_index().rename(columns={'c': 'c_lqg'}).drop(
        columns='times')

    df3 = pd.merge(df1.reset_index(), df2.reset_index(),
                   how='left').drop(columns='index')

    path = os.path.join(path_figures, 'cost_scatter_rnn_lqg.png')
    plot_cost_scatter(df3, path)


if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    # filename1 = 'config_test_rnn.py'
    filename1 = 'config_test_rnn_small.py'
    # filename1 = 'config_test_rnn_generalization.py'
    filename2 = 'config_collect_training_data.py'
    _config1 = get_config(os.path.join(base_path, filename1))
    _config2 = get_config(os.path.join(base_path, filename2))

    main(_config1, _config2)

    sys.exit()
