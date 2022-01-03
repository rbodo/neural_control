import sys

import os
from collections import OrderedDict

import pandas as pd

from src.double_integrator import configs
from src.double_integrator.plotting import (plot_cost_vs_noise,
                                            plot_cost_scatter)
from src.double_integrator.utils import split_train_test, apply_config


def main(config_dict: OrderedDict):
    label2, config2 = config_dict.popitem()
    label1, config1 = config_dict.popitem()

    validation_fraction = config2.training.VALIDATION_FRACTION

    # Use output data file generated during testing.
    path_data1 = config1.paths.FILEPATH_OUTPUT_DATA
    df1 = pd.read_pickle(path_data1)
    df1['controller'] = label1

    path_data2 = config2.paths.FILEPATH_OUTPUT_DATA
    df2 = pd.read_pickle(path_data2)
    _, df2 = split_train_test(df2, validation_fraction)
    df2['controller'] = label2

    df = pd.concat([df1, df2], join='inner')

    path_figures = config1.paths.PATH_FIGURES
    path = os.path.join(path_figures,
                        f'cost_vs_noise_combined_{label1}_{label2}.png')
    plot_cost_vs_noise(df, path)

    df1 = df1.groupby(
        ['process_noise', 'observation_noise', 'times'])['c'].mean()
    df1 = df1.reset_index().rename(columns={'c': f'c_{label1}'}).drop(
        columns='times')

    df2 = df2.groupby(
        ['process_noise', 'observation_noise', 'times'])['c'].mean()
    df2 = df2.reset_index().rename(columns={'c': f'c_{label2}'}).drop(
        columns='times')

    df3 = pd.merge(df1.reset_index(), df2.reset_index(),
                   how='left').drop(columns='index')

    path = os.path.join(path_figures, f'cost_scatter_{label1}_{label2}.png')
    plot_cost_scatter(df3, path)


if __name__ == '__main__':

    # _config1 = configs.config_test_rnn.get_config()
    # _config1 = configs.config_test_rnn_generalization.get_config()
    # _config1 = configs.config_test_rnn_small.get_config()
    _config1 = configs.config_test_rnn_all_noises.get_config()
    _config2 = configs.config_test_rnn_ood.get_config()
    # _config2 = configs.config_collect_training_data.get_config()

    apply_config(_config1)
    apply_config(_config2)

    # main(OrderedDict([('rnn', _config1), ('lqg', _config2)]))
    main(OrderedDict([('rnn_id', _config1), ('rnn_ood', _config2)]))

    sys.exit()
