import sys

import os
import pandas as pd

from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import (
    plot_trajectories_vs_noise, plot_cost_vs_noise, plot_cost_heatmap)
from src.double_integrator.utils import split_train_test


def main(config):

    # Use output data file generated during testing.
    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES
    validation_fraction = config.training.VALIDATION_FRACTION

    df = pd.read_pickle(path_data)

    _, df = split_train_test(df, validation_fraction)

    path = os.path.join(path_figures, 'cost_heatmap.png')
    plot_cost_heatmap(df, path, 1)

    path = os.path.join(path_figures, 'trajectories_vs_noise.png')
    plot_trajectories_vs_noise(df, path)

    path = os.path.join(path_figures, 'cost_vs_noise.png')
    plot_cost_vs_noise(df, path)


if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    # filename = 'config_collect_training_data.py'
    # filename = 'config_test_rnn.py'
    filename = 'config_test_rnn_generalization.py'
    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()
