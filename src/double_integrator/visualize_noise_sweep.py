import sys

import os
import pandas as pd

from src.double_integrator import configs
from src.double_integrator.plotting import (
    plot_trajectories_vs_noise, plot_cost_vs_noise, plot_cost_heatmap)
from src.double_integrator.utils import split_train_test, apply_config


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

    _config = configs.config_test_rnn_generalization.get_config()
    # _config = configs.config_collect_training_data.get_config()
    # _config = configs.config_test_rnn.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
