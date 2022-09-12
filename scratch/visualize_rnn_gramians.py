import os
import sys

import pandas as pd

from scratch import configs
from src.plotting import plot_rnn_gramians
from src.utils import apply_config


def main(config):
    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES
    df = pd.read_pickle(path_data)

    # n = -1  # Show all eigenvalues
    n = 10  # Show first few eigenvalues
    remove_outliers_below = 1e-8
    path = os.path.join(path_figures, f'rnn_gramians_first{n}.png')
    plot_rnn_gramians(df, path, n, remove_outliers_below)


if __name__ == '__main__':
    _config = configs.config_train_rnn_gramian_high_noise.get_config()
    # _config = configs.config_train_rnn_gramian_low_noise.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
