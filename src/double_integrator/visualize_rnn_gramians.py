import os
import sys

import pandas as pd

from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import plot_rnn_gramians


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
    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    # filename = 'config_train_rnn_gramian_low_noise.py'
    filename = 'config_train_rnn_gramian_high_noise.py'
    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()
