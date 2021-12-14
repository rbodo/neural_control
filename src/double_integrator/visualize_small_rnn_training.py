import sys

import os
import pandas as pd

from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import plot_loss_heatmap


def main(config):

    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES

    df = pd.read_pickle(path_data)

    path = os.path.join(path_figures, 'cost_heatmap.png')
    plot_loss_heatmap(df, path, 1)


if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    filename = 'config_train_rnn_small.py'
    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()
