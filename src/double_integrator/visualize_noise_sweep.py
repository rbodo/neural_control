import sys

import os
import pandas as pd

from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import (
    plot_trajectories_vs_noise, plot_cost_vs_noise, plot_cost_heatmap)
from src.double_integrator.utils import split_train_test


def main(config):

    path_data = config.paths.PATH_TRAINING_DATA
    path_out = config.paths.PATH_OUT

    df = pd.read_pickle(path_data)

    _, df = split_train_test(df)

    path = os.path.join(path_out, 'cost_heatmap.png')
    plot_cost_heatmap(df, path, 50)

    path = os.path.join(path_out, 'trajectories_vs_noise.png')
    plot_trajectories_vs_noise(df, path)

    path = os.path.join(path_out, 'cost_vs_noise.png')
    plot_cost_vs_noise(df, path)


if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    # filename = 'config_collect_training_data.py'
    filename = 'config_rnn.py'
    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()
