import sys

import os
import pandas as pd

from src.double_integrator import configs
from src.double_integrator.plotting import plot_loss_heatmap
from src.double_integrator.utils import apply_config


def main(config):

    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES

    df = pd.read_pickle(path_data)

    path = os.path.join(path_figures, 'cost_heatmap.png')
    plot_loss_heatmap(df, path, 1)


if __name__ == '__main__':

    _config = configs.config_train_rnn_small.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
