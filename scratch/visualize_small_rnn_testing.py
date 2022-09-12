import sys

import os
import pandas as pd

from scratch import configs
from src.plotting import plot_rnn_states_vs_lqe_estimates
from src.utils import apply_config


def main(config):

    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES

    df = pd.read_pickle(path_data)

    path = os.path.join(path_figures, 'rnn_states_vs_lqe_estimates.png')
    plot_rnn_states_vs_lqe_estimates(df, path)


if __name__ == '__main__':

    _config = configs.config_test_rnn_small.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
