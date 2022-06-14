import sys

import os
import pandas as pd

from src.double_integrator import configs
from src.double_integrator.plotting import (
    plot_cost_vs_noise_control, plot_trajectories_vs_noise_control)
from src.double_integrator.utils import apply_config


def main(config):

    # Use output data file generated during testing.
    path_data = config.paths.FILEPATH_OUTPUT_DATA
    path_figures = config.paths.PATH_FIGURES

    df = pd.read_pickle(path_data)

    path = os.path.join(path_figures, 'cost_vs_noise.png')
    plot_cost_vs_noise_control(df, path)

    path = os.path.join(path_figures, 'trajectories_vs_noise.png')
    plot_trajectories_vs_noise_control(df, path, [-1, 1], [-1, 1])

if __name__ == '__main__':

    _config = configs.config_di_rnn_pid.get_config('20220613_215649')

    apply_config(_config)

    main(_config)

    sys.exit()
