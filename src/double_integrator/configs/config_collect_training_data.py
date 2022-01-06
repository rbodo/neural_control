import os

import numpy as np

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'lqg_grid.pkl')

    config.simulation.GRID_SIZE = 100

    config.process.PROCESS_NOISES = np.logspace(-2, -1, 5,
                                                dtype='float32').tolist()
    config.process.OBSERVATION_NOISES = np.logspace(-1, 0, 5,
                                                    dtype='float32').tolist()

    return config
