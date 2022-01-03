import os

import numpy as np

from src.double_integrator import configs


def get_config():
    config = configs.config.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'lqg_grid_ood.pkl')

    config.simulation.GRID_SIZE = 100

    config.process.PROCESS_NOISES = np.logspace(-1.9, -0.9, 5,
                                                dtype='float32').tolist()
    config.process.OBSERVATION_NOISES = np.logspace(-0.9, 0.1, 5,
                                                    dtype='float32').tolist()

    return config
