import os

import numpy as np

from src.double_integrator import configs


def get_config(timestamp_workdir=None):

    config = configs.config_collect_training_data.get_config(timestamp_workdir)

    base_path = os.path.dirname(config.paths.FILEPATH_OUTPUT_DATA)

    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'lqg_grid_ood.pkl')

    config.process.PROCESS_NOISES = np.logspace(-1.9, -0.9, 5,
                                                dtype='float32').tolist()
    config.process.OBSERVATION_NOISES = np.logspace(-0.9, 0.1, 5,
                                                    dtype='float32').tolist()

    return config
