import os
import time

from src.double_integrator import configs


def get_config():
    config = configs.config_rnn_defaults.get_config()

    RANDOMIZE_WORKDIR = False

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn'
    if RANDOMIZE_WORKDIR:
        base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

    return config
