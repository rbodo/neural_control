import os

from scratch import configs
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'reinforce'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.pth')

    config.training.BATCH_SIZE = 1

    config.simulation.GRID_SIZE = 100

    return config
