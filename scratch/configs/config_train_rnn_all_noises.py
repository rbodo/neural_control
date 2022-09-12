import os

from scratch import configs
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    timestamp_dataset = '20211231_000000'
    config = configs.config_rnn_defaults.get_config(timestamp_dataset)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn/all_noises'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

    # Reduce relative size of training set because we are sampling training
    # data from across all noise levels.
    f = config.training.VALIDATION_FRACTION
    c = (len(config.process.PROCESS_NOISES) *
         len(config.process.OBSERVATION_NOISES))
    config.training.VALIDATION_FRACTION = 1 + (f - 1) / c

    return config
