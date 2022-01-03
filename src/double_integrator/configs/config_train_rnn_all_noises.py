import os
import time

from src.double_integrator import configs


def get_config(randomize_workdir=False):
    config = configs.config_rnn_defaults.get_config()

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn/all_noises'
    if randomize_workdir:
        base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

    # Reduce relative size of training set because we are sampling training
    # data from across all noise levels.
    f = config.training.VALIDATION_FRACTION
    c = (len(config.process.PROCESS_NOISES) *
         len(config.process.OBSERVATION_NOISES))
    config.training.VALIDATION_FRACTION = 1 + (f - 1) / c

    return config
