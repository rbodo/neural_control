import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)
    config2 = \
        configs.config_collect_training_data.get_config(timestamp_workdir)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'gramian/first_epoch/high_noise'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path,
                                               'models/rnn_gramian.params')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'rnn_gramians.pkl')

    config.training.NUM_EPOCHS = 1

    config.process.PROCESS_NOISES = config2.process.PROCESS_NOISES[-2:-1]
    config.process.OBSERVATION_NOISES = \
        config2.process.OBSERVATION_NOISES[-2:-1]

    return config
