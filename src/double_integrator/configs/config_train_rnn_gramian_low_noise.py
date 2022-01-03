import os
import time

from src.double_integrator import configs



def get_config():
    config = configs.config_rnn_defaults.get_config()
    config2 = configs.config_collect_training_data.get_config()

    RANDOMIZE_WORKDIR = False

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'gramian/first_epoch/low_noise'
    if RANDOMIZE_WORKDIR:
        base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path,
                                               'models/rnn_gramian.params')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'rnn_gramians.pkl')

    config.training.NUM_EPOCHS = 1

    config.process.PROCESS_NOISES = config2.process.PROCESS_NOISES[0:1]
    config.process.OBSERVATION_NOISES = config2.process.OBSERVATION_NOISES[0:1]

    return config
