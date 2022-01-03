import os
import time

from src.double_integrator import configs



def get_config():
    config = configs.config_train_rnn_small.get_config()

    RANDOMIZE_WORKDIR = False

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn/two_neurons'
    if RANDOMIZE_WORKDIR:
        base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')

    config.training.VALIDATION_FRACTION = 0.01

    return config
