import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_train_rnn_small.get_config(timestamp_workdir)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn/two_neurons'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')

    config.training.VALIDATION_FRACTION = 0.01

    return config
