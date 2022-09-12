import os

from scratch import configs
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    timestamp_dataset = '20211231_000000'
    timestamp_model = '20211231_000001'
    config = configs.config_rnn_defaults.get_config(timestamp_dataset)
    config2 = configs.config_train_rnn_all_noises.get_config(timestamp_model)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'all_noises'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')
    config.paths.FILEPATH_MODEL = config2.paths.FILEPATH_MODEL

    config.model.USE_SINGLE_MODEL_IN_SWEEP = True

    return config
