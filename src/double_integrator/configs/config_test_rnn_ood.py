import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    timestamp_dataset = '20211231_000000'
    timestamp_model = '20211231_000001'
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)
    config2 = configs.config_train_rnn_all_noises.get_config(timestamp_model)
    config3 = configs.config_collect_ood_data.get_config(timestamp_dataset)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'all_noises'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config3.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'output_ood.pkl')
    config.paths.FILEPATH_MODEL = config2.paths.FILEPATH_MODEL

    config.process.PROCESS_NOISES = config3.process.PROCESS_NOISES
    config.process.OBSERVATION_NOISES = config3.process.OBSERVATION_NOISES

    config.model.USE_SINGLE_MODEL_IN_SWEEP = True

    return config
