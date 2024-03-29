import os.path

from scratch import configs
from scratch.train_rnn import get_model_name
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)
    config2 = configs.config_train_rnn.get_config(timestamp_workdir)
    config3 = \
        configs.config_collect_training_data.get_config(timestamp_workdir)

    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    path, model_name = os.path.split(config2.paths.FILEPATH_MODEL)
    model_name = \
        get_model_name(model_name, process_noises[2], observation_noises[2])

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'generalization/' + os.path.splitext(model_name)[0]
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config3.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
    config.paths.FILEPATH_MODEL = os.path.join(path, model_name)

    config.model.USE_SINGLE_MODEL_IN_SWEEP = True

    config.process.PROCESS_NOISES = process_noises
    config.process.OBSERVATION_NOISES = observation_noises

    return config
