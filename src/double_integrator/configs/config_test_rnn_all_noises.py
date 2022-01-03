import os

from src.double_integrator import configs


def get_config():
    config = configs.config_rnn_defaults.get_config()
    config2 = configs.config_train_rnn_all_noises.get_config()
    config3 = configs.config_collect_training_data.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'all_noises'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config3.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path,
                                                     'output.pkl')
    config.paths.FILEPATH_MODEL = config2.paths.FILEPATH_MODEL

    config.process.PROCESS_NOISES = config3.process.PROCESS_NOISES
    config.process.OBSERVATION_NOISES = config3.process.OBSERVATION_NOISES

    config.model.USE_SINGLE_MODEL_IN_SWEEP = True

    return config
