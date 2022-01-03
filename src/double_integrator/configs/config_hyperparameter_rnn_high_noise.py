import os

from src.double_integrator import configs



def get_config():
    config = configs.config_rnn_defaults.get_config()
    config2 = configs.config_collect_training_data.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/' \
                'hyperparameter_rnn/high_noise'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config2.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
    config.paths.STUDY_NAME = 'rnn_high_noise'
    config.paths.FILEPATH_OUTPUT_DATA = \
        os.path.join(base_path, config.paths.STUDY_NAME + '.db')

    config.process.PROCESS_NOISES = config2.paths.PROCESS_NOISES[-2:-1]
    config.process.OBSERVATION_NOISES = config2.paths.OBSERVATION_NOISES[-2:-1]

    return config
