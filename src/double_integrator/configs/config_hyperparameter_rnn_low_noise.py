import os

from src.double_integrator import configs



def get_config():
    config = configs.config_rnn_defaults.get_config()
    config2 = configs.config_collect_training_data.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/' \
                'hyperparameter_rnn/low_noise'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config2.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
    config.paths.STUDY_NAME = 'rnn_low_noise'
    config.paths.FILEPATH_OUTPUT_DATA = \
        os.path.join(base_path, config.paths.STUDY_NAME + '.db')

    return config
