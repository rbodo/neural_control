import os

from src.double_integrator import configs



def get_config():
    config = configs.config_rnn_defaults.get_config()
    config2 = configs.config_train_rnn.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
    config.paths.FILEPATH_MODEL = config2.paths.FILEPATH_MODEL

    return config
