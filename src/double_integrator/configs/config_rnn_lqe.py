import os

from src.double_integrator import configs



def get_config():
    config = configs.config_rnn_defaults.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn_lqe'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path,
                                               'models/rnn_lqe.params')

    config.process.PROCESS_NOISES = [1e-2]
    config.process.OBSERVATION_NOISES = [1e-2]

    return config
