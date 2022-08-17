import os

from src.double_integrator import configs


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn_controller'
    config.paths.BASE_PATH = base_path
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models',
                                               'rnn.params')
    config.paths.FILEPATH_INPUT_DATA = \
        os.path.abspath(os.path.join(base_path, '..', 'lqg_grid.pkl'))

    config.process.PROCESS_NOISES = config.process.PROCESS_NOISES[:1]
    config.process.OBSERVATION_NOISES = config.process.OBSERVATION_NOISES[:2]

    return config
