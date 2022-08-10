import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn_ppo/' \
                'rnn/maximize_rewards'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.abspath(os.path.join(
        base_path, '..', 'models', 'lqg_rnn_ppo'))
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')

    config.simulation.GRID_SIZE = 10

    return config
