import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config.get_config()
    config2 = configs.config_collect_ood_data.get_config()

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/lqg/generalization'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'lqg_ood.pkl')

    config.simulation.GRID_SIZE = 100

    config.process.PROCESS_NOISES = config2.process.PROCESS_NOISES
    config.process.OBSERVATION_NOISES = config2.process.OBSERVATION_NOISES

    return config
