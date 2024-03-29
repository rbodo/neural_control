import os

from examples import configs
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'lqg.pkl')

    config.simulation.T = 10
    config.simulation.NUM_STEPS = 100
    config.simulation.GRID_SIZE = 1

    config.process.PROCESS_NOISES = [1e-2]
    config.process.OBSERVATION_NOISES = [1e-2]
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]

    return config
