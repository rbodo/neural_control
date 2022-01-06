import os

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/open'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')

    config.simulation.T = 10
    config.simulation.NUM_STEPS = 100

    config.process.PROCESS_NOISES = [1e-2]
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]

    return config
