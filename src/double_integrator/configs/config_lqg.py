import os

from src.double_integrator import configs



def get_config():
    config = configs.config.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
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
