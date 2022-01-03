from src.double_integrator import configs



def get_config():
    config = configs.config.get_config()

    config.paths.PATH_FIGURES = \
        '/home/bodrue/Data/neural_control/double_integrator/open/figures/'

    config.simulation.T = 10
    config.simulation.NUM_STEPS = 100

    config.process.PROCESS_NOISES = [1e-2]
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]

    return config
