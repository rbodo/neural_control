import os

from yacs.config import CfgNode

from src.double_integrator import configs


def get_config():
    config = configs.config.get_config()
    config2 = configs.config_collect_training_data.get_config()

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn'
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config2.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

    config.model = CfgNode()
    config.model.NUM_HIDDEN = 50
    config.model.NUM_LAYERS = 1
    config.model.ACTIVATION = 'relu'

    config.training.OPTIMIZER = 'adam'

    config.simulation.DO_WARMUP = False

    config.process.PROCESS_NOISES = config2.process.PROCESS_NOISES
    config.process.OBSERVATION_NOISES = config2.process.OBSERVATION_NOISES
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]

    return config
