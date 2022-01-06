import os

from yacs.config import CfgNode

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config.get_config()
    config2 = \
        configs.config_collect_training_data.get_config(timestamp_workdir)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/mlp_lqe'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_INPUT_DATA = config2.paths.FILEPATH_OUTPUT_DATA
    config.paths.FILEPATH_MODEL = os.path.join(base_path,
                                               'models/mlp_lqe.params')

    config.model = CfgNode()
    config.model.NUM_HIDDEN = 10
    config.model.NUM_LAYERS = 1

    config.training.BATCH_SIZE = 32
    config.training.LEARNING_RATE = 1e-2
    config.training.NUM_EPOCHS = 10

    config.simulation.T = 10
    config.simulation.NUM_STEPS = 100

    config.process.PROCESS_NOISES = [1e-2]
    config.process.OBSERVATION_NOISES = [1e-2]
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]

    return config
