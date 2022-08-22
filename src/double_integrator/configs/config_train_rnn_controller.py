import os

import numpy as np
from yacs.config import CfgNode

from src.double_integrator import configs


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn_controller'
    config.paths.BASE_PATH = base_path
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'rnn.params')
    config.paths.FILEPATH_INPUT_DATA = \
        os.path.abspath(os.path.join(base_path, '..', 'lqg_grid.pkl'))

    config.process.PROCESS_NOISES = config.process.PROCESS_NOISES[:1]
    config.process.OBSERVATION_NOISES = config.process.OBSERVATION_NOISES[:1]

    config.training.NUM_EPOCHS = 10
    config.training.BATCH_SIZE = 32

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1
    config.model.REGULARIZATION_LEVELS = \
        np.logspace(-6, -4, 3, dtype='float32').tolist()
    config.model.SPARSITY_THRESHOLD = 1e-4

    config.perturbation = CfgNode()
    config.perturbation.PERTURBATION_TYPES = \
        ['sensor', 'actuator', 'processor']
    config.perturbation.PERTURBATION_LEVELS = \
        np.logspace(-2, -1, 3, dtype='float32').tolist()
    config.perturbation.DELTA = 1e-5

    return config
