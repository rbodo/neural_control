import os

import numpy as np

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = 7
    config.EXPERIMENT_NAME = 'linear_rnn_lqg'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path
    config.paths.FILEPATH_INPUT_DATA = \
        os.path.abspath(os.path.join(base_path, '..', 'lqg_grid.pkl'))

    # Environment
    config.process.NUM_INPUTS = 1
    config.process.NUM_STATES = 2
    config.process.NUM_OUTPUTS = 1
    config.process.PROCESS_NOISES = [0.01]
    config.process.OBSERVATION_NOISES = \
        np.logspace(-1, 0, 5, dtype='float32').tolist()[:1]

    config.training.NUM_EPOCHS_NEURALSYSTEM = 10
    config.training.NUM_EPOCHS_CONTROLLER = 4
    config.training.BATCH_SIZE = 32
    config.training.OPTIMIZER = 'adam'

    # Neural system and controller
    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1
    config.model.CLOSE_ENVIRONMENT_LOOP = False  # Env is not part of graph

    # Perturbation of neural system
    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.PERTURBATIONS = [
        ('sensor', [0.1, 0.5, 1, 2, 3]),
        ('actuator', [0.1, 0.5, 1, 2, 3]),
        ('processor', [0.1, 0.5, 1, 2, 3])]
    config.perturbation.DROPOUT_PROBABILITIES = [0]#, 0.1, 0.5, 0.7, 0.9]

    config.SEEDS = [42]#, 234, 55, 2, 5632]

    return config
