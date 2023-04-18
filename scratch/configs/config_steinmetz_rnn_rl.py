import os

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = '6'
    config.EXPERIMENT_NAME = 'steinmetz_rnn_rl'
    config.RESUME_EXPERIMENT = 'bidirectional_larger'  # 2023-03-16'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path

    # Environment
    config.process.CONTRAST_LEVELS = [0, 0.25, 0.5, 1]
    config.process.TIME_STIMULUS = 0.2  # 0.5 s in paper
    config.process.TIMEOUT_WAIT = 0.5  # 1.5 s in paper
    config.process.GOCUE_WAIT = 0.2  # 0.7 s in paper
    config.process.DT = 0.01  # 0.01 s in paper

    config.training.NUM_EPOCHS_NEURALSYSTEM = 5e4
    config.training.NUM_EPOCHS_CONTROLLER = 5e4
    config.training.BATCH_SIZE = 0
    config.training.LEARNING_RATE = 1e-4
    config.training.EVALUATE_EVERY_N = 500
    config.training.NUM_TEST = 100

    config.model.USE_BIDIRECTIONAL_CONTROLLER = True
    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 18
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 32
    config.model.NUM_LAYERS_CONTROLLER = 2

    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.DROPOUT_PROBABILITIES = [0]
    config.perturbation.PERTURBATIONS = [('linear', [1e-1, 1e-3, 1e-5]),
                                         ('random', [1e-5, 1e-3, 1e-1]),
                                         ('noise', [1e-5, 1e-3, 1e-1])]

    config.COMPUTE_GRAMIANS = False

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
