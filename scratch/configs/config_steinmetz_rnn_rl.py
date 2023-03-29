import os

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = '8'
    config.EXPERIMENT_NAME = 'steinmetz_rnn_rl'
    config.RESUME_EXPERIMENT = 'test35'  # 2023-03-16'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path

    config.simulation.NUM_STEPS = 50
    config.simulation.T = config.simulation.NUM_STEPS / 10

    # Environment
    config.process.NUM_CONTRAST_LEVELS = 5
    config.process.TIME_STIMULUS = 20

    config.training.NUM_EPOCHS_NEURALSYSTEM = 1e4
    config.training.NUM_EPOCHS_CONTROLLER = 1e5
    config.training.BATCH_SIZE = 0
    config.training.LEARNING_RATE = 1e-4
    config.training.EVALUATE_EVERY_N = 500
    config.training.NUM_TEST = 100

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 18
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 30
    config.model.NUM_LAYERS_CONTROLLER = 1

    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.DROPOUT_PROBABILITIES = [0]  # , 0.1, 0.5, 0.7, 0.9]
    config.perturbation.PERTURBATIONS = [('sensor', [1])]

    config.COMPUTE_GRAMIANS = False

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
