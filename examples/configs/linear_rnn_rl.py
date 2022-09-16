import os

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = 8
    config.EXPERIMENT_NAME = 'linear_rnn_rl'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path

    config.simulation.NUM_STEPS = 300
    config.simulation.T = config.simulation.NUM_STEPS / 10

    # Environment
    config.process.NUM_INPUTS = 1
    config.process.NUM_STATES = 2
    config.process.NUM_OUTPUTS = 1
    config.process.PROCESS_NOISES = [0.01]
    config.process.OBSERVATION_NOISES = [0.1]

    config.training.NUM_EPOCHS = 5e5
    config.training.BATCH_SIZE = None
    config.training.LEARNING_RATE = 2e-4

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1

    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.PERTURBATION_TYPES = \
        ['sensor', 'actuator', 'processor']
    config.perturbation.PERTURBATION_LEVELS = [0.5]  # , 1, 2]  # Todo: Adapt per perturbation type
    config.perturbation.DROPOUT_PROBABILITIES = [0.1]  # [0, 0.1, 0.5, 0.7, 0.9]

    config.SEEDS = [42]  # , 234]

    return config
