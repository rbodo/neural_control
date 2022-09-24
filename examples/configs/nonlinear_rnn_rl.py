import os

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = 8
    config.EXPERIMENT_NAME = 'nonlinear_rnn_rl'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path

    config.simulation.NUM_STEPS = 1000
    config.simulation.T = config.simulation.NUM_STEPS / 10

    # Environment
    config.process.OBSERVATION_INDICES = [0, 1]  # Position, angle
    config.process.PROCESS_NOISES = [0]
    config.process.OBSERVATION_NOISES = [0]

    config.training.NUM_EPOCHS_NEURALSYSTEM = 8e5
    config.training.NUM_EPOCHS_CONTROLLER = 1e5
    config.training.BATCH_SIZE = None
    config.training.LEARNING_RATE = 2e-4

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 256
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1

    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.PERTURBATIONS = [
        ('sensor', [0.1, 0.5, 1, 2, 3]),
        ('actuator', [0.1, 0.5, 1, 2, 3]),
        ('processor', [0.1, 0.5, 1, 2, 3])]
    config.perturbation.DROPOUT_PROBABILITIES = [0, 0.1, 0.5, 0.7, 0.9]

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
