import os

from examples import configs


def get_config():
    config = configs.config.get_config()

    config.GPU = 'cuda'
    config.EXPERIMENT_NAME = 'linear_rnn_rl'
    config.RESUME_EXPERIMENT = '2022-11-12'

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

    config.training.NUM_EPOCHS_NEURALSYSTEM = 5e5
    config.training.NUM_EPOCHS_CONTROLLER = 1e5
    config.training.BATCH_SIZE = None
    config.training.LEARNING_RATE = 2e-4

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 64
    config.model.NUM_LAYERS_CONTROLLER = 1

    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.ELECTRODE_SELECTIONS = ['random']
    config.perturbation.PERTURBATIONS = [
        ('sensor', [0.5, 1, 2, 3, 4]),
        ('processor', [0.1, 0.2, 0.3, 0.4, 0.5]),
        ('actuator', [0.1, 0.2, 0.3, 0.4, 0.5])
    ]
    config.perturbation.DROPOUT_PROBABILITIES = [0, 0.1, 0.5, 0.7, 0.9]

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
