import os

from examples import configs


def get_config():
    config = configs.config.get_config()

    config.GPU = 4
    config.EXPERIMENT_NAME = 'linear_rnn_lqr'
    config.RESUME_EXPERIMENT = '2022-11-15'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path
    config.paths.FILEPATH_INPUT_DATA = \
        os.path.abspath(os.path.join(base_path, '..', 'lqr_grid.pkl'))

    # Environment
    config.process.NUM_INPUTS = 1
    config.process.NUM_STATES = 2
    config.process.NUM_OUTPUTS = 2  # Fully observable
    config.process.PROCESS_NOISES = [0.01]
    config.process.OBSERVATION_NOISES = [0]  # Noiseless observations

    config.training.NUM_EPOCHS_NEURALSYSTEM = 10
    config.training.NUM_EPOCHS_CONTROLLER = 4
    config.training.BATCH_SIZE = 32
    config.training.OPTIMIZER = 'adam'

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1
    config.model.CLOSE_ENVIRONMENT_LOOP = True  # Make env part of graph

    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.ELECTRODE_SELECTIONS = ['random', 'gramian']
    config.perturbation.PERTURBATIONS = [
        ('sensor', [1, 2, 4, 8, 16]),
        ('actuator', [0.1, 0.5, 1, 2, 3]),
        ('processor', [0.1, 0.2, 0.3, 0.4, 0.5])]
    config.perturbation.DROPOUT_PROBABILITIES = [0, 0.1, 0.5, 0.7, 0.9, 1]

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
