import os

from yacs.config import CfgNode

from scratch import configs


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn_controller'
    config.paths.BASE_PATH = base_path
    config.paths.FILEPATH_INPUT_DATA = \
        os.path.abspath(os.path.join(base_path, '..', 'lqr_grid.pkl'))

    config.process.PROCESS_NOISES = config.process.PROCESS_NOISES[:1]
    config.process.OBSERVATION_NOISES = [0]  # Fully observable, noiseless

    config.training.NUM_EPOCHS = 10
    config.training.BATCH_SIZE = 32

    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 50
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 40
    config.model.NUM_LAYERS_CONTROLLER = 1

    config.perturbation = CfgNode()
    config.perturbation.PERTURBATION_TYPES = \
        ['sensor', 'actuator', 'processor']
    config.perturbation.PERTURBATION_LEVELS = [0.5, 1, 2]  # Todo: Adapt per perturbation type
    config.perturbation.DROPOUT_PROBABILITIES = [0]#, 0.1, 0.5, 0.7, 0.9]

    config.SEEDS = [42]#, 234]

    return config
