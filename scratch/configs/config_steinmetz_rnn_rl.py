import os

from examples import configs
from yacs.config import CfgNode


def get_config():
    config = configs.config.get_config()

    config.GPU = '5'
    config.EXPERIMENT_NAME = 'steinmetz_rnn_rl'
    config.RESUME_EXPERIMENT = 'visual_bidirectional14'  # 2023-03-16'

    base_path = os.path.join(os.path.expanduser(
        '~/Data/neural_control'), config.EXPERIMENT_NAME)
    config.paths.BASE_PATH = base_path
    config.paths.MODEL_NAME = 'steinmetz_weights_visual'

    # Data
    config.data = CfgNode()
    # config.data.AREAS = ['VISa', 'VISp', 'LD', 'LP', 'PO', 'CA1', 'CA3',
    #                      'DG', 'SUB', 'ILA', 'MOs', 'PL', 'TT', 'LSr']
    # config.data.SUBJECT_INDEX = 7
    # config.data.AREAS = ['MOp']
    # config.data.SUBJECT_INDEX = 10
    config.data.AREAS = ['VISl', 'VISp', 'VISrl']
    config.data.SUBJECT_INDEX = 9

    # Environment
    config.process.CONTRAST_LEVELS = [0, 0.25, 0.5, 1]
    config.process.TIME_STIMULUS = 0.5  # 0.5 s in paper
    config.process.TIMEOUT_WAIT = 1.5  # 1.5 s in paper
    config.process.GOCUE_WAIT = 0.7  # 0.7 s in paper
    config.process.DT = 0.01  # 0.01 s in paper

    config.training.NUM_EPOCHS_NEURALSYSTEM = 1e5
    config.training.NUM_EPOCHS_CONTROLLER = 5e4
    config.training.BATCH_SIZE = 0
    config.training.LEARNING_RATE = 2e-4
    config.training.EVALUATE_EVERY_N = 500
    config.training.NUM_TEST = 100
    config.training.AGENT_PRETRAINING = True

    config.model.USE_BIDIRECTIONAL_CONTROLLER = True
    config.model.ACTIVATION = 'tanh'
    config.model.NUM_HIDDEN_NEURALSYSTEM = 32
    config.model.NUM_LAYERS_NEURALSYSTEM = 1
    config.model.NUM_HIDDEN_CONTROLLER = 30
    config.model.NUM_LAYERS_CONTROLLER = 2

    config.perturbation = CfgNode()
    config.perturbation.SKIP_PERTURBATION = False
    config.perturbation.DROPOUT_PROBABILITIES = [0]
    config.perturbation.PERTURBATIONS = [('random', [1]),
                                         ('linear', [1e-6, 1e-3, 9e-1]),
                                         ('noise', [1e-6, 1e-3, 1e-1])]

    config.COMPUTE_GRAMIANS = False

    config.SEEDS = [43, 234, 55, 2, 5632]

    return config
