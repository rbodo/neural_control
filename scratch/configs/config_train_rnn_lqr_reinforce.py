import os

from scratch import configs
from src.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)
    config2 = \
        configs.config_collect_training_data.get_config(timestamp_workdir)

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'reinforce/lqr'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_BASE = base_path
    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')
    config.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.pth')

    config.simulation.T = 100
    config.simulation.NUM_STEPS = 1000
    config.simulation.GRID_SIZE = 100

    config.model.NUM_HIDDEN = 128

    config.training.BATCH_SIZE = 1
    config.training.OPTIMIZER = 'adam'
    config.training.LEARNING_RATE = 1e-3

    config.process.PROCESS_NOISES = config2.process.PROCESS_NOISES
    config.process.OBSERVATION_NOISES = [0]
    config.process.STATE_MEAN = [1, 0]
    config.process.STATE_COVARIANCE = 1e-1

    config.controller.STATE_TARGET = [0, 0]
    config.controller.cost.lqr.Q = 0.5
    config.controller.cost.lqr.R = 0.5

    return config
