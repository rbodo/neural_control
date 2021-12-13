import os

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_collect_training_data import (
    PROCESS_NOISES, OBSERVATION_NOISES, FILEPATH_OUTPUT_DATA)

base_path = \
    '/home/bodrue/Data/neural_control/double_integrator/hyperparameter_rnn'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

cfg.process.PROCESS_NOISES = PROCESS_NOISES[-2:-1]
cfg.process.OBSERVATION_NOISES = OBSERVATION_NOISES[-2:-1]
