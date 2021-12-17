import os

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_train_rnn_all_noises import \
    FILEPATH_MODEL
from src.double_integrator.configs.config_collect_ood_data import (
    PROCESS_NOISES, OBSERVATION_NOISES, FILEPATH_OUTPUT_DATA)

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/all_noises'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output_ood.pkl')
cfg.paths.FILEPATH_MODEL = FILEPATH_MODEL

cfg.process.PROCESS_NOISES = PROCESS_NOISES
cfg.process.OBSERVATION_NOISES = OBSERVATION_NOISES

cfg.model.USE_SINGLE_MODEL_IN_SWEEP = True
