import os

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_collect_training_data import \
    FILEPATH_OUTPUT_DATA

base_path = '/home/bodrue/Data/neural_control/double_integrator/' \
            'hyperparameter_rnn/low_noise'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
cfg.paths.STUDY_NAME = 'rnn_low_noise'
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, cfg.paths.STUDY_NAME +
                                              '.db')
