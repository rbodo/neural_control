import os

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_train_rnn import FILEPATH_MODEL

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
cfg.paths.FILEPATH_MODEL = FILEPATH_MODEL
