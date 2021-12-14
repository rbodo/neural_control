import os
import time

from src.double_integrator.configs.config_rnn_defaults import cfg

RANDOMIZE_WORKDIR = False

base_path = \
    '/home/bodrue/Data/neural_control/double_integrator/rnn/two_neurons'
if RANDOMIZE_WORKDIR:
    base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures_training')
FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
cfg.paths.FILEPATH_MODEL = FILEPATH_MODEL
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'training_loss.pkl')

cfg.model.NUM_HIDDEN = 2
