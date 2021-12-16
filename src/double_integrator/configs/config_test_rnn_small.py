import os
import time

from src.double_integrator.configs.config_train_rnn_small import cfg

RANDOMIZE_WORKDIR = False

base_path = \
    '/home/bodrue/Data/neural_control/double_integrator/rnn/two_neurons'
if RANDOMIZE_WORKDIR:
    base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
cfg.paths.FILEPATH_MODEL = FILEPATH_MODEL
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
