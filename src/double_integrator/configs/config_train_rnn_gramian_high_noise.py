import os
import time

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_collect_training_data import (
    PROCESS_NOISES, OBSERVATION_NOISES)

RANDOMIZE_WORKDIR = False

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/gramian/' \
            'first_epoch/high_noise'
if RANDOMIZE_WORKDIR:
    base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn_gramian.params')
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn_gramians.pkl')

cfg.training.NUM_EPOCHS = 1

cfg.process.PROCESS_NOISES = PROCESS_NOISES[-2:-1]
cfg.process.OBSERVATION_NOISES = OBSERVATION_NOISES[-2:-1]
