import os
import time

from src.double_integrator.configs.config_rnn_defaults import cfg

RANDOMIZE_WORKDIR = False

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/all_noises'
if RANDOMIZE_WORKDIR:
    base_path = os.path.join(base_path, time.strftime('%Y%m%d_%H%M%S'))
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')
cfg.paths.FILEPATH_MODEL = FILEPATH_MODEL

# Reduce relative size of training set because we are sampling training data
# from across all noise levels.
f = cfg.training.VALIDATION_FRACTION
c = len(cfg.process.PROCESS_NOISES) * len(cfg.process.OBSERVATION_NOISES)
cfg.training.VALIDATION_FRACTION = 1 + (f - 1) / c
