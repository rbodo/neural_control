import os

from yacs.config import CfgNode

from src.double_integrator.configs.config import config as cfg
from src.double_integrator.configs.config_collect_training_data import (
    PROCESS_NOISES, OBSERVATION_NOISES, FILEPATH_OUTPUT_DATA)

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')
cfg.paths.FILEPATH_MODEL = os.path.join(base_path, 'models/rnn.params')

cfg.model = CfgNode()
cfg.model.NUM_HIDDEN = 50
cfg.model.NUM_LAYERS = 1
cfg.model.ACTIVATION = 'relu'

cfg.training.OPTIMIZER = 'adam'

cfg.simulation.DO_WARMUP = False

cfg.process.PROCESS_NOISES = PROCESS_NOISES
cfg.process.OBSERVATION_NOISES = OBSERVATION_NOISES
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]