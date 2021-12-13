import os

from yacs.config import CfgNode

from src.double_integrator.configs.config import config as cfg
from src.double_integrator.configs.config_collect_training_data import \
    FILEPATH_OUTPUT_DATA

base_path = '/home/bodrue/Data/neural_control/double_integrator/mlp'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.PATH_MODEL = os.path.join(base_path, 'models/mlp.params')

cfg.model = CfgNode()
cfg.model.NUM_HIDDEN = 10
cfg.model.NUM_LAYERS = 1

cfg.training.BATCH_SIZE = 32
cfg.training.LEARNING_RATE = 1e-2
cfg.training.NUM_EPOCHS = 10

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100

cfg.process.PROCESS_NOISES = [1e-2]
cfg.process.OBSERVATION_NOISES = [1e-2]
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
