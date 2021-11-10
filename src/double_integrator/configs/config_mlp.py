from yacs.config import CfgNode

from src.double_integrator.configs.config import config as cfg

cfg.paths.PATH_OUT = '/home/bodrue/Data/neural_control/figures'
cfg.paths.PATH_TRAINING_DATA = '/home/bodrue/Data/neural_control/training_data'
cfg.paths.PATH_MODEL = '/home/bodrue/Data/neural_control/models/mlp.params'

cfg.model = CfgNode()
cfg.model.NUM_HIDDEN = 10
cfg.model.NUM_LAYERS = 1

cfg.training.BATCH_SIZE = 32
cfg.training.LEARNING_RATE = 1e-2
cfg.training.NUM_EPOCHS = 10

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100

cfg.process.PROCESS_NOISE = 1e-2
cfg.process.OBSERVATION_NOISE = 1e-2
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
