from yacs.config import CfgNode

from src.double_integrator.configs.config import config as cfg
from src.double_integrator.configs.config_collect_training_data import cfg as c

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/'
cfg.paths.PATH_OUT = base_path + 'figures/'
cfg.paths.PATH_TRAINING_DATA = c.paths.PATH_TRAINING_DATA
cfg.paths.PATH_MODEL = base_path + 'models/rnn.params'

cfg.model = CfgNode()
cfg.model.NUM_HIDDEN = 50
cfg.model.NUM_LAYERS = 1
cfg.model.ACTIVATION = 'relu'

cfg.training.BATCH_SIZE = 32
cfg.training.LEARNING_RATE = 1e-3
cfg.training.NUM_EPOCHS = 10
cfg.training.OPTIMIZER = 'adam'

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100
cfg.simulation.DO_WARMUP = False

cfg.process.PROCESS_NOISES = c.process.PROCESS_NOISES
cfg.process.OBSERVATION_NOISES = c.process.OBSERVATION_NOISES
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
