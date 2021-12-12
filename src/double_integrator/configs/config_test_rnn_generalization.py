import os.path

from yacs.config import CfgNode

from src.double_integrator.configs.config import config as cfg
from src.double_integrator.configs.config_collect_training_data import cfg as c
from src.double_integrator.configs.config_train_rnn import cfg as c2
from src.double_integrator.train_rnn import get_model_name

process_noises = c2.process.PROCESS_NOISES
observation_noises = c2.process.OBSERVATION_NOISES
path_model = c2.paths.FILEPATH_MODEL
path, model_name = os.path.split(path_model)
model_name = \
    get_model_name(model_name, process_noises[2], observation_noises[2])

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
            'generalization_' + os.path.splitext(model_name)[0]
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = c.paths.FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
cfg.paths.FILEPATH_MODEL = os.path.join(path, model_name)

cfg.model = CfgNode()
cfg.model.NUM_HIDDEN = 50
cfg.model.NUM_LAYERS = 1
cfg.model.ACTIVATION = 'relu'
cfg.model.USE_SINGLE_MODEL_IN_SWEEP = True

cfg.training.BATCH_SIZE = 32
cfg.training.LEARNING_RATE = 1e-3
cfg.training.NUM_EPOCHS = 10
cfg.training.OPTIMIZER = 'adam'

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100
cfg.simulation.DO_WARMUP = False

cfg.process.PROCESS_NOISES = process_noises
cfg.process.OBSERVATION_NOISES = observation_noises
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
