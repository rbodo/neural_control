import os.path

from src.double_integrator.configs.config_rnn_defaults import cfg
from src.double_integrator.configs.config_train_rnn import FILEPATH_MODEL
from src.double_integrator.configs.config_collect_training_data import (
    PROCESS_NOISES, OBSERVATION_NOISES, FILEPATH_OUTPUT_DATA)
from src.double_integrator.train_rnn import get_model_name

path, model_name = os.path.split(FILEPATH_MODEL)
model_name = \
    get_model_name(model_name, PROCESS_NOISES[2], OBSERVATION_NOISES[2])

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
            'generalization/' + os.path.splitext(model_name)[0]
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_INPUT_DATA = FILEPATH_OUTPUT_DATA
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'rnn.pkl')
cfg.paths.FILEPATH_MODEL = os.path.join(path, model_name)

cfg.model.USE_SINGLE_MODEL_IN_SWEEP = True
