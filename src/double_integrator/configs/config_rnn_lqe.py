import os

from src.double_integrator.configs.config_rnn_defaults import cfg

base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn_lqe'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.PATH_MODEL = os.path.join(base_path, 'models/rnn_lqe.params')

cfg.process.PROCESS_NOISES = [1e-2]
cfg.process.OBSERVATION_NOISES = [1e-2]
