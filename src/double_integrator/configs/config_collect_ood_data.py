import os

import numpy as np

from src.double_integrator.configs.config import config as cfg

PROCESS_NOISES = np.logspace(-1.5, -0.5, 5, dtype='float32').tolist()
OBSERVATION_NOISES = np.logspace(-0.5, 0.6, 5, dtype='float32').tolist()

base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'lqg_grid_ood.pkl')
cfg.paths.FILEPATH_OUTPUT_DATA = FILEPATH_OUTPUT_DATA

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100
cfg.simulation.GRID_SIZE = 100

cfg.process.PROCESS_NOISES = PROCESS_NOISES
cfg.process.OBSERVATION_NOISES = OBSERVATION_NOISES
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.cost.lqr.Q = 0.5
cfg.controller.cost.lqr.R = 0.5
