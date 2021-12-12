import os

import numpy as np

from src.double_integrator.configs.config import config as cfg

base_path = '/home/bodrue/Data/neural_control/double_integrator/lqg'
cfg.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
cfg.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'lqg_grid.pkl')

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100
cfg.simulation.GRID_SIZE = 100

cfg.process.PROCESS_NOISES = np.logspace(-2, -1, 5, dtype='float32').tolist()
cfg.process.OBSERVATION_NOISES = np.logspace(-1, 0, 5,
                                             dtype='float32').tolist()
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.cost.lqr.Q = 0.5
cfg.controller.cost.lqr.R = 0.5
