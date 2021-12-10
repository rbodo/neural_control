from src.double_integrator.configs.config import config as cfg

cfg.paths.PATH_OUT = \
    '/home/bodrue/Data/neural_control/double_integrator/lqg/figures/'

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 1000
cfg.simulation.GRID_SIZE = 1

cfg.process.PROCESS_NOISES = [1e-2]
cfg.process.OBSERVATION_NOISES = [1e-2]
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
cfg.controller.cost.lqr.Q = 0.5
cfg.controller.cost.lqr.R = 0.5
