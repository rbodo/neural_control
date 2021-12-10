from src.double_integrator.configs.config import config as cfg

cfg.paths.PATH_OUT = \
    '/home/bodrue/Data/neural_control/double_integrator/open/figures/'

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 100

cfg.process.PROCESS_NOISES = [1e-2]
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
