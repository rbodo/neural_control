from src.double_integrator.configs.config import config as cfg

# cfg.paths.PATH_OUT = 'C:\\Users\\u714174\\Data\\neural_control\\figures'
cfg.paths.PATH_OUT = '/home/bodrue/Data/neural_control/figures'

cfg.simulation.T = 10
cfg.simulation.NUM_STEPS = 1000

cfg.process.PROCESS_NOISE = 1e-2
cfg.process.OBSERVATION_NOISE = 1e-2
cfg.process.STATE_MEAN = [1, 0]
cfg.process.STATE_COVARIANCE = 1e-1

cfg.controller.STATE_TARGET = [0, 0]
cfg.controller.cost.lqr.Q = 0.5
cfg.controller.cost.lqr.R = 0.5
