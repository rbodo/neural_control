"""This file defines the default values for scripts in double_integrator."""

from yacs.config import CfgNode


def get_config() -> CfgNode:

    config = CfgNode()

    config.SEED = 42
    config.paths = CfgNode()
    config.paths.PATH_FIGURES = './'  # Where to save plots
    config.paths.FILEPATH_INPUT_DATA = ''  # Location of input data
    config.paths.FILEPATH_OUTPUT_DATA = ''  # Where to save output data
    config.paths.FILEPATH_MODEL = ''

    config.training = CfgNode()
    config.training.BATCH_SIZE = 32
    config.training.LEARNING_RATE = 1e-3
    config.training.NUM_EPOCHS = 10
    config.training.VALIDATION_FRACTION = 0.2

    config.simulation = CfgNode()
    config.simulation.T = 10  # Simulation duration
    config.simulation.NUM_STEPS = 100  # Number of steps in simulation duration
    config.simulation.GRID_SIZE = 1  # Number of grid lines along each dim

    config.process = CfgNode()
    config.process.PROCESS_NOISES = [0.]
    config.process.OBSERVATION_NOISES = [0.]
    config.process.STATE_MEAN = [1, 0]  # For sampling initial state values
    config.process.STATE_COVARIANCE = 1e-1  # For sampling initial state values

    config.controller = CfgNode()
    config.controller.cost = CfgNode()
    config.controller.cost.lqr = CfgNode()
    config.controller.cost.lqr.Q = 0.5  # Scale factor for state cost
    config.controller.cost.lqr.R = 0.5  # Scale factor for control cost
    config.controller.STATE_TARGET = [0, 0]

    config.model = CfgNode()
    config.model.USE_SINGLE_MODEL_IN_SWEEP = False  # Concerns RNN noise sweep.

    config.set_new_allowed(True)

    return config
