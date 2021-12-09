"""This file defines the default values for scripts in double_integrator."""
import os
from typing import List, Optional, Union

from yacs.config import CfgNode

config = CfgNode()

config.SEED = 42
config.paths = CfgNode()
config.paths.PATH_OUT = './'  # Where to save plots
config.paths.PATH_TRAINING_DATA = './'  # Where to save or load training data
config.paths.PATH_MODEL = './'

config.training = CfgNode()
config.training.BATCH_SIZE = 32
config.training.LEARNING_RATE = 1e-3
config.training.NUM_EPOCHS = 100

config.simulation = CfgNode()
config.simulation.T = 10  # Simulation duration
config.simulation.NUM_STEPS = 100  # Number of steps in simulation duration

config.process = CfgNode()
config.process.PROCESS_NOISE = 0.
config.process.OBSERVATION_NOISE = 0.
config.process.STATE_MEAN = []  # For sampling initial state values
config.process.STATE_COVARIANCE = 1e-3  # For sampling initial state values

config.controller = CfgNode()
config.controller.cost = CfgNode()
config.controller.cost.lqr = CfgNode()
config.controller.cost.lqr.Q = 0.5  # Scale factor for state cost
config.controller.cost.lqr.R = 0.5  # Scale factor for control cost
config.controller.STATE_TARGET = []

config.set_new_allowed(True)


def get_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None) -> CfgNode:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config). For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """

    CONFIG_FILE_SEPARATOR = ','

    _config = config.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            _config.merge_from_file(config_path)

    if opts:
        _config.merge_from_list(opts)

    _config.freeze()
    with open(_config.paths.PATH_OUT + '_config.txt', 'w') as f:
        f.write(_config.dump())

    for p in _config.paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)

    return _config
