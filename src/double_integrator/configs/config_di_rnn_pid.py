import os

import numpy as np
from yacs.config import CfgNode

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config('20211231_000000')

    base_path = '/home/bodrue/Data/neural_control/double_integrator/rnn_pid'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = os.path.abspath(os.path.join(
        base_path, '..', '..', 'rnn', 'models', 'rnn.params'))
    config.paths.FILEPATH_OUTPUT_DATA = os.path.join(base_path, 'output.pkl')

    config.perturbation = CfgNode()
    config.perturbation.DRIFT_LEVELS = np.logspace(-3, -1, 5,
                                                   dtype='float32').tolist()
    config.perturbation.DELTA = 1e-5
    config.controller.KP = 1.
    config.controller.KI = 0.
    config.controller.KD = 0.5

    return config
