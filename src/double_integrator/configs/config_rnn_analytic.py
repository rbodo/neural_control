import os

import mxnet as mx

from src.double_integrator import configs
from src.double_integrator.utils import apply_timestamp


def get_config(timestamp_workdir=None):
    config = configs.config_rnn_defaults.get_config(timestamp_workdir)

    base_path = \
        '/home/bodrue/Data/neural_control/double_integrator/rnn_analytic'
    base_path = apply_timestamp(base_path, timestamp_workdir)

    config.paths.PATH_FIGURES = os.path.join(base_path, 'figures')
    config.paths.FILEPATH_MODEL = None

    config.model.NUM_HIDDEN = 2
    config.model.ACTIVATION = 'linear'

    return config
