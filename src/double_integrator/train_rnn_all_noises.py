import os
import sys

from src.double_integrator.configs.config import get_config
from src.double_integrator.train_rnn import train_single


if __name__ == '__main__':
    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs'
    filename = 'config_train_rnn_all_noises.py'
    _config = get_config(os.path.join(base_path, filename))

    train_single(_config)

    sys.exit()
