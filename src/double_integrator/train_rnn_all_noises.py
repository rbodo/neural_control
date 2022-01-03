import sys

from src.double_integrator import configs
from src.double_integrator.train_rnn import train_single
from src.double_integrator.utils import apply_config


if __name__ == '__main__':
    _config = configs.config_train_rnn_all_noises.get_config()

    apply_config(_config)

    train_single(_config)

    sys.exit()
