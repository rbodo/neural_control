import sys

from scratch import configs
from scratch.train_rnn import train_single
from src.utils import apply_config


if __name__ == '__main__':
    _config = configs.config_train_rnn_all_noises.get_config()

    apply_config(_config)

    print(_config)

    train_single(_config)

    sys.exit()
