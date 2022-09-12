import sys

from scratch import configs
from scratch.di_lqg import main
from src.utils import apply_config

if __name__ == '__main__':

    # _config = configs.config_collect_training_data.get_config()
    _config = configs.config_collect_ood_data.get_config()

    apply_config(_config)

    print(_config)

    main(_config)

    sys.exit()
