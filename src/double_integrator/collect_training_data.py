import sys

from src.double_integrator import configs
from src.double_integrator.di_lqg import main

if __name__ == '__main__':

    # _config = configs.config_collect_training_data.get_config()
    _config = configs.config_collect_ood_data.get_config()

    main(_config)

    sys.exit()
