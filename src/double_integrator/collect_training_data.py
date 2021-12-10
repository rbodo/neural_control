import sys

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_lqg import main

if __name__ == '__main__':

    _config = get_config(
        '/home/bodrue/PycharmProjects/neural_control/src/double_integrator/'
        'configs/config_collect_training_data.py')

    main(_config)

    sys.exit()
