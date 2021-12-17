import os.path
import sys

from src.double_integrator.configs.config import get_config
from src.double_integrator.di_lqg import main

if __name__ == '__main__':

    base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
                'double_integrator/configs/'
    # filename = 'config_collect_training_data.py'
    filename = 'config_collect_ood_data.py'

    _config = get_config(os.path.join(base_path, filename))

    main(_config)

    sys.exit()
