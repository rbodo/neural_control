import sys
import os

import pandas as pd

from scratch import configs
from src.plotting import plot_cost_vs_time
from src.utils import apply_timestamp, split_train_test


def main(config_lqg_train, config_lqg_test, config_rnn_train, config_rnn_test,
         path_out):

    df1 = pd.read_pickle(config_lqg_train.paths.FILEPATH_OUTPUT_DATA)
    # Loaded full LQG dataset here, thus need to reduce to test set.
    validation_fraction = config_lqg_train.training.VALIDATION_FRACTION
    _, df1 = split_train_test(df1, validation_fraction)
    df1['controller'] = 'lqg'
    df1['testset'] = 'id'

    df2 = pd.read_pickle(config_lqg_test.paths.FILEPATH_OUTPUT_DATA)
    df2['controller'] = 'lqg'
    df2['testset'] = 'ood'

    df3 = pd.read_pickle(config_rnn_train.paths.FILEPATH_OUTPUT_DATA)
    df3['controller'] = 'rnn'
    df3['testset'] = 'id'

    df4 = pd.read_pickle(config_rnn_test.paths.FILEPATH_OUTPUT_DATA)
    df4['controller'] = 'rnn'
    df4['testset'] = 'ood'

    df = pd.concat([df1, df2, df3, df4], join='inner')

    path = os.path.join(path_out, f'cost_vs_time_generalization_lqg_rnn.png')
    plot_cost_vs_time(df, path)


if __name__ == '__main__':

    _config_lqg_train = \
        configs.config_collect_training_data.get_config('20211231_000000')
    _config_lqg_test = \
        configs.config_lqg_generalization.get_config('20220106_161006')
    _config_rnn_train = \
        configs.config_test_rnn_all_noises.get_config('20211231_000002')
    _config_rnn_test = \
        configs.config_test_rnn_ood.get_config('20211231_000002')

    _path_out = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'generalization/figures'
    _path_out = apply_timestamp(_path_out)
    os.makedirs(_path_out, exist_ok=True)

    main(_config_lqg_train,
         _config_lqg_test,
         _config_rnn_train,
         _config_rnn_test,
         _path_out)

    sys.exit()
