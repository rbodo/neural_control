import sys

import os
from collections import OrderedDict

import pandas as pd

from src.double_integrator import configs
from src.double_integrator.plotting import plot_cost_vs_time

from src.double_integrator.utils import apply_timestamp


def main(config_dict: OrderedDict, path_out: str):
    label2, config2 = config_dict.popitem()
    label1, config1 = config_dict.popitem()

    # Use output data file generated during testing.
    path_data1 = config1.paths.FILEPATH_OUTPUT_DATA
    df1 = pd.read_pickle(path_data1)
    df1['controller'] = label1

    path_data2 = config2.paths.FILEPATH_OUTPUT_DATA
    df2 = pd.read_pickle(path_data2)
    df2['controller'] = label2

    df = pd.concat([df1, df2], join='inner')

    path = os.path.join(path_out,
                        f'cost_vs_time_combined_{label1}_{label2}.png')
    plot_cost_vs_time(df, path)


if __name__ == '__main__':

    _config1 = configs.config_test_rnn_all_noises.get_config('20211231_000002')
    _config2 = configs.config_test_rnn_ood.get_config('20211231_000002')

    print(_config1)
    print(_config2)

    _path_out = '/home/bodrue/Data/neural_control/double_integrator/rnn/' \
                'figures/generalization'
    _path_out = apply_timestamp(_path_out)
    os.makedirs(_path_out, exist_ok=True)

    main(OrderedDict([('rnn_id', _config1), ('rnn_ood', _config2)]), _path_out)

    sys.exit()
