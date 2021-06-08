import os
import time
import configparser
import itertools

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from pid_dnn import main
from pid_mnist_cnn import GaussianNoisePerturbation, BrownianPerturbation


def make_config():
    model_type = 'cnn'
    dataset_name = 'cifar10'
    num_timesteps = 10

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': '/home/rbodo/Data/neural_control',
                       'path_wd': os.path.join('%(path_base)s', 'log', 'tmp',
                                               'pid_parameter_sweep',
                                               str(time.time())),
                       'path_models': os.path.join('%(path_base)s', 'models'),
                       'path_model': os.path.join('%(path_models)s',
                                                  f'pid_{dataset_name}_'
                                                  f'{model_type}.h5'),
                       'path_plots': os.path.join('%(path_wd)s', 'plots'),
                       'path_data': os.path.join('%(path_wd)s', 'data'),
                       'filename_data': 'data_pid_sweep'}
    config['model'] = {'model_type': model_type}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': 20,
                            'reset_weights_between_samples': True,
                            'seed': 123,
                            'window_ratio': 0.2}
    config['controller'] = {'k_p': 0,
                            'k_i': 0,
                            'k_d': 0,
                            'layer_names': str(['layer_2']),
                            'cumulative': True}
    config['dataset'] = {'name': dataset_name}
    config['output'] = {'make_plots': False, 'plot_fileformat': '.png'}
    config['perturbations'] = {
        'input_perturbation_scales': str([0.2]),
        'model_perturbation_scales': str([0.2]),
        'input_perturbation_kwargs': {'normalize': False, 'static': False},
        'model_perturbation_kwargs': {'delta': 1e-3, 'drift': 1e-3,
                                      'num_timesteps': num_timesteps}}

    return config


if __name__ == '__main__':

    _config = make_config()

    seed = _config.getint('experiment', 'seed')
    rng = np.random.default_rng(seed)

    input_perturbation_kwargs = eval(_config.get(
        'perturbations', 'input_perturbation_kwargs'))
    model_perturbation_kwargs = eval(_config.get(
        'perturbations', 'model_perturbation_kwargs'))
    input_perturbation = GaussianNoisePerturbation(rng,
                                                   **input_perturbation_kwargs)
    model_perturbation = BrownianPerturbation(rng=rng,
                                              **model_perturbation_kwargs)

    path_data = os.path.join(_config['paths']['path_data'],
                             _config['paths']['filename_data'] + '.hdf')

    k_p_values = np.insert(np.geomspace(1e-2, 2, 9), 0, 0)
    k_i_values = np.insert(np.geomspace(1e-2, 2, 9), 0, 0)
    k_d_values = np.insert(np.geomspace(1e-6, 2, 9), 0, 0)

    n = len(k_p_values) * len(k_i_values) * len(k_d_values)
    i = 0
    data = None
    for k_p, k_i, k_d in itertools.product(k_p_values, k_i_values, k_d_values):
        print(f"k_p: {k_p}, k_i: {k_i}, k_d: {k_d}")
        _config.set('controller', 'k_p', str(k_p))
        _config.set('controller', 'k_i', str(k_i))
        _config.set('controller', 'k_d', str(k_d))
        main(_config, input_perturbation, model_perturbation, False)
        d = pd.DataFrame(pd.read_hdf(path_data))
        d['k_p'] = k_p
        d['k_i'] = k_i
        d['k_d'] = k_d
        data = d if data is None else data.merge(d, 'outer')
        i += 1
        print('{:.2%}'.format(i/n))

    data.to_hdf(path_data, 'data', mode='w')

    data_sub = data[['classifications', 'k_p', 'k_i', 'k_d']]
    data_sub.loc[:, 'classifications'] = \
        pd.to_numeric(data_sub['classifications'])
    acc = data_sub.groupby(['k_p', 'k_i', 'k_d']).mean().reset_index()

    g = sb.FacetGrid(acc, col='k_d', col_wrap=3, margin_titles=True,
                     despine=False)
    for i, (k_d, ax) in enumerate(g.axes_dict.items()):
        a = acc[acc['k_d'] == k_d].pivot('k_p', 'k_i', 'classifications')
        cbar = i + 1 == len(g.axes_dict)
        h = sb.heatmap(a, ax=ax, square=True, cbar=cbar, annot=False, vmin=0,
                       vmax=1)
        h.set_xticklabels(['{:.2g}'.format(float(v.get_text())) for v in
                           h.get_xticklabels()])
        h.set_yticklabels(['{:.2g}'.format(float(v.get_text())) for v in
                           h.get_yticklabels()])
    g.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.set_titles('$k_d = {col_name:.2g}$')
    g.set_xlabels('$k_i$')
    g.set_ylabels('$k_p$')
    plt.savefig(os.path.join(_config['paths']['path_plots'], 'pid_sweep.png'))
    plt.show()
    print()
