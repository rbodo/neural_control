import os
import time
import configparser

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from pid_dnn import main
from pid_mnist_cnn import GaussianNoisePerturbation, BrownianPerturbation


def make_config():
    model_type = 'cnn'
    dataset_name = 'mnist'
    num_timesteps = 10

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': '/home/rbodo/Data/neural_control',
                       'path_wd': os.path.join('%(path_base)s', 'log', 'tmp',
                                               'noise_parameter_sweep',
                                               str(time.time())),
                       'path_models': os.path.join('%(path_base)s', 'models'),
                       'path_model': os.path.join('%(path_models)s',
                                                  f'pid_{dataset_name}_'
                                                  f'{model_type}.h5'),
                       'path_plots': os.path.join('%(path_wd)s', 'plots'),
                       'path_data': os.path.join('%(path_wd)s', 'data')}
    config['model'] = {'model_type': model_type}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': 20,
                            'reset_weights_between_samples': True,
                            'seed': 123,
                            'window_ratio': 1}
    config['controller'] = {'k_p': 0.1,
                            'k_i': 0.2,
                            'k_d': 0.1,
                            'layer_names':
                                str(['layer_1', 'layer_2', 'layer_3']),
                            'cumulative': True}
    config['dataset'] = {'name': dataset_name}
    config['output'] = {'make_plots': True, 'plot_fileformat': '.png'}
    config['perturbations'] = {
        'input_perturbation_scales': str([0]),
        'model_perturbation_scales': str([0, 0.2, 0.4, 0.7, 1]),
        'input_perturbation_kwargs': {'normalize': False, 'static': False},
        'model_perturbation_kwargs': {'delta': 0, 'drift': 0,
                                      'num_timesteps': num_timesteps}}

    return config


if __name__ == '__main__':

    _config = make_config()

    layer_names = eval(_config['controller']['layer_names'])

    seed = _config.getint('experiment', 'seed')

    input_perturbation_kwargs = eval(_config.get(
        'perturbations', 'input_perturbation_kwargs'))
    model_perturbation_kwargs = eval(_config.get(
        'perturbations', 'model_perturbation_kwargs'))

    filename_data = 'data_drift'
    _config['paths']['filename_data'] = filename_data
    path_data = os.path.join(_config['paths']['path_data'], filename_data)
    drift_values = np.insert(np.geomspace(1e-4, 1e-2, 8), 0, 0)
    data = None
    for drift in drift_values:
        print(f"drift: {drift}")
        model_perturbation_kwargs['drift'] = drift
        _config.set('perturbations', 'model_perturbation_kwargs',
                    str(model_perturbation_kwargs))
        rng = np.random.default_rng(seed)
        input_perturbation = GaussianNoisePerturbation(
            rng, **input_perturbation_kwargs)
        model_perturbation = BrownianPerturbation(rng=rng,
                                                  **model_perturbation_kwargs)
        main(_config, input_perturbation, model_perturbation, True)
        d = pd.DataFrame(pd.read_hdf(path_data + '.hdf'))
        d['drift'] = drift
        data = d if data is None else data.merge(d, 'outer')

    data.to_hdf(path_data + '.hdf', 'data', mode='w')
    data_reduced = data[data['acc_window']]
    data_reduced.loc[:, 'classifications'] = \
        pd.to_numeric(data_reduced['classifications'])

    g = sb.relplot(
        x='model_perturbation_scales', y='classifications', hue='layer_name',
        style='use_pid', style_order=[True, False], col='drift', col_wrap=3,
        palette='rocket', markers=True, dashes=True, hue_order=layer_names,
        kind='line', data=data_reduced)

    g.set_axis_labels("Model perturbation level", "Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(_config['paths']['path_plots'],
                             'drift_sweep.png'))
    plt.show()

    filename_data = 'data_delta'
    _config['paths']['filename_data'] = filename_data
    path_data = os.path.join(_config['paths']['path_data'], filename_data)
    delta_values = np.insert(np.geomspace(1e-4, 1e-2, 8), 0, 0)
    data = None
    for delta in delta_values:
        print(f"delta: {delta}")
        model_perturbation_kwargs['delta'] = delta
        _config.set('perturbations', 'model_perturbation_kwargs',
                    str(model_perturbation_kwargs))
        rng = np.random.default_rng(seed)
        input_perturbation = GaussianNoisePerturbation(
            rng, **input_perturbation_kwargs)
        model_perturbation = BrownianPerturbation(rng=rng,
                                                  **model_perturbation_kwargs)
        main(_config, input_perturbation, model_perturbation, True)
        d = pd.DataFrame(pd.read_hdf(path_data + '.hdf'))
        d['delta'] = delta
        data = d if data is None else data.merge(d, 'outer')

    data.to_hdf(path_data + '.hdf', 'data', mode='w')
    data_reduced = data[data['acc_window']]
    data_reduced.loc[:, 'classifications'] = \
        pd.to_numeric(data_reduced['classifications'])

    g = sb.relplot(
        x='model_perturbation_scales', y='classifications', hue='layer_name',
        style='use_pid', style_order=[True, False], col='delta', col_wrap=3,
        palette='rocket', markers=True, dashes=True, #hue_order=layer_names,
        kind='line', data=data_reduced)

    g.set_axis_labels("Model perturbation level", "Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(_config['paths']['path_plots'],
                             'delta_sweep.png'))
    plt.show()
