import configparser
import os
import time

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from tensorflow.keras import layers

from src.pid import PID
from pid_mnist_cnn import save_config, save_output, get_dataset, \
    ModelPerturber, get_model, GaussianNoisePerturbation, BrownianPerturbation


class ActivationPID(layers.Activation):
    def __init__(self, activation='linear', **kwargs):
        super().__init__(activation, **kwargs)
        self.b = 0

    def call(self, inputs):
        return inputs + self.b


def get_model_pid(x_test, y_test, model_path, evaluate=False):

    input_shape = x_test.shape[1:]
    num_classes = y_test.shape[-1]

    model_pid = keras.Sequential([
        layers.InputLayer(input_shape, name='layer_0'),
        layers.Conv2D(64, (3, 3), activation='relu', name='layer_1'),
        ActivationPID(name='layer_1_activ'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', name='layer_2'),
        ActivationPID(name='layer_2_activ'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(num_classes, activation='relu', name='layer_3'),
        ActivationPID(name='layer_3_activ'),
    ])

    optimizer = keras.optimizers.Adam(1e-5)
    loss = 'mse'  # 'categorical_crossentropy'
    model_pid.compile(optimizer, loss, ['accuracy'])

    model = keras.models.load_model(model_path)
    model_pid.set_weights(model.get_weights())

    # model_pid.summary()

    if evaluate:
        score = model_pid.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {score[1]:.2%}")

    return model_pid


def apply_pid(pid, y_perturbed, setpoint, t, model, layer_name, cumulative):
    control_variable = pid.update(y_perturbed, setpoint, t)
    layer = model.get_layer(layer_name + '_activ')
    if cumulative:
        layer.b += control_variable
    else:
        layer.b = control_variable


def step(model, x, layer_name, submodel=None, pid=None, setpoint=None, t=None,
         cumulative=False):

    x_ = np.expand_dims(x, 0)

    output = model(x_)[0]
    label_predicted = np.argmax(output)

    # Submodel not needed if we control the output layer.
    if submodel is None:
        y_perturbed = output
    else:
        y_perturbed = submodel(x_)[0]

    use_pid = pid is not None
    if use_pid:
        apply_pid(pid, y_perturbed, setpoint, t, model, layer_name, cumulative)

    return label_predicted


def update_data(data, **kwargs):
    data['classifications'].append(
        kwargs['label_predicted'] == kwargs['label'])
    data['use_pid'].append('pid' in kwargs)
    data['input_perturbation_scales'].append(
        kwargs['input_perturbation_scale'])
    data['model_perturbation_scales'].append(
        kwargs['model_perturbation_scale'])
    data['layer_name'].append(kwargs['layer_name'])
    data['acc_window'].append(kwargs['t'] >= kwargs['window_start'])


def make_config():
    model_type = 'cnn'  # 'cnn', 'mlp', 'dae'
    dataset_name = 'cifar10'  # 'mnist', 'cifar10'

    num_timesteps = 10

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': '/home/rbodo/Data/neural_control',
                       'path_wd': os.path.join('%(path_base)s', 'log', 'tmp',
                                               str(time.time())),
                       'path_models': os.path.join('%(path_base)s', 'models'),
                       'path_model': os.path.join('%(path_models)s',
                                                  f'pid_{dataset_name}_'
                                                  f'{model_type}.h5'),
                       'path_plots': os.path.join('%(path_wd)s', 'plots'),
                       'path_data': os.path.join('%(path_wd)s', 'data'),
                       'filename_data': 'data'}
    config['model'] = {'model_type': model_type}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': 100,
                            'reset_weights_between_samples': True,
                            'seed': 123,
                            'window_ratio': 1}
    config['controller'] = {'k_p': 0.3,
                            'k_i': 0.3,
                            'k_d': 0.3,
                            'layer_names':
                                str(['layer_1', 'layer_2', 'layer_3']),
                            'cumulative': True}
    config['dataset'] = {'name': dataset_name}
    config['output'] = {'make_plots': True, 'plot_fileformat': '.png'}
    config['perturbations'] = {
        'input_perturbation_scales': str([0, 0.2, 0.4, 0.7, 1]),
        'model_perturbation_scales': str([0, 0.2, 0.4, 0.7, 1]),
        'input_perturbation_kwargs': {'normalize': False, 'static': False},
        'model_perturbation_kwargs': {'delta': 1e-4, 'drift': 1e-4,
                                      'num_timesteps': num_timesteps}}

    return config


def reset_control(model):
    for layer in model.layers:
        if type(layer).__name__ == 'ActivationPID':
            layer.b = 0


def get_setpoints(model, x, submodel=None):
    if submodel is None:
        return model(x)
    else:
        return submodel(x)


class InputPerturber:
    def __init__(self, x, num_timesteps, perturbation, scales):
        self.x = x
        self.num_timesteps = num_timesteps
        self.perturbation = perturbation
        self.scales = scales
        self.x_perturbed = None

    def apply(self):
        self.x_perturbed = {}
        x = self.x
        if self.num_timesteps > 1:
            if x.shape[0] > 1:
                x = np.expand_dims(x, 0)
            x = np.repeat(x, self.num_timesteps, 0)
        for scale in self.scales:
            self.x_perturbed[scale] = self.perturbation.apply(x, scale)

    def step(self, i, t, scale):
        return self.x_perturbed[scale][t, i]


def main(config, input_perturbation, model_perturbation,
         run_noncontrolled=True):

    path_wd = config['paths']['path_wd']
    path_plots = config['paths']['path_plots']
    path_models = config['paths']['path_models']
    path_model = config['paths']['path_model']
    path_data = config['paths']['path_data']
    filename_data = config['paths']['filename_data']

    model_type = config['model']['model_type']

    num_timesteps = config.getint('experiment', 'num_timesteps')
    num_to_test = config.getint('experiment', 'num_to_test')
    reset_weights_between_samples = config.getboolean(
        'experiment', 'reset_weights_between_samples')
    window_ratio = config.getfloat('experiment', 'window_ratio')
    # Optionally use only the last few timesteps for accuracy computation.
    assert 0 < window_ratio <= 1
    window_start = int(num_timesteps * (1 - window_ratio))

    make_plots = config.getboolean('output', 'make_plots')
    plot_fileformat = config['output']['plot_fileformat']

    cumulative_control = config.getboolean('controller', 'cumulative')
    k_p = config.getfloat('controller', 'k_p')
    k_i = config.getfloat('controller', 'k_i')
    k_d = config.getfloat('controller', 'k_d')
    pid = PID(k_p=k_p, k_i=k_i, k_d=k_d)

    layer_names = eval(config['controller']['layer_names'])
    dataset_name = config['dataset']['name']

    os.makedirs(path_wd, exist_ok=True)
    os.makedirs(path_plots, exist_ok=True)
    os.makedirs(path_data, exist_ok=True)
    os.makedirs(path_models, exist_ok=True)

    num_classes = 10
    flatten = model_type == 'mlp'
    x_train, y_train, x_test, y_test = get_dataset(dataset_name, num_classes,
                                                   flatten)

    offset = 0
    test_idxs = np.arange(num_to_test) + offset
    labels = np.argmax(y_test[test_idxs], -1)

    config['dataset']['test_idxs'] = str(test_idxs)

    input_perturbation_scales = eval(config.get(
        'perturbations', 'input_perturbation_scales'))
    model_perturbation_scales = eval(config.get(
        'perturbations', 'model_perturbation_scales'))

    config.set('perturbations', 'input_perturbation',
               str(input_perturbation.__dict__))
    config.set('perturbations', 'model_perturbation',
               str(model_perturbation.__dict__))

    model_kwargs = {'model_type': model_type, 'model_path': path_model}
    model = submodel = None
    model_perturber = None
    if run_noncontrolled:
        model = get_model(x_train, y_train, x_test, y_test, model_kwargs)
        model_perturber = ModelPerturber(
            model, layer_names, model_perturbation, model_perturbation_scales)
    model_setpoint = get_model(x_train, y_train, x_test, y_test, model_kwargs,
                               evaluate=False)
    model_pid = get_model_pid(x_test, y_test, path_model, evaluate=False)
    model_perturber_pid = ModelPerturber(
        model_pid, layer_names, model_perturbation, model_perturbation_scales)

    config['model']['model_config'] = str(model_pid.get_config())
    save_config(path_wd, config)

    # For efficiency, prepare perturbation for all time steps in advance.
    # For consistency, prepare perturbation for all layers and scales in
    # advance. If we produced the perturbations on demand in some inner loop,
    # the random state would be different in case we change loop parameters.
    model_perturber_pid.apply()
    if run_noncontrolled:
        # Copy over params so the perturbation is identical.
        model_perturber.params_perturbed = \
            model_perturber_pid.params_perturbed.copy()

    input_perturber = InputPerturber(x_test[test_idxs], num_timesteps,
                                     input_perturbation,
                                     input_perturbation_scales)
    input_perturber.apply()

    data_dict = {'classifications': [],
                 'input_perturbation_scales': [],
                 'model_perturbation_scales': [],
                 'use_pid': [],
                 'layer_name': [],
                 'acc_window': []}
    data_kwargs = {'window_start': window_start}
    for layer_name in layer_names:
        print(f"Layer: {layer_name}.")
        data_kwargs['layer_name'] = layer_name
        is_output_layer = layer_name in model_setpoint.layers[-1].name
        if is_output_layer:
            # Auxiliary and inverse models not needed for output layer.
            submodel = submodel_pid = submodel_setpoint = None
        else:
            output_layer = model_pid.get_layer(layer_name + '_activ')
            submodel_pid = keras.models.Model(model_pid.input,
                                              output_layer.output)
            output_layer = model_setpoint.get_layer(layer_name)
            submodel_setpoint = keras.models.Model(model_setpoint.input,
                                                   output_layer.output)
            if run_noncontrolled:
                output_layer = model.get_layer(layer_name)
                submodel = keras.models.Model(model.input, output_layer.output)

        setpoints = get_setpoints(model_setpoint, x_test[test_idxs],
                                  submodel_setpoint)
        for input_perturbation_scale in input_perturbation_scales:
            print(f"\tInput perturbation scale: {input_perturbation_scale}")
            data_kwargs['input_perturbation_scale'] = input_perturbation_scale
            for model_perturbation_scale in model_perturbation_scales:
                print("\t\tModel perturbation scale: "
                      f"{model_perturbation_scale}")
                data_kwargs['model_perturbation_scale'] = \
                    model_perturbation_scale
                for sample_idx_rel, label in enumerate(labels):
                    data_kwargs['label'] = label
                    for t in range(num_timesteps):
                        data_kwargs['t'] = t
                        if reset_weights_between_samples:
                            t_abs = t
                        else:
                            t_abs = t + num_timesteps * sample_idx_rel
                        x_perturbed = input_perturber.step(
                            sample_idx_rel, t, input_perturbation_scale)
                        model_perturber_pid.step(t, model_perturbation_scale,
                                                 [layer_name])
                        y = step(model_pid, x_perturbed, layer_name,
                                 submodel_pid, pid, setpoints[sample_idx_rel],
                                 t_abs, cumulative_control)
                        data_kwargs['pid'] = pid
                        data_kwargs['label_predicted'] = y
                        update_data(data_dict, **data_kwargs)
                        if run_noncontrolled:
                            model_perturber.step(t, model_perturbation_scale,
                                                 [layer_name])
                            y = step(model, x_perturbed, layer_name, submodel)
                            data_kwargs.pop('pid')
                            data_kwargs['label_predicted'] = y
                            update_data(data_dict, **data_kwargs)
                    # Resetting biases between samples could be replaced by a
                    # decay of the biases back to initial values.
                    if reset_weights_between_samples:
                        if run_noncontrolled:
                            model.set_weights(model_setpoint.get_weights())
                        model_pid.set_weights(model_setpoint.get_weights())
                        reset_control(model_pid)
                        pid.reset()
                if run_noncontrolled:
                    model.set_weights(model_setpoint.get_weights())
                model_pid.set_weights(model_setpoint.get_weights())
                reset_control(model_pid)
                pid.reset()

    filepath_data_out = os.path.join(path_data, filename_data + '.hdf')

    save_output(data_dict, filepath_data_out)

    if make_plots:
        plot_results(path_plots, filepath_data_out, layer_names,
                     plot_fileformat)


def plot_results(path: str,
                 path_data: str,
                 layer_names: list[str],
                 output_format: str = '.png'):

    data = pd.DataFrame(pd.read_hdf(path_data, 'data'))

    data_reduced = data[data['acc_window']]

    g = sb.relplot(
        x='model_perturbation_scales', y='classifications', hue='layer_name',
        style='use_pid', style_order=[True, False],
        col='input_perturbation_scales', col_wrap=3, palette='rocket',
        markers=True, dashes=True, hue_order=layer_names, kind='line',
        data=data_reduced)

    g.set_axis_labels("Model perturbation level", "Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(path, f'accuracy_vs_perturbation{output_format}'))
    plt.show()


if __name__ == '__main__':

    _config = make_config()

    _seed = _config.getint('experiment', 'seed')
    rng = np.random.default_rng(_seed)

    input_perturbation_kwargs = eval(_config.get(
        'perturbations', 'input_perturbation_kwargs'))
    model_perturbation_kwargs = eval(_config.get(
        'perturbations', 'model_perturbation_kwargs'))
    _input_perturbation = GaussianNoisePerturbation(
        rng, **input_perturbation_kwargs)
    _model_perturbation = BrownianPerturbation(
        rng=rng, **model_perturbation_kwargs)

    main(_config, _input_perturbation, _model_perturbation)
