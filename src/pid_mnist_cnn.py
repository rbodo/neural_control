import configparser
import os
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import List

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from brownian import brownian
from src.pid import PID


def relu_inverse(x, slope):
    return keras.activations.relu(x, 1/slope)


def get_dataset(dataset_name, num_classes, flatten=False):
    dataset_name = str(dataset_name).lower()
    if dataset_name == 'mnist':
        from tensorflow.keras.datasets import mnist as dataset
    elif dataset_name == 'cifar10':
        from tensorflow.keras.datasets import cifar10 as dataset
    else:
        raise NotImplementedError

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if flatten:
        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
    elif dataset_name == 'mnist':
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_model(x_train, y_train, x_test, y_test, model_kwargs, evaluate=False):
    model_path = model_kwargs['model_path']
    model_type = model_kwargs['model_type']
    relu_slope = model_kwargs.get('relu_slope', 0)
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path)
        # for layer in model.layers:#
        #     if len(layer.weights):
        #         w, b = layer.get_weights()
        #         b = np.zeros_like(b)
        #         layer.set_weights([w, b])
    else:
        batch_size = 128
        epochs = 20
        input_shape = x_test.shape[1:]
        num_classes = y_test.shape[-1]

        use_bias = True
        if model_type == 'mlp':
            g = model_kwargs['growth_factor']
            num_neurons = input_shape[0]
            model = keras.Sequential([
                layers.InputLayer(input_shape, name='layer_0'),
                layers.Dense(int(g * num_neurons), name='layer_1',
                             use_bias=use_bias),
                layers.ReLU(negative_slope=relu_slope, name='layer_1_activ'),
                layers.Dense(int(g * g * num_neurons), name='layer_2',
                             use_bias=use_bias),
                layers.ReLU(negative_slope=relu_slope, name='layer_2_activ'),
                layers.Dense(num_classes, name='layer_3', use_bias=use_bias),
                layers.ReLU(negative_slope=relu_slope, name='layer_3_activ')
            ])
        elif model_type == 'cnn':
            model = keras.Sequential([
                layers.InputLayer(input_shape, name='layer_0'),
                layers.Conv2D(64, (3, 3), activation='relu', name='layer_1'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, (3, 3), activation='relu', name='layer_2'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                # layers.Dropout(0.5),
                layers.Dense(num_classes, activation='relu', name='layer_3')
            ])
        elif model_type == 'dae':
            model = keras.Sequential([
                layers.InputLayer(input_shape, name='layer_0'),
                layers.Dense(128, activation='relu', name='layer_1'),
                layers.Dense(128, activation='relu', name='layer_2'),
                layers.Dense(num_classes, activation='relu', name='layer_3')
            ])
        else:
            raise NotImplementedError

        optimizer = keras.optimizers.Adam(1e-5)
        loss = 'mse'  # 'categorical_crossentropy'
        model.compile(optimizer, loss, ['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_split=0.1)
        # for layer in model.layers:#
        #     if len(layer.weights):
        #         w, b = layer.get_weights()
        #         b = np.zeros_like(b)
        #         layer.set_weights([w, b])

        model.save(model_path)

    # model.summary()

    if evaluate:
        score = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {score[1]:.2%}")

    return model


def get_auxiliary_model(model, x_test, y_test, model_kwargs, rng):
    g = model_kwargs['growth_factor']

    params = model.get_weights()
    w1, b1, w2, b2, w3, b3 = params

    num_neurons_0, num_neurons_1 = w1.shape
    num_neurons_2, num_classes = w3.shape
    num_neurons_3 = int(g * num_neurons_2)
    w3_ = rng.choice(w3.ravel(), (num_neurons_2, num_neurons_3))
    w3_[:, :num_classes] = w3
    b3_ = np.zeros(num_neurons_3)
    b3_[:num_classes] = b3

    relu_slope = model_kwargs['relu_slope']
    input_shape = x_test.shape[1:]
    model_aux = keras.Sequential([
        layers.InputLayer(input_shape, name='layer_0'),
        layers.Dense(num_neurons_1, name='layer_1', weights=[w1, b1]),
        layers.ReLU(negative_slope=relu_slope, name='layer_1_activ'),
        layers.Dense(num_neurons_2, name='layer_2', weights=[w2, b2]),
        layers.ReLU(negative_slope=relu_slope, name='layer_2_activ'),
        layers.Dense(num_neurons_3, name='layer_3', weights=[w3_, b3_]),
        layers.ReLU(negative_slope=relu_slope, name='layer_3_activ')
    ])

    model_aux.summary()

    model_aux.compile('sgd', 'mse', ['accuracy'])

    y_test_ = np.zeros((len(y_test), num_neurons_3))
    y_test_[:, :num_classes] = y_test
    loss, acc = model_aux.evaluate(x_test, y_test_)
    print(f"Accuracy of auxiliary model: {acc:.2%}")

    return model_aux


def get_inverse_model(model, x_test, model_kwargs, dataset_name):
    params = model.get_weights()
    w1, b1, w2, b2, w3, b3 = params

    c3 = np.linalg.cond(w3)
    c2 = np.linalg.cond(w2)
    c1 = np.linalg.cond(w1)
    print(f"Condition numbers (should be below 30): {c1}, {c2}, {c3}")

    w2_ = np.linalg.pinv(w3)
    w1_ = np.linalg.pinv(w2)
    w0_ = np.linalg.pinv(w1)
    b2_ = -np.dot(b3, w2_)
    b1_ = -np.dot(b2, w1_)
    b0_ = -np.dot(b1, w0_)

    relu_slope = model_kwargs['relu_slope']
    activation = partial(relu_inverse, slope=relu_slope)
    y = model(x_test)
    num_neurons_0, num_neurons_1 = w1.shape
    num_neurons_2, num_neurons_3 = w3.shape
    input_shape = (num_neurons_3,)
    model_inverse = keras.Sequential([
        layers.InputLayer(input_shape, name='layer_3'),
        layers.Activation(activation, name='layer_3_activ'),
        layers.Dense(num_neurons_2, name='layer_2', weights=[w2_, b2_]),
        layers.Activation(activation, name='layer_2_activ'),
        layers.Dense(num_neurons_1, name='layer_1', weights=[w1_, b1_]),
        layers.Activation(activation, name='layer_1_activ'),
        layers.Dense(num_neurons_0, name='layer_0', weights=[w0_, b0_])
    ])

    model_inverse.summary()

    model_inverse.compile('sgd', 'mse')

    loss = model_inverse.evaluate(y, x_test)
    print(f"Reconstruction MSE of inverted model: {loss}")
    shape = get_shape(dataset_name)
    plt.imshow(np.reshape(model_inverse(y[:1]), shape))
    plt.show()

    return model_inverse


def apply_pid(pid, y_perturbed, setpoint, t, model, layer_name):
    control_variable = pid.update(y_perturbed, setpoint, t)
    layer = model.get_layer(layer_name)
    w, b = layer.get_weights()
    b_new = b + control_variable
    layer.set_weights([w, b_new])


class Perturbation(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, x, scale):
        return scale * x


class PoissonNoisePerturbation(Perturbation):
    def __init__(self, rng=None, normalize=False, static=False):
        super().__init__('poisson_noise')
        self.rng = np.random.default_rng() if rng is None else rng
        self.normalize = normalize
        self.static = static

    def apply(self, x, scale):
        lam = x[0]  # Use pixel values at first time step (pixels are static).
        if self.static:
            noise = self.rng.poisson(lam, (1,) + (x.shape[1:]))
            noise = np.repeat(noise, len(x), 0)
        else:
            noise = self.rng.poisson(lam, x.shape)
        x_ = x + scale * noise
        return normalize_array(x_) if self.normalize else x_


class GaussianNoisePerturbation(Perturbation):
    def __init__(self, rng=None, normalize=False, static=False):
        super().__init__('gaussian_noise')
        self.rng = np.random.default_rng() if rng is None else rng
        self.normalize = normalize
        self.static = static

    def apply(self, x, scale):
        if self.static:
            noise = self.rng.standard_normal((1,) + x.shape[1:])
            noise = np.repeat(noise, len(x), 0)
        else:
            noise = self.rng.standard_normal(x.shape)
        noise[noise < 0] = 0
        x_ = x + scale * noise
        return normalize_array(x_) if self.normalize else x_


class GaussianNoiseWithDriftPerturbation(GaussianNoisePerturbation):
    def __init__(self, rng=None, normalize=False, static=False, drift=0,
                 homogeneous_drift=True):
        super().__init__(rng, normalize, static)
        self.drift = drift
        self.homogeneous_drift = homogeneous_drift

    def apply_drift(self, x, scale):
        if self.static:
            drift = np.zeros_like(x)
            if self.homogeneous_drift:  # Add drift impulse at first time step.
                drift[0] = self.drift
            else:  # Add random drift impulse at first time step.
                drift[0] = self.drift * self.rng.standard_normal((1,) +
                                                                 x.shape[1:])
        else:
            if self.homogeneous_drift:
                drift = self.drift  # Add same amount of drift at each step.
            else:  # Add same random amount of drift at each time step.
                drift = self.drift * self.rng.standard_normal((1,) +
                                                              x.shape[1:])
                drift = np.repeat(drift, len(x), 0)
        x = x + scale * drift
        return x

    def apply(self, x, scale):
        x = self.apply_drift(x, scale)
        return super().apply(x, scale)


def normalize_array(x):
    return x / np.max(x)


class ContrastPerturbation(Perturbation):
    def __init__(self):
        super().__init__('contrast')

    @staticmethod
    def contrast(c, f):
        return 0.5 + f * (c - 0.5)

    def apply(self, x, scale):
        factor = (1 - scale) / (1 + scale * 255 / 259)
        return self.contrast(x, factor)


class BrownianPerturbation(Perturbation):
    def __init__(self, delta, drift=0, num_timesteps=1, rng=None):
        super(BrownianPerturbation, self).__init__('brownian')
        self.delta = delta
        self.drift = drift
        self.num_timesteps = num_timesteps
        self.rng = rng

    def apply(self, x, scale):
        return brownian(x, self.num_timesteps, scale * self.delta, scale,
                        scale * self.drift, rng=self.rng)


def apply_perturbation(x, num_timesteps, perturbation, scale, **kwargs):

    if num_timesteps > 1:
        assert x.shape[0] == 1, "Axis 0 of image sample must have size 1."
        x = np.repeat(x, num_timesteps, 0)
    return perturbation.apply(x, scale, **kwargs)


class ModelPerturber:
    def __init__(self, model, layer_names, perturbation, scales):
        self.model = model
        self.layer_names = layer_names
        self.perturbation = perturbation
        self.scales = scales
        self.params_perturbed = None

    def apply(self):
        self.params_perturbed = {}
        for layer_name in self.layer_names:
            layer = self.model.get_layer(layer_name)
            w, b = layer.get_weights()
            self.params_perturbed.setdefault(layer_name, {})
            for scale in self.scales:
                w_perturbed = self.perturbation.apply(w, scale)
                self.params_perturbed[layer_name][scale] = [w_perturbed, b]

    def step(self, t, scale, layer_names=None):
        layer_names = self.layer_names if layer_names is None else layer_names
        for layer_name in layer_names:
            layer = self.model.get_layer(layer_name)
            w_t, b = self.params_perturbed[layer_name][scale]
            # Set weights with current time slice.
            layer.set_weights([w_t[t], b])


def step(model, x, data, label, scale, perturbation_name, layer_name,
         submodel=None, pid=None, setpoint=None, t=None):

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
        apply_pid(pid, y_perturbed, setpoint, t, model, layer_name)

    data['process_values'].append(y_perturbed)
    data['classifications'].append(label_predicted == label)
    data['use_pid'].append(use_pid)
    data['perturbation_scale'].append(scale)
    data['perturbation_type'].append(perturbation_name)
    data['layer_name'].append(layer_name)


def plot_accuracy_vs_perturbation_scale(path, data, name, window_ratio,
                                        num_timesteps, num_samples,
                                        output_format='.png'):
    groups = ['perturbation_scale', 'layer_name', 'use_pid']

    # Optionally use only the last few timesteps.
    assert 0 < window_ratio <= 1
    window_start = max(int(num_timesteps * window_ratio), 1)
    idxs = np.arange(-window_start, 0)
    rows = list(np.concatenate([(i + 1) * num_timesteps + idxs
                                for i in range(num_samples)]))
    data_reduced = data.loc[data['perturbation_type'] == name]
    data_reduced = data_reduced.groupby(groups).nth(rows)

    sb.lineplot(x='perturbation_scale', y='classifications', hue='layer_name',
                style='use_pid', style_order=[True, False], palette='rocket',
                markers=True, dashes=True,
                hue_order=['layer_1', 'layer_2', 'layer_3'], data=data_reduced)
    plt.ylim(0, 1)
    plt.xlabel("Perturbation level")
    plt.ylabel("Accuracy")
    plt.title(f"Perturbation: {name}")
    plt.savefig(os.path.join(
        path, f'accuracy_vs_perturbation_scale_{name}{output_format}'))
    plt.show()


def plot_process_values(path, data, scale, perturbation_type, num_timesteps,
                        output_format, use_pid, labels=None):
    y = data.loc[(data['perturbation_type'] == perturbation_type) &
                 (data['perturbation_scale'] == scale) &
                 (data['use_pid'] == use_pid),
                 'process_values']
    plt.figure(figsize=(20, 5))
    sb.heatmap(np.stack(y).transpose(), cmap='Blues')
    if labels is not None:
        lt = np.repeat(labels, num_timesteps)
        sb.scatterplot(x=np.arange(len(lt)) + 0.5, y=lt + 0.5, marker='_',
                       color='k')
    plt.title("Perturbation {} @ scale {:.2f} w/{} pid".format(
        perturbation_type, scale, '' if use_pid else 'o'))
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.savefig(os.path.join(path, 'process_values_{}_{}{}{}'.format(
        perturbation_type, scale, '_pid' if use_pid else '', output_format)))
    plt.show()


def get_setpoints(model, x, submodel=None):
    y = model(x)
    if submodel is not None:
        return submodel(y)
    return y


def save_output(data, path):
    pd.DataFrame(data).to_hdf(path, 'data', mode='w')


def plot_results(path: str,
                 path_data: str,
                 perturbations: List[dict],
                 num_timesteps: int,
                 num_samples: int,
                 labels: np.array = None,
                 plot_process_variable: bool = False,
                 output_format: str = '.png',
                 window_ratio: float = 1.0):

    data = pd.DataFrame(pd.read_hdf(path_data, 'data'))

    for perturbation_dict in perturbations:
        perturbation = perturbation_dict['function']
        perturbation_type = perturbation.name
        plot_accuracy_vs_perturbation_scale(path, data, perturbation_type,
                                            window_ratio, num_timesteps,
                                            num_samples, output_format)

        if plot_process_variable:
            scales = perturbation_dict['scales']
            for scale in scales:
                plot_process_values(path, data, scale, perturbation_type,
                                    num_timesteps, output_format, use_pid=True,
                                    labels=labels)
                plot_process_values(path, data, scale, perturbation_type,
                                    num_timesteps, output_format,
                                    use_pid=False, labels=labels)


def plot_input(path, x, dataset_name, perturbation_list, output_format):
    for perturbation_dict in perturbation_list:
        perturbation = perturbation_dict['function']
        scales = perturbation_dict['scales']
        name = perturbation.name
        kwargs = perturbation_dict['kwargs']
        shape = get_shape(dataset_name)
        cmap = 'Greys' if dataset_name == 'mnist' else None
        x = np.reshape(x, shape)
        x = np.expand_dims(x, 0)
        for scale in scales:
            x_ = apply_perturbation(x, 1, perturbation, scale, **kwargs)
            plt.imshow(x_[0], vmin=0, vmax=1, cmap=cmap)
            plt.title(f"Perturbation: {name} @ {scale:.1%}")
            plt.savefig(os.path.join(path,
                                     f'input_{name}_{scale}' + output_format))


def get_shape(dataset_name):
    if dataset_name == 'mnist':
        return 28, 28
    if dataset_name == 'cifar10':
        return 32, 32, 3
    raise NotImplementedError


def make_config():
    model_type = 'mlp'  # 'cnn', 'mlp', 'dae'
    dataset_name = 'mnist'  # 'mnist', 'cifar10'
    relu_slope = 0.1
    growth_factor = 1.05

    path_base = '/home/rbodo/Data/neural_control'
    path_wd = os.path.join(path_base, 'log', 'tmp', str(time.time()))
    path_plots = os.path.join(path_wd, 'plots')
    path_data = os.path.join(path_wd, 'data')
    path_models = os.path.join(path_base, 'models')
    path_model = os.path.join(path_models,
                              f'pid_{dataset_name}_{model_type}.h5')

    k_p = 0.5
    k_i = 0  # Even k_i=0.1 reduces accuracy.
    k_d = 0.5

    reset_weights_between_samples = False
    plot_process_variable = False
    plot_fileformat = '.png'
    num_timesteps = 10
    num_to_test = 20
    window_ratio = 0.2

    layer_names = ['layer_1', 'layer_2', 'layer_3']

    seed = 123

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': path_base,
                       'path_wd': path_wd,
                       'path_models': path_models,
                       'path_model': path_model,
                       'path_plots': path_plots,
                       'path_data': path_data}
    config['model'] = {'model_type': model_type,
                       'relu_slope': relu_slope,
                       'growth_factor': growth_factor}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': str(num_to_test),
                            'reset_weights_between_samples':
                                str(reset_weights_between_samples),
                            'seed': seed,
                            'window_ratio': window_ratio}
    config['controller'] = {'k_p': str(k_p),
                            'k_i': str(k_i),
                            'k_d': str(k_d),
                            'layer_names': layer_names}
    config['dataset'] = {'name': dataset_name}
    config['output'] = {'plot_process_variable': str(plot_process_variable),
                        'plot_fileformat': plot_fileformat}
    config['perturbations'] = {}

    return config


def save_config(path, config):
    with open(os.path.join(path, 'config.ini'), 'w') as configfile:
        config.write(configfile)


def main():

    config = make_config()

    path_wd = config['paths']['path_wd']
    path_plots = config['paths']['path_plots']
    path_models = config['paths']['path_models']
    path_model = config['paths']['path_model']
    path_data = config['paths']['path_data']

    model_type = config['model']['model_type']
    relu_slope = config.getfloat('model', 'relu_slope')
    growth_factor = config.getfloat('model', 'growth_factor')

    num_timesteps = config.getint('experiment', 'num_timesteps')
    num_to_test = config.getint('experiment', 'num_to_test')
    reset_weights_between_samples = config.getboolean(
        'experiment', 'reset_weights_between_samples')
    seed = config.getint('experiment', 'seed')
    window_ratio = config.getfloat('experiment', 'window_ratio')

    plot_process_variable = config.getboolean('output',
                                              'plot_process_variable')
    plot_fileformat = config['output']['plot_fileformat']

    k_p = config.getfloat('controller', 'k_p')
    k_i = config.getfloat('controller', 'k_i')
    k_d = config.getfloat('controller', 'k_d')
    layer_names = eval(config['controller']['layer_names'])

    dataset_name = config['dataset']['name']

    rng = np.random.default_rng(seed)

    os.makedirs(path_wd)
    os.makedirs(path_plots)
    os.makedirs(path_data)
    os.makedirs(path_models, exist_ok=True)

    num_classes = 10
    flatten = model_type == 'mlp'
    x_train, y_train, x_test, y_test = get_dataset(dataset_name, num_classes,
                                                   flatten)

    offset = 0
    test_idxs = np.arange(num_to_test) + offset
    labels = np.argmax(y_test[test_idxs], -1)

    config['dataset']['test_idxs'] = str(test_idxs)

    perturbation_list = [
        {'function': PoissonNoisePerturbation(rng),
         'scales': [0, 0.1, 0.2, 0.3, 0.4, 0.7, 1],
         'kwargs': {}},
        {'function': GaussianNoisePerturbation(rng),
         'scales': [0, 0.1, 0.2, 0.3, 0.4, 0.7, 1],
         'kwargs': {}},
        {'function': ContrastPerturbation(),
         'scales': [0, 0.1, 0.2, 0.3, 0.4, 0.7, 1],
         'kwargs': {}}
    ]

    for perturbation_dict in perturbation_list:
        p = perturbation_dict.copy()
        name = p.pop('function').name
        config['perturbations'][name] = str(p)

    do_plot_input = False
    if do_plot_input:
        plot_input(path_plots, x_test[0], dataset_name, perturbation_list,
                   plot_fileformat)

    model_kwargs = {'model_type': model_type, 'model_path': path_model,
                    'relu_slope': relu_slope, 'growth_factor': growth_factor}

    model = get_model(x_train, y_train, x_test, y_test, model_kwargs)
    model_pid = get_model(x_train, y_train, x_test, y_test, model_kwargs)
    config['model']['model_config'] = str(model.get_config())

    data_dict = {'process_values': [],
                 'classifications': [],
                 'perturbation_scale': [],
                 'perturbation_type': [],
                 'use_pid': [],
                 'layer_name': []}

    for layer_name in layer_names:
        print(f"Layer: {layer_name}.")
        is_output_layer = layer_name in model.layers[-1].name
        if is_output_layer:
            # Auxiliary and inverse models not needed for output layer.
            submodel = submodel_pid = submodel_inv = None
            model_aux = model
        else:
            layer_name_activ = layer_name + '_activ'
            output_layer = model.get_layer(layer_name_activ)
            submodel = keras.models.Model(model.input, output_layer.output)
            output_layer = model_pid.get_layer(layer_name_activ)
            submodel_pid = keras.models.Model(model_pid.input,
                                              output_layer.output)
            model_aux = get_auxiliary_model(model_pid, x_test, y_test,
                                            model_kwargs, rng)
            model_inv = get_inverse_model(model_aux, x_test, model_kwargs,
                                          dataset_name)
            output_layer = model_inv.get_layer(layer_name)
            submodel_inv = keras.models.Model(model_inv.input,
                                              output_layer.output)

        setpoints = get_setpoints(model_aux, x_test[test_idxs], submodel_inv)
        for perturbation_dict in perturbation_list:
            perturbation = perturbation_dict['function']
            scales = perturbation_dict['scales']
            name = perturbation.name
            kwargs = perturbation_dict['kwargs']
            print("Perturbation: {}.".format(name))
            for scale in scales:
                print("\tScale: {:.2f}.".format(scale))
                pid = PID(k_p=k_p, k_i=k_i, k_d=k_d)
                for sample_idx_rel, sample_idx_abs in enumerate(test_idxs):
                    label = labels[sample_idx_rel]
                    x = x_test[sample_idx_abs: sample_idx_abs + 1]
                    x_perturbed_array = apply_perturbation(
                        x, num_timesteps, perturbation, scale, **kwargs)
                    # x_perturbed_array *= 0#
                    for t, x_perturbed in enumerate(x_perturbed_array):
                        t_abs = t + num_timesteps * sample_idx_rel
                        step(model, x_perturbed, data_dict, label, scale, name,
                             layer_name, submodel)
                        step(model_pid, x_perturbed, data_dict, label, scale,
                             name, layer_name, submodel_pid, pid,
                             setpoints[sample_idx_rel], t_abs)
                    # Resetting biases between samples is expected to help but
                    # in fact reduced accuracy a bit. Not tested on many
                    # samples yet. Should be replaced by a decay of the biases
                    # back to initial values.
                    if reset_weights_between_samples:
                        model_pid.set_weights(model.get_weights())
                model_pid.set_weights(model.get_weights())

    filepath_data_out = os.path.join(path_data, 'data.hdf')

    save_output(data_dict, filepath_data_out)

    plot_results(path_plots, filepath_data_out, perturbation_list,
                 num_timesteps, num_to_test, labels, plot_process_variable,
                 plot_fileformat, window_ratio)

    save_config(path_wd, config)


if __name__ == '__main__':
    main()
