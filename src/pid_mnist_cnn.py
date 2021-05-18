import configparser
import os
import time
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from src.pid import PID


def relu_inverse(x, slope):
    return keras.activations.relu(x, 1/slope)


def get_dataset(num_classes, flatten=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if flatten:
        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_model(x_train, y_train, x_test, y_test, model_kwargs):
    model_path = model_kwargs['model_path']
    model_type = model_kwargs['model_type']
    relu_slope = model_kwargs['relu_slope']
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path)
    else:
        batch_size = 128
        epochs = 10
        input_shape = x_test.shape[1:]
        num_classes = y_test.shape[-1]

        if model_type == 'mlp':
            model = keras.Sequential([
                layers.InputLayer(input_shape, name='layer_0'),
                layers.Dense(784, name='layer_1'),
                layers.ReLU(negative_slope=relu_slope, name='layer_1_activ'),
                layers.Dense(784, name='layer_2'),
                layers.ReLU(negative_slope=relu_slope, name='layer_2_activ'),
                layers.Dense(num_classes, name='layer_3'),
                layers.ReLU(negative_slope=relu_slope, name='layer_3_activ')
            ])
        elif model_type == 'cnn':
            model = keras.Sequential([
                layers.InputLayer(input_shape, name='layer_0'),
                layers.Conv2D(32, (3, 3), activation='relu', name='layer_1'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, (3, 3), activation='relu', name='layer_2'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dropout(0.5),
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
        model.save(model_path)

    model.summary()

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {score[1]:.2%}")

    return model


def get_inverse_model(model, x_test, model_kwargs):
    params = model.get_weights()
    w1, b1, w2, b2, w3, b3 = params

    w3_ = np.linalg.inv(w3)
    w2_ = np.linalg.inv(w2)
    w1_ = np.linalg.inv(w1)
    b2_ = -np.dot(b3, w3_)
    b1_ = -np.dot(b2, w2_)
    b0_ = -np.dot(b1, w1_)

    relu_slope = model_kwargs['relu_slope']
    activation = partial(relu_inverse, slope=relu_slope)
    y = model.predict(x_test)
    num_classes = y.shape[-1]
    input_shape = (num_classes,)
    output_shape = x_test.shape[1:]
    model_inverse = keras.Sequential([
        layers.InputLayer(input_shape, name='layer_3'),
        layers.Activation(activation, name='layer_3_activ'),
        layers.Dense(784, name='layer_2', weights=[w3_, b2_]),
        layers.Activation(activation, name='layer_2_activ'),
        layers.Dense(784, name='layer_1', weights=[w2_, b1_]),
        layers.Activation(activation, name='layer_1_activ'),
        layers.Dense(np.prod(output_shape), name='layer_0', weights=[w1_, b0_])
    ])

    model_inverse.summary()

    model_inverse.compile('sgd', 'mse')

    loss = model_inverse.evaluate(y, x_test)
    print(f"Reconstruction MSE of inverted model: {loss}")
    plt.imshow(np.reshape(model_inverse.predict(y[:1]), (28, 28)))
    plt.show()

    return model_inverse


def get_auxiliary_model(model, x_test, y_test, model_kwargs, rng):
    params = model.get_weights()
    w1, b1, w2, b2, w3, b3 = params

    d_min, d_max = sorted(w3.shape)
    w3_ = rng.choice(w3.ravel(), (d_max, d_max))
    w3_[:, :d_min] = w3
    b3_ = np.zeros(d_max)
    b3_[:d_min] = b3

    relu_slope = model_kwargs['relu_slope']
    num_classes = y_test.shape[-1]
    input_shape = x_test.shape[1:]
    output_shape = input_shape
    model_aux = keras.Sequential([
        layers.InputLayer(input_shape, name='layer_0'),
        layers.Dense(784, name='layer_1', weights=[w1, b1]),
        layers.ReLU(negative_slope=relu_slope, name='layer_1_activ'),
        layers.Dense(784, name='layer_2', weights=[w2, b2]),
        layers.ReLU(negative_slope=relu_slope, name='layer_2_activ'),
        layers.Dense(np.prod(output_shape), name='layer_3',
                     weights=[w3_, b3_]),
        layers.ReLU(negative_slope=relu_slope, name='layer_3_activ')
    ])

    model_aux.summary()

    model_aux.compile('sgd', 'mse', ['accuracy'])

    y_test_ = np.zeros((len(y_test), d_max))
    y_test_[:, :num_classes] = y_test
    loss, acc = model_aux.evaluate(x_test, y_test_)
    print(f"Accuracy of auxiliary model: {acc:.2%}")

    return model_aux


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
    def __init__(self, rng=None):
        super().__init__('poisson_noise')
        self.rng = np.random.default_rng() if rng is None else rng

    def apply(self, x, scale, static=False):
        if static:
            noise = self.rng.poisson(1, (1,) + (x.shape[1:]))
            noise = np.repeat(noise, len(x), 0)
        else:
            noise = self.rng.poisson(1, x.shape)
        return x + scale * noise


class ContrastPerturbation(Perturbation):
    def __init__(self):
        super().__init__('contrast')

    def apply(self, x, scale):
        return scale * x


def apply_perturbation(x, num_timesteps, perturbation, scale, **kwargs):

    x_p = np.repeat(x, num_timesteps, 0)
    return perturbation.apply(x_p, scale, **kwargs)


def step(model, submodel, x, data, label, scale, perturbation_name,
         pid=None, setpoint=None, layer_name=None, t=None):

    x_ = np.expand_dims(x, 0)
    y_perturbed = submodel.predict(x_)[0]
    label_predicted = np.argmax(model.predict(x_)[0])

    use_pid = pid is not None
    if use_pid:
        apply_pid(pid, y_perturbed, setpoint, t, model, layer_name)

    data['process_values'].append(y_perturbed)
    data['classifications'].append(label_predicted == label)
    data['use_pid'].append(use_pid)
    data['perturbation_scale'].append(scale)
    data['perturbation_type'].append(perturbation_name)


def plot_accuracy_vs_perturbation_scale(path, data, name,
                                        output_format='.png'):
    sb.lineplot(x='scales', y='accuracy', hue='use_pid', data=data)
    plt.ylim(0, 1)
    plt.xlabel("Perturbation level")
    plt.title(f"Perturbation {name}")
    plt.savefig(os.path.join(path,
                             'accuracy_vs_perturbation_scale'+output_format))
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
    y = model.predict(x)
    if submodel is not None:
        return submodel.predict(y)
    return y


def save_output(data, perturbations, path):
    data = pd.DataFrame(data)
    data.to_hdf(path, 'data', mode='w')

    for perturbation_dict in perturbations:
        perturbation = perturbation_dict['function']
        scales = perturbation_dict['scales']
        perturbation_type = perturbation.name
        data_acc = {'scales': [], 'accuracy': [], 'use_pid': []}
        for scale in scales:
            mask = (data['perturbation_type'] == perturbation_type) & \
                   (data['perturbation_scale'] == scale)

            mask_pid = mask & data['use_pid']
            acc = data.loc[mask_pid, 'classifications'].mean()
            data_acc['scales'].append(scale)
            data_acc['accuracy'].append(acc)
            data_acc['use_pid'].append(True)

            mask_no_pid = mask & ~data['use_pid']
            acc = data.loc[mask_no_pid, 'classifications'].mean()
            data_acc['scales'].append(scale)
            data_acc['accuracy'].append(acc)
            data_acc['use_pid'].append(False)

        data_acc = pd.DataFrame(data_acc)
        data_acc.to_hdf(path, 'data_acc_{}'.format(perturbation_type))


def plot_results(path: str,
                 path_data: str,
                 perturbations: list[dict],
                 num_timesteps: int,
                 labels: np.array = None,
                 plot_process_variable: bool = False,
                 output_format: str = '.png'):

    data = pd.DataFrame(pd.read_hdf(path_data, 'data'))

    for perturbation_dict in perturbations:
        perturbation = perturbation_dict['function']
        perturbation_type = perturbation.name
        scales = perturbation_dict['scales']
        data_acc = pd.read_hdf(path_data,
                               'data_acc_{}'.format(perturbation_type))
        plot_accuracy_vs_perturbation_scale(path, data_acc, perturbation_type,
                                            output_format)

        if plot_process_variable:
            for scale in scales:
                plot_process_values(path, data, scale, perturbation_type,
                                    num_timesteps, output_format, use_pid=True,
                                    labels=labels)
                plot_process_values(path, data, scale, perturbation_type,
                                    num_timesteps, output_format,
                                    use_pid=False, labels=labels)


def make_config():
    model_type = 'mlp'  # 'cnn', 'mlp', 'dae'
    relu_slope = 0.1

    path_base = '/home/rbodo/Data/neural_control'
    path_wd = os.path.join(path_base, 'log', 'tmp', str(time.time()))
    path_plots = os.path.join(path_wd, 'plots')
    path_data = os.path.join(path_wd, 'data')
    path_models = os.path.join(path_base, 'models')
    path_model = os.path.join(path_models,
                              'pid_mnist_{}.h5'.format(model_type))

    k_p = 0.5
    k_i = 0  # Even k_i=0.1 reduces accuracy.
    k_d = 0.5

    reset_weights_between_samples = False
    plot_process_variable = False
    plot_fileformat = '.png'
    num_timesteps = 20
    num_to_test = 10

    layer_name = 'layer_2'

    seed = 123

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': path_base,
                       'path_wd': path_wd,
                       'path_models': path_models,
                       'path_model': path_model,
                       'path_plots': path_plots,
                       'path_data': path_data}
    config['model'] = {'model_type': model_type,
                       'relu_slope': relu_slope}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': str(num_to_test),
                            'reset_weights_between_samples':
                                str(reset_weights_between_samples),
                            'seed': seed}
    config['controller'] = {'k_p': str(k_p),
                            'k_i': str(k_i),
                            'k_d': str(k_d),
                            'layer_name': layer_name}
    config['dataset'] = {'name': 'MNIST'}
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

    num_timesteps = config.getint('experiment', 'num_timesteps')
    num_to_test = config.getint('experiment', 'num_to_test')
    reset_weights_between_samples = config.getboolean(
        'experiment', 'reset_weights_between_samples')
    seed = config.getint('experiment', 'seed')

    plot_process_variable = config.getboolean('output',
                                              'plot_process_variable')
    plot_fileformat = config['output']['plot_fileformat']

    k_p = config.getfloat('controller', 'k_p')
    k_i = config.getfloat('controller', 'k_i')
    k_d = config.getfloat('controller', 'k_d')
    layer_name = config['controller']['layer_name']

    rng = np.random.default_rng(seed)

    os.makedirs(path_wd)
    os.makedirs(path_plots)
    os.makedirs(path_data)
    os.makedirs(path_models, exist_ok=True)

    num_classes = 10
    flatten = model_type == 'mlp'
    x_train, y_train, x_test, y_test = get_dataset(num_classes, flatten)
    model_kwargs = {'model_type': model_type, 'model_path': path_model,
                    'relu_slope': relu_slope}
    model = get_model(x_train, y_train, x_test, y_test, model_kwargs)
    model_pid = get_model(x_train, y_train, x_test, y_test, model_kwargs)
    model_aux = get_auxiliary_model(model_pid, x_test, y_test, model_kwargs,
                                    rng)
    model_inv = get_inverse_model(model_aux, x_test, model_kwargs)

    layer_name_activ = layer_name + '_activ'
    output_layer = model.get_layer(layer_name_activ)
    submodel = keras.models.Model(model.input, output_layer.output)
    output_layer = model_pid.get_layer(layer_name_activ)
    submodel_pid = keras.models.Model(model_pid.input, output_layer.output)
    output_layer = model_inv.get_layer(layer_name)
    submodel_inv = keras.models.Model(model_inv.input, output_layer.output)

    config['model']['model_config'] = str(model.get_config())

    perturbation_list = [{'function': PoissonNoisePerturbation(rng),
                          'scales': np.linspace(0, 1, 5),
                          'kwargs': {'static': False}},
                         # {'function': ContrastPerturbation(),
                         #  'scales': np.linspace(0, 1, 5)}
                         ]

    for perturbation_dict in perturbation_list:
        p = perturbation_dict.copy()
        name = p.pop('function').name
        config['perturbations'][name] = str(p)

    data_dict = {'process_values': [],
                 'classifications': [],
                 'perturbation_scale': [],
                 'perturbation_type': [],
                 'use_pid': []}

    offset = 0
    test_idxs = np.arange(num_to_test) + offset

    config['dataset']['test_idxs'] = str(test_idxs)

    setpoints = get_setpoints(model_aux, x_test[test_idxs], submodel_inv)
    labels = np.argmax(y_test[test_idxs], -1)
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
                for t, x_perturbed in enumerate(x_perturbed_array):
                    t_abs = t + num_timesteps * sample_idx_rel
                    step(model, submodel, x_perturbed, data_dict, label, scale,
                         name)
                    step(model_pid, submodel_pid, x_perturbed, data_dict,
                         label, scale, name, pid, setpoints[sample_idx_rel],
                         layer_name, t_abs)
                # Resetting biases between samples is expected to help but in
                # fact reduced accuracy a bit. Not tested on many samples yet.
                # Should be replaced by a decay of the biases back to initial
                # values.
                if reset_weights_between_samples:
                    model_pid.set_weights(model.get_weights())

    filepath_data_out = os.path.join(path_data, 'data.hdf')

    save_output(data_dict, perturbation_list, filepath_data_out)

    plot_results(path_plots, filepath_data_out, perturbation_list,
                 num_timesteps, labels, plot_process_variable, plot_fileformat)

    config = make_config()

    save_config(path_wd, config)


if __name__ == '__main__':
    main()
