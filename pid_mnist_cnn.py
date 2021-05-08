import configparser
import os
import time
from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from pid import PID


def get_dataset(num_classes, use_mlp=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if use_mlp:
        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_model(model_path, x_train, y_train, x_test, y_test, use_mlp=False):
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path)
    else:
        batch_size = 128
        epochs = 10
        input_shape = x_test.shape[1:]
        num_classes = y_test.shape[-1]

        if use_mlp:
            model = keras.Sequential([
                keras.Input(input_shape),
                layers.Dense(128, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='relu')
            ])
        else:
            model = keras.Sequential([
                keras.Input(input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='relu')
            ])

        optimizer = keras.optimizers.Adam(1e-5)
        loss = 'mse'  # 'categorical_crossentropy'
        model.compile(optimizer, loss, ['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_split=0.1)
        model.save(model_path)

    model.summary()

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", score[1])

    return model


def apply_pid(pid, y_perturbed, setpoint, t, model):
    control_variable = pid.update(y_perturbed, setpoint, t)
    w, b = model.layers[-1].get_weights()
    b_new = b + control_variable
    model.layers[-1].set_weights([w, b_new])


class Perturbation(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, x, scale):
        return scale * x


class PoissonNoisePerturbation(Perturbation):
    def __init__(self, seed=123):
        super().__init__('poisson_noise')
        self.rng = np.random.default_rng(seed)

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


def step(model, x, data, label, scale, perturbation_name,
         pid=None, setpoint=None, t=None):

    y_perturbed = model.predict(np.expand_dims(x, 0))[0]
    label_predicted = np.argmax(y_perturbed)

    use_pid = pid is not None
    if use_pid:
        apply_pid(pid, y_perturbed, setpoint, t, model)

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
    plt.title(name)
    plt.savefig(os.path.join(path,
                             'accuracy_vs_perturbation_scale'+output_format))
    plt.show()


def plot_output_layer(path, data, scale, perturbation_type, labels,
                      num_timesteps, use_pid, output_format='.png'):
    y = data.loc[(data['perturbation_type'] == perturbation_type) &
                 (data['perturbation_scale'] == scale) &
                 (data['use_pid'] == use_pid),
                 'process_values']
    plt.figure(figsize=(20, 5))
    sb.heatmap(np.stack(y).transpose(), cmap='Blues')
    lt = np.repeat(labels, num_timesteps)
    sb.scatterplot(x=np.arange(len(lt)) + 0.5, y=lt + 0.5, marker='_',
                   color='k')
    plt.title("Perturbation {} @ scale {:.2f} w/{} pid".format(
        perturbation_type, scale, '' if use_pid else 'o'))
    plt.xlabel("Time")
    plt.ylabel("Class")
    plt.savefig(os.path.join(path, 'process_variable'+output_format))
    plt.show()


def main():
    path_base = '/home/rbodo/Data/neural_control'
    path_wd = os.path.join(path_base, 'log', str(time.time()))
    os.makedirs(path_wd)
    path_plots = os.path.join(path_wd, 'plots')
    os.makedirs(path_plots)
    path_data = os.path.join(path_wd, 'data')
    os.makedirs(path_data)
    path_models = os.path.join(path_base, 'models')
    os.makedirs(path_models, exist_ok=True)
    filepath_data_out = os.path.join(path_data, 'data.hdf')

    k_p = 0.5
    k_i = 0  # Even k_i=0.1 reduces accuracy.
    k_d = 0.5

    reset_weights_between_samples = False
    use_mlp = False
    plot_process_variable = False
    plot_fileformat = '.png'

    num_classes = 10
    path_model = os.path.join(path_models, 'pid_mnist_{}.h5'.format(
        'mlp' if use_mlp else 'cnn'))

    x_train, y_train, x_test, y_test = get_dataset(num_classes, use_mlp)
    model = get_model(path_model, x_train, y_train, x_test, y_test, use_mlp)
    model_pid = get_model(path_model, x_train, y_train, x_test, y_test,
                          use_mlp)

    perturbation_list = [{'function': PoissonNoisePerturbation(),
                          'scales': np.linspace(0, 1, 5),
                          'kwargs': {'static': False}},
                         # {'function': ContrastPerturbation(),
                         #  'scales': np.linspace(0, 1, 5)}
                         ]

    num_timesteps = 20
    num_to_test = 10

    data = {'process_values': [],
            'classifications': [],
            'perturbation_scale': [],
            'perturbation_type': [],
            'use_pid': []}
    offset = 0
    test_idxs = np.arange(num_to_test) + offset
    setpoints = model.predict(x_test[test_idxs])
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
                    step(model, x_perturbed, data, label, scale, name)
                    step(model_pid, x_perturbed, data, label, scale, name, pid,
                         setpoints[sample_idx_rel], t_abs)
                # Resetting biases between samples is expected to help but in
                # fact reduced accuracy a bit. Not tested on many samples yet.
                # Should be replaced by a decay of the biases back to initial
                # values.
                if reset_weights_between_samples:
                    model_pid.set_weights(model.get_weights())

    data = pd.DataFrame(data)
    data.to_hdf(filepath_data_out, 'data', mode='w')

    for perturbation_dict in perturbation_list:
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

            if plot_process_variable:
                plot_output_layer(path_plots, data, scale, perturbation_type,
                                  labels, num_timesteps, use_pid=True)
                plot_output_layer(path_plots, data, scale, perturbation_type,
                                  labels, num_timesteps, use_pid=False)
        data_acc = pd.DataFrame(data_acc)
        data_acc.to_hdf(filepath_data_out,
                        'data_acc_{}'.format(perturbation_type), mode='a')
        plot_accuracy_vs_perturbation_scale(path_plots, data_acc,
                                            perturbation_type)

    config = configparser.ConfigParser()
    config['paths'] = {'path_base': path_base,
                       'path_wd': path_wd,
                       'path_models': path_models,
                       'path_model': path_model,
                       'path_plots': path_plots,
                       'path_data': path_data}
    config['model'] = {'use_mlp': str(use_mlp),
                       'model_config': model.get_config()}
    config['experiment'] = {'num_timesteps': str(num_timesteps),
                            'num_to_test': str(num_to_test),
                            'reset_weights_between_samples':
                                str(reset_weights_between_samples)}
    config['controller'] = {'k_p': str(k_p),
                            'k_i': str(k_i),
                            'k_d': str(k_d)}
    config['dataset'] = {'name': 'MNIST',
                         'test_idxs': str(test_idxs)}
    config['output'] = {'plot_process_variable': str(plot_process_variable),
                        'plot_fileformat': plot_fileformat}
    config['perturbations'] = {}
    for perturbation_dict in perturbation_list:
        p = perturbation_dict.copy()
        name = p.pop('function').name
        config['perturbations'][name] = str(p)

    with open(os.path.join(path_wd, 'config.ini'), 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    main()
