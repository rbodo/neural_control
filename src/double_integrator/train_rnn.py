import os
import sys
import time
from itertools import product

import numpy as np
import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import autograd
from tqdm import tqdm

from src.double_integrator.configs.config import get_config
from src.double_integrator.utils import split_train_test, select_noise_subset


class RNNModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1,
                 activation='relu', **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden
        self.num_layers = num_layers

        with self.name_scope():
            self.rnn = mx.gluon.rnn.RNN(num_hidden, num_layers, activation)
            self.decoder = mx.gluon.nn.Dense(num_outputs, activation='tanh',
                                             in_units=num_hidden,
                                             flatten=False)

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args, **kwargs):
        output, hidden = self.rnn(x, args[0])
        decoded = self.decoder(output)
        return decoded, hidden


def get_model_name(filename, w, v):
    return f'w{w:.4f}_v{v:.4f}' + filename


def get_trajectories(data, num_steps, variable: str):
    if variable == 'estimates':
        print("Using Kalman-filtered state estimates.")
        x0 = data[r'$\hat{x}$']
        x1 = data[r'$\hat{v}$']
        x0 = np.reshape(x0.to_numpy(), (-1, num_steps))
        x1 = np.reshape(x1.to_numpy(), (-1, num_steps))
        x = np.stack([x0, x1], 1)
    elif variable == 'observations':
        print("Using noisy partial observations.")
        x = data['y']
        x = np.reshape(x.to_numpy(), (-1, 1, num_steps))
    elif variable == 'states':
        print("Using states.")
        x0 = data['x']
        x1 = data['v']
        x0 = np.reshape(x0.to_numpy(), (-1, num_steps))
        x1 = np.reshape(x1.to_numpy(), (-1, num_steps))
        x = np.stack([x0, x1], 1)
    else:
        raise NotImplementedError
    return x.astype(np.float32)


def get_control(data, num_steps):
    y = data['u']
    y = np.reshape(y.to_numpy(), (-1, 1, num_steps))
    return y.astype(np.float32)


def get_data_loaders(data, config, variable):
    num_cpus = max(os.cpu_count() // 2, 1)
    num_steps = config.simulation.NUM_STEPS
    batch_size = config.training.BATCH_SIZE
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]

    data = select_noise_subset(data, process_noise, observation_noise)

    data_train, data_test = split_train_test(data)

    x_train = get_trajectories(data_train, num_steps, variable)
    y_train = get_control(data_train, num_steps)
    x_test = get_trajectories(data_test, num_steps, variable)
    y_test = get_control(data_test, num_steps)

    train_dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
    train_data_loader = mx.gluon.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_cpus,
        last_batch='rollover')
    test_dataset = mx.gluon.data.dataset.ArrayDataset(x_test, y_test)
    test_data_loader = mx.gluon.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_cpus,
        last_batch='discard')

    return test_data_loader, train_data_loader


def evaluate(model, test_data_loader, loss_function, hidden_init, context):
    valid_loss = 0
    for data, label in test_data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        output, hidden = model(data, hidden_init)
        output = mx.nd.moveaxis(output, 0, -1)
        loss = loss_function(output, label)
        valid_loss += loss.mean().asscalar()
    return valid_loss


def train_single(config, verbose=True, plot_control=True, save_model=True):
    context = mx.gpu(1) if mx.context.num_gpus() > 0 else mx.cpu()

    num_hidden = config.model.NUM_HIDDEN
    num_layers = config.model.NUM_LAYERS
    num_outputs = 1
    activation = config.model.ACTIVATION
    path_dataset = config.paths.PATH_TRAINING_DATA
    batch_size = config.training.BATCH_SIZE
    lr = config.training.LEARNING_RATE
    num_epochs = config.training.NUM_EPOCHS
    optimizer = config.training.OPTIMIZER

    data = pd.read_pickle(path_dataset)

    test_data_loader, train_data_loader = get_data_loaders(data, config,
                                                           'observations')

    model = RNNModel(num_hidden, num_layers, num_outputs, activation)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)
    # model.load_parameters(config.paths.PATH_MODEL, ctx=context)

    loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(model.collect_params(), optimizer,
                               {'learning_rate': lr,
                                'rescale_grad': 1 / batch_size})

    hidden_init = mx.nd.zeros((num_layers, batch_size, model.num_hidden),
                              ctx=context)
    for epoch in range(num_epochs):
        train_loss = 0
        label = None
        output = None
        tic = time.time()
        for data, label in train_data_loader:
            # Move time axis from last to first position to conform to RNN
            # convention.
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output, hidden = model(data, hidden_init)
                output = mx.nd.moveaxis(output, 0, -1)
                loss = loss_function(output, label)

            loss.backward()

            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()

        if plot_control:
            plt.plot(output[0, 0].asnumpy(), label='RNN')
            plt.plot(label[0, 0].asnumpy(), label='LQR')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Control')
            plt.show()

        if verbose:
            valid_loss = evaluate(model, test_data_loader, loss_function,
                                  hidden_init, context)
            print("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                  "".format(epoch, time.time() - tic,
                            train_loss / len(train_data_loader),
                            valid_loss / len(test_data_loader)))

    if save_model:
        model.save_parameters(config.paths.PATH_MODEL)
        print("Saved model to {}.".format(config.paths.PATH_MODEL))


def train_sweep(config):
    path, filename = os.path.split(config.paths.PATH_MODEL)
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    config.defrost()
    for w, v in tqdm(product(process_noises, observation_noises)):
        path_model = os.path.join(path, get_model_name(filename, w, v))
        config.paths.PATH_MODEL = path_model
        config.process.PROCESS_NOISES = [w]
        config.process.OBSERVATION_NOISES = [v]

        train_single(config, verbose=True, plot_control=False)


if __name__ == '__main__':
    path_config = '/home/bodrue/PycharmProjects/neural_control/src/' \
                  'double_integrator/configs/config_rnn.py'
    _config = get_config(path_config)

    train_sweep(_config)

    sys.exit()
