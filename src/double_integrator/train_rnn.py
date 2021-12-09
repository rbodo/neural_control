import os
import sys
import time

import numpy as np
import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import autograd

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


def get_trajectories(data, num_steps, use_filter=False):
    if use_filter:
        print("Using Kalman-filtered state estimates.")
        x0 = data[r'$\hat{x}$']
        x1 = data[r'$\hat{v}$']
        x0 = np.reshape(x0.to_numpy(), (-1, num_steps))
        x1 = np.reshape(x1.to_numpy(), (-1, num_steps))
        x = np.stack([x0, x1], 1)
    else:
        print("Using noisy partial observations.")
        x = data['y']
        x = np.reshape(x.to_numpy(), (-1, 1, num_steps))
    return x.astype(np.float32)


def get_control(data, num_steps):
    y = data['u']
    y = np.reshape(y.to_numpy(), (-1, 1, num_steps))
    return y.astype(np.float32)


def get_data_loaders(data, config):
    num_cpus = max(os.cpu_count() // 2, 1)
    num_steps = config.simulation.NUM_STEPS
    batch_size = config.training.BATCH_SIZE
    use_filter = '$\\hat{x}$' in data.columns
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE

    data = select_noise_subset(data, process_noise, observation_noise)

    data_train, data_test = split_train_test(data)

    x_train = get_trajectories(data_train, num_steps, use_filter)
    y_train = get_control(data_train, num_steps)
    x_test = get_trajectories(data_test, num_steps, use_filter)
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


def main(config):
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

    test_data_loader, train_data_loader = get_data_loaders(data, config)

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
        train_loss, valid_loss = 0, 0
        label = None
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

        for data, label in test_data_loader:
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            output, hidden = model(data, hidden_init)
            output = mx.nd.moveaxis(output, 0, -1)
            loss = loss_function(output, label)
            valid_loss += loss.mean().asscalar()

        plt.plot(output[0, 0].asnumpy(), label='RNN')
        plt.plot(label[0, 0].asnumpy(), label='LQR')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.show()

        print("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}.".format(
            epoch, time.time() - tic,
            train_loss / len(train_data_loader),
            valid_loss / len(test_data_loader)))

    model.save_parameters(config.paths.PATH_MODEL)
    print("Saved model to {}.".format(config.paths.PATH_MODEL))


if __name__ == '__main__':
    path_config = '/home/bodrue/PycharmProjects/neural_control/src/' \
                  'double_integrator/configs/config_rnn_lqe.py'
    _config = get_config(path_config)

    main(_config)

    sys.exit()
