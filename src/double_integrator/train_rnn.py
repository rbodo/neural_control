import os
import sys
import time

import numpy as np
import mxnet as mx
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.double_integrator.configs.config import get_config


class RNNModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1, **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden

        with self.name_scope():
            self.rnn = mx.gluon.rnn.RNN(num_hidden, num_layers,
                                        activation='tanh')
            self.decoder = mx.gluon.nn.Dense(num_outputs, activation='tanh',
                                             in_units=num_hidden,
                                             flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        output, hidden = self.rnn(x, args[0])
        decoded = self.decoder(output)
        return decoded, hidden


def main():
    num_cpus = min(os.cpu_count() // 2, 1)
    num_gpus = mx.context.num_gpus()
    context = mx.gpu(1) if num_gpus > 0 else mx.cpu()
    print(context)

    config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_rnn.py')

    num_hidden = config.model.NUM_HIDDEN
    num_layers = config.model.NUM_LAYERS
    num_outputs = 1

    path_dataset = config.paths.PATH_TRAINING_DATA
    batch_size = config.training.BATCH_SIZE
    lr = config.training.LEARNING_RATE
    num_epochs = config.training.NUM_EPOCHS

    data = np.load(os.path.join(path_dataset, 'lqg.npz'))
    x = data['X']
    y = data['Y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    train_dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
    train_data_loader = mx.gluon.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_cpus,
        last_batch='rollover')

    test_dataset = mx.gluon.data.dataset.ArrayDataset(x_test, y_test)
    test_data_loader = mx.gluon.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_cpus,
        last_batch='discard')

    model = RNNModel(num_hidden, num_layers, num_outputs)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)
    # model.load_parameters(config.paths.PATH_MODEL, ctx=context)

    loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd',
                               {'learning_rate': lr})

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
            with mx.autograd.record():
                output, hidden = model(data, hidden_init)
                output = mx.nd.moveaxis(output, 0, -1)
                loss = loss_function(output, label)

            loss.backward()

            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()

        for data, label in test_data_loader:
            data = np.moveaxis(data, -1, 0)
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


if __name__ == '__main__':
    main()

    sys.exit()
