import os
import sys
import time

import numpy as np
import mxnet as mx
from mxnet import autograd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from scratch import configs
from src.control_systems_mxnet import MlpModel
from src.utils import apply_config


def main():
    num_cpus = min(os.cpu_count() // 2, 1)
    num_gpus = mx.context.num_gpus()
    context = mx.gpu(1) if num_gpus > 0 else mx.cpu()
    print(context)

    config = configs.config_mlp.get_config()

    apply_config(config)

    num_hidden = config.model.NUM_HIDDEN
    num_outputs = 1

    path_data = config.paths.FILEPATH_INPUT_DATA
    batch_size = config.training.BATCH_SIZE
    lr = config.training.LEARNING_RATE
    num_epochs = config.training.NUM_EPOCHS

    data = np.load(path_data)
    # x = data['X']  # 'X' are the noise free process states.
    x = data['Y']  # 'Y' is the process output (used as input here).
    y = data['U']  # 'U' is the control signal (used as labels here).

    x = np.reshape(np.swapaxes(x, 1, 2), (-1, x.shape[1]))
    y = np.reshape(np.swapaxes(y, 1, 2), (-1, y.shape[1]))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    train_dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
    train_data_loader = mx.gluon.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_cpus,
        last_batch='rollover')

    test_dataset = mx.gluon.data.dataset.ArrayDataset(x_test, y_test)
    test_data_loader = mx.gluon.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_cpus,
        last_batch='discard')

    model = MlpModel(num_hidden, num_outputs)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)
    # model.load_parameters(config.paths.PATH_MODEL, ctx=context)

    loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd',
                               {'learning_rate': lr})

    for epoch in range(num_epochs):
        train_loss, valid_loss = 0, 0
        tic = time.time()
        for data, label in train_data_loader:
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output = model(data)
                loss = loss_function(output, label)

            loss.backward()

            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()

        for data, label in test_data_loader:
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            output = model(data)
            loss = loss_function(output, label)
            valid_loss += loss.mean().asscalar()

        num_steps = config.simulation.NUM_STEPS
        x0 = mx.nd.array(x[:num_steps], context)
        plt.plot(model(x0)[:, 0].asnumpy(), label='MLP')
        plt.plot(y[:num_steps, 0], label='LQR')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.show()

        print("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}.".format(
            epoch, time.time() - tic,
            train_loss / len(train_data_loader),
            valid_loss / len(test_data_loader)))

    model.save_parameters(config.paths.FILEPATH_MODEL)
    print("Saved model to {}.".format(config.paths.FILEPATH_MODEL))


if __name__ == '__main__':
    main()

    sys.exit()
