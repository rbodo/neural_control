import os
import sys
import time
import logging

import optuna
import pandas as pd
import mxnet as mx
from mxnet import autograd
from matplotlib import pyplot as plt

from scratch import configs
from src.utils import apply_config, get_data_loaders


def create_model(trial):
    num_layers = trial.suggest_int('num_layers', 1, 2)
    num_hidden = trial.suggest_int('num_hidden', 2, 64, log=True)
    activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
    dropout = trial.suggest_float('dropout', 0, 1)
    neuron_model = trial.suggest_categorical('neuron_model',
                                             ['rnn', 'gru', 'lstm'])
    return RNNModel(num_hidden, num_layers, 1, activation, dropout,
                    neuron_model)


def create_optimizer(trial, batch_size=1):
    # We optimize over the type of optimizer to use, and over the learning
    # rate and weight decay.
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer',
                                               ['Adam', 'MomentumSGD'])

    rescale_grad = 1 / batch_size

    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_float('adam_lr', 1e-5, 1e-1, log=True)
        optimizer = mx.optimizer.Adam(adam_lr, wd=weight_decay,
                                      rescale_grad=rescale_grad)
    else:
        momentum_sgd_lr = trial.suggest_float('momentum_sgd_lr', 1e-5, 1e-1,
                                              log=True)
        optimizer = mx.optimizer.SGD(momentum_sgd_lr, wd=weight_decay,
                                     rescale_grad=rescale_grad)

    return optimizer


class RNNModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1,
                 activation='tanh', dropout=0, neuron_model='rnn',
                 **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden
        self.num_layers = num_layers

        with self.name_scope():
            if neuron_model == 'rnn':
                ModelClass = mx.gluon.rnn.RNN
                cell_kwargs = {'hidden_size': num_hidden,
                               'num_layers': num_layers,
                               'activation': activation,
                               'dropout': dropout}
            elif neuron_model == 'gru':
                ModelClass = mx.gluon.rnn.GRU
                cell_kwargs = {'hidden_size': num_hidden,
                               'num_layers': num_layers,
                               'dropout': dropout}
            elif neuron_model == 'lstm':
                ModelClass = mx.gluon.rnn.LSTM
                cell_kwargs = {'hidden_size': num_hidden,
                               'num_layers': num_layers,
                               'dropout': dropout}
            else:
                raise NotImplementedError

            self.rnn = ModelClass(**cell_kwargs)
            self.decoder = mx.gluon.nn.Dense(num_outputs, activation='tanh',
                                             in_units=num_hidden,
                                             flatten=False)

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args, **kwargs):
        output = self.rnn(x)
        decoded = self.decoder(output)
        return decoded


def objective(trial, config, verbose=0, plot_accuracy=False, save_model=False):
    num_gpus = mx.context.num_gpus()
    context = mx.gpu(1) if num_gpus > 0 else mx.cpu()
    print(context)

    path_data = config.paths.FILEPATH_INPUT_DATA
    batch_size = config.training.BATCH_SIZE
    num_epochs = config.training.NUM_EPOCHS

    data = pd.read_pickle(path_data)

    test_data_loader, train_data_loader = get_data_loaders(data, config,
                                                           'observations')

    model = create_model(trial)
    optimizer = create_optimizer(trial, batch_size)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)
    # model.load_parameters(config.paths.PATH_MODEL, ctx=context)

    loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(model.collect_params(), optimizer)

    valid_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        label = None
        tic = time.time()
        for data, label in train_data_loader:
            # Move time axis from last to first position to conform to RNN
            # convention.
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output = model(data)
                output = mx.nd.moveaxis(output, 0, -1)
                loss = loss_function(output, label)

            loss.backward()

            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()

        for data, label in test_data_loader:
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            output = model(data)
            output = mx.nd.moveaxis(output, 0, -1)
            loss = loss_function(output, label)
            valid_loss += loss.mean().asscalar()

        if plot_accuracy:
            plt.plot(output[0, 0].asnumpy(), label='RNN')
            plt.plot(label[0, 0].asnumpy(), label='LQR')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Control')
            plt.show()

        if verbose:
            print("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                  "".format(epoch, time.time() - tic,
                            train_loss / len(train_data_loader),
                            valid_loss / len(test_data_loader)))

    if save_model:
        path_model = config.paths.PATH_MODEL
        path, ext = os.path.splitext(path_model)
        path_model = path + str(trial.number) + ext
        model.save_parameters(path_model)
        print(f"Saved model to {path_model}.")

    return valid_loss / len(test_data_loader)


if __name__ == '__main__':
    optuna.logging.get_logger('optuna').addHandler(
        logging.StreamHandler(sys.stdout))
    _config = configs.config_hyperparameter_rnn_low_noise.get_config()
    # _config = configs.config_hyperparameter_rnn_high_noise.get_config()

    apply_config(_config)

    study_name = _config.paths.STUDY_NAME
    filepath_output = _config.paths.FILEPATH_OUTPUT_DATA
    storage_name = f'sqlite:///{filepath_output}'
    study = optuna.create_study(storage_name, study_name=study_name,
                                direction='minimize', load_if_exists=True)

    study.optimize(lambda t: objective(t, _config), n_trials=1000,
                   timeout=None, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    sys.exit()
