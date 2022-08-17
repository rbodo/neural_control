import logging
import os
import sys
import time
from itertools import product

import mlflow
import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import autograd
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

from src.double_integrator import configs
from src.double_integrator.control_systems import RNNModel
from src.double_integrator.utils import apply_config
from src.double_integrator.plotting import plot_training_curve, float2str
from src.double_integrator.train_rnn import (get_data_loaders, get_model_name,
                                             evaluate)


def train_single(config, plot_control=True, plot_loss=True, save_model=True):
    context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()

    num_hidden = config.model.NUM_HIDDEN
    num_layers = config.model.NUM_LAYERS
    num_inputs = 1
    num_outputs = 1
    activation = config.model.ACTIVATION
    path_data = config.paths.FILEPATH_INPUT_DATA
    batch_size = config.training.BATCH_SIZE
    lr = config.training.LEARNING_RATE
    num_epochs = config.training.NUM_EPOCHS
    optimizer = config.training.OPTIMIZER

    data = pd.read_pickle(path_data)

    test_data_loader, train_data_loader = get_data_loaders(data, config,
                                                           'observations')

    model = RNNModel(num_hidden, num_layers, num_outputs, num_inputs,
                     activation)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)

    loss_function = mx.gluon.loss.L2Loss()  # NRMSD()
    trainer = mx.gluon.Trainer(model.collect_params(), optimizer,
                               {'learning_rate': lr,
                                'rescale_grad': 1 / batch_size})

    hidden_init = mx.nd.zeros((num_layers, batch_size, model.num_hidden),
                              ctx=context)
    training_losses = []
    validation_losses = []
    for epoch in trange(num_epochs, desc='epoch'):
        training_loss = 0
        label = None
        output = None
        tic = time.time()
        for batch_idx, (data, label) in tenumerate(train_data_loader,
                                                   desc='batch', leave=False):
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

            training_loss += loss.mean().asscalar()

        training_loss_mean = training_loss / len(train_data_loader)
        training_losses.append(training_loss_mean)

        if plot_control:
            fig = plt.figure()
            plt.plot(output[0, 0].asnumpy(), label='RNN')
            plt.plot(label[0, 0].asnumpy(), label='LQR')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Control')
            # plt.show()
            mlflow.log_figure(fig, 'figures/control_{}.png'.format(epoch))

        validation_loss = evaluate(model, test_data_loader, loss_function,
                                   hidden_init, context)
        validation_loss_mean = validation_loss / len(test_data_loader)
        validation_losses.append(validation_loss_mean)
        logging.debug("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                      "".format(epoch, time.time() - tic,
                                training_loss_mean, validation_loss_mean))
        mlflow.log_metric('mean_validation_loss', validation_loss_mean, epoch)
        mlflow.log_metric('mean_training_loss', training_loss_mean, epoch)

    if plot_loss:
        path_figures = config.paths.PATH_FIGURES
        w = float2str(config.process.PROCESS_NOISES[0])
        v = float2str(config.process.OBSERVATION_NOISES[0])
        filename = f'training_curve_{w}_{v}.png'
        path_plot = os.path.join(path_figures, filename)
        plot_training_curve(training_losses, validation_losses, path_plot,
                            show=False)
        mlflow.log_figure(plt.gcf(), os.path.join('figures', filename))

    if save_model:
        model.save_parameters(config.paths.FILEPATH_MODEL)
        logging.info("Saved model to {}.".format(config.paths.FILEPATH_MODEL))
        mlflow.log_artifact(config.paths.FILEPATH_MODEL, 'models')

    return training_losses, validation_losses


def train_sweep(config):
    path, filename = os.path.split(config.paths.FILEPATH_MODEL)
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    dfs = []
    config.defrost()
    mlflow.set_tracking_uri(
        os.path.join('file:' + config.paths.BASE_PATH, 'mlruns'))
    mlflow.set_experiment('train_rnn_controller')
    mlflow.start_run(run_name='Noise sweep parent')
    for w, v in tqdm(product(process_noises, observation_noises), 'noise',
                     leave=False):
        mlflow.start_run(run_name='Noise sweep child', nested=True)
        path_model = os.path.join(path, get_model_name(filename, w, v))
        config.paths.FILEPATH_MODEL = path_model
        config.process.PROCESS_NOISES = [w]
        config.process.OBSERVATION_NOISES = [v]

        t_loss, v_loss = train_single(config, plot_control=True)

        dfs.append(pd.DataFrame({'process_noise': w,
                                 'observation_noise': v,
                                 'training_loss': t_loss,
                                 'validation_loss': v_loss}))
        mlflow.log_params({'process_noise': w, 'observation_noise': v})
        mlflow.end_run()

    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(config.paths.FILEPATH_OUTPUT_DATA)
    mlflow.log_artifact(config.paths.FILEPATH_OUTPUT_DATA)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    GPU = 2

    _config = configs.config_train_rnn_controller.get_config()

    apply_config(_config)

    train_sweep(_config)

    sys.exit()
