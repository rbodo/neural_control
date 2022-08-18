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
from src.double_integrator.plotting import plot_training_curve
from src.double_integrator.utils import get_artifact_path, get_data_loaders


class CombinedRNN(mx.gluon.HybridBlock):

    def __init__(self, model: RNNModel, controller: RNNModel, **kwargs):

        super().__init__(**kwargs)

        self.model = model
        self.controller = controller

    def hybrid_forward(self, F, x, *args):
        model_hidden, controller_hidden = args
        model_outputs = []
        for model_input in x:
            if F is mx.ndarray:
                model_input = F.expand_dims(model_input, 0)
            controller_output, controller_hidden = self.controller(
                model_hidden[0], controller_hidden)
            model_output, model_hidden = self.model(
                model_input, [model_hidden[0] + controller_output])
            model_outputs.append(model_output)
        return F.concat(*model_outputs, dim=0)


def evaluate(model, test_data_loader, loss_function, init, context):
    validation_loss = 0
    for data, label in test_data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        output = model(data, *init)
        output = mx.nd.moveaxis(output, 0, -1)
        loss = loss_function(output, label)
        validation_loss += loss.mean().asscalar()
    return validation_loss


def train_single(config, plot_control=True, plot_loss=True, save_model=True):
    context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()
    os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'

    model_num_hidden = config.model.NUM_HIDDEN
    model_num_layers = config.model.NUM_LAYERS
    controller_num_hidden = 40
    controller_num_layers = 1
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

    model = RNNModel(model_num_hidden, model_num_layers, num_outputs,
                     num_inputs, activation, prefix='model_')

    controller = RNNModel(controller_num_hidden, controller_num_layers,
                          model_num_hidden, model_num_hidden, activation,
                          prefix='controller_')

    combined = CombinedRNN(model, controller)
    combined.initialize(mx.init.Xavier(), context)
    combined.hybridize(active=True, static_alloc=True, static_shape=True)
    # controller.collect_params().setattr('grad_req', 'null')

    loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(combined.collect_params(), optimizer,
                               {'learning_rate': lr,
                                'rescale_grad': 1 / batch_size})

    model_hidden = model.rnn.begin_state(batch_size, ctx=context)
    controller_hidden = controller.rnn.begin_state(batch_size, ctx=context)
    init = (model_hidden, controller_hidden)
    training_losses = []
    validation_losses = []
    for epoch in trange(num_epochs, desc='epoch'):
        training_loss = 0
        label = None
        tic = time.time()
        for batch_idx, (data, label) in tenumerate(train_data_loader,
                                                   desc='batch', leave=False):
            # Move time axis from last to first position to conform to RNN
            # convention.
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output = combined(data, *init)
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
            mlflow.log_figure(fig, f'figures/control_{epoch}.png')
            plt.close(fig)

        validation_loss = evaluate(combined, test_data_loader, loss_function,
                                   init, context)
        validation_loss_mean = validation_loss / len(test_data_loader)
        validation_losses.append(validation_loss_mean)
        logging.debug("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                      "".format(epoch, time.time() - tic,
                                training_loss_mean, validation_loss_mean))
        mlflow.log_metric('mean_validation_loss', validation_loss_mean, epoch)
        mlflow.log_metric('mean_training_loss', training_loss_mean, epoch)

    if plot_loss:
        f = plot_training_curve(training_losses, validation_losses, show=False)
        mlflow.log_figure(f, os.path.join('figures', 'training_curve.png'))

    if save_model:
        path_model = get_artifact_path('models/rnn.params')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        combined.save_parameters(path_model)
        logging.info(f"Saved model to {path_model}.")

    return training_losses, validation_losses


def train_sweep(config):
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    dfs = []
    config.defrost()
    mlflow.set_tracking_uri(
        os.path.join('file:' + config.paths.BASE_PATH, 'mlruns'))
    mlflow.set_experiment('train_rnn_controller')
    mlflow.start_run(run_name='Noise sweep parent')
    with open(get_artifact_path('config.txt'), 'w') as f:
        f.write(_config.dump())
    for w, v in tqdm(product(process_noises, observation_noises), 'noise',
                     leave=False):
        mlflow.start_run(run_name='Noise sweep child', nested=True)
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
    df.to_pickle(get_artifact_path('output.pkl'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    GPU = 2

    _config = configs.config_train_rnn_controller.get_config()

    train_sweep(_config)

    sys.exit()
