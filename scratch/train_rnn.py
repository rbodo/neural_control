import os
import sys
import time
from itertools import product

import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import autograd
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

from src.empirical_gramians import emgr
from scratch import configs
from src.control_systems_mxnet import RnnModel
from src.plotting import plot_training_curve, float2str
from src.utils import apply_config, get_data_loaders


def get_model_name(filename, w, v):
    return f'w{w:.4f}_v{v:.4f}' + filename


def evaluate(model, test_data_loader, loss_function, hidden_init, context):
    validation_loss = 0
    for data, label in test_data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        output, hidden = model(data, hidden_init)
        output = mx.nd.moveaxis(output, 0, -1)
        loss = loss_function(output, label)
        validation_loss += loss.mean().asscalar()
    return validation_loss


class NRMSD(mx.gluon.loss.L2Loss):
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        loss = super(NRMSD, self).hybrid_forward(F, pred, label, sample_weight)
        if mx.is_np_array():
            return loss / (F.np.max(label) - F.np.min(label))
        else:
            return loss / (F.max(label) - F.min(label))


def train_single(config, verbose=True, plot_control=True, plot_loss=True,
                 save_model=True, compute_gramians=False):
    context = mx.gpu(1) if mx.context.num_gpus() > 0 else mx.cpu()

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
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # noinspection PyUnusedLocal
    def h(x, u, p, t):
        _batch_size = 1
        _num_steps = 1
        u0 = mx.nd.empty((_num_steps, _batch_size, num_inputs), ctx=context)
        u0[0, 0, :] = u
        x0 = mx.nd.empty((num_layers, _batch_size, num_hidden), ctx=context)
        x0[:, 0, :] = x
        y, x1 = model.rnn.forward(u0, [x0])
        return y, x1[0]

    def f(x, u, p, t):
        y, x1 = h(x, u, p, t)
        return x1.asnumpy()[0, 0]

    def g(x, u, p, t):
        y, x1 = h(x, u, p, t)
        z = model.decoder(y)
        return z.asnumpy()[0]

    data = pd.read_pickle(path_data)

    test_data_loader, train_data_loader = get_data_loaders(data, config,
                                                           'observations')

    model = RnnModel(num_hidden, num_layers, num_outputs, num_inputs,
                     activation)
    model.hybridize()
    model.initialize(mx.init.Xavier(), context)

    loss_function = mx.gluon.loss.L2Loss()  # NRMSD()
    trainer = mx.gluon.Trainer(model.collect_params(), optimizer,
                               {'learning_rate': lr,
                                'rescale_grad': 1 / batch_size})

    hidden_init = mx.nd.zeros((num_layers, batch_size, model.num_hidden),
                              ctx=context)
    s = [num_inputs, num_hidden, num_outputs]
    _t = [dt, T]
    training_losses = []
    validation_losses = []
    controllability_gramians = []
    observability_gramians = []
    for epoch in trange(num_epochs):
        training_loss = 0
        label = None
        output = None
        tic = time.time()
        for batch_idx, (data, label) in tenumerate(train_data_loader,
                                                   leave=False):
            if compute_gramians and batch_idx % 25 == 0:
                g_c = emgr(f, g, s, _t, 'c')
                g_o = emgr(f, g, s, _t, 'o')
                controllability_gramians.append(g_c)
                observability_gramians.append(g_o)

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
            plt.plot(output[0, 0].asnumpy(), label='RNN')
            plt.plot(label[0, 0].asnumpy(), label='LQR')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Control')
            plt.show()

        validation_loss = evaluate(model, test_data_loader, loss_function,
                                   hidden_init, context)
        validation_loss_mean = validation_loss / len(test_data_loader)
        validation_losses.append(validation_loss_mean)
        if verbose:
            print("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                  "".format(epoch, time.time() - tic,
                            training_loss_mean, validation_loss_mean))

    if plot_loss:
        path_figures = config.paths.PATH_FIGURES
        w = float2str(config.process.PROCESS_NOISES[0])
        v = float2str(config.process.OBSERVATION_NOISES[0])
        path_plot = os.path.join(path_figures, f'training_curve_{w}_{v}.png')
        plot_training_curve(training_losses, validation_losses, path_plot)

    if save_model:
        model.save_parameters(config.paths.FILEPATH_MODEL)
        print("Saved model to {}.".format(config.paths.FILEPATH_MODEL))

    if compute_gramians:
        df = pd.DataFrame({'controllability': controllability_gramians,
                           'observability': observability_gramians})
        path_out = config.paths.FILEPATH_OUTPUT_DATA
        df.to_pickle(path_out)

    return training_losses, validation_losses


def train_sweep(config):
    path, filename = os.path.split(config.paths.FILEPATH_MODEL)
    process_noises = config.process.PROCESS_NOISES
    observation_noises = config.process.OBSERVATION_NOISES

    dfs = []
    config.defrost()
    for w, v in tqdm(product(process_noises, observation_noises), leave=False):
        path_model = os.path.join(path, get_model_name(filename, w, v))
        config.paths.FILEPATH_MODEL = path_model
        config.process.PROCESS_NOISES = [w]
        config.process.OBSERVATION_NOISES = [v]

        t_loss, v_loss = train_single(config, verbose=True, plot_control=False)

        dfs.append(pd.DataFrame({'process_noise': w,
                                 'observation_noise': v,
                                 'training_loss': t_loss,
                                 'validation_loss': v_loss}))

    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(config.paths.FILEPATH_OUTPUT_DATA)


if __name__ == '__main__':
    _config = configs.config_train_rnn_small.get_config()
    # _config = configs.config_train_rnn.get_config()

    apply_config(_config)

    train_sweep(_config)

    sys.exit()
