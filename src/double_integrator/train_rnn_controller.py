import logging
import os
import sys
import time

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import autograd
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

from src.double_integrator import configs
from src.double_integrator.control_systems import RNNModel
from src.double_integrator.emgr import emgr
from src.double_integrator.utils import get_artifact_path, get_data_loaders
from src.ff_pid.brownian import brownian


def plot_control_output(output, label):
    plt.close()
    plt.plot(output[0, 0].asnumpy(), label='RNN')
    plt.plot(label[0, 0].asnumpy(), label='LQR')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Control')
    return plt.gcf()


def plot_weight_histogram(model):
    plt.close()

    w = model.controller.rnn.l0_i2h_weight.data().asnumpy()
    r = np.count_nonzero(w) / w.size
    mlflow.log_metric('observability', r)
    plt.hist(np.ravel(w), 100, histtype='step', log=True,
             label=f'observability: {r:.2%}', align='left')

    w = model.controller.decoder.weight.data().asnumpy()
    r = np.count_nonzero(w) / w.size
    mlflow.log_metric('controllability', r)
    plt.hist(np.ravel(w), 100, histtype='step', log=True,
             label=f'controllability: {r:.2%}', align='right')

    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.legend()
    return plt.gcf()


class ControlledNeuralSystem(mx.gluon.HybridBlock):

    def __init__(self, neuralsystem: RNNModel, controller: RNNModel, **kwargs):
        super().__init__(**kwargs)
        self.neuralsystem = neuralsystem
        self.controller = controller

    def hybrid_forward(self, F, x, *args):
        neuralsystem_hidden, controller_hidden = args
        neuralsystem_outputs = []
        for neuralsystem_input in x:
            if F is mx.ndarray:
                neuralsystem_input = F.expand_dims(neuralsystem_input, 0)
            controller_output, controller_hidden = self.controller(
                neuralsystem_hidden[0], controller_hidden)
            neuralsystem_output, neuralsystem_hidden = self.neuralsystem(
                neuralsystem_input, [neuralsystem_hidden[0]+controller_output])
            neuralsystem_outputs.append(neuralsystem_output)
        return F.concat(*neuralsystem_outputs, dim=0)

    def begin_state(self, batch_size, ctx):
        return (self.neuralsystem.rnn.begin_state(batch_size, ctx=ctx),
                self.controller.rnn.begin_state(batch_size, ctx=ctx))

    def apply_drift(self, where, dt, delta, drift, rng):
        if where in (None, 'None', 'none', ''):
            return
        elif where == 'sensor':
            parameters = self.neuralsystem.rnn.l0_i2h_weight
            scale = 1
        elif where == 'processor':
            parameters = self.neuralsystem.rnn.l0_h2h_weight
            scale = 1e-2
        elif where == 'actuator':
            parameters = self.neuralsystem.decoder.weight
            scale = 1e-3
        else:
            raise NotImplementedError
        shape = parameters.shape
        w = np.ravel(parameters.data().asnumpy())
        w = brownian(w, 1, dt, delta, scale * drift, None, rng)[0]
        parameters.data()[:] = np.reshape(w, shape)

    def get_reg_weights(self):
        return [self.controller.decoder.weight,
                self.controller.rnn.l0_i2h_weight]

    def sparsify(self, atol):
        weight_list = self.get_reg_weights()
        for weights in weight_list:
            idxs = np.nonzero(weights.data().abs().asnumpy() < atol)
            if len(idxs[0]):
                weights.data()[idxs] = 0


class L2L1(mx.gluon.loss.L2Loss):
    def __init__(self, lambda_: float, context, **kwargs):
        super().__init__(**kwargs)
        self.l1 = mx.gluon.loss.L1Loss(lambda_)
        self.context = context

    def hybrid_forward(self, F, pred, label, weight_list: list = None,
                       sample_weight=None):
        l2 = super().hybrid_forward(F, pred, label, sample_weight)
        if weight_list is not None:
            for weights in weight_list:
                l2 = l2 + self.l1(
                    weights.data(), F.zeros(weights.shape,
                                            ctx=self.context)).mean()
        return l2


def evaluate(model, test_data_loader, init, context):
    loss_function = mx.gluon.loss.L2Loss()
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


def get_model(config, context, freeze_neuralsystem, freeze_controller,
              load_weights_from: str = None):
    neuralsystem_num_hidden = config.model.NUM_HIDDEN_NEURALSYSTEM
    neuralsystem_num_layers = config.model.NUM_LAYERS_NEURALSYSTEM
    controller_num_hidden = config.model.NUM_HIDDEN_CONTROLLER
    controller_num_layers = config.model.NUM_LAYERS_CONTROLLER
    activation = config.model.ACTIVATION
    num_inputs = 1
    num_outputs = 1

    neuralsystem = RNNModel(neuralsystem_num_hidden, neuralsystem_num_layers,
                            num_outputs, num_inputs, activation,
                            prefix='neuralsystem_')
    if freeze_neuralsystem:
        neuralsystem.collect_params().setattr('grad_req', 'null')

    controller = RNNModel(controller_num_hidden, controller_num_layers,
                          neuralsystem_num_hidden, neuralsystem_num_hidden,
                          activation, prefix='controller_')
    if freeze_controller:
        controller.collect_params().setattr('grad_req', 'null')

    model = ControlledNeuralSystem(neuralsystem, controller)
    model.initialize(mx.init.Xavier(), context)
    model.hybridize(active=True, static_alloc=True, static_shape=True)

    if load_weights_from is not None:
        model.load_parameters(load_weights_from, ctx=context)

    return model


# noinspection PyUnusedLocal
class Gramians:
    def __init__(self, model: ControlledNeuralSystem, context, dt, T):
        self.model = model
        self.context = context
        self.dt = dt
        self.T = T
        self.num_steps = 1
        self.batch_size = 1
        self.num_inputs = model.neuralsystem.num_hidden
        self.num_layers = model.neuralsystem.num_layers
        self.num_hidden = model.neuralsystem.num_hidden
        self.num_outputs = model.neuralsystem.num_hidden

    def h(self, x, u):
        u0 = mx.nd.zeros((self.num_steps, self.batch_size, self.num_inputs),
                         ctx=self.context)
        u0[0, 0, :] = u
        x0 = mx.nd.zeros((self.num_layers, self.batch_size, self.num_hidden),
                         ctx=self.context)
        x0[:, 0, :] = x
        y, x1 = self.model.controller.rnn(u0, [x0])
        return y, x1[0]

    def f(self, x, u, p, t):
        _, x1 = self.h(x, u)
        return x1.asnumpy()[0, 0]

    def g(self, x, u, p, t):
        y, _ = self.h(x, u)
        z = self.model.controller.decoder(y)
        return z.asnumpy()[0]

    def compute_gramian(self, kind):
        return emgr(self.f, self.g,
                    [self.num_inputs, self.num_hidden, self.num_outputs],
                    [self.dt, self.T], kind)

    def compute_controllability(self):
        return self.compute_gramian('c')

    def compute_observability(self):
        return self.compute_gramian('o')


def train(config, perturbation_type, perturbation_level, regularization_level,
          data_train, data_test, context, plot_control=True, save_model=True):

    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    delta = config.perturbation.DELTA
    batch_size = config.training.BATCH_SIZE
    path_model = config.paths.FILEPATH_MODEL
    atol = config.model.SPARSITY_THRESHOLD

    model = get_model(config, context, freeze_neuralsystem=True,
                      freeze_controller=False, load_weights_from=path_model)

    if perturbation_level > 0:
        model.apply_drift(perturbation_type, dt, delta, perturbation_level,
                          RNG)

    if regularization_level > 0:
        loss_function = L2L1(regularization_level, context)
    else:
        loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(
        model.collect_params(), config.training.OPTIMIZER,
        {'learning_rate': config.training.LEARNING_RATE,
         'rescale_grad': 1 / batch_size})

    init = model.begin_state(batch_size, context)

    baseline = evaluate(model, data_test, init, context) / len(data_test)
    mlflow.log_metric('Baseline_perturbed_uncontrolled', baseline)

    training_loss_mean = validation_loss_mean = None
    for epoch in trange(config.training.NUM_EPOCHS, desc='epoch'):
        training_loss = 0
        label = None
        tic = time.time()
        for batch_idx, (data, label) in tenumerate(data_train, desc='batch',
                                                   leave=False):
            # Move time axis from last to first position to conform to RNN
            # convention.
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output = model(data, *init)
                output = mx.nd.moveaxis(output, 0, -1)
                if regularization_level > 0:
                    loss = loss_function(output, label,
                                         model.get_reg_weights())
                else:
                    loss = loss_function(output, label)

            loss.backward()
            trainer.step(batch_size)
            model.sparsify(atol)
            training_loss += loss.mean().asscalar()

        training_loss_mean = training_loss / len(data_train)
        validation_loss = evaluate(model, data_test, init, context)
        validation_loss_mean = validation_loss / len(data_test)
        logging.debug("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                      "".format(epoch, time.time() - tic,
                                training_loss_mean, validation_loss_mean))
        mlflow.log_metric('mean_training_loss', training_loss_mean, epoch)
        mlflow.log_metric('mean_validation_loss', validation_loss_mean, epoch)

        if plot_control:
            f = plot_control_output(output, label)
            mlflow.log_figure(f, f'figures/control_{epoch}.png')

    if save_model:
        path_model = get_artifact_path('models/rnn.params')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        model.save_parameters(path_model)
        logging.info(f"Saved model to {path_model}.")

    f = plot_weight_histogram(model)
    mlflow.log_figure(f, f'figures/weight_distribution.png')

    # gramians = Gramians(model, context, dt, T)
    g_c = None#gramians.compute_controllability()
    g_o = None#gramians.compute_observability()

    return {'perturbation_type': perturbation_type,
            'perturbation_level': perturbation_level,
            'regularization_level': regularization_level,
            'training_loss': training_loss_mean,
            'validation_loss': validation_loss_mean,
            'controllability': g_c, 'observability': g_o}


def main(config):
    context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()
    os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'
    perturbation_types = config.perturbation.PERTURBATION_TYPES
    perturbation_levels = config.perturbation.PERTURBATION_LEVELS
    regularization_levels = config.model.REGULARIZATION_LEVELS
    path_data = config.paths.FILEPATH_INPUT_DATA
    data = pd.read_pickle(path_data)
    data_test, data_train = get_data_loaders(data, config, 'observations')

    dfs = []
    mlflow.set_tracking_uri(os.path.join('file:' + config.paths.BASE_PATH,
                                         'mlruns'))
    mlflow.set_experiment('train_rnn_controller')
    mlflow.start_run(run_name='Main')
    with open(get_artifact_path('config.txt'), 'w') as f:
        f.write(_config.dump())
    for perturbation_type in tqdm(perturbation_types, 'perturbation_type',
                                  leave=False):
        mlflow.start_run(run_name='Perturbation type', nested=True)
        mlflow.log_param('perturbation_type', perturbation_type)
        for perturbation_level in tqdm(perturbation_levels,
                                       'perturbation_level', leave=False):
            mlflow.start_run(run_name='Perturbation level', nested=True)
            mlflow.log_param('perturbation_level', perturbation_level)
            for regularization_level in tqdm(regularization_levels,
                                             'regularization_level',
                                             leave=False):
                mlflow.start_run(run_name='Regularization level', nested=True)
                mlflow.log_param('regularization_level', regularization_level)
                out = train(config, perturbation_type, perturbation_level,
                            regularization_level, data_train, data_test,
                            context)
                dfs.append(out)
                mlflow.end_run()
            mlflow.end_run()
        mlflow.end_run()
    df = pd.DataFrame(dfs)
    df.to_pickle(get_artifact_path('output.pkl'))
    mlflow.end_run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    GPU = 1
    RNG = np.random.default_rng(42)

    _config = configs.config_train_rnn_controller.get_config()

    main(_config)

    sys.exit()
