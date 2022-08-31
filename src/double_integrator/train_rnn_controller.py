import logging
import os
import sys
import time

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

from src.double_integrator import configs
from src.double_integrator.control_systems import RnnModel, DIMx, \
    ClosedControlledNeuralSystem
from src.double_integrator.plotting import plot_phase_diagram, \
    plot_control_output
from src.double_integrator.utils import get_artifact_path, get_data_loaders, \
    Gramians


def evaluate(model: ClosedControlledNeuralSystem, data_loader, loss_function,
             filename=None):
    validation_loss = 0
    environment_states = data = None
    for data, label in data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(model.context)
        neuralsystem_outputs, environment_states = model(data[:1])
        neuralsystem_outputs = mx.nd.moveaxis(neuralsystem_outputs, 0, -1)
        environment_states = mx.nd.moveaxis(environment_states, 0, -1)
        loss = loss_function(environment_states, neuralsystem_outputs)
        validation_loss += loss.mean().asscalar()
    if filename is not None:
        fig = plot_phase_diagram({'x': environment_states[0, 0].asnumpy(),
                                  'v': environment_states[0, 1].asnumpy()},
                                 xt=[0, 0], show=False, xlim=[-1, 1],
                                 ylim=[-1, 1], line_label='RNN')
        fig = plot_phase_diagram({'x': data[:, 0, 0].asnumpy(),
                                  'v': data[:, 0, 1].asnumpy()},
                                 show=False, fig=fig, line_label='LQG')
        fig.legend()
        mlflow.log_figure(fig, os.path.join('figures', filename))
    return validation_loss / len(data_loader)


def get_model(config, context, freeze_neuralsystem, freeze_controller,
              load_weights_from: str = None):
    neuralsystem_num_hidden = config.model.NUM_HIDDEN_NEURALSYSTEM
    neuralsystem_num_layers = config.model.NUM_LAYERS_NEURALSYSTEM
    controller_num_hidden = config.model.NUM_HIDDEN_CONTROLLER
    controller_num_layers = config.model.NUM_LAYERS_CONTROLLER
    activation_rnn = config.model.ACTIVATION
    activation_decoder = None  # Defaults to 'linear'
    batch_size = config.training.BATCH_SIZE
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    num_inputs = 1
    num_outputs = 1
    num_states = 2

    environment = DIMx(num_inputs, num_outputs, num_states, context,
                       process_noise, observation_noise, dt,
                       prefix='environment_')

    neuralsystem = RnnModel(neuralsystem_num_hidden, neuralsystem_num_layers,
                            num_outputs, num_inputs, activation_rnn,
                            activation_decoder, prefix='neuralsystem_')
    if load_weights_from is None:
        neuralsystem.initialize(mx.init.Xavier(), context)
    if freeze_neuralsystem:
        neuralsystem.collect_params().setattr('grad_req', 'null')

    controller = RnnModel(controller_num_hidden, controller_num_layers,
                          neuralsystem_num_hidden, neuralsystem_num_hidden,
                          activation_rnn, activation_decoder,
                          prefix='controller_')
    if load_weights_from is None:
        controller.initialize(mx.init.Zero(), context)
    if freeze_controller:
        controller.collect_params().setattr('grad_req', 'null')

    model = ClosedControlledNeuralSystem(
        environment, neuralsystem, controller, context, batch_size, num_steps)
    if load_weights_from is not None:
        model.load_parameters(load_weights_from, ctx=context)

    model.hybridize(active=True, static_alloc=True, static_shape=True)

    return model


class LQRLoss(mx.gluon.loss.Loss):
    def __init__(self, weight=1, batch_axis=0, **kwargs):
        self.Q = kwargs.pop('Q')
        self.R = kwargs.pop('R')
        super().__init__(weight, batch_axis, **kwargs)

    # noinspection PyMethodOverriding,PyProtectedMember
    def hybrid_forward(self, F, x, u, sample_weight=None):
        loss = (F.sum(x * F.batch_dot(F.tile(self.Q, (len(x), 1, 1)), x), 1) +
                F.sum(u * F.batch_dot(F.tile(self.R, (len(x), 1, 1)), u), 1))
        loss = mx.gluon.loss._apply_weighting(F, loss, self._weight,
                                              sample_weight)
        if mx.is_np_array():
            if F is mx.nd.ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)


class Masker:
    def __init__(self, model: ClosedControlledNeuralSystem, p,
                 rng: np.random.Generator):
        self.model = model
        self.p = p
        n = self.model.neuralsystem.num_hidden
        self._controllability_mask = np.nonzero(rng.binomial(1, self.p, n))
        self._observability_mask = np.nonzero(rng.binomial(1, self.p, n))

    def apply_mask(self):
        if self.p == 0:
            return
        self.model.controller.decoder.weight.data()[
            self._controllability_mask] = 0
        self.model.controller.rnn.l0_i2h_weight.data()[
            :, self._observability_mask] = 0


def train(config, perturbation_type, perturbation_level, dropout_probability,
          data_train, data_test, context, plot_control=True, save_model=True):
    T = config.simulation.T
    dt = T / config.simulation.NUM_STEPS
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R
    seed = config.SEED
    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
    if is_perturbed:
        path_model = config.paths.FILEPATH_MODEL
        model = get_model(config, context, freeze_neuralsystem=True,
                          freeze_controller=False,
                          load_weights_from=path_model)
    else:
        model = get_model(config, context, freeze_neuralsystem=False,
                          freeze_controller=True)

    if is_perturbed and perturbation_level > 0:
        model.add_noise(perturbation_type, perturbation_level, dt, rng)

    Q = q * mx.nd.eye(model.environment.num_states, ctx=context)
    R = r * mx.nd.eye(model.environment.num_inputs, ctx=context)
    loss_function = LQRLoss(dt, Q=Q, R=R)

    logging.info("Computing baseline performances...")
    test_loss = evaluate(model, data_train, loss_function)
    mlflow.log_metric('training_loss', test_loss, -1)
    test_loss = evaluate(model, data_test, loss_function, 'trajectory_-1.png')
    mlflow.log_metric('test_loss', test_loss, -1)

    if is_perturbed:
        # Initialize controller to non-zero weights only after computing
        # baseline accuracy of perturbed network before training. Otherwise the
        # untrained controller output will degrade accuracy further.
        model.controller.initialize(mx.init.Xavier(), context,
                                    force_reinit=True)

    trainer = mx.gluon.Trainer(
        model.collect_params(), config.training.OPTIMIZER,
        {'learning_rate': config.training.LEARNING_RATE,
         'rescale_grad': 1 / model.batch_size})

    masker = Masker(model, dropout_probability, rng)
    masker.apply_mask()

    logging.info("Training...")
    training_loss = test_loss = None
    for epoch in trange(config.training.NUM_EPOCHS, desc='epoch'):
        training_loss = 0
        label = None
        tic = time.time()
        for batch_idx, (data, label) in tenumerate(data_train, desc='batch',
                                                   leave=False):
            # Move time axis from last to first position to conform to RNN
            # convention.
            data = mx.nd.moveaxis(data, -1, 0)
            # Use only first time step.
            x0 = data[:1].as_in_context(context)
            with autograd.record():
                u, x = model(x0)
                u = mx.nd.moveaxis(u, 0, -1)
                x = mx.nd.moveaxis(x, 0, -1)
                loss = loss_function(x, u)

            loss.backward()
            trainer.step(model.batch_size)
            masker.apply_mask()
            training_loss += loss.mean().asscalar()

        training_loss /= len(data_train)
        test_loss = evaluate(model, data_test, loss_function,
                             f'trajectory_{epoch}.png')
        logging.debug("Epoch {:3} ({:2.1f} s): loss {:.3e}, test loss {:.3e}."
                      "".format(epoch, time.time() - tic, training_loss,
                                test_loss))
        mlflow.log_metric('training_loss', training_loss, epoch)
        mlflow.log_metric('test_loss', test_loss, epoch)

        if plot_control:
            f = plot_control_output(u, label)
            mlflow.log_figure(f, f'figures/control_{epoch}.png')

    if save_model:
        path_model = get_artifact_path('models/rnn.params')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        model.save_parameters(path_model)
        logging.info(f"Saved model to {path_model}.")

    if is_perturbed:
        logging.info("Computing Gramians...")
        gramians = Gramians(model, context, dt, T)
        g_c = gramians.compute_controllability()
        g_o = gramians.compute_observability()
        g_c_eig = np.linalg.eig(g_c)
        g_o_eig = np.linalg.eig(g_o)
        mlflow.log_metrics({'controllability': np.prod(g_c_eig[0]).item(),
                            'observability': np.prod(g_o_eig[0]).item()})
    else:
        g_c = g_o = None

    return {'perturbation_type': perturbation_type,
            'perturbation_level': perturbation_level,
            'dropout_probability': dropout_probability,
            'training_loss': training_loss, 'test_loss': test_loss,
            'controllability': g_c, 'observability': g_o}


def main(config):
    context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()
    os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'
    perturbation_types = config.perturbation.PERTURBATION_TYPES
    perturbation_levels = config.perturbation.PERTURBATION_LEVELS
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    seeds = config.SEEDS
    path_data = config.paths.FILEPATH_INPUT_DATA
    data = pd.read_pickle(path_data)
    data_test, data_train = get_data_loaders(data, config, 'states')

    dfs = []
    mlflow.set_tracking_uri(os.path.join('file:' + config.paths.BASE_PATH,
                                         'mlruns'))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name='Main')
    out = train(config, None, 0, 0, data_train, data_test, context)
    dfs.append(out)
    config.defrost()
    config.paths.FILEPATH_MODEL = get_artifact_path('models/rnn.params')
    with open(get_artifact_path('config.txt'), 'w') as f:
        f.write(config.dump())
    for perturbation_type in tqdm(perturbation_types, 'perturbation_type',
                                  leave=False):
        mlflow.start_run(run_name='Perturbation type', nested=True)
        mlflow.log_param('perturbation_type', perturbation_type)
        for perturbation_level in tqdm(perturbation_levels,
                                       'perturbation_level', leave=False):
            mlflow.start_run(run_name='Perturbation level', nested=True)
            mlflow.log_param('perturbation_level', perturbation_level)
            for dropout_probability in tqdm(
                    dropout_probabilities, 'dropout_probability', leave=False):
                mlflow.start_run(run_name='Dropout probability', nested=True)
                mlflow.log_param('dropout_probability', dropout_probability)
                for seed in tqdm(seeds, 'seed', leave=False):
                    config.SEED = seed
                    mlflow.start_run(run_name='seed', nested=True)
                    mlflow.log_param('seed', seed)
                    out = train(config, perturbation_type, perturbation_level,
                                dropout_probability, data_train, data_test,
                                context)
                    dfs.append(out)
                    mlflow.end_run()
                mlflow.end_run()
            mlflow.end_run()
        mlflow.end_run()
    df = pd.DataFrame(dfs)
    df.to_pickle(get_artifact_path('output.pkl'))
    mlflow.end_run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    GPU = 2
    EXPERIMENT_NAME = 'train_rnn_controller'

    _config = configs.config_train_rnn_controller.get_config()

    main(_config)

    sys.exit()
