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
from src.double_integrator.control_systems import RNNModel, DIMx, \
    ControlledNeuralSystem, ClosedControlledNeuralSystem
from src.double_integrator.plotting import plot_phase_diagram, \
    plot_control_output, plot_weight_histogram
from src.double_integrator.utils import get_artifact_path, get_data_loaders, \
    L2L1, Gramians


def evaluate(model: ControlledNeuralSystem, data_loader):
    loss_function = mx.gluon.loss.L2Loss()
    validation_loss = 0
    for data, label in data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(model.context)
        label = label.as_in_context(model.context)
        output = model(data)
        output = mx.nd.moveaxis(output, 0, -1)
        loss = loss_function(output, label)
        validation_loss += loss.mean().asscalar()
    return validation_loss / len(data_loader)


def evaluate_closed_loop(model: ClosedControlledNeuralSystem, data_loader,
                         filename):
    loss_function = mx.gluon.loss.L2Loss()
    validation_loss = 0
    environment_states = data = None
    for data, label in data_loader:
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(model.context)
        label = label.as_in_context(model.context)
        neuralsystem_outputs, environment_states = model(data[:1])
        neuralsystem_outputs = mx.nd.moveaxis(neuralsystem_outputs, 0, -1)
        loss = loss_function(neuralsystem_outputs, label)
        validation_loss += loss.mean().asscalar()
    fig = plot_phase_diagram({'x': environment_states[:, 0, 0].asnumpy(),
                              'v': environment_states[:, 0, 1].asnumpy()},
                             xt=[0, 0], show=False, xlim=[-1, 1], ylim=[-1, 1],
                             line_label='RNN')
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
    num_inputs = 1
    num_outputs = 1

    neuralsystem = RNNModel(neuralsystem_num_hidden, neuralsystem_num_layers,
                            num_outputs, num_inputs, activation_rnn,
                            activation_decoder, prefix='neuralsystem_')
    neuralsystem.initialize(mx.init.Xavier(), context)
    if freeze_neuralsystem:
        neuralsystem.collect_params().setattr('grad_req', 'null')

    controller = RNNModel(controller_num_hidden, controller_num_layers,
                          neuralsystem_num_hidden, neuralsystem_num_hidden,
                          activation_rnn, activation_decoder,
                          prefix='controller_')
    controller.initialize(mx.init.Zero(), context)
    if freeze_controller:
        controller.collect_params().setattr('grad_req', 'null')

    model = ControlledNeuralSystem(neuralsystem, controller, context,
                                   batch_size)
    model.hybridize(active=True, static_alloc=True, static_shape=True)

    if load_weights_from is not None:
        model.load_parameters(load_weights_from, ctx=context)

    return model


def train(config, perturbation_type, perturbation_level, regularization_level,
          data_train, data_test, data_test_full_obs, context,
          plot_control=True, save_model=True):
    rng = np.random.default_rng(config.SEED)
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    delta = config.perturbation.DELTA
    batch_size = config.training.BATCH_SIZE
    atol = config.model.SPARSITY_THRESHOLD
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]

    is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
    if is_perturbed:
        path_model = config.paths.FILEPATH_MODEL
        model = get_model(config, context, freeze_neuralsystem=True,
                          freeze_controller=False,
                          load_weights_from=path_model)
    else:
        model = get_model(config, context, freeze_neuralsystem=False,
                          freeze_controller=True)

    if perturbation_level > 0:
        model.apply_drift(perturbation_type, dt, delta, perturbation_level,
                          rng)

    environment = DIMx(1, 1, 2, context, process_noise, observation_noise, dt,
                       prefix='environment_', freeze=True)
    model_closed = ClosedControlledNeuralSystem(
        environment, model.neuralsystem, model.controller, context, batch_size,
        num_steps)

    if regularization_level > 0:
        loss_function = L2L1(regularization_level, context)
    else:
        loss_function = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(
        model.collect_params(), config.training.OPTIMIZER,
        {'learning_rate': config.training.LEARNING_RATE,
         'rescale_grad': 1 / batch_size})

    logging.info("Computing baseline performances...")
    test_loss = evaluate_closed_loop(model_closed, data_test_full_obs,
                                     'test_trajectory_before_training.png')
    mlflow.log_metric('test_loss', test_loss, 0)
    if is_perturbed:
        # Initialize controller to non-zero weights only after computing
        # baseline accuracy of perturbed network before training. Otherwise the
        # untrained controller output will degrade accuracy further.
        model.controller.initialize(mx.init.Xavier(), context)

    logging.info("Training...")
    training_loss = validation_loss = None
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
                output = model(data)
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

        training_loss /= len(data_train)
        validation_loss = evaluate(model, data_test)
        logging.debug("Epoch {:3} ({:2.1f} s): loss {:.3e}, val loss {:.3e}."
                      "".format(epoch, time.time() - tic, training_loss,
                                validation_loss))
        mlflow.log_metric('training_loss', training_loss, epoch)
        mlflow.log_metric('validation_loss', validation_loss, epoch)

        if plot_control:
            f = plot_control_output(output, label)
            mlflow.log_figure(f, f'figures/control_{epoch}.png')

    if save_model:
        path_model = get_artifact_path('models/rnn.params')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        model.save_parameters(path_model)
        logging.info(f"Saved model to {path_model}.")

    if is_perturbed:
        f = plot_weight_histogram(model)
        mlflow.log_figure(f, f'figures/weight_distribution.png')

    logging.info("Computing test performance...")
    test_loss = evaluate_closed_loop(model_closed, data_test_full_obs,
                                     'test_trajectory_after_training.png')
    mlflow.log_metric('test_loss', test_loss, 1)

    if is_perturbed:
        logging.info("Computing Gramians...")
        gramians = Gramians(model_closed, context, dt, T)
        g_c = gramians.compute_controllability()
        g_o = gramians.compute_observability()
    else:
        g_c = g_o = None

    return {'perturbation_type': perturbation_type,
            'perturbation_level': perturbation_level,
            'regularization_level': regularization_level,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
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
    data_test_full_obs, _ = get_data_loaders(data, config, 'states')

    dfs = []
    mlflow.set_tracking_uri(os.path.join('file:' + config.paths.BASE_PATH,
                                         'mlruns'))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name='Main')
    out = train(config, None, 0, 0, data_train, data_test, data_test_full_obs,
                context)
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
            for regularization_level in tqdm(regularization_levels,
                                             'regularization_level',
                                             leave=False):
                mlflow.start_run(run_name='Regularization level', nested=True)
                mlflow.log_param('regularization_level', regularization_level)
                out = train(config, perturbation_type, perturbation_level,
                            regularization_level, data_train, data_test,
                            data_test_full_obs, context)
                dfs.append(out)
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
