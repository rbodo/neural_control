import logging
import os
import sys
import time
from abc import ABC, abstractmethod

import mlflow
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from yacs.config import CfgNode

from examples import configs
from src.plotting import plot_phase_diagram
from src.utils import get_artifact_path, get_data_loaders
from src.control_systems_mxnet import (RnnModel, DI, Gramians, LQRLoss,
                                       ClosedControlledNeuralSystem, Masker,
                                       get_device)


class NeuralPerturbationPipeline(ABC):
    def __init__(self, config: CfgNode, data_dict: dict = None):
        self.config = config
        self.data_dict = data_dict or {}

    @abstractmethod
    def get_model(self, context, freeze_neuralsystem, freeze_controller,
                  load_weights_from: str = None):
        raise NotImplementedError

    @abstractmethod
    def train(self, perturbation_type, perturbation_level, dropout_probability,
              device, save_model=True, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate(model, data_loader, loss_function,
                 filename=None):
        raise NotImplementedError

    def main(self):
        # Select hardware backend.
        device = get_device(self.config)

        # We use the mlflow package for tracking experiment results.
        mlflow.set_tracking_uri(os.path.join(
            'file:' + self.config.paths.BASE_PATH, 'mlruns'))
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)

        # Train unperturbed and uncontrolled baseline model (i.e. neural system
        # controlling the double integrator environment).
        mlflow.start_run(run_name='Main')
        out = self.train(None, 0, 0, device, **self.data_dict)
        dfs = [out]

        # Remember where the baseline model has been saved.
        self.config.paths.FILEPATH_MODEL = get_artifact_path(
            'models/rnn.params')
        with open(get_artifact_path('config.txt'), 'w') as f:
            f.write(self.config.dump())

        # Return here if all we want is the baseline model.
        if self.config.perturbation.SKIP_PERTURBATION:
            pd.DataFrame(dfs).to_pickle(get_artifact_path('output.pkl'))
            mlflow.end_run()
            return

        # Get perturbation configuration.
        perturbation_types = self.config.perturbation.PERTURBATION_TYPES
        perturbation_levels = self.config.perturbation.PERTURBATION_LEVELS
        dropout_probabilities = self.config.perturbation.DROPOUT_PROBABILITIES

        # Sweep over perturbation types and levels, degrees of controllability
        # and observability, and repeat for multiple random seeds.
        for perturbation_type in tqdm(perturbation_types, 'perturbation_type', leave=False):
            mlflow.start_run(run_name='Perturbation type', nested=True)
            mlflow.log_param('perturbation_type', perturbation_type)
            for perturbation_level in tqdm(perturbation_levels, 'perturbation_level', leave=False):
                mlflow.start_run(run_name='Perturbation level', nested=True)
                mlflow.log_param('perturbation_level', perturbation_level)
                for dropout_probability in tqdm(dropout_probabilities, 'dropout_probability', leave=False):
                    mlflow.start_run(run_name='Dropout probability', nested=True)
                    mlflow.log_param('dropout_probability', dropout_probability)
                    for seed in tqdm(self.config.SEEDS, 'seed', leave=False):
                        self.config.SEED = seed
                        mlflow.start_run(run_name='seed', nested=True)
                        mlflow.log_param('seed', seed)
                        out = self.train(self.config, perturbation_type,
                                         perturbation_level,
                                         dropout_probability, device,
                                         **self.data_dict)
                        out.update(perturbation_type=perturbation_type,
                                   perturbation_level=perturbation_level,
                                   dropout_probability=dropout_probability)
                        dfs.append(out)
                        mlflow.end_run()
                    mlflow.end_run()
                mlflow.end_run()
            mlflow.end_run()
        pd.DataFrame(dfs).to_pickle(get_artifact_path('output.pkl'))
        mlflow.end_run()


class LqrPipeline(NeuralPerturbationPipeline):
    def get_model(self, context, freeze_neuralsystem, freeze_controller,
                  load_weights_from: str = None):
        environment_num_states = 2
        environment_num_outputs = 2
        neuralsystem_num_states = self.config.model.NUM_HIDDEN_NEURALSYSTEM
        neuralsystem_num_layers = self.config.model.NUM_LAYERS_NEURALSYSTEM
        neuralsystem_num_outputs = 1
        controller_num_states = self.config.model.NUM_HIDDEN_CONTROLLER
        controller_num_layers = self.config.model.NUM_LAYERS_CONTROLLER
        activation_rnn = self.config.model.ACTIVATION
        activation_decoder = None  # Defaults to 'linear'
        batch_size = self.config.training.BATCH_SIZE
        process_noise = self.config.process.PROCESS_NOISE
        observation_noise = self.config.process.OBSERVATION_NOISE
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps

        environment = DI(neuralsystem_num_outputs, environment_num_outputs,
                         environment_num_states, context, process_noise,
                         observation_noise, dt, prefix='environment_')

        neuralsystem = RnnModel(neuralsystem_num_states,
                                neuralsystem_num_layers,
                                neuralsystem_num_outputs,
                                environment_num_outputs,
                                activation_rnn, activation_decoder,
                                prefix='neuralsystem_')
        if load_weights_from is None:
            neuralsystem.initialize(mx.init.Xavier(), context)
        if freeze_neuralsystem:
            neuralsystem.collect_params().setattr('grad_req', 'null')

        controller = RnnModel(controller_num_states, controller_num_layers,
                              neuralsystem_num_states, neuralsystem_num_states,
                              activation_rnn, activation_decoder,
                              prefix='controller_')
        if load_weights_from is None:
            controller.initialize(mx.init.Zero(), context)
        if freeze_controller:
            controller.collect_params().setattr('grad_req', 'null')

        model = ClosedControlledNeuralSystem(
            environment, neuralsystem, controller, context, batch_size,
            num_steps)
        if load_weights_from is not None:
            model.load_parameters(load_weights_from, ctx=context)

        model.hybridize(active=True, static_alloc=True, static_shape=True)

        return model

    def train(self, perturbation_type, perturbation_level, dropout_probability,
              device, save_model=True, **kwargs):
        T = self.config.simulation.T
        dt = T / self.config.simulation.NUM_STEPS
        q = self.config.controller.cost.lqr.Q
        r = self.config.controller.cost.lqr.R
        seed = self.config.SEED

        # Set random seed.
        mx.random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Create model consisting of neural system, controller and environment.
        is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
        if is_perturbed:
            # Initialize model using unperturbed, uncontrolled baseline from
            # previous run.
            path_model = self.config.paths.FILEPATH_MODEL
            model = self.get_model(device, freeze_neuralsystem=True,
                                   freeze_controller=False,
                                   load_weights_from=path_model)
        else:
            model = self.get_model(device, freeze_neuralsystem=False,
                                   freeze_controller=True)

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            model.add_noise(perturbation_type, perturbation_level, dt, rng)

        # Define loss function based on LQR.
        Q = q * mx.nd.eye(model.environment.num_states, ctx=device)
        R = r * mx.nd.eye(model.environment.num_inputs, ctx=device)
        loss_function = LQRLoss(dt, Q=Q, R=R)

        data_train = kwargs['data_train']
        data_test = kwargs['data_test']

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        test_loss = self.evaluate(model, data_train, loss_function)
        mlflow.log_metric('training_loss', test_loss, -1)
        test_loss = self.evaluate(model, data_test, loss_function,
                                  'trajectory_-1.png')
        mlflow.log_metric('test_loss', test_loss, -1)

        # Initialize controller weights.
        if is_perturbed:
            # Initialize controller to non-zero weights only after computing
            # baseline accuracy of perturbed network before training. Otherwise
            # the untrained controller output will degrade accuracy further.
            model.controller.initialize(mx.init.Xavier(), device,
                                        force_reinit=True)

        trainer = mx.gluon.Trainer(
            model.collect_params(), self.config.training.OPTIMIZER,
            {'learning_rate': self.config.training.LEARNING_RATE,
             'rescale_grad': 1 / model.batch_size})

        # Reduce controllability and observability of neural system by
        # dropping out rows from the stimulation and readout matrices of the
        # controller.
        masker = Masker(model, dropout_probability, rng)
        masker.apply_mask()

        # Train neural system (if unperturbed) or controller (if perturbed).
        logging.info("Training...")
        training_loss = test_loss = None
        for epoch in trange(self.config.training.NUM_EPOCHS, desc='epoch'):
            training_loss = 0
            tic = time.time()
            for batch_idx, (data, label) in tenumerate(data_train,
                                                       desc='batch',
                                                       leave=False):
                # Move time axis from last to first position to conform to RNN
                # convention.
                data = mx.nd.moveaxis(data, -1, 0)
                # Use only first time step.
                x0 = data[:1].as_in_context(device)
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
            test_loss = self.evaluate(model, data_test, loss_function,
                                      f'trajectory_{epoch}.png')
            logging.debug(
                "Epoch {:3} ({:2.1f} s): loss {:.3e}, test loss {:.3e}."
                "".format(epoch, time.time() - tic, training_loss,
                          test_loss))
            mlflow.log_metric('training_loss', training_loss, epoch)
            mlflow.log_metric('test_loss', test_loss, epoch)

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            model.save_parameters(path_model)
            logging.info(f"Saved model to {path_model}.")

        # Compute controllability and observability Gramians.
        if is_perturbed:
            logging.info("Computing Gramians...")
            gramians = Gramians(model, model.environment, device, dt, T)
            g_c = gramians.compute_controllability()
            g_o = gramians.compute_observability()
            g_c_eig = np.linalg.eig(g_c)
            g_o_eig = np.linalg.eig(g_o)
            # Use product of eigenvalue spectrum as scalar matric for
            # controllability and observability. The larger the better.
            mlflow.log_metrics({'controllability': np.prod(g_c_eig[0]).item(),
                                'observability': np.prod(g_o_eig[0]).item()})
        else:
            g_c = g_o = None

        return {'training_loss': training_loss, 'test_loss': test_loss,
                'controllability': g_c, 'observability': g_o}

    @staticmethod
    def evaluate(model: ClosedControlledNeuralSystem, data_loader,
                 loss_function, filename=None):
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


def get_data(config, variable):
    """Return training and test data loader as dict.

    Contains trajectories of a classic LQR controller in the double integrator
    state space with initial values sampled from a jittered rectangular grid.
    """

    path_data = config.paths.FILEPATH_INPUT_DATA
    data = pd.read_pickle(path_data)
    data_test, data_train = get_data_loaders(data, config, variable)
    return dict(data_train=data_train, data_test=data_test)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_lqr.get_config()

    # Get trajectories of a classic LQR controller in the double integrator
    # state space. In this study we use only the initial values for training
    # the RNN, and plot the LQR trajectories as comparison.
    _data_dict = get_data(_config, 'states')

    pipeline = NeuralPerturbationPipeline(_config, _data_dict)
    pipeline.main()

    sys.exit()
