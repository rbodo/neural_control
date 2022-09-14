import logging
import os
import sys
from typing import Union, Optional, Tuple

import mlflow
import mxnet as mx
import numpy as np
import torch
from mxnet import autograd
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.contrib import tenumerate

from examples import configs
from examples.linear_rnn_lqr import NeuralPerturbationPipeline, get_data
from src.plotting import plot_phase_diagram
from src.utils import get_artifact_path
from src.control_systems_mxnet import (RnnModel, ControlledNeuralSystem, DI,
                                       Masker, Gramians)


class LqgPipeline(NeuralPerturbationPipeline):
    def get_model(self, device, freeze_neuralsystem, freeze_controller,
                  load_weights_from=None) -> Tuple[ControlledNeuralSystem, DI]:
        environment_num_states = 2
        environment_num_outputs = 1
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
                         environment_num_states, device, process_noise,
                         observation_noise, dt, prefix='environment_')

        neuralsystem = RnnModel(
            neuralsystem_num_states, neuralsystem_num_layers,
            neuralsystem_num_outputs, environment_num_outputs,
            activation_rnn, activation_decoder, prefix='neuralsystem_')
        if load_weights_from is None:
            neuralsystem.initialize(mx.init.Xavier(), device)
        if freeze_neuralsystem:
            neuralsystem.collect_params().setattr('grad_req', 'null')

        controller = RnnModel(controller_num_states, controller_num_layers,
                              neuralsystem_num_states, neuralsystem_num_states,
                              activation_rnn, activation_decoder,
                              prefix='controller_')
        if load_weights_from is None:
            controller.initialize(mx.init.Zero(), device)
        if freeze_controller:
            controller.collect_params().setattr('grad_req', 'null')

        model = ControlledNeuralSystem(neuralsystem, controller, device,
                                       batch_size)
        if load_weights_from is not None:
            model.load_parameters(load_weights_from, ctx=device)

        model.hybridize(active=True, static_alloc=True, static_shape=True)

        return model, environment

    @staticmethod
    def evaluate(model: ControlledNeuralSystem, data_loader: DataLoader,
                 loss_function: Union[torch.nn.Module, mx.gluon.HybridBlock],
                 filename: Optional[str] = None) -> float:
        """
        Evaluate model.

        Parameters
        ----------
        model
            Neural system and controller. Possibly includes the differentiable
            environment in the computational graph.
        data_loader
            Data loaders for train and test set. Contain trajectories in state
            space to provide initial values or learning signal.
        loss_function
            Loss function used to evaluate performance. E.g. L2 or LQR.
        filename
            If specified, create an example phase diagram and save under given
            name.

        Returns
        -------
        loss
            The average performance when evaluating `loss_function` on the
            samples in `data_loader`.
        """
        validation_loss = 0
        environment_states = data = None
        for data, label in data_loader:
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(model.context)
            label = label.as_in_context(model.context)
            neuralsystem_outputs, environment_states = model(data)
            neuralsystem_outputs = mx.nd.moveaxis(neuralsystem_outputs, 0, -1)
            environment_states = mx.nd.moveaxis(environment_states, 0, -1)
            loss = loss_function(neuralsystem_outputs, label)
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

    def train(self, perturbation_type, perturbation_level, dropout_probability,
              device, save_model=True, **kwargs):
        T = self.config.simulation.T
        dt = T / self.config.simulation.NUM_STEPS
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
            model, environment = self.get_model(
                device, freeze_neuralsystem=True, freeze_controller=False,
                load_weights_from=path_model)
        else:
            model, environment = self.get_model(
                device, freeze_neuralsystem=False, freeze_controller=True)

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            model.add_noise(perturbation_type, perturbation_level, dt, rng)

        # Define loss function as the mean square error between RNN output
        # and LQG oracle.
        loss_function = mx.gluon.loss.L2Loss()

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
            # baseline accuracy of perturbed network before training.
            # Otherwise the untrained controller output will degrade
            # accuracy further.
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
            for batch_idx, (data, label) in tenumerate(
                    data_train, desc='batch', leave=False):
                # Move time axis from last to first position to conform to RNN
                # convention.
                data = mx.nd.moveaxis(data, -1, 0)
                data = data.as_in_context(device)
                label = label.as_in_context(device)
                with autograd.record():
                    u, x = model(data)
                    u = mx.nd.moveaxis(u, 0, -1)
                    loss = loss_function(u, label)

                loss.backward()
                trainer.step(model.batch_size)
                masker.apply_mask()
                training_loss += loss.mean().asscalar()

            training_loss /= len(data_train)
            test_loss = self.evaluate(model, data_test, loss_function,
                                      f'trajectory_{epoch}.png')
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
            gramians = Gramians(model, environment, device, dt, T)
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_lqg.get_config()

    # Get training set for RNN. The data consists of partial noisy observations
    # of a classic LQG controller in the double integrator state space. Labels
    # are provided by the LQG control output.
    _data_dict = get_data(_config, 'observations')

    pipeline = LqgPipeline(_config, _data_dict)
    pipeline.main()

    sys.exit()
