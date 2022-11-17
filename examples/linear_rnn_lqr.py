import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
import time
from typing import Optional, Union, Any, Tuple

import gym
import mlflow
import mxnet as mx
import numpy as np
from mxnet import autograd
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from yacs.config import CfgNode

from examples import configs
from src.plotting import plot_phase_diagram
from src.utils import get_artifact_path, get_data
from src.control_systems_mxnet import (
    RnnModel, DI, Gramians, LqrLoss, ClosedControlledNeuralSystem, Masker,
    get_device, ControlledNeuralSystem, StochasticLinearIOSystem)


class NeuralPerturbationPipeline(ABC):
    """Base class for pipeline objects.

    Methods
    -------
    main
        Sweeps over various perturbation types and levels, degrees of
        controllability and observability, and random seeds. Each iteration,
        a controller RNN is trained to maintain performance of a perturbed
        neural system while solving a task in a dynamic environment.

    All other methods need to be overwritten, namely a method to create, train,
    and evaluate the model.

    Notes
    -----
    The objective is to train a neural system (represented e.g. by an RNN) to
    solve a task in a dynamical environment (e.g. inverted pendulum). The
    trained neural system is then perturbed (e.g. degradation of sensory,
    association, or actuator populations). An external controller is interfaced
    with the neural system via readout and stimulation connections. The
    controller is trained to compensate for any performance loss by providing
    feedback to the neural system.

    Here we consider the following training strageties for the neural system
    and controller:

    - Direct training on some convex cost function (e.g. LQR). Fastest and
      cleanest, but requires knowledge of the system, full observability, and
      for the system to be linear, differentiable and part of the computational
      graph. Also, not all environments / tasks allow defining a suitable cost
      function.
    - Using the output of an optimal controller (e.g. LQR, LQG) as oracle to
      provide the learning signal. Assumes knowledge of system and for the
      system to be linear.
    - Reinforcement learning. Does not assume knowledge of system, linearity,
      or full observability.
    """
    def __init__(self, config: CfgNode, data_dict: Optional[dict] = None):
        self.config = config
        self.data_dict = data_dict or {}
        self.device = None
        self.model = None
        self._resume_experiment = None
        self._runs = None  # Previous runs to resume.
        self.argparser = None

    @abstractmethod
    def get_environment(self, num_inputs: int, num_states: int,
                        num_outputs: int, **kwargs
                        ) -> Union[gym.Env, StochasticLinearIOSystem]:
        """Create the environment."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self, freeze_neuralsystem: bool, freeze_controller: bool,
                  environment: Union[gym.Env, mx.gluon.HybridBlock],
                  load_weights_from: Optional[str] = None
                  ) -> ControlledNeuralSystem:
        """Create the model.

        Parameters
        ----------
        environment
            The environment with which the neural system interacts.
        freeze_neuralsystem
            Whether to train the neural system. Usually turned off while
            perturbing the neural system and training the controller.
        freeze_controller
            Whether to train the controller. Usually turned off while training
            the neural system.
        load_weights_from
            Path to weights that are used to initialize the model.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, perturbation_type: str, perturbation_level: float,
              dropout_probability: float, save_model: Optional[bool] = True,
              **kwargs) -> dict:
        """Train the model (consisting of a neural system and a controller).

        Parameters
        ----------
        perturbation_type
            Which part of the neural system to perturb. One of 'sensor',
            'actuator', 'processor'. Will add gaussian noise to the weights of
            the corresponding population.
        perturbation_level
            The weight given to the perturbation. Proportional to the standard
            deviation of the gaussian noise distribution.
        dropout_probability
            Probability of dropping out a row in the readout and stimulation
            connection matrix of the controller. Tunes the controllability and
            observability of the neural system.
        save_model
            Whether to save the model in the mlflow artifact directory.

        Returns
        -------
        results
            A dictionary containing the performance metrics to be logged.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Evaluate model."""
        raise NotImplementedError

    @abstractmethod
    def get_device(self) -> Union[mx.context.Context, torch.device]:
        """Get hardware backend to run on."""
        raise NotImplementedError

    def get_run_id(self, conditions: dict) -> Union[Tuple[str, bool], None]:
        """Get id of run that meets `conditions` and is unfinished.

        Parameters
        ----------
        conditions
            Dictionary of conditions (on tags, metrics, params, ...) that the
            run should meet.

        Returns
        -------
        run_id, is_done
            If there exists a run that meets `conditions`, return its ID
            together with a flag that tells whether the run is completed.
            If no run could be found, return an empty string, which starts a
            new run.
        """

        run_id = ''
        is_done = False
        if self._runs is None:
            self._runs = mlflow.search_runs(
                experiment_names=[self.config.EXPERIMENT_NAME],
                filter_string='tags.resume_experiment = ' +
                              f'"{self._resume_experiment}"')
        try:
            mask = np.prod([self._runs[key].astype(str) == str(value)
                            for key, value in conditions.items()], 0)
        except KeyError:
            logging.info("Could not find run that matches conditions "
                         f"{conditions}.")
            return run_id, is_done
        idx = self._runs.loc[mask.astype(bool)].index
        if len(idx) > 1:
            raise AssertionError("Could not find unique run to resume.")
        if len(idx) == 1:
            idx = idx[0]
            run_id = self._runs['run_id'][idx]
            is_done = run_id in ['UNFINISHED', 'RUNNING']
            if not is_done:
                logging.info(f"Resuming unfinished run {run_id} "
                             f"with conditions {conditions}.")
        return run_id, is_done

    def set_resume_experiment(self):
        """Decide whether to resume a previous run."""
        from_args = self.argparser.parse_args().resume_experiment
        from_config = self.config.get('RESUME_EXPERIMENT', '')
        if from_args:  # Command line argument takes precedence over config.
            self._resume_experiment = from_args
        elif from_config:
            self._resume_experiment = from_config
        else:
            self._resume_experiment = time.strftime('%Y-%m-%d_%H:%M:%S')
        self.config.RESUME_EXPERIMENT = self._resume_experiment

    def set_argparser(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--sweep_id', type=int, required=False,
                                    default=-1)
        self.argparser.add_argument('--resume_experiment', type=str,
                                    required=False, default='')

    def main(self):
        """Run the pipeline.

        Consists of two parts. First, a baseline model is trained (without
        perturbations or external control). Then we sweep over perturbation
        types and levels, degrees of controllability and observability, and
        repeat for multiple random seeds.

        The results are logged via mlflow.
        """

        # Select hardware backend.
        self.device = self.get_device()

        self.set_argparser()
        sweep_id = self.argparser.parse_args().sweep_id
        self.set_resume_experiment()

        # We use the mlflow package for tracking experiment results.
        mlflow.set_tracking_uri(os.path.join(
            'file:' + self.config.paths.BASE_PATH, 'mlruns'))
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)

        run_name = 'Main'
        conditions = {'tags.mlflow.runName': run_name,
                      'params.sweep_id': sweep_id}
        run_id, is_done = self.get_run_id(conditions)
        if is_done:
            return

        tags = {'resume_experiment': self._resume_experiment}
        mlflow.start_run(run_id, run_name=run_name, tags=tags)
        mlflow.log_param('sweep_id', sweep_id)

        if sweep_id < 0:
            for seed in tqdm(self.config.SEEDS, 'seed', leave=False):
                self._main(seed)
        else:
            logging.info("Using seed from command line.")
            seed = self.config.SEEDS[sweep_id]
            self._main(seed)
        mlflow.end_run()

    def _main(self, seed):
        run_name = 'seed'
        self.config.SEED = seed
        conditions = {'tags.mlflow.runName': run_name,
                      'params.seed': seed}
        run_id, is_done = self.get_run_id(conditions)
        tags = {'resume_experiment': self._resume_experiment}
        mlflow.start_run(run_id, run_name=run_name, tags=tags, nested=True)

        # Train unperturbed and uncontrolled baseline model (i.e. neural
        # system controlling the double integrator environment).
        perturbation_type = ''
        perturbation_level = 0
        dropout_probability = 0
        mlflow.log_params({run_name: seed,
                           'perturbation_type': perturbation_type,
                           'perturbation_level': perturbation_level,
                           'dropout_probability': dropout_probability})
        self.train(perturbation_type, perturbation_level, dropout_probability,
                   **self.data_dict)

        # Remember where the baseline model has been saved.
        self.config.paths.FILEPATH_MODEL = get_artifact_path(
            'models/rnn.params')
        with open(get_artifact_path('config.txt'), 'w') as f:
            f.write(self.config.dump())

        # Return here if all we want is the baseline model.
        if self.config.perturbation.SKIP_PERTURBATION:
            mlflow.end_run()
            return

        # Get perturbation configuration.
        perturbations = dict(self.config.perturbation.PERTURBATIONS)
        dropout_probabilities = self.config.perturbation.DROPOUT_PROBABILITIES

        # Sweep over perturbation types and levels, degrees of controllability
        # and observability, and repeat for multiple random seeds.
        for perturbation_type in tqdm(perturbations.keys(),
                                      'Perturbation type', leave=False):
            conditions['tags.mlflow.runName'] = 'Perturbation type'
            conditions['params.perturbation_type'] = perturbation_type
            run_id, is_done = self.get_run_id(conditions)
            if is_done:
                continue
            mlflow.start_run(run_id, run_name='Perturbation type', tags=tags,
                             nested=True)
            levels = perturbations[perturbation_type]
            for perturbation_level in tqdm(levels, 'Perturbation level',
                                           leave=False):
                conditions['tags.mlflow.runName'] = 'Perturbation level'
                conditions['params.perturbation_level'] = \
                    str(perturbation_level)
                run_id, is_done = self.get_run_id(conditions)
                if is_done:
                    continue
                mlflow.start_run(run_id, run_name='Perturbation level',
                                 tags=tags, nested=True)
                for dropout_probability in tqdm(dropout_probabilities,
                                                'Dropout probability',
                                                leave=False):
                    conditions['tags.mlflow.runName'] = 'Dropout probability'
                    conditions['params.dropout_probability'] = \
                        str(dropout_probability)
                    run_id, is_done = self.get_run_id(conditions)
                    if is_done:
                        continue
                    mlflow.start_run(run_id, run_name='Dropout probability',
                                     tags=tags, nested=True)
                    mlflow.log_params({
                        'seed': seed,
                        'perturbation_type': perturbation_type,
                        'perturbation_level': perturbation_level,
                        'dropout_probability': dropout_probability})
                    self.train(perturbation_type, perturbation_level,
                               dropout_probability, **self.data_dict)
                    mlflow.end_run()
                mlflow.end_run()
            mlflow.end_run()
        mlflow.end_run()


class LqrPipeline(NeuralPerturbationPipeline):
    def __init__(self, config: CfgNode, data_dict: Optional[dict] = None):
        super().__init__(config, data_dict)
        self.loss_function = None

    def get_device(self) -> mx.context.Context:
        return get_device(self.config)

    def get_environment(self, *args, **kwargs) -> StochasticLinearIOSystem:
        num_inputs = self.config.process.NUM_INPUTS
        num_states = self.config.process.NUM_STATES
        num_outputs = self.config.process.NUM_OUTPUTS
        process_noise = self.config.process.PROCESS_NOISES[0]
        observation_noise = self.config.process.OBSERVATION_NOISES[0]
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps
        return DI(num_inputs, num_outputs, num_states, self.device,
                  process_noise,
                  observation_noise, dt, prefix='environment_')

    def get_model(self, freeze_neuralsystem, freeze_controller,
                  environment: StochasticLinearIOSystem,
                  load_weights_from: Optional[str] = None
                  ) -> Union[ControlledNeuralSystem,
                             ClosedControlledNeuralSystem]:
        neuralsystem_num_states = self.config.model.NUM_HIDDEN_NEURALSYSTEM
        neuralsystem_num_layers = self.config.model.NUM_LAYERS_NEURALSYSTEM
        controller_num_states = self.config.model.NUM_HIDDEN_CONTROLLER
        controller_num_layers = self.config.model.NUM_LAYERS_CONTROLLER
        activation_rnn = self.config.model.ACTIVATION
        activation_decoder = None  # Defaults to 'linear'
        batch_size = self.config.training.BATCH_SIZE

        neuralsystem = RnnModel(
            neuralsystem_num_states, neuralsystem_num_layers,
            environment.num_inputs, environment.num_outputs,
            activation_rnn, activation_decoder, prefix='neuralsystem_')
        if load_weights_from is None:
            neuralsystem.initialize(mx.init.Xavier(), self.device)
        if freeze_neuralsystem:
            neuralsystem.collect_params().setattr('grad_req', 'null')

        controller = RnnModel(controller_num_states, controller_num_layers,
                              neuralsystem_num_states, neuralsystem_num_states,
                              activation_rnn, activation_decoder,
                              prefix='controller_')
        if load_weights_from is None:
            controller.initialize(mx.init.Zero(), self.device)
        if freeze_controller:
            controller.collect_params().setattr('grad_req', 'null')

        if self.config.model.CLOSE_ENVIRONMENT_LOOP:
            model = ClosedControlledNeuralSystem(
                environment, neuralsystem, controller, self.device, batch_size,
                self.config.simulation.NUM_STEPS)
        else:
            model = ControlledNeuralSystem(neuralsystem, controller,
                                           self.device, batch_size)
        if load_weights_from is not None:
            model.load_parameters(load_weights_from, ctx=self.device)

        model.hybridize(active=True, static_alloc=True, static_shape=True)

        return model

    def get_loss_function(self, *args, **kwargs) -> mx.gluon.HybridBlock:
        """Define loss function based on LQR."""
        q = self.config.controller.cost.lqr.Q
        r = self.config.controller.cost.lqr.R
        Q = q * mx.nd.eye(self.model.environment.num_states, ctx=self.device)
        R = r * mx.nd.eye(self.model.environment.num_inputs, ctx=self.device)
        return LqrLoss(kwargs['dt'], Q=Q, R=R)

    def get_loss(self, data: mx.nd.NDArray,
                 label: Optional[mx.nd.NDArray] = None) -> mx.nd.NDArray:
        # Move time axis from last to first position to conform to RNN
        # convention.
        data = mx.nd.moveaxis(data, -1, 0)
        # Use only first time step.
        x0 = data[:1].as_in_context(self.device)
        with autograd.record():
            u, x = self.model(x0)
            u = mx.nd.moveaxis(u, 0, -1)
            x = mx.nd.moveaxis(x, 0, -1)
            return self.loss_function(x, u)

    def train(self, perturbation_type, perturbation_level, dropout_probability,
              save_model=True, **kwargs):
        T = self.config.simulation.T
        dt = T / self.config.simulation.NUM_STEPS
        seed = self.config.SEED

        # Set random seed.
        mx.random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Create environment.
        environment = self.get_environment()

        # Create model consisting of neural system, controller and environment.
        is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
        if is_perturbed:
            freeze_neuralsystem = True
            freeze_controller = False
            # Initialize model using unperturbed, uncontrolled baseline from
            # previous run.
            path_model = self.config.paths.FILEPATH_MODEL
            self.model = self.get_model(
                freeze_neuralsystem, freeze_controller, environment,
                load_weights_from=path_model)
            num_epochs = self.config.training.NUM_EPOCHS_CONTROLLER
        else:
            freeze_neuralsystem = False
            freeze_controller = True
            self.model = self.get_model(
                freeze_neuralsystem, freeze_controller, environment)
            num_epochs = self.config.training.NUM_EPOCHS_NEURALSYSTEM

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            self.model.add_noise(perturbation_type, perturbation_level, dt,
                                 rng)

        self.loss_function = self.get_loss_function(dt=dt)

        data_train = kwargs['data_train']
        data_test = kwargs['data_test']

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        training_loss = self.evaluate(data_train, environment)
        mlflow.log_metric('training_loss', training_loss, -1)
        test_loss = self.evaluate(data_test, environment, 'trajectory_-1.png')
        mlflow.log_metric('test_loss', test_loss, -1)

        # Initialize controller weights.
        if is_perturbed:
            # Initialize controller to non-zero weights only after computing
            # baseline accuracy of perturbed network before training. Otherwise
            # the untrained controller output will degrade accuracy further.
            self.model.controller.initialize(mx.init.Xavier(), self.device,
                                             force_reinit=True)

        trainer = mx.gluon.Trainer(
            self.model.collect_params(), self.config.training.OPTIMIZER,
            {'learning_rate': self.config.training.LEARNING_RATE,
             'rescale_grad': 1 / self.model.batch_size})

        # Reduce controllability and observability of neural system by
        # dropping out rows from the stimulation and readout matrices of the
        # controller.
        masker = Masker(self.model, dropout_probability, rng)
        masker.apply_mask()

        # Store the hash values of model weights so we can check later that
        # only the parts were trained that we wanted.
        self.model.cache_weight_hash()

        # Train neural system (if unperturbed) or controller (if perturbed).
        logging.info("Training...")
        for epoch in trange(num_epochs, desc='epoch'):
            training_loss = 0
            for batch_idx, (data, label) in tenumerate(
                    data_train, desc='batch', leave=False):
                loss = self.get_loss(data, label)
                loss.backward()
                trainer.step(self.model.batch_size)
                masker.apply_mask()
                training_loss += loss.mean().asscalar()

            training_loss /= len(data_train)
            test_loss = self.evaluate(data_test, environment,
                                      f'trajectory_{epoch}.png')
            mlflow.log_metric('training_loss', training_loss, epoch)
            mlflow.log_metric('test_loss', test_loss, epoch)

        self.model.assert_plasticity(freeze_neuralsystem, freeze_controller)

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            self.model.save_parameters(path_model)
            logging.info(f"Saved model to {path_model}.")

        if self.config.get('COMPUTE_GRAMIANS', True):
            # Compute controllability and observability Gramians.
            logging.info("Computing Gramians...")
            gramians = Gramians(self.model, environment, self.device, dt, T)
            g_c = gramians.compute_controllability()
            g_o = gramians.compute_observability()

            np.savez_compressed(get_artifact_path('gramians.npz'),
                                controllability_gramian=g_c,
                                observability_gramian=g_o)
        mlflow.log_metrics({'controllability': masker.controllability,
                            'observability': masker.observability})

    def evaluate(self, data_loader: DataLoader,
                 environment: Optional[StochasticLinearIOSystem] = None,
                 filename: Optional[str] = None) -> float:
        """
        Evaluate model.

        Parameters
        ----------
        environment
            The environment with which the neural system interacts. Can be None
            if we evaluate only on pre-recorded trajectories from
            `data_loader`.
        data_loader
            Data loaders for train and test set. Contain trajectories in state
            space to provide initial values or learning signal.
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
            data = data.as_in_context(self.device)
            neuralsystem_outputs, environment_states = self.model(data[:1])
            neuralsystem_outputs = mx.nd.moveaxis(neuralsystem_outputs, 0, -1)
            environment_states = mx.nd.moveaxis(environment_states, 0, -1)
            loss = self.loss_function(environment_states, neuralsystem_outputs)
            validation_loss += loss.mean().asscalar()
        if filename is not None:
            fig = plot_phase_diagram({'x': environment_states[0, 0].asnumpy(),
                                      'v': environment_states[0, 1].asnumpy()},
                                     xt=[0, 0], show=False, xlim=[-1, 1],
                                     ylim=[-1, 1], line_label='RNN')
            fig = plot_phase_diagram({'x': data[:, 0, 0].asnumpy(),
                                      'v': data[:, 0, 1].asnumpy()},
                                     show=False, fig=fig, line_label='LQR')
            fig.legend()
            mlflow.log_figure(fig, os.path.join('figures', filename))
        return validation_loss / len(data_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_lqr.get_config()

    # Get trajectories of a classic LQR controller in the double integrator
    # state space. In this study we use only the initial values for training
    # the RNN, and plot the LQR trajectories as comparison.
    _data_dict = get_data(_config, 'states')

    pipeline = LqrPipeline(_config, _data_dict)
    pipeline.main()

    sys.exit()
