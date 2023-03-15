import logging
import os
import sys
from typing import Callable, Optional, Tuple, Union, Sized

import mlflow
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Figure
from yacs.config import CfgNode

from examples import configs
from examples.linear_rnn_lqr import NeuralPerturbationPipeline
from src.control_systems_torch import (DiGym, Masker, RnnModel, Gramians,
                                       get_device)
from src.plotting import plot_phase_diagram
from src.ppo_recurrent import RecurrentPPO, MlpRnnPolicy
from src.utils import (get_artifact_path, Monitor,
                       get_additive_white_gaussian_noise, atleast_3d)


class POMDP(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, sigma: float, rng: np.random.Generator,
                 observation_indices: np.iterable, dt: float):
        """Custom gym ObservationWrapper to remove part of the observations and
        add noise to the others.

        Parameters
        ----------
        env
            Environment to wrap.
        sigma
            Standard deviation of gaussian noise distribution.
        rng
            Pseudo-random number generator.
        observation_indices
            Indices of states to observe.
        dt
            Time constant for stepping through the environment.
        """
        super().__init__(env)
        self.rng = rng
        self.observation_indexes = observation_indices
        self.dt = dt
        num_observations = len(self.observation_indexes)
        self.sigma = np.eye(num_observations) * sigma
        self._add_noise = sigma > 0
        # We add a dummy batch and time dimension to the observations to make
        # the environment compatible with the torch RNN pipeline.
        shape = (1, 1, num_observations)
        # noinspection PyUnresolvedReferences
        self.observation_space = gym.spaces.Box(
            low=atleast_3d(env.observation_space.low[observation_indices]),
            high=atleast_3d(env.observation_space.high[observation_indices]),
            shape=shape, dtype=env.observation_space.dtype)
        self.states = None

    def observation(self, observation):
        self.states = atleast_3d(observation)
        noise = 0
        if self._add_noise:
            noise = get_additive_white_gaussian_noise(self.sigma, rng=self.rng)
        partial_observation = observation[self.observation_indexes]
        return atleast_3d(partial_observation + noise)


class EvaluationCallback(BaseCallback):
    """
    Custom torch callback to evaluate an RL agent and log a phase diagram.
    """
    def __init__(self, num_test: int, evaluate_function: Callable):
        super().__init__()
        self.num_test = num_test
        self.evaluate = evaluate_function

    def _on_step(self) -> bool:
        n = self.n_calls
        env = self.locals['env'].envs[0]
        episode_reward, episode_length, figure = self.evaluate(
            env, self.num_test, f'trajectory_{n}.png')
        self.logger.record('trajectory/eval', Figure(figure, close=True),
                           exclude=('stdout', 'log', 'json', 'csv'))
        self.logger.record('test/episode_reward', episode_reward)
        mlflow.log_metric(key='test_reward', value=episode_reward, step=n)
        self.logger.record('test/episode_length', episode_length)
        mlflow.log_metric(key='test_episode_length', value=episode_length,
                          step=n)
        mlflow.log_metric(key='training_reward',
                          value=self.model.ep_info_buffer[-1]['r'], step=n)
        plt.close()
        return True

    def reset(self):
        self.n_calls = 0


def run_n(n: int, *args, **kwargs) -> Tuple[list, list]:
    """Run an RL agent for `n` episodes and return reward and run lengths."""
    reward_means = []
    episode_lengths = []
    for i in range(n):
        states, rewards = run_single(*args, **kwargs)
        reward_means.append(np.sum(rewards).item())
        episode_lengths.append(len(rewards))
    return reward_means, episode_lengths


def run_single(env: Union[DiGym, POMDP],
               model: Union[RecurrentPPO, BaseAlgorithm],
               monitor: Optional[Monitor] = None,
               deterministic: Optional[bool] = True
               ) -> Tuple[Union[np.ndarray, Sized], list]:
    """Run an RL agent for one episode and return states and rewards.

    Parameters
    ----------
    model
        The policy to evaluate.
    env
        The environment to evaluate `model` in.
    monitor
        A logging container.
    deterministic
        If `True`, the actions are selected deterministically.

    Returns
    -------
    results
        A tuple containing:

        - The environment states with shape (num_timesteps, 1, num_states). The
          second axis is a placeholder for the batch size.
        - Episode rewards with shape (num_timesteps,).
    """
    t = 0
    y, _ = env.reset()
    x = env.states
    reward = 0

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    states = []
    rewards = []
    while True:
        u, lstm_states = model.predict(y, lstm_states, episode_starts,
                                       deterministic)
        states.append(x)
        if monitor is not None:
            monitor.update_variables(t, states=x, outputs=y, control=u,
                                     cost=-reward)

        y, reward, terminated, truncated, info = env.step(u)
        done = terminated or truncated
        rewards.append(reward)
        x = env.states
        t += env.dt
        episode_starts = done
        if done:
            env.reset()
            break
    return np.concatenate(states), rewards


def linear_schedule(initial_value: float, q: float = 0.01
                    ) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Parameters
    ----------
    initial_value
        Initial learning rate.
    q
        Fraction of initial learning rate at end of progress.

    Returns
    -------
    out
        Schedule that computes current learning rate depending on remaining
        progress.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        Parameters
        ----------
        progress_remaining
            Fraction of remaining steps till end of training.
        Returns
        -------
        out
            current learning rate
        """
        return initial_value * (progress_remaining * (1 - q) + q)

    return func


def add_noise(model: RecurrentPPO, where: str, sigma: float, dt: float,
              rng: np.random.Generator):
    """Add gaussian noise to weights of neural system."""
    if where in (None, 'None', 'none', ''):
        return
    elif where == 'sensor':
        parameters = model.policy.lstm_actor.neuralsystem.weight_ih_l0
    elif where == 'processor':
        parameters = model.policy.lstm_actor.neuralsystem.weight_hh_l0
    elif where == 'actuator':
        parameters = model.policy.action_net.weight
    else:
        raise NotImplementedError
    noise = rng.standard_normal(parameters.shape) * sigma * np.sqrt(dt)
    parameters[:] = parameters + torch.tensor(noise, dtype=parameters.dtype,
                                              device=parameters.device)


class MaskingCallback(BaseCallback):
    """Custom torch callback to set some rows of weight matrix to zero."""
    def __init__(self, masker: Masker):
        super().__init__()
        self.masker = masker

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        self.masker.apply_mask()

    def _on_training_end(self):
        self.masker.apply_mask()


class LinearRlPipeline(NeuralPerturbationPipeline):
    def __init__(self, config: CfgNode):
        super().__init__(config)
        self.evaluation_callback = EvaluationCallback(100, self.evaluate)

    def get_device(self) -> torch.device:
        return get_device(self.config)

    def evaluate(self, env: DiGym, n: int, filename: Optional[str] = None,
                 deterministic: Optional[bool] = True
                 ) -> Tuple[float, float, Optional[plt.Figure]]:
        """Evaluate RL agent and optionally plot phase diagram.

        Parameters
        ----------
        env
            The environment to evaluate model in.
        n
            Number of evaluation runs.
        filename
            If specified, plot phase diagram and save under given name in the
            mlflow artifact directory.
        deterministic
            If `True`, the actions are selected deterministically.

        Returns
        -------
        results
            A tuple with the average reward and episode length.
        """
        reward_means, episode_lengths = run_n(n, env, self.model,
                                              deterministic=deterministic)
        f = None
        if filename is not None:
            states, rewards = run_single(env, self.model)
            f = plot_phase_diagram(
                {'x': states[:, 0, 0], 'v': states[:, 0, 1]}, xt=[0, 0],
                show=False, xlim=[-1, 1], ylim=[-1, 1], draw_endpoints=True)
            mlflow.log_figure(f, os.path.join('figures', filename))
        return np.mean(reward_means).item(), np.mean(episode_lengths).item(), f

    def get_environment(self, *args, **kwargs) -> Union[DiGym, TimeLimit]:
        """Create a double integrator gym environment."""
        num_inputs = self.config.process.NUM_INPUTS
        num_states = self.config.process.NUM_STATES
        num_outputs = self.config.process.NUM_OUTPUTS
        process_noise = self.config.process.PROCESS_NOISES[0]
        observation_noise = self.config.process.OBSERVATION_NOISES[0]
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps

        environment = DiGym(num_inputs, num_outputs, num_states, self.device,
                            process_noise, observation_noise, dt,
                            cost_threshold=1e-4, use_observations_in_cost=True)
        environment = TimeLimit(environment, num_steps)
        return environment

    def get_model(self, freeze_neuralsystem, freeze_controller, environment,
                  load_weights_from=None) -> RecurrentPPO:
        neuralsystem_num_states = self.config.model.NUM_HIDDEN_NEURALSYSTEM
        neuralsystem_num_layers = self.config.model.NUM_LAYERS_NEURALSYSTEM
        controller_num_states = self.config.model.NUM_HIDDEN_CONTROLLER
        controller_num_layers = self.config.model.NUM_LAYERS_CONTROLLER
        activation_rnn = self.config.model.ACTIVATION
        activation_decoder = None  # Defaults to 'linear'
        learning_rate = self.config.training.LEARNING_RATE
        num_steps = self.config.simulation.NUM_STEPS
        dtype = torch.float32

        controller = RnnModel(controller_num_states, controller_num_layers,
                              neuralsystem_num_states, neuralsystem_num_states,
                              activation_rnn, activation_decoder, self.device,
                              dtype)
        if load_weights_from is None:
            controller.init_zero()

        policy_kwargs = {'lstm_hidden_size': neuralsystem_num_states,
                         'n_lstm_layers': neuralsystem_num_layers,
                         'activation_fn': activation_rnn,
                         'net_arch': [],
                         'controller': controller,
                         # 'shared_lstm': True,
                         # 'enable_critic_lstm': False,
                         }
        log_dir = get_artifact_path('tensorboard_log')
        model = RecurrentPPO(
            MlpRnnPolicy, environment, verbose=0, device=self.device,
            seed=self.config.SEED,
            tensorboard_log=log_dir, policy_kwargs=policy_kwargs, n_epochs=10,
            learning_rate=linear_schedule(learning_rate, 0.005),
            n_steps=num_steps,
            batch_size=None)

        if load_weights_from is not None:
            model.set_parameters(load_weights_from)

        controller.requires_grad_(not freeze_controller)
        model.policy.action_net.requires_grad_(not freeze_neuralsystem)
        model.policy.lstm_actor.neuralsystem.requires_grad_(
            not freeze_neuralsystem)

        return model

    def train(self, perturbation_type, perturbation_level, dropout_probability,
              save_model=True, **kwargs):
        num_steps = self.config.simulation.NUM_STEPS
        T = self.config.simulation.T
        dt = T / num_steps
        seed = self.config.SEED
        evaluate_every_n = 5000
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        num_test = 100

        # Create environment.
        environment = self.get_environment(rng=rng)

        # Create model consisting of neural system, controller, and RL agent.
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
            num_epochs = int(self.config.training.NUM_EPOCHS_CONTROLLER)
        else:
            freeze_neuralsystem = False
            freeze_controller = True
            self.model = self.get_model(
                freeze_neuralsystem, freeze_controller, environment)
            num_epochs = int(self.config.training.NUM_EPOCHS_NEURALSYSTEM)

        controlled_neuralsystem = self.model.policy.lstm_actor

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            add_noise(self.model, perturbation_type, perturbation_level, dt,
                      rng)

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        reward, length, _ = self.evaluate(environment, num_test,
                                          deterministic=False)
        mlflow.log_metric('training_reward', reward, -1)
        reward, length, _ = self.evaluate(environment, num_test,
                                          'trajectory_-1.png')
        mlflow.log_metric('test_reward', reward, -1)

        # Initialize controller weights.
        if is_perturbed:
            # Initialize controller to non-zero weights only after computing
            # baseline accuracy of perturbed network before training.
            # Otherwise the untrained controller output will degrade
            # accuracy further.
            controlled_neuralsystem.controller.init_nonzero()

        # Reduce controllability and observability of neural system by
        # dropping out rows from the stimulation and readout matrices of the
        # controller.
        masker = Masker(controlled_neuralsystem, dropout_probability, rng)
        masker.apply_mask()
        callbacks = [EveryNTimesteps(evaluate_every_n,
                                     self.evaluation_callback),
                     MaskingCallback(masker)]
        self.evaluation_callback.reset()

        # Store the hash values of model weights so we can check later that
        # only the parts were trained that we wanted.
        controlled_neuralsystem.cache_weight_hash()

        logging.info("Training...")
        self.model.learn(num_epochs, callback=callbacks)

        controlled_neuralsystem.assert_plasticity(freeze_neuralsystem,
                                                  freeze_controller)

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            self.model.save(path_model)
            logging.info(f"Saved model to {path_model}.")

        # Compute controllability and observability Gramians.
        if self.config.get('COMPUTE_GRAMIANS', True):
            logging.info("Computing Gramians...")
            gramians = Gramians(controlled_neuralsystem, environment,
                                self.model.policy.action_net, T)
            with torch.no_grad():
                g_c = gramians.compute_controllability()
                g_o = gramians.compute_observability()

            np.savez_compressed(get_artifact_path('gramians.npz'),
                                controllability_gramian=g_c,
                                observability_gramian=g_o)
        mlflow.log_metrics({'controllability': masker.controllability,
                            'observability': masker.observability})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_rl.get_config()

    pipeline = LinearRlPipeline(_config)
    pipeline.main()

    sys.exit()
