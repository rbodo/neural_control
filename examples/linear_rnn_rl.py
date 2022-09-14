import logging
import os
import sys
from typing import Callable, Optional, Tuple, Union

import gym
import mlflow
import numpy as np
import torch
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Figure

from examples import configs
from examples.linear_rnn_lqr import NeuralPerturbationPipeline
from src.control_systems_torch import DiGym, Masker, RnnModel, Gramians
from src.plotting import plot_phase_diagram
from src.ppo_recurrent import RecurrentPPO, MlpRnnPolicy
from src.utils import get_artifact_path, Monitor


class EvaluationCallback(BaseCallback):
    """
    Custom torch callback to evaluate an RL agent and log a phase diagram.
    """
    def _on_step(self) -> bool:
        n = self.n_calls
        env = self.locals['env'].envs[0]
        episode_reward, episode_length = evaluate(self.model, env, 100)
        states, rewards = run_single(env, self.model)
        figure = plot_phase_diagram({'x': states[:, 0, 0],
                                     'v': states[:, 0, 1]}, xt=[0, 0],
                                    show=False, xlim=[-1, 1], ylim=[-1, 1],
                                    draw_endpoints=True)
        self.logger.record('trajectory/eval', Figure(figure, close=True),
                           exclude=('stdout', 'log', 'json', 'csv'))
        self.logger.record('test/episode_reward', episode_reward)
        self.logger.record('test/episode_length', episode_length)
        mlflow.log_figure(figure, f'figures/trajectory_{n}.png')
        mlflow.log_metric(key='test_reward', value=episode_reward, step=n)
        mlflow.log_metric(key='test_episode_length', value=episode_length,
                          step=n)
        mlflow.log_metric(key='training_reward',
                          value=self.model.ep_info_buffer[-1]['r'], step=n)
        plt.close()
        return True


def evaluate(model: Union[RecurrentPPO, BaseAlgorithm], env: DiGym, n: int,
             filename: Optional[str] = None,
             deterministic: Optional[bool] = True) -> Tuple[float, float]:
    """Evaluate RL agent and optionally plot phase diagram.

    Parameters
    ----------
    model
        The policy to evaluate.
    env
        The environment to evaluate `model` in.
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
    reward_means, episode_lengths = run_n(n, env, model,
                                          deterministic=deterministic)
    if filename is not None:
        states, rewards = run_single(env, model)
        fig = plot_phase_diagram({'x': states[:, 0, 0], 'v': states[:, 0, 1]},
                                 xt=[0, 0], show=False, xlim=[-1, 1],
                                 ylim=[-1, 1], draw_endpoints=True)
        mlflow.log_figure(fig, os.path.join('figures', filename))
    return np.mean(reward_means).item(), np.mean(episode_lengths).item()


def run_n(n: int, *args, **kwargs) -> Tuple[list, list]:
    """Run an RL agent for `n` episodes and return reward and run lengths."""
    reward_means = []
    episode_lengths = []
    for i in range(n):
        states, rewards = run_single(*args, **kwargs)
        reward_means.append(np.sum(rewards).item())
        episode_lengths.append(len(rewards))
    return reward_means, episode_lengths


def run_single(env: DiGym, model: Union[RecurrentPPO, BaseAlgorithm],
               x0: Optional[np.ndarray] = None,
               monitor: Optional[Monitor] = None,
               deterministic: Optional[bool] = True
               ) -> Tuple[np.ndarray, list]:
    """Run an RL agent for one episode and return states and rewards.

    Parameters
    ----------
    model
        The policy to evaluate.
    env
        The environment to evaluate `model` in.
    x0
        Initial states. Zero if not specified.
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
    y = env.reset(state_init=x0)
    x = env.states.cpu().numpy()

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    states = []
    rewards = []
    while True:
        u, lstm_states = model.predict(y, state=lstm_states,
                                       episode_start=episode_starts,
                                       deterministic=deterministic)
        states.append(x)
        if monitor is not None:
            monitor.update_variables(t, states=x, outputs=y, control=u,
                                     cost=env.cost)

        y, reward, done, info = env.step(u)
        rewards.append(reward)
        x = env.states.cpu().numpy()
        t += env.process.dt
        episode_starts = done
        if done:
            env.reset()
            break
    return np.expand_dims(states, 1), rewards


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

    @staticmethod
    def evaluate(model, env, n, filename=None, deterministic=True):
        return evaluate(model, env, n, filename, deterministic)

    def get_environment(self, **kwargs) -> gym.Env:
        """Create a double integrator gym environment."""
        neuralsystem_num_outputs = 1
        environment_num_outputs = 1
        environment_num_states = 2
        process_noise = self.config.process.PROCESS_NOISE
        observation_noise = self.config.process.OBSERVATION_NOISE
        T = self.config.simulation.T
        num_steps = self.config.simulation.NUM_STEPS
        dt = T / num_steps
        device = kwargs['device']

        environment = DiGym(neuralsystem_num_outputs, environment_num_outputs,
                            environment_num_states, device, process_noise,
                            observation_noise, dt, cost_threshold=1e-4,
                            use_observations_in_cost=True)
        environment = TimeLimit(environment, num_steps)
        return environment

    def get_model(self, device, freeze_neuralsystem, freeze_controller,
                  load_weights_from=None, environment: gym.Env = None
                  ) -> RecurrentPPO:
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
                              activation_rnn, activation_decoder, device,
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
            MlpRnnPolicy, environment, verbose=0, device=device,
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
              device, save_model=True, **kwargs):
        num_epochs = int(self.config.training.NUM_EPOCHS)
        num_steps = self.config.simulation.NUM_STEPS
        T = self.config.simulation.T
        dt = T / num_steps
        seed = self.config.SEED
        evaluation_rate = 0.2
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        num_test = 100

        # Create environment.
        environment = self.get_environment(device=device, rng=rng)

        # Create model consisting of neural system, controller, and RL agent.
        is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
        if is_perturbed:
            # Initialize model using unperturbed, uncontrolled baseline from
            # previous run.
            path_model = self.config.paths.FILEPATH_MODEL
            model = self.get_model(
                device, freeze_neuralsystem=True, freeze_controller=False,
                environment=environment, load_weights_from=path_model)
        else:
            model = self.get_model(
                device, freeze_neuralsystem=False, freeze_controller=True,
                environment=environment)

        controlled_neuralsystem = model.policy.lstm_actor

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            add_noise(model, perturbation_type, perturbation_level, dt, rng)

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        reward, length = self.evaluate(model, environment, num_test,
                                       deterministic=False)
        mlflow.log_metric('training_reward', reward, -1)
        reward, length = self.evaluate(model, environment, num_test,
                                       'trajectory_-1.png')
        mlflow.log_metric('test_reward', reward, -1)

        # Initialize controller weights.
        if is_perturbed:
            # Initialize controller to non-zero weights only after computing
            # baseline accuracy of perturbed network before training.
            # Otherwise the untrained controller output will degrade
            # accuracy further.
            controlled_neuralsystem.controller.init_nonzero()

        masker = Masker(controlled_neuralsystem, dropout_probability, rng)
        callbacks = [EveryNTimesteps(int(num_epochs * evaluation_rate),
                                     EvaluationCallback()),
                     MaskingCallback(masker)]

        logging.info("Training...")
        model.learn(num_epochs, callback=callbacks)

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            model.save(path_model)
            logging.info(f"Saved model to {path_model}.")

        # Compute controllability and observability Gramians.
        if is_perturbed:
            with torch.no_grad():
                logging.info("Computing Gramians...")
                gramians = Gramians(controlled_neuralsystem, environment,
                                    model.policy.action_net, T)
                g_c = gramians.compute_controllability()
                g_o = gramians.compute_observability()
                g_c_eig = np.linalg.eig(g_c)
                g_o_eig = np.linalg.eig(g_o)
                # Use product of eigenvalue spectrum as scalar matric for
                # controllability and observability. The larger the better.
                controllability = np.prod(g_c_eig[0]).item()
                observability = np.prod(g_o_eig[0]).item()
                mlflow.log_metrics({'controllability': controllability,
                                    'observability': observability})
        else:
            g_c = g_o = None

        training_reward = model.ep_info_buffer[-1]['r']
        test_reward, length = self.evaluate(model, environment, num_test)
        return {'training_reward': training_reward, 'test_reward': test_reward,
                'controllability': g_c, 'observability': g_o}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_rl.get_config()

    pipeline = LinearRlPipeline(_config)
    pipeline.main()

    sys.exit()
