import os

import logging
import mlflow
import numpy as np
import sys
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EveryNTimesteps
from typing import Union, Tuple, Optional

from gymnasium.wrappers import TimeLimit

from examples.linear_rnn_rl import LinearRlPipeline, run_n, run_single, \
    EvaluationCallback, linear_schedule
from scratch import configs
from src.control_systems_torch import SteinmetzGym, RnnModel
from src.ppo_recurrent import RecurrentPPO, ControlledExtractorMlpRnnPolicy
from src.utils import get_artifact_path


def plot_trajectory(actions, rewards, xt=None, path=None, show=True,
                    ylim=None):

    plt.close()

    # Draw trajectory.
    plt.plot(actions, label='Behavior (wheel speed)')
    plt.xlabel('Time')
    plt.ylabel('A.U.')

    plt.fill_between(np.arange(len(rewards)), *ylim, where=rewards,
                     color='green', alpha=0.2)

    # Draw target line.
    if xt is not None:
        plt.plot(xt, linestyle='--', color='k',
                 label='Stimulus (contrast grating)')

    plt.legend()

    if ylim is not None:
        plt.ylim(ylim)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()

    return plt.gcf()


def set_weights_neuralsystem(model, device):
    weights = np.load('/home/bodrue/PycharmProjects/Thesis/empirical/'
                      'steinmetz_weights.npz')
    J = torch.tensor(weights['J'], device=device)
    B = torch.tensor(weights['B'], device=device)
    b = torch.tensor(weights['b'], device=device)
    W = torch.tensor(weights['W'], device=device)
    model.policy.lstm_actor.weight_ih_l0.data = B
    model.policy.lstm_actor.weight_hh_l0.data = J
    model.policy.lstm_actor.bias_hh_l0.data = b
    model.policy.lstm_actor.bias_ih_l0.data[:] = 0
    model.policy.mlp_extractor.policy_net.neuralsystem[0].weight.data = W
    model.policy.lstm_actor.flatten_parameters()


class SteinmetzRlPipeline(LinearRlPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.reward_norm = (config.simulation.NUM_STEPS -
                            config.process.TIME_STIMULUS) + 1
        num_test = config.training.NUM_TEST
        self.evaluation_callback = EvaluationCallback(num_test, self.evaluate,
                                                      self.reward_norm)

    def evaluate(self, env: SteinmetzGym, n: int,
                 filename: Optional[str] = None,
                 deterministic: Optional[bool] = True
                 ) -> Tuple[float, float, None]:
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
            states, rewards, actions = run_single(env, self.model,
                                                  return_actions=True)
            f = plot_trajectory(actions, rewards, xt=np.diff(states),
                                show=False, ylim=[-1.1, 1.1])
            mlflow.log_figure(f, os.path.join('figures', filename))
        mean_reward = np.mean(reward_means).item()
        mean_length = np.mean(episode_lengths).item()
        return mean_reward, mean_length, f

    def get_environment(self, *args, **kwargs) -> Union[SteinmetzGym,
                                                        TimeLimit]:
        """Create a double integrator gym environment."""
        num_contrast_levels = self.config.process.NUM_CONTRAST_LEVELS
        time_stimulus = self.config.process.TIME_STIMULUS
        num_steps = self.config.simulation.NUM_STEPS
        T = self.config.simulation.T
        dt = T / num_steps
        environment = SteinmetzGym(num_contrast_levels=num_contrast_levels,
                                   time_stimulus=time_stimulus,
                                   time_end=num_steps, dt=dt)
        environment = TimeLimit(environment, num_steps)
        return environment

    def get_model(self, freeze_neuralsystem, freeze_controller, environment,
                  load_weights_from=None) -> RecurrentPPO:
        neuralsystem_num_inputs = 2
        neuralsystem_num_states = 3
        neuralsystem_num_hidden = self.config.model.NUM_HIDDEN_NEURALSYSTEM
        neuralsystem_num_layers = self.config.model.NUM_LAYERS_NEURALSYSTEM
        controller_num_states = self.config.model.NUM_HIDDEN_CONTROLLER
        controller_num_layers = self.config.model.NUM_LAYERS_CONTROLLER
        activation_rnn = self.config.model.ACTIVATION
        activation_decoder = 'relu'  # Defaults to 'linear'
        learning_rate = self.config.training.LEARNING_RATE
        num_steps = self.config.simulation.NUM_STEPS
        dtype = torch.float32
        use_bidirectional_controller = True

        controller_num_inputs = neuralsystem_num_inputs
        if use_bidirectional_controller:
            controller_num_inputs += neuralsystem_num_states

        controller = RnnModel(
            controller_num_states, controller_num_layers,
            neuralsystem_num_states, controller_num_inputs,
            activation_rnn, activation_decoder, self.device, dtype)
        if load_weights_from is None:
            controller.init_zero()

        policy_kwargs = {'lstm_hidden_size': neuralsystem_num_hidden,
                         'n_lstm_layers': neuralsystem_num_layers,
                         'activation_fn': torch.nn.Tanh,
                         'net_arch': {'pi': [3]},
                         'controller': controller,
                         'log_std_init': -1,
                         }
        log_dir = get_artifact_path('tensorboard_log')
        model = RecurrentPPO(
            ControlledExtractorMlpRnnPolicy, environment, verbose=0,
            device=self.device, seed=self.config.SEED, tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs, n_epochs=10, n_steps=num_steps,
            learning_rate=linear_schedule(learning_rate, 0.005),
            batch_size=None)

        if load_weights_from is None:
            set_weights_neuralsystem(model, self.device)
        else:
            model.set_parameters(load_weights_from)

        controller.requires_grad_(not freeze_controller)
        model.policy.action_net.requires_grad_(not freeze_neuralsystem)
        model.policy.lstm_actor.requires_grad_(False)
        model.policy.mlp_extractor.policy_net.neuralsystem.requires_grad_(
            False)

        return model

    def train(self, perturbation_type, perturbation_level, dropout_probability,
              save_model=True, **kwargs):
        seed = self.config.SEED
        evaluate_every_n = self.config.training.EVALUATE_EVERY_N
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        num_test = self.config.training.NUM_TEST

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

        controlled_neuralsystem = self.model.policy.mlp_extractor.policy_net

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            self.apply_perturbation()

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        reward, length, _ = self.evaluate(environment, num_test,
                                          deterministic=False)
        mlflow.log_metric('training_reward', reward / self.reward_norm, -1)
        reward, length, _ = self.evaluate(environment, num_test,
                                          'trajectory_-1.png')
        mlflow.log_metric('test_reward', reward / self.reward_norm, -1)

        # Initialize controller weights.
        if is_perturbed:
            # Initialize controller to non-zero weights only after computing
            # baseline accuracy of perturbed network before training.
            # Otherwise the untrained controller output will degrade
            # accuracy further.
            controlled_neuralsystem.controller.init_nonzero()

        callbacks = [EveryNTimesteps(evaluate_every_n,
                                     self.evaluation_callback)]
        self.evaluation_callback.reset()

        # Store the hash values of model weights so we can check later that
        # only the parts were trained that we wanted.
        controlled_neuralsystem.cache_weight_hash()

        logging.info("Training...")
        self.model.learn(num_epochs, callback=callbacks)

        controlled_neuralsystem.assert_plasticity(
            freeze_controller=freeze_controller)

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            self.model.save(path_model)
            logging.info(f"Saved model to {path_model}.")

    def apply_perturbation(self):
        # self.model.policy.lstm_actor.weight_ih_l0.data[:] = 0
        self.model.policy.lstm_actor.weight_ih_l0.data[:] *= -0.001


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.config_steinmetz_rnn_rl.get_config()

    pipeline = SteinmetzRlPipeline(_config)
    pipeline.main()

    sys.exit()
