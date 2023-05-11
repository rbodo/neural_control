import os
import sys
import logging
from stable_baselines3.common.logger import Figure
from typing import Union, Tuple, Optional, Callable

import mlflow
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
import seaborn as sns

from scratch import configs
from examples.linear_rnn_rl import (LinearRlPipeline, run_single,
                                    linear_schedule)
from src.control_systems_torch import (SteinmetzGym, RnnModel, ControlledMlp,
                                       BidirectionalControlledMlp)
from src.ppo_recurrent import (RecurrentPPO, ControlledExtractorMlpRnnPolicy,
                               CnnExtractor)
from src.utils import get_artifact_path


class QuadraticDecoder(torch.nn.Module):
    def __init__(self, a, b, device):
        super().__init__()
        self.a = a
        self.b = b
        self.device = device

    def forward(self, x):
        y = []
        for xt in x:
            y.append(self.a * xt * (self.b * xt - 1))
        return torch.unsqueeze(torch.stack(y, 0), 0)


def plot_trajectory(infos, z, path=None, show=True, ylim=None):

    plt.close()

    times = []
    responses = []
    positions = []
    for info in infos:
        times.append(info['time'])
        responses.append(info['response'])
        positions.append(info['position'])
    info = infos[-1]
    time_stimulus = info['time_stimulus']
    time_gocue = info['time_gocue']
    time_end = min(info['time_end'], info['time'])  # May have stopped early.
    correct_response = info['correct_response'] or 0
    correct_response *= 90
    reward = info['reward']

    fig, (ax0, ax1) = plt.subplots(2, sharex='all')

    # Draw trajectory.
    ax0.plot(times, positions, label='Wheel position')
    ax0.hlines(correct_response, time_gocue, time_end,
               linestyle='--', color='k', label='Target')
    ax0.axvline(time_stimulus, linestyle=':', color='k')
    ax0.axvline(time_gocue, linestyle=':', color='k')
    ax0.axvline(time_end, linestyle=':', color='k')
    ax0.set_ylabel('Stimulus position [dva]')
    if reward > 0:
        ax0.set_title(f'Response: {responses[-1]} (correct).',
                      dict(color='green'))
    else:
        ax0.set_title(f'Response: {responses[-1]} (false).', dict(color='red'))

    ax1.plot(times, np.concatenate(z, 0))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Firing rate [a.u.]')

    ax0.legend()

    if ylim is not None:
        ax0.set_ylim(-ylim, ylim)
        ax1.set_ylim(0, 10)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def set_weights_neuralsystem(model, device):
    weights = np.load('/home/bodrue/PycharmProjects/Thesis/empirical/'
                      'steinmetz_weights.npz')
    J = torch.tensor(weights['J'], device=device)
    B = torch.tensor(weights['B'], device=device)
    b = torch.tensor(weights['b'], device=device)
    W = torch.tensor(weights['W'], device=device)
    W_cnn = torch.tensor(weights['W_cnn'], device=device)
    b_cnn = torch.tensor(weights['b_cnn'], device=device)
    W_fc = torch.tensor(weights['W_fc'], device=device)
    b_fc = torch.tensor(weights['b_fc'], device=device)
    tau = torch.tensor(weights['tau'], device=device)
    model.policy.features_extractor.cnn[0].weight.data = W_cnn
    model.policy.features_extractor.cnn[0].bias.data = b_cnn
    model.policy.features_extractor.linear[0].weight.data = W_fc
    model.policy.features_extractor.linear[0].bias.data = b_fc
    model.policy.lstm_actor.weight_ih_l0.data = B
    model.policy.lstm_actor.weight_hh_l0.data = J
    model.policy.lstm_actor.bias_hh_l0.data = b
    model.policy.lstm_actor.bias_ih_l0.data[:] = 0
    model.policy.lstm_actor.tau.data = tau
    model.policy.mlp_extractor.policy_net.neuralsystem[0].weight.data = W
    model.policy.lstm_actor.flatten_parameters()
    use_quadratic_decoder = False
    if use_quadratic_decoder:
        weights = dict(np.load('/home/bodrue/PycharmProjects/Thesis/empirical/'
                               'steinmetz_weights_decoder.npz'))
        W = torch.tensor(weights.pop('W'), device=device)
        b = torch.tensor(weights.pop('b'), device=device)
        model.policy.action_net.weight.data = W
        model.policy.action_net.bias.data = b
        for k, v in weights.items():
            getattr(model.policy.mlp_extractor.policy_net.decoder, k).data = \
                torch.tensor(v, device=device)
        model.policy.mlp_extractor.policy_net.decoder.flatten_parameters()


class EvaluationCallback(BaseCallback):
    """
    Custom torch callback to evaluate an RL agent and log a phase diagram.
    """
    def __init__(self, num_test: int, evaluate_function: Callable,
                 reward_norm: Optional[float] = 1):
        super().__init__()
        self.num_test = num_test
        self.evaluate = evaluate_function
        self.reward_norm = reward_norm

    def _on_step(self) -> bool:
        n = self.n_calls
        env = self.locals['env'].envs[0]
        test_accuracy, episode_length, figure = self.evaluate(
            env, self.num_test, f'trajectory_{n}.png')
        rewards = [b['r'] for b in self.model.ep_info_buffer]
        train_accuracy = np.mean(np.greater(rewards, 0)).item()
        self.logger.record('trajectory/eval', Figure(figure, close=True),
                           exclude=('stdout', 'log', 'json', 'csv'))
        self.logger.record('test/episode_reward', test_accuracy)
        mlflow.log_metric(key='test_accuracy', value=test_accuracy, step=n)
        self.logger.record('test/episode_length', episode_length)
        mlflow.log_metric(key='test_episode_length', value=episode_length,
                          step=n)
        mlflow.log_metric(key='training_accuracy', value=train_accuracy,
                          step=n)
        plt.close()
        return True

    def reset(self):
        self.n_calls = 0


def run_n(n: int, *args, **kwargs) -> Tuple[list, list]:
    """Run an RL agent for `n` episodes and return reward and run lengths."""
    final_rewards = []
    episode_lengths = []
    for i in range(n):
        states, rewards = run_single(*args, **kwargs)
        final_rewards.append(rewards[-1])
        episode_lengths.append(len(rewards))
    return final_rewards, episode_lengths


class SteinmetzRlPipeline(LinearRlPipeline):
    def __init__(self, config):
        super().__init__(config)
        num_test = config.training.NUM_TEST
        self.evaluation_callback = EvaluationCallback(num_test, self.evaluate)

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
            A tuple with the accuracy and episode length.
        """
        final_rewards, episode_lengths = run_n(n, env, self.model,
                                               deterministic=deterministic)
        f = None
        if filename is not None:
            policy_net = self.model.policy.mlp_extractor.policy_net
            policy_net.neuralsystem_outputs = []
            _, _, infos = run_single(env, self.model, return_infos=True)
            with sns.axes_style('ticks'):
                f = plot_trajectory(infos, policy_net.neuralsystem_outputs,
                                    show=False, ylim=100)
            mlflow.log_figure(f, os.path.join('figures', filename))
            del policy_net.neuralsystem_outputs
        accuracy = np.mean(np.greater(final_rewards, 0)).item()
        mean_length = np.mean(episode_lengths).item()
        return accuracy, mean_length, f

    def get_environment(self, *args, **kwargs) -> Union[SteinmetzGym,
                                                        TimeLimit]:
        """Create a double integrator gym environment."""
        contrast_levels = self.config.process.CONTRAST_LEVELS
        time_stimulus = self.config.process.TIME_STIMULUS
        timeout_wait = self.config.process.TIMEOUT_WAIT
        gocue_wait = self.config.process.GOCUE_WAIT
        dt = self.config.process.DT
        environment = SteinmetzGym(contrast_levels=contrast_levels,
                                   time_stimulus=time_stimulus,
                                   timeout_wait=timeout_wait,
                                   gocue_wait=gocue_wait, dt=dt)
        return environment

    def get_model(self, freeze_neuralsystem, freeze_actor, freeze_controller,
                  environment, load_weights_from=None) -> RecurrentPPO:
        neuralsystem_num_inputs = 32
        neuralsystem_num_states = 3
        neuralsystem_num_hidden = self.config.model.NUM_HIDDEN_NEURALSYSTEM
        neuralsystem_num_layers = self.config.model.NUM_LAYERS_NEURALSYSTEM
        controller_num_states = self.config.model.NUM_HIDDEN_CONTROLLER
        controller_num_layers = self.config.model.NUM_LAYERS_CONTROLLER
        activation_rnn = self.config.model.ACTIVATION
        activation_decoder = 'relu'  # Defaults to 'linear'
        learning_rate = self.config.training.LEARNING_RATE
        time_stimulus = self.config.process.TIME_STIMULUS
        timeout_wait = self.config.process.TIMEOUT_WAIT
        gocue_wait = self.config.process.GOCUE_WAIT
        dt = self.config.process.DT
        num_steps = int((time_stimulus + gocue_wait + timeout_wait) / dt)
        dtype = torch.float32
        use_bidirectional_controller = \
            self.config.model.USE_BIDIRECTIONAL_CONTROLLER

        controller_num_inputs = neuralsystem_num_inputs
        if use_bidirectional_controller:
            controller_num_inputs += neuralsystem_num_states

        controller = RnnModel(
            controller_num_states, controller_num_layers,
            neuralsystem_num_states, controller_num_inputs,
            activation_rnn, activation_decoder, self.device, dtype)
        if load_weights_from is None:
            controller.init_zero()

        decoder = torch.nn.RNN(neuralsystem_num_states, 64, 1, dtype=dtype,
                               device=self.device, nonlinearity=activation_rnn)

        policy_kwargs = {
            'lstm_hidden_size': neuralsystem_num_hidden,
            'n_lstm_layers': neuralsystem_num_layers,
            'activation_fn': torch.nn.Identity,  # Applied at mlp_extractor
            'net_arch': {'pi': [neuralsystem_num_states],
                         'controller': controller,
                         'decoder': decoder,
                         'a_max': 6.25,
                         'mlp_extractor_class': BidirectionalControlledMlp if
                         use_bidirectional_controller else ControlledMlp},
            'features_extractor_class': CnnExtractor,
            'features_extractor_kwargs': {
                'features_dim': neuralsystem_num_inputs},
            'dt': dt
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

        model.policy.features_extractor.requires_grad_(not freeze_neuralsystem)
        model.policy.lstm_actor.requires_grad_(not freeze_neuralsystem)
        model.policy.mlp_extractor.policy_net.neuralsystem.requires_grad_(
            not freeze_neuralsystem)
        controller.requires_grad_(not freeze_controller)
        model.policy.mlp_extractor.policy_net.decoder.requires_grad_(
            not freeze_actor)
        model.policy.action_net.requires_grad_(not freeze_actor)

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
        is_perturbed = perturbation_type in ['linear', 'random', 'noise']
        if is_perturbed:
            freeze_neuralsystem = True
            freeze_actor = False#
            freeze_controller = False
            # Initialize model using unperturbed, uncontrolled baseline from
            # previous run.
            path_model = self.config.paths.FILEPATH_MODEL
            self.model = self.get_model(
                freeze_neuralsystem, freeze_actor, freeze_controller,
                environment, load_weights_from=path_model)
            num_epochs = int(self.config.training.NUM_EPOCHS_CONTROLLER)
        else:
            freeze_neuralsystem = True  # Use pretrained weights.
            freeze_actor = False
            freeze_controller = True
            self.model = self.get_model(
                freeze_neuralsystem, freeze_actor, freeze_controller,
                environment)
            num_epochs = int(self.config.training.NUM_EPOCHS_NEURALSYSTEM)

        controlled_neuralsystem = self.model.policy.mlp_extractor.policy_net

        # Apply perturbation to neural system.
        if is_perturbed and perturbation_level > 0:
            self.apply_perturbation(perturbation_type)

        # Get baseline performance before training.
        logging.info("Computing baseline performances...")
        reward, length, _ = self.evaluate(environment, num_test,
                                          deterministic=False)
        mlflow.log_metric('training_accuracy', reward, -1)
        reward, length, _ = self.evaluate(environment, num_test,
                                          'trajectory_-1.png')
        mlflow.log_metric('test_accuracy', reward, -1)

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
            dict(controller=freeze_controller,
                 neuralsystem=freeze_neuralsystem,
                 decoder=freeze_actor))

        if save_model:
            path_model = get_artifact_path('models/rnn.params')
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            self.model.save(path_model)
            logging.info(f"Saved model to {path_model}.")

    # def apply_perturbation(self, rng: np.random.Generator):
    #     with torch.no_grad():
    #         w = self.model.policy.mlp_extractor.policy_net.neuralsystem[0].weight
    #         w_np = w.data.cpu().numpy()
    #         rng.shuffle(w_np)
    #         w.data[:] = torch.tensor(w_np, device=self.device)
    #         w.requires_grad_(False)
    #         # self.model.policy.lstm_actor.weight_ih_l0.data[:] = 0
    #         # self.model.policy.lstm_actor.weight_ih_l0.data[:] *= -0.001

    def apply_perturbation(self, kind='linear', perturbation_level=1):
        with torch.no_grad():
            w = self.model.policy.lstm_actor.weight_ih_l0.data
            if kind == 'linear':
                w[:] = w * (1 - perturbation_level)
            elif kind == 'random':
                w[:] = torch.rand_like(w) * perturbation_level
            elif kind == 'noise':
                w[:] = w + torch.randn_like(w) * w.std() * perturbation_level
            else:
                raise NotImplementedError


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.config_steinmetz_rnn_rl.get_config()

    pipeline = SteinmetzRlPipeline(_config)
    pipeline.main()

    sys.exit()
