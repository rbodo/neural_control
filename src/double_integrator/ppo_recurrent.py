import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union, Tuple, NamedTuple, \
    Generator

import gym
from gym import spaces
import numpy as np
import torch as th
from sb3_contrib.common.recurrent.type_aliases import \
    RecurrentRolloutBufferSamples
from torch import nn
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, \
    create_sequencers
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, \
    FlattenExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, \
    Schedule
from stable_baselines3.common.utils import explained_variance, \
    get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.utils import zip_strict
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


class RNNStates(NamedTuple):
    pi: th.Tensor
    vf: th.Tensor


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer :param
    observation_space: Observation space :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm
    states (n_steps, lstm.num_layers, n_envs, lstm.hidden_size) :param
    device: PyTorch device :param gae_lambda: Factor for trade-off of bias
    vs variance for Generalized Advantage Estimator Equivalent to classic
    advantage when set to 1. :param gamma: Discount factor :param n_envs:
    Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.hidden_states_pi = None
        self.hidden_states_vf = None
        super().__init__(buffer_size, observation_space, action_space, device,
                         gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, np.float32)

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param lstm_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi.cpu().numpy()
        self.hidden_states_vf[self.pos] = lstm_states.vf.cpu().numpy()

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> \
            Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers,
            # self.n_envs, lstm.hidden_size) swap first to (self.n_steps,
            # self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "hidden_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape)
            #   -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape)
            #   -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "hidden_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = \
                    self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:],
                                  indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
            self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx: start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    # noinspection PyMethodOverriding
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = \
            create_sequencers(self.episode_starts[batch_inds],
                              env_change[batch_inds], self.device)

        n_layers = self.hidden_states_pi.shape[1]
        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
        lstm_states_pi = self.hidden_states_pi[batch_inds][
            self.seq_start_indices].reshape(n_layers, n_seq, -1)
        # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
        lstm_states_vf = self.hidden_states_vf[batch_inds][
            self.seq_start_indices].reshape(n_layers, n_seq, -1)
        lstm_states_pi = self.to_torch(lstm_states_pi)
        lstm_states_vf = self.to_torch(lstm_states_vf)

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim)
            #   -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape(
                (padded_batch_size,) + self.obs_shape),
            actions=self.pad(self.actions[batch_inds]).reshape(
                (padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(
                self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


# noinspection PyIncorrectDocstring,PyMethodOverriding
class MlpRnnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class:
            Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.RNN(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (self.shared_lstm and self.enable_critic_lstm), \
            "You must choose between shared LSTM, seperate or no LSTM for " \
            "the critic"

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.RNN(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        # noinspection PyArgumentList
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
        lstm: nn.RNN,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous cell and hidden states of the LSTM
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic (sequence length, batch size, features dim) (batch size
        # = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states.shape[1]
        # Batch to sequence (padded batch size, features_dim) -> (n_seq,
        # max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data
        # collection
        features_sequence = features.reshape((n_seq, -1,
                                              lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0,
                                     end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence,
                                                  episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                # Reset the states at the beginning of a new episode
                (1.0 - episode_start).view(1, n_seq, 1) * lstm_states)
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1),
                                 start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation :param lstm_states: The last
        hidden and memory states for the LSTM. :param episode_starts:
        Whether the observations correspond to new episodes or not (we reset
        the lstm states in that case). :param deterministic: Whether to
        sample or use deterministic actions :return: action, value and log
        probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = lstm_states_pi.detach()
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi,
                                                    lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, th.Tensor]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation. :param lstm_states: The last hidden and
        memory states for the LSTM. :param episode_starts: Whether the
        observations correspond to new episodes or not (we reset the lstm
        states in that case). :return: the action distribution and new
        hidden states.
        """
        features = self.extract_features(obs)
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the
        observations.

        :param obs: Observation. :param lstm_states: The last hidden and
        memory states for the LSTM. :param episode_starts: Whether the
        observations correspond to new episodes or not (we reset the lstm
        states in that case). :return: the estimated values.
        """
        features = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation. :param actions: :param lstm_states: The
        last hidden and memory states for the LSTM. :param episode_starts:
        Whether the observations correspond to new episodes or not (we reset
        the lstm states in that case). :return: estimated value,
        log likelihood of taking those actions and entropy of the action
        distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi,
                                              episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation: :param lstm_states: The last hidden and memory
        states for the LSTM. :param episode_starts: Whether the observations
        correspond to new episodes or not (we reset the lstm states in that
        case). :param deterministic: Whether to use stochastic or
        deterministic actions :return: Taken action according to the policy
        and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), \
            lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden
        state). Includes sugar-coating to handle different observations (
        e.g. normalizing images).

        :param observation: the input observation :param lstm_states: The
        last hidden and memory states for the LSTM. :param episode_starts:
        Whether the observations correspond to new episodes or not (we reset
        the lstm states in that case). :param deterministic: Whether or not
        to return deterministic actions. :return: the model's action and the
        next hidden state (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape)
                                    for _ in range(n_envs)], axis=1)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state, dtype=th.float).to(self.device)
            episode_starts = th.tensor(episode_start).float().to(self.device)
            actions, states = self._predict(
                observation, lstm_states=states, episode_starts=episode_starts,
                deterministic=deterministic
            )
            states = states.cpu().numpy()

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions
                # to avoid out of bound error (e.g. if sampling from a
                # Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, states


class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms
    /ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy,
    ...) :param env: The environment to learn from (if registered in Gym,
    can be str) :param learning_rate: The learning rate, it can be a
    function of the current progress remaining (from 1 to 0) :param n_steps:
    The number of steps to run for each environment per update (i.e. batch
    size is n_steps * n_env where n_env is number of environment copies
    running in parallel) :param batch_size: Minibatch size :param n_epochs:
    Number of epoch when optimizing the surrogate loss :param gamma:
    Discount factor :param gae_lambda: Factor for trade-off of bias vs
    variance for Generalized Advantage Estimator :param clip_range: Clipping
    parameter, it can be a function of the current progress remaining (from
    1 to 0). :param clip_range_vf: Clipping parameter for the value
    function, it can be a function of the current progress remaining (from 1
    to 0). This is a parameter specific to the OpenAI implementation. If
    None is passed (default), no clipping will be done on the value
    function. IMPORTANT: this clipping depends on the reward scaling. :param
    normalize_advantage: Whether to normalize or not the advantage :param
    ent_coef: Entropy coefficient for the loss calculation :param vf_coef:
    Value function coefficient for the loss calculation :param
    max_grad_norm: The maximum value for the gradient clipping :param
    target_kl: Limit the KL divergence between updates, because the clipping
    is not enough to prevent large update see issue #213 (cf
    https://github.com/hill-a/stable-baselines/issues/213) By default,
    there is no limit on the kl div. :param tensorboard_log: the log
    location for tensorboard (if None, no logging) :param create_eval_env:
    Whether to create a second environment that will be used for evaluating
    the agent periodically. (Only available when passing string for the
    environment) :param policy_kwargs: additional arguments to be passed to
    the policy on creation :param verbose: the verbosity level: 0 no output,
    1 info, 2 debug :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    Setting it to auto, the code will be run on the GPU if possible. :param
    _init_setup_model: Whether or not to build the network at the creation
    of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 128,
            batch_size: Optional[int] = 128,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (RecurrentDictRolloutBuffer if
                      isinstance(self.observation_space, gym.spaces.Dict)
                      else RecurrentRolloutBuffer)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        single_hidden_state_shape = (lstm.num_layers, self.n_envs,
                                     lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device))

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers,
                                     self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, \
                    "`clip_range_vf` must be positive, pass `None` to " \
                    "deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
            self,
            total_timesteps: int,
            eval_env: Optional[GymEnv],
            callback: MaybeCallback = None,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "RecurrentPPO",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to
        train on :param eval_env: Environment to use for evaluation. :param
        callback: Callback(s) called at every step with state of the
        algorithm. :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be
        saved :param reset_num_timesteps: Whether to reset or not the
        ``num_timesteps`` attribute :param tb_log_name: the name of the run
        for tensorboard log :return:
        """

        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        return total_timesteps, callback

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a
        ``RolloutBuffer``. The term rollout here refers to the model-free
        notion and should not be used with the concept of rollout used in
        model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per
            environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer,
            (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, \
            "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        new_obs = None
        dones = None
        while n_steps < n_rollout_steps:
            if (self.use_sde and self.sde_sample_freq > 0 and
                    n_steps % self.sde_sample_freq == 0):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts).float().to(self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low,
                                          self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                        done_
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = lstm_states.vf[:, idx: idx + 1]
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(
                            self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state,
                            episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.vf,
                episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                     dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        loss = None

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                            advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the
                # first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range,
                                                      1 + clip_range)
                policy_loss = -th.mean(
                    th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = \
                    th.mean(th.gt(th.abs(th.subtract(ratio, 1)),
                                  clip_range).to(th.float)[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf,
                        clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(
                    ((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss +
                        self.vf_coef * value_loss)

                # Calculate approximate form of reverse KL Divergence for
                # early stopping see issue #417:
                # https://github.com/DLR-RM/stable-baselines3/issues/417 and
                # discussion in PR #419:
                # https://github.com/DLR-RM/stable-baselines3/pull/419 and
                # Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        ((th.exp(log_ratio) - 1) - log_ratio)[
                            mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > \
                        1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to "
                              f"reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                            self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std",
                               th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "RecurrentPPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "RecurrentPPO":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer,
                n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps,
                                                    total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / (
                                time.time() - self.start_time))
                self.logger.record("time/iterations", iteration,
                                   exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(
                        self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/moving_avg_of_total_episode_rewards",
                        safe_mean([ep_info["r"] for ep_info in
                                   self.ep_info_buffer]))
                    self.logger.record("rollout/moving_avg_of_episode_lengths",
                                       safe_mean([ep_info["l"] for ep_info
                                                  in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed",
                                   int(time.time() - self.start_time),
                                   exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps,
                                   exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
