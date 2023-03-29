import logging
import os
from gymnasium.core import RenderFrame, ActType, ObsType
from typing import Union, Tuple, Optional, Callable, List, Dict, Iterator, \
    SupportsFloat, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from yacs.config import CfgNode

from src.empirical_gramians import emgr
from src.utils import get_lqr_cost, atleast_3d


class MlpModel(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, num_inputs, num_hidden, num_outputs, activation_hidden,
                 activation_output, dtype, device):

        super().__init__()

        self.num_hidden = num_hidden

        self.hidden = nn.Linear(num_inputs, num_hidden, dtype=dtype,
                                device=device)
        self.activation_hidden = activation_hidden
        self.output = nn.Linear(num_hidden, num_outputs, dtype=dtype,
                                device=device)
        self.activation_output = activation_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation_hidden(self.hidden(x))
        output = self.activation_output(self.output(hidden))
        # Add dummy time dimension for compatibility with RNN interface.
        return output.unsqueeze(0)


class RnnModel(nn.Module):
    """Multi-layer Elman RNN with fully-connected decoder."""
    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1, input_size=1,
                 activation_rnn=None, activation_decoder=None, device=None,
                 dtype=None):

        super().__init__()

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.tkwargs = dict(dtype=dtype, device=device)

        self.rnn = nn.RNN(input_size, num_hidden, num_layers,
                          nonlinearity=activation_rnn, **self.tkwargs)
        self.decoder = nn.Linear(num_hidden, num_outputs, **self.tkwargs)
        if activation_decoder == 'relu':
            self.activation = nn.ReLU()
        elif activation_decoder == 'tanh':
            self.activation = nn.Tanh()
        elif activation_decoder in ['linear', None]:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError

    def init_zero(self):
        # noinspection PyProtectedMember
        for w in self.rnn._flat_weights:
            w.data.zero_()
        self.decoder.weight.data.zero_()
        self.decoder.bias.data.zero_()

    def init_nonzero(self):
        self.rnn.reset_parameters()
        self.decoder.reset_parameters()

    def begin_state(self, batch_size: Optional[int] = None) -> torch.Tensor:
        shape = [self.num_layers, batch_size, self.num_hidden] if batch_size \
            else [self.num_layers, self.num_hidden]
        return torch.zeros(*shape, **self.tkwargs)

    def forward(self, x: torch.Tensor, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(x, h)
        decoded = self.activation(self.decoder(output))
        return decoded, hidden


class StochasticLinearIOSystem(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_states, device, dt=0.1,
                 dtype=torch.float32):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_states = num_states

        self.tkwargs = dict(dtype=dtype, device=device)
        self.dt = nn.Parameter(torch.tensor(
            (dt,), **self.tkwargs), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(
            self.num_states, self.num_states, **self.tkwargs),
            requires_grad=False)
        self.B = nn.Parameter(torch.zeros(
            self.num_states, self.num_inputs, **self.tkwargs),
            requires_grad=False)
        self.C = nn.Parameter(torch.zeros(
            self.num_outputs, self.num_states, **self.tkwargs),
            requires_grad=False)
        self.D = nn.Parameter(torch.zeros(
            self.num_outputs, self.num_inputs, **self.tkwargs),
            requires_grad=False)
        self.W = None
        self.V = None

    def begin_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.num_states, **self.tkwargs)

    def step(self, x: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        dxdt = self.dynamics(x, u)
        x = self.integrate(x, dxdt, **kwargs)
        return self.add_process_noise(x, **kwargs)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (torch.tensordot(x, self.A.T, 1) +
                torch.tensordot(u, self.B.T, 1))

    def integrate(self, x: torch.Tensor, dxdt: torch.Tensor, **kwargs
                  ) -> torch.Tensor:
        method = kwargs.get('method', 'euler-maruyama')
        if method == 'euler-maruyama':  # x + dt * dx/dt
            return x + self.dt * dxdt
        else:
            raise NotImplementedError

    def add_process_noise(self, x: torch.Tensor, deterministic=False
                          ) -> torch.Tensor:
        if self.W is None or deterministic:
            return x
        dW = self.get_additive_white_gaussian_noise(
            self.num_states,
            torch.ones(self.num_states, **self.tkwargs) *
            torch.sqrt(self.dt))
        return x + torch.tensordot(dW.unsqueeze(0), self.W.T, 1).squeeze(0)

    def output(self, x: torch.Tensor, u: torch.Tensor,
               deterministic: Optional[bool] = False) -> torch.Tensor:
        y = (torch.tensordot(x, self.C.T, 1) +
             torch.tensordot(u, self.D.T, 1))
        return self.add_observation_noise(y, deterministic)

    def add_observation_noise(self, y: torch.Tensor,
                              deterministic: Optional[bool] = False
                              ) -> torch.Tensor:
        if self.V is None or deterministic:
            return y
        return y + self.get_additive_white_gaussian_noise(self.num_outputs,
                                                          torch.diag(self.V))

    def get_additive_white_gaussian_noise(self, n: int, scale: torch.Tensor
                                          ) -> torch.Tensor:
        return torch.normal(torch.zeros(n, **self.tkwargs), scale)

    def forward(self, x: torch.Tensor, u: torch.Tensor, **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.step(x, u, **kwargs)
        return self.output(x, u, **kwargs), x


class DI(StochasticLinearIOSystem):
    """Double integrator dynamical system."""
    def __init__(self, num_inputs, num_outputs, num_states, var_x: float = 0,
                 var_y: float = 0, dt: float = 0.1, dtype=torch.float32,
                 device='cuda'):

        super().__init__(num_inputs, num_outputs, num_states, device, dt,
                         dtype)

        self.A[0, 1] = 1
        self.B[1, 0] = 1  # Control only second state (acceleration).
        self.C.data = torch.eye(self.num_outputs, self.num_states,
                                **self.tkwargs)
        if var_x:
            self.W = nn.Parameter(
                torch.mul(var_x, torch.eye(self.num_states, **self.tkwargs)),
                requires_grad=False)
        if var_y:
            self.V = nn.Parameter(
                torch.mul(var_y, torch.eye(self.num_outputs, **self.tkwargs)),
                requires_grad=False)


class MLP:
    """Dynamical system controlled by a multi-layer perceptron."""
    def __init__(self, process: StochasticLinearIOSystem, num_hidden,
                 activation_hidden, activation_output, q=0.5, r=0.5,
                 path_model=None):

        self.process = process
        self.dt = self.process.dt.data.numpy()
        self.tkwargs = self.process.tkwargs
        dtype = np.dtype(str(self.process.dtype).split('.')[1])

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=dtype)

        self.model = MlpModel(self.process.num_outputs, num_hidden,
                              self.process.num_inputs, activation_hidden,
                              activation_output, **self.tkwargs)
        if path_model is not None:
            self.model.load_state_dict(torch.load(path_model))

    def get_cost(self, x: torch.Tensor, u: torch.Tensor) -> float:
        return get_lqr_cost(asnumpy(x), asnumpy(u), self.Q, self.R,
                            self.dt)

    def get_control(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, x: torch.Tensor, y: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        u = self.get_control(y)
        x = self.process.step(x, u)
        y = self.process.output(x, u)
        c = self.get_cost(x, u)
        return x, y, u, c

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = self.process.output(x, u)
        u = self.get_control(y)
        return self.process.dynamics(x, u)


class ControlledNeuralSystem(nn.Module):
    """Perturbed neural system stabilized by a controller RNN."""
    def __init__(self, neural_system: nn.Module, controller: RnnModel):
        super().__init__()
        self.neuralsystem = neural_system
        self.controller = controller
        self.tkwargs = self.controller.tkwargs
        self._weight_hashes = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tensordot(x, self.controller.rnn.weight_ih_l0.T, 1) +
                self.controller.rnn.bias_ih_l0)

    def readin(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tensordot(x, self.controller.decoder.weight.T, 1) +
                self.controller.decoder.bias)

    def begin_state(self, batch_size: int) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        neuralsystem_states = torch.zeros(self.num_layers, batch_size,
                                          self.hidden_size, **self.tkwargs)
        controller_states = self.controller.begin_state(batch_size)
        return neuralsystem_states, controller_states

    def get_weight_hash(self) -> Dict[str, List[int]]:
        """Get the hash values of the model parameters."""
        return {'neuralsystem': to_hash(self.neuralsystem.parameters()),
                'controller': to_hash(self.controller.parameters())}

    def cache_weight_hash(self):
        """Store the hash values of the model parameters."""
        self._weight_hashes = self.get_weight_hash()

    def is_static(self, key: str, weight_hashes: Dict[str, List[int]]) -> bool:
        return np.array_equal(weight_hashes[key], self._weight_hashes[key])

    def assert_plasticity(self, where: Dict[str, bool]):
        """Make sure only the allowed weights have been modified."""
        hashes = self.get_weight_hash()
        for key, is_frozen in where.items():
            assert self.is_static(key, hashes) == is_frozen


class ControlledRnn(ControlledNeuralSystem):
    """Perturbed RNN stabilized by a controller RNN."""
    def __init__(self, rnn: nn.RNN, controller: RnnModel):
        super().__init__(rnn, controller)
        self.input_size = self.neuralsystem.input_size
        self.num_layers = self.neuralsystem.num_layers
        self.hidden_size = self.neuralsystem.hidden_size

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        neuralsystem_states, controller_states = self.begin_state(x.shape[1])
        if h is not None:
            neuralsystem_states = h
        neuralsystem_outputs = []
        for neuralsystem_input in x:
            controller_output, controller_states = self.controller(
                neuralsystem_states, controller_states)
            neuralsystem_output, neuralsystem_states = self.neuralsystem(
                neuralsystem_input.unsqueeze(0),
                neuralsystem_states + controller_output)
            neuralsystem_outputs.append(neuralsystem_output)
        return torch.concat(neuralsystem_outputs, dim=0), neuralsystem_states


class ControlledMlp(ControlledNeuralSystem):
    """Perturbed MLP stabilized by a controller RNN."""
    def __init__(self, mlp: nn.Module, controller: RnnModel,
                 a_max: Optional[float] = None):
        super().__init__(mlp, controller)
        self.a_max = a_max

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        controller_states = self.begin_state()
        neuralsystem_outputs = []
        for neuralsystem_input, environment_output in zip(x, u):
            neuralsystem_input = neuralsystem_input.unsqueeze(0)
            environment_output = environment_output.unsqueeze(0)
            neuralsystem_output = self.neuralsystem(neuralsystem_input)
            controller_output, controller_states = self.controller(
                environment_output, controller_states)
            controller_output = torch.clip(controller_output, max=self.a_max)
            neuralsystem_output += controller_output
            neuralsystem_outputs.append(neuralsystem_output)
        return torch.concat(neuralsystem_outputs, dim=0)

    def begin_state(self, batch_size: Optional[int] = None) -> torch.Tensor:
        return self.controller.begin_state(batch_size)


class BidirectionalControlledMlp(ControlledMlp):
    """Perturbed MLP stabilized by a controller RNN which uses measurements
    from the MLP itself."""

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        controller_states = self.begin_state()
        neuralsystem_outputs = []
        for neuralsystem_input, environment_output in zip(x, u):
            neuralsystem_input = neuralsystem_input.unsqueeze(0)
            environment_output = environment_output.unsqueeze(0)
            neuralsystem_output = self.neuralsystem(neuralsystem_input)
            controller_input = torch.concat([neuralsystem_output,
                                             environment_output], -1)
            controller_output, controller_states = self.controller(
                controller_input, controller_states)
            controller_output = torch.clip(controller_output, max=self.a_max)
            neuralsystem_output += controller_output
            neuralsystem_outputs.append(neuralsystem_output)
        return torch.concat(neuralsystem_outputs, dim=0)


class DiMlp(MLP):
    """Double integrator controlled by a multi-layer perceptron."""
    def __init__(self, num_inputs, num_outputs, num_states, num_hidden,
                 activation_hidden, activation_output, var_x=0, var_y=0,
                 dt=0.1, q=0.5, r=0.5, path_model=None, dtype=torch.float32,
                 device='cuda'):
        process = DI(num_inputs, num_outputs, num_states, var_x, var_y, dt,
                     dtype, device)
        super().__init__(process, num_hidden, activation_hidden,
                         activation_output, q, r, path_model)


class StatefulGym(gym.Env):
    """gym.Env with a state and dt attribute."""

    def __init__(self, dt: float):
        super().__init__()
        self.dt = dt
        self.states = None

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool,
                                             bool, dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError


class DiGym(StatefulGym):
    """Double integrator that follows gym interface."""

    metadata = {'render.modes': ['console']}

    def __init__(self, num_inputs, num_outputs, num_states, device,
                 var_x: float = 0, var_y: float = 0, dt: float = 0.1,
                 cost_threshold=1e-3, state_threshold=None,
                 q: Union[float, np.iterable] = 0.5, r=0.5,
                 dtype=torch.float32, use_observations_in_cost=False):
        super().__init__(dt)
        self.tkwargs = dict(dtype=dtype, device=device)
        self.dtype = np.dtype(str(dtype).split('.')[1]).type
        self.use_observations_in_cost = use_observations_in_cost
        self.process = DI(num_inputs, num_outputs, num_states, var_x, var_y,
                          self.dt, **self.tkwargs)

        # Define action and observation spaces. For compatibility with RNNs,
        # we add two dummy dimensions to observation and state space for the
        # number of time steps and the batch size.
        self.min = -1
        self.max = 1
        self.action_space = spaces.Box(-10, 10, (1,), self.dtype)
        self.observation_space = spaces.Box(self.min, self.max,
                                            (1, 1, num_outputs), self.dtype)
        self.init_state_space = spaces.Box(self.min / 2, self.max / 2,
                                           (1, 1, num_states), self.dtype)
        self.cost_threshold = cost_threshold
        self.state_threshold = state_threshold or self.max

        # State cost matrix:
        if np.isscalar(q):
            dim = self.process.num_outputs if self.use_observations_in_cost \
                else self.process.num_states
            self.Q = q * np.eye(dim, dtype=self.dtype)
            self.Q_states = q * np.eye(self.process.num_states,
                                       dtype=self.dtype)
        else:
            self.Q = np.diag(q)
            self.Q_states = q * np.eye(self.process.num_states,
                                       dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self._states = None
        self.cost = None
        self.t = None

    @property
    def states(self) -> np.ndarray:
        """Get environment states as numpy array."""
        return asnumpy(self._states)

    def get_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        return get_lqr_cost(x[0, 0], u, self.Q, self.R, self.dt)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool,
                                                dict]:
        action_tensor = astensor(action, **self.tkwargs)
        self._states = self.process.step(self._states, action_tensor)
        torch.clip(self._states, self.min, self.max, out=self._states)

        observation = asnumpy(self.process.output(self._states, action_tensor))
        np.clip(observation, self.min, self.max, observation)

        x = observation if self.use_observations_in_cost else self.states
        self.cost = self.get_cost(x, action)

        # Use full state to determine whether agent succeeded.
        terminated = self.is_done(self.states, action)

        # Reward consists of negative LQR cost plus a bonus when reaching the
        # target in state space. This bonus decays over time to encourage fast
        # termination.
        reward = -self.cost + terminated * 10 * np.exp(-self.t / 4)

        self.t += self.dt

        return observation, reward, terminated, False, {}

    def is_done(self, x: np.ndarray, u: np.ndarray) -> bool:
        cost = get_lqr_cost(x[0, 0], u, self.Q_states, self.R, self.dt)
        return cost < self.cost_threshold

    def begin_state(self) -> np.ndarray:
        return self.init_state_space.sample()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._states = astensor(self.begin_state(), **self.tkwargs)
        action = torch.zeros(self.process.num_outputs, **self.tkwargs)

        self.t = 0

        observation = asnumpy(self.process.output(self._states, action))

        x = observation if self.use_observations_in_cost else self.states
        self.cost = self.get_cost(x, asnumpy(action))

        return observation, {}

    def render(self, mode='human'):
        if mode != 'console':
            raise NotImplementedError()
        logging.info("States: ", self.states, "\tCost: ", self.cost)


class SteinmetzGym(StatefulGym):
    """Gym environment that reproduces the Steinmetz behavioral study.

    A mouse views a grating on the left and right. The gratings may have
    different contrast values. The mouse must turn a wheel to the side with the
    higher contrast.
    """

    metadata = {'render.modes': ['console']}

    def __init__(self, render_mode=None, num_contrast_levels=5,
                 time_stimulus=50, time_end=350, dt=0.1, action_threshold=0.1):
        super().__init__(dt)
        self.observation_space = spaces.Box(0, 1, (2,))
        self.action_space = spaces.Box(-1, 1, (1,))
        assert render_mode is None \
            or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self._contrast_levels = np.linspace(0, 1, num_contrast_levels)
        self._contrast_left, self._contrast_right = None, None
        self.time = None
        self._time_stimulus = time_stimulus
        self._time_end = time_end
        self.threshold = action_threshold

    @property
    def is_post_stimulus(self):
        return self.time >= self._time_stimulus

    def _get_obs(self):
        if self.is_post_stimulus:
            return self._contrast_left, self._contrast_right
        else:
            return 0, 0

    def _get_info(self):
        return {'Time': self.time,
                'Contrast': (self._contrast_left, self._contrast_right)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            self._contrast_left = self.np_random.choice(self._contrast_levels)
            self._contrast_right = self.np_random.choice(self._contrast_levels)
            if self._contrast_left != self._contrast_right:
                break

        self.time = 0

        observation = self._get_obs()
        info = self._get_info()
        self.states = np.expand_dims(observation, 0)

        self.render()

        return observation, info

    def step(self, action):
        terminated = self.time >= self._time_end
        self.time += 1
        observation = self._get_obs()
        contrast_left, contrast_right = observation
        correct_action = np.sign(contrast_right - contrast_left)
        if correct_action == 0:
            is_correct = np.abs(action) < self.threshold
        else:
            is_correct = np.sign(action) == correct_action

        reward = is_correct * self.is_post_stimulus

        info = self._get_info()
        self.states = np.expand_dims(observation, 0)

        self.render()

        return observation, float(reward), terminated, False, info

    def render(self):
        if self.render_mode == 'console':
            logging.info(self._get_info())


class Masker:
    """Helper class to set certain rows in the readout and stimulation matrix
    of a controller to zero."""
    def __init__(self, model: ControlledRnn, p, rng: np.random.Generator):
        self.model = model
        self.p = p
        n = self.model.hidden_size if hasattr(self.model, 'hidden_size') else 1
        self._controllability_mask = np.flatnonzero(rng.binomial(1, self.p, n))
        self._observability_mask = np.flatnonzero(rng.binomial(1, self.p, n))
        self.controllability = 1 - len(self._controllability_mask) / n
        self.observability = 1 - len(self._observability_mask) / n

    def apply_mask(self):
        if self.p == 0:
            return
        with torch.no_grad():
            self.model.controller.decoder.weight[
                self._controllability_mask] = 0
            self.model.controller.rnn.weight_ih_l0[
                :, self._observability_mask] = 0


class Gramians(nn.Module):
    """Estimator for empirical controllability and observability Gramians."""
    def __init__(self, model: ControlledRnn,
                 environment: Union[gym.Env, StochasticLinearIOSystem],
                 decoder: 'nn.Linear', T):
        super().__init__()
        self.model = model
        self.environment = environment
        self.decoder = decoder
        self.tkwargs = self.model.tkwargs
        self.dt = self.environment.dt
        self.T = T
        self.num_inputs = self.model.controller.num_hidden
        self.num_hidden = self.model.neuralsystem.hidden_size
        self.num_outputs = self.model.controller.num_hidden
        if hasattr(self.environment, 'action_space'):
            self.num_controls = self.environment.action_space.shape[0]
        else:
            self.num_controls = self.environment.process.num_inputs
        self._return_observations = None

    def forward(self, x: Callable, h: torch.Tensor) -> torch.Tensor:
        self.environment.reset()
        neuralsystem_states = h
        neuralsystem_output = atleast_3d(torch.zeros(self.num_controls,
                                                     **self.tkwargs))
        outputs = []
        for t in np.arange(0, self.T, self.dt):
            ut = astensor(atleast_3d(x(t)), **self.tkwargs)
            environment_output = self.environment.step(
                asnumpy(neuralsystem_output))[0]
            neuralsystem_states, neuralsystem_states = self.model.neuralsystem(
                atleast_3d(astensor(environment_output, **self.tkwargs)),
                neuralsystem_states + self.model.readin(ut))
            neuralsystem_output = self.decoder(neuralsystem_states)
            if self._return_observations:
                outputs.append(self.model.readout(neuralsystem_states))
            else:
                outputs.append(neuralsystem_states)
        return torch.concat(outputs, dim=0).reshape((len(outputs), -1)).T

    # noinspection PyUnusedLocal
    def _ode(self, f, g, t, x0: np.ndarray, u: Callable, p) -> np.ndarray:
        x0 = astensor(atleast_3d(x0), **self.tkwargs)
        return asnumpy(self.__call__(u, x0))

    def compute_gramian(self, kind: str) -> np.ndarray:
        return emgr(None, 1,
                    [self.num_inputs, self.num_hidden, self.num_outputs],
                    [self.dt, self.T - self.dt], kind, ode=self._ode,
                    us=np.zeros(self.num_inputs), xs=np.zeros(self.num_hidden),
                    nf=12*[0]+[3])  # Normalize trajectories

    def compute_controllability(self) -> np.ndarray:
        self._return_observations = False
        return self.compute_gramian('c')

    def compute_observability(self) -> np.ndarray:
        self._return_observations = True
        return self.compute_gramian('o')


def get_device(config: CfgNode) -> torch.device:
    """Return hardware backend to run on."""

    gpu = str(config.GPU)
    if gpu == 'cuda':  # Leave device ID unspecified.
        return torch.device(gpu)
    # Otherwise expect integer or ''.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # Always set GPU ID to 0 here because we allow only one visible device in
    # the environment variable.
    return torch.device('cpu' if not gpu else 'cuda:0')


def asnumpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def astensor(x: np.ndarray, **kwargs) -> torch.Tensor:
    return torch.tensor(x, **kwargs)


def to_hash(params: Iterator[torch.nn.Parameter]) -> List[int]:
    """Compute the hash values of a list of weight matrices."""
    return [hash(tuple(param.data.cpu().numpy().ravel())) for param in params]
