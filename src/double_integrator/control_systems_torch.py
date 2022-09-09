import logging
from typing import Union

import numpy as np
import gym
from gym import spaces
import torch
from torch import nn

from py.emgr import emgr
from src.double_integrator.utils import get_lqr_cost


class MlpModel(nn.Module):
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

    def forward(self, x):
        hidden = self.activation_hidden(self.hidden(x))
        return self.activation_output(self.output(hidden))


class RnnModel(nn.Module):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1, input_size=1,
                 activation_rnn=None, activation_decoder=None, device=None,
                 dtype=None):

        super().__init__()

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.rnn = nn.RNN(input_size, num_hidden, num_layers, dtype=self.dtype,
                          device=self.device, nonlinearity=activation_rnn)
        self.decoder = nn.Linear(num_hidden, num_outputs, device=self.device,
                                 dtype=self.dtype)
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

    def begin_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.num_hidden,
                           device=self.device, dtype=self.dtype)

    def forward(self, x, h):
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
        self.device = device
        self.dtype = dtype

        specs = dict(dtype=self.dtype, device=self.device)
        self.dt = nn.Parameter(torch.tensor((dt,),
                                            **specs), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(self.num_states, self.num_states,
                                          **specs), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(self.num_states, self.num_inputs,
                                          **specs), requires_grad=False)
        self.C = nn.Parameter(torch.zeros(self.num_outputs, self.num_states,
                                          **specs), requires_grad=False)
        self.D = nn.Parameter(torch.zeros(self.num_outputs, self.num_inputs,
                                          **specs), requires_grad=False)
        self.W = None
        self.V = None

    def begin_state(self, batch_size):
        return torch.zeros(1, batch_size, self.num_states, dtype=self.dtype,
                           device=self.device)

    def step(self, x, u, **kwargs):
        dxdt = self.dynamics(x, u)
        x = self.integrate(x, dxdt, **kwargs).squeeze(0)
        return self.add_process_noise(x, **kwargs)

    def dynamics(self, x, u):
        u = torch.tensor(u, dtype=self.dtype, device=self.device)
        return (torch.tensordot(torch.atleast_2d(x), self.A.T, 1) +
                torch.tensordot(torch.atleast_2d(u), self.B.T, 1))

    def integrate(self, x, dxdt, **kwargs):
        method = kwargs.get('method', 'euler-maruyama')
        if method == 'euler-maruyama':  # x + dt * dx/dt
            return x + self.dt * dxdt
        else:
            raise NotImplementedError

    def add_process_noise(self, x, deterministic=False):
        if self.W is None or deterministic:
            return x
        dW = self.get_additive_white_gaussian_noise(
            self.num_states,
            torch.ones(self.num_states, dtype=self.dtype, device=self.device) *
            torch.sqrt(self.dt))
        return x + torch.tensordot(torch.atleast_2d(dW),
                                   self.W.T, 1).squeeze(0)

    def output(self, x, u, deterministic=False):
        u = torch.tensor(u, dtype=self.dtype, device=self.device)
        y = (torch.tensordot(torch.atleast_2d(x), self.C.T, 1) +
             torch.tensordot(torch.atleast_2d(u), self.D.T, 1)).squeeze(0)
        return self.add_observation_noise(y, deterministic)

    def add_observation_noise(self, y, deterministic=False):
        if self.V is None or deterministic:
            return y
        return y + self.get_additive_white_gaussian_noise(self.num_outputs,
                                                          torch.diag(self.V))

    def get_additive_white_gaussian_noise(self, n, scale):
        return torch.normal(torch.zeros(n, dtype=self.dtype,
                                        device=self.device), scale)

    def forward(self, x, u, **kwargs):
        x = self.step(x, u, **kwargs)
        return self.output(x, u, **kwargs), x


class DI(StochasticLinearIOSystem):
    def __init__(self, num_inputs, num_outputs, num_states, var_x: float = 0,
                 var_y: float = 0, dt: float = 0.1, dtype=torch.float32,
                 device='cuda'):

        super().__init__(num_inputs, num_outputs, num_states, device, dt,
                         dtype)

        specs = dict(dtype=self.dtype, device=self.device)
        self.A[0, 1] = 1
        self.B[1, 0] = 1  # Control only second state (acceleration).
        self.C.data = torch.eye(self.num_outputs, self.num_states, **specs)
        if var_x:
            self.W = nn.Parameter(
                torch.mul(var_x, torch.eye(self.num_states, **specs)),
                requires_grad=False)
        if var_y:
            self.V = nn.Parameter(
                torch.mul(var_y, torch.eye(self.num_outputs, **specs)),
                requires_grad=False)


class MLP:
    def __init__(self, process, num_hidden, activation_hidden,
                 activation_output, q=0.5, r=0.5, path_model=None):

        self.process = process
        dtype_np = np.dtype(str(self.process.dtype).split('.')[1])

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=dtype_np)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=dtype_np)

        self.model = MlpModel(self.process.num_outputs, num_hidden,
                              self.process.num_inputs, activation_hidden,
                              activation_output, self.process.dtype,
                              self.process.device)
        if path_model is not None:
            self.model.load_state_dict(torch.load(path_model))

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt)

    def get_control(self, x):
        # Add dummy dimension for batch size.
        x = torch.tensor(np.expand_dims(x, 0), dtype=self.process.dtype,
                         device=self.process.device)
        u = self.model(x)
        return u.cpu().numpy()[0]

    def step(self, x, y):
        u = self.get_control(y)
        x = self.process.step(x, u)
        y = self.process.output(x, u)
        c = self.get_cost(x, u)

        return x, y, u, c

    def dynamics(self, x, u):
        y = self.process.output(x, u)
        u = self.get_control(y)

        return self.process.dynamics(x, u)


class ControlledRnn(nn.Module):
    def __init__(self, rnn: nn.RNN, controller: RnnModel):
        super().__init__()
        self.neuralsystem = rnn
        self.controller = controller
        self.device = self.controller.device
        self.dtype = self.controller.dtype
        self.input_size = self.neuralsystem.input_size
        self.num_layers = self.neuralsystem.num_layers
        self.hidden_size = self.neuralsystem.hidden_size

    def forward(self, x, h=None):
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

    def readout(self, x):
        return (torch.tensordot(x, self.controller.rnn.weight_ih_l0.T, 1) +
                self.controller.rnn.bias_ih_l0)

    def readin(self, x):
        return (torch.tensordot(x, self.controller.decoder.weight.T, 1) +
                self.controller.decoder.bias)

    def begin_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size,
                            device=self.device, dtype=self.dtype),
                self.controller.begin_state(batch_size))


class DiMlp(MLP):
    def __init__(self, num_inputs, num_outputs, num_states, num_hidden,
                 activation_hidden, activation_output, var_x=0, var_y=0,
                 dt=0.1, q=0.5, r=0.5, path_model=None, dtype=torch.float32,
                 device='cuda'):
        process = DI(num_inputs, num_outputs, num_states, var_x, var_y, dt,
                     dtype, device)
        super().__init__(process, num_hidden, activation_hidden,
                         activation_output, q, r, path_model)


class DiGym(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {'render.modes': ['console']}

    def __init__(self, num_inputs, num_outputs, num_states, device,
                 var_x: float = 0, var_y: float = 0, dt: float = 0.1,
                 cost_threshold=1e-3, state_threshold=None,
                 q: Union[float, np.iterable] = 0.5, r=0.5,
                 dtype=torch.float32, use_observations_in_cost=False):
        super().__init__()
        self.dt = dt
        self.dtype = dtype
        self.dtype_np = np.dtype(str(self.dtype).split('.')[1])
        self.device = device
        self.use_observations_in_cost = use_observations_in_cost
        self.process = DI(num_inputs, num_outputs, num_states, var_x, var_y,
                          self.dt, self.dtype, self.device)

        self.min = -1
        self.max = 1
        self.action_space = spaces.Box(-10, 10, (1,), self.dtype_np)
        self.observation_space = spaces.Box(self.min, self.max, (num_outputs,),
                                            self.dtype_np)
        self.init_state_space = spaces.Box(self.min / 2, self.max / 2,
                                           (num_states,), self.dtype_np)
        self.cost_threshold = cost_threshold
        self.state_threshold = state_threshold or self.max

        # State cost matrix:
        if np.isscalar(q):
            dim = self.process.num_outputs if self.use_observations_in_cost \
                else self.process.num_states
            self.Q = q * np.eye(dim, dtype=self.dtype_np)
            self.Q_states = q * np.eye(self.process.num_states,
                                       dtype=self.dtype_np)
        else:
            self.Q = np.diag(q)
            self.Q_states = q * np.eye(self.process.num_states,
                                       dtype=self.dtype_np)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype_np)

        self.states = None
        self.cost = None
        self.t = None

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt).item()

    def step(self, action):

        self.states = self.process.step(self.states, action)
        torch.clip(self.states, self.min, self.max, out=self.states)

        observation = self.process.output(self.states, action).cpu().numpy()
        np.clip(observation, self.min, self.max, observation)

        states_np = self.states.cpu().numpy()
        x = observation if self.use_observations_in_cost else states_np
        self.cost = self.get_cost(x, action)

        done = self.is_done(states_np, action)
        # or abs(self.states[0].item()) > self.state_threshold

        reward = -self.cost + done * 10 * np.exp(-self.t / 4)

        self.t += self.dt

        return observation, reward, done, {}

    def is_done(self, x, u):
        cost = get_lqr_cost(np.squeeze(x), np.squeeze(u), self.Q_states,
                            self.R, self.dt).item()
        return cost < self.cost_threshold

    def begin_state(self):
        return torch.tensor(self.init_state_space.sample(), dtype=self.dtype,
                            device=self.device)

    def reset(self, state_init=None):

        self.states = state_init or self.begin_state()
        action = np.zeros(self.process.num_outputs, self.dtype_np)

        self.t = 0

        observation = self.process.output(self.states, action).cpu().numpy()

        x = observation if self.use_observations_in_cost else self.states
        self.cost = self.get_cost(x, action)

        return observation

    def render(self, mode='human'):
        if mode != 'console':
            raise NotImplementedError()
        logging.info("States: ", self.states, "\tCost: ", self.cost)


class Masker:
    def __init__(self, model: 'ControlledRnn', p, rng: np.random.Generator):
        self.model = model
        self.p = p
        n = self.model.hidden_size
        self._controllability_mask = np.flatnonzero(rng.binomial(1, self.p, n))
        self._observability_mask = np.flatnonzero(rng.binomial(1, self.p, n))

    def apply_mask(self):
        if self.p == 0:
            return
        with torch.no_grad():
            self.model.controller.decoder.weight[
                self._controllability_mask] = 0
            self.model.controller.rnn.weight_ih_l0[
                :, self._observability_mask] = 0


class Gramians(nn.Module):
    def __init__(self, model: 'ControlledRnn', environment,
                 decoder: 'nn.Linear', T):
        super().__init__()
        self.model = model
        self.environment = environment
        self.decoder = decoder
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.dt = self.environment.dt
        self.T = T
        self.num_inputs = self.model.controller.num_hidden
        self.num_hidden = self.model.neuralsystem.hidden_size
        self.num_outputs = self.model.controller.num_hidden
        self._return_observations = None

    def forward(self, x, h):
        self.environment.reset()
        neuralsystem_states = h
        neuralsystem_output = \
            torch.zeros(1, 1, self.environment.process.num_inputs,
                        dtype=self.dtype, device=self.device)
        outputs = []
        for t in np.arange(0, self.T, self.dt):
            ut = torch.tensor(np.expand_dims(x(t), (0, 1)), dtype=self.dtype,
                              device=self.device)
            environment_output = self.environment.step(
                neuralsystem_output.cpu().numpy())[0]
            neuralsystem_states, neuralsystem_states = self.model.neuralsystem(
                torch.tensor(environment_output, dtype=self.dtype,
                             device=self.device).unsqueeze(0),
                neuralsystem_states + self.model.readin(ut))
            neuralsystem_output = self.decoder(neuralsystem_states)
            if self._return_observations:
                outputs.append(self.model.readout(neuralsystem_states))
            else:
                outputs.append(neuralsystem_states)
        return torch.concat(outputs, dim=0).reshape((len(outputs), -1)).T

    # noinspection PyUnusedLocal
    def _ode(self, f, g, t, x0, u, p):
        x0 = torch.tensor(np.expand_dims(x0, (0, 1)), dtype=self.dtype,
                          device=self.device)
        return self.__call__(u, x0).cpu().numpy()

    def compute_gramian(self, kind):
        return emgr(None, 1,
                    [self.num_inputs, self.num_hidden, self.num_outputs],
                    [self.dt, self.T - self.dt], kind, ode=self._ode,
                    us=np.zeros(self.num_inputs), xs=np.zeros(self.num_hidden))

    def compute_controllability(self):
        self._return_observations = False
        return self.compute_gramian('c')

    def compute_observability(self):
        self._return_observations = True
        return self.compute_gramian('o')
