import logging
import time

from abc import ABC, abstractmethod
from typing import Union, Optional

import control
import numpy as np
import gym
from gym import spaces

from src.utils import (get_lqr_cost, get_initial_states,
                       get_additive_white_gaussian_noise)


class StochasticLinearIOSystem(control.LinearIOSystem):

    def __rdiv__(self, other):
        control.StateSpace.__rdiv__(self, other)

    def __div__(self, other):
        control.StateSpace.__div__(self, other)

    def __init__(self, linsys, W=None, V=None, rng=None, **kwargs):
        super().__init__(linsys, **kwargs)
        self.W = W
        self.V = V
        self.rng = rng
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float32'
        self.num_inputs = linsys.ninputs
        self.num_states = linsys.nstates
        self.num_outputs = linsys.noutputs

    def step(self, t, x, u, method='euler-maruyama', deterministic=False):
        dxdt = super().dynamics(t, x, u).astype(self.dtype)

        if method == 'euler-maruyama':
            x_new = x + self.dt * dxdt
            if self.W is not None and not deterministic:
                dW = get_additive_white_gaussian_noise(
                    np.eye(len(x), dtype=self.dtype) * np.sqrt(self.dt),
                    rng=self.rng)
                x_new += np.dot(self.W, dW)
            return x_new
        else:
            raise NotImplementedError

    def dynamics(self, t, x, u):
        return super().dynamics(t, x, u)

    def output(self, t, x, u, deterministic=False):
        out = super().output(t, x, u).astype(self.dtype)
        if self.V is not None and not deterministic:
            out += get_additive_white_gaussian_noise(self.V, rng=self.rng)
        return out

    def get_initial_states(self, mu, Sigma, n=1):
        return get_initial_states(mu, Sigma, len(self.A), n, self.rng)


class LQR:
    """Linear Quadratic Regulator."""
    def __init__(self, process: StochasticLinearIOSystem,
                 q: Optional[float] = 0.5, r: Optional[float] = 0.5,
                 dtype: Optional[str] = 'float32',
                 normalize_cost: Optional[bool] = False):

        self.process = process
        self.dtype = dtype
        self.normalize_cost = normalize_cost

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        # Feedback gain matrix:
        self.K = self.get_feedback_gain()

    def get_feedback_gain(self):
        """Solve LQR.

        Returns state feedback gain K, solution S to Riccati equation, and
        eigenvalues E of closed-loop system.
        """
        K, S, E = control.lqr(self.process.A, self.process.B, self.Q, self.R)
        return K

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt,
                            normalize=self.normalize_cost)

    def get_control(self, x):
        return -self.K.dot(x)

    def step(self, t, x):
        u = self.get_control(x)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        c = self.get_cost(x, u)

        return x, y, u, c

    # noinspection PyUnusedLocal
    def dynamics(self, t, x, u):
        u = self.get_control(x)

        return self.process.dynamics(t, x, u)


class LQE:
    """Linear Quadratic Estimator."""
    def __init__(self, process, dtype='float32'):
        self.process = process
        self.dtype = dtype

        # Kalman gain matrix:
        self.L = self.get_Kalman_gain()

    def get_Kalman_gain(self):
        """Solve LQE.

        Returns Kalman estimator gain L, solution P to Riccati equation, and
        eigenvalues F of estimator poles A-LC.
        """
        L, P, F = control.lqe(self.process.A,
                              np.eye(len(self.process.A), dtype=self.dtype),
                              self.process.C,
                              self.process.W,
                              self.process.V)
        return L

    def step(self, t, mu, Sigma, u, y, asymptotic=True):
        mu = self.process.step(t, mu, u, deterministic=True)

        if asymptotic:
            L = self.L
        else:
            A = self.process.A
            C = self.process.C
            V = self.process.V
            W = self.process.W
            Sigma = A @ Sigma @ A.T + W
            L = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + V)
            Sigma = (1 - L @ C) @ Sigma

        mu += self.process.dt * L @ (y - self.process.output(t, mu, u, True))

        return mu, Sigma


class LQG:
    """Linear Quadratic Gaussian."""
    def __init__(self, process, q=0.5, r=0.5, normalize_cost=False):
        self.process = process
        self.estimator = LQE(self.process)
        self.control = LQR(self.process, q, r, normalize_cost=normalize_cost)

    def step(self, t, x, x_est, Sigma):
        u = self.control.get_control(x_est)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        x_est, Sigma = self.estimator.step(t, x_est, Sigma, u, y)
        c = self.control.get_cost(x, u)

        return x, y, u, c, x_est, Sigma

    # noinspection PyUnusedLocal
    def dynamics(self, t, x, u):
        # Skipping estimator step here, because this method is only evaluated
        # for one time step to draw vector field.
        u = self.control.get_control(x)

        return self.process.dynamics(t, x, u)


class DI(StochasticLinearIOSystem):
    """Double integrator dynamical system."""
    def __init__(self, num_inputs, num_outputs, num_states, var_x=0., var_y=0.,
                 dt=0.1, rng=None, **kwargs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_states = num_states
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else 'float32'

        # Dynamics matrix:
        A = np.zeros((self.num_states, self.num_states), self.dtype)
        A[0, 1] = 1

        # Input matrix:
        B = np.zeros((self.num_states, self.num_inputs), self.dtype)
        B[1, 0] = 1  # Control only second state (acceleration).

        # Output matrices:
        C = np.eye(self.num_outputs, self.num_states, dtype=self.dtype)
        D = np.zeros((self.num_outputs, self.num_inputs), self.dtype)

        # Process noise:
        W = var_x * np.eye(self.num_states, dtype=self.dtype) if var_x \
            else None

        # Output noise:
        V = var_y * np.eye(self.num_outputs, dtype=self.dtype) if var_y \
            else None

        ss = control.StateSpace(A, B, C, D, dt)
        super().__init__(ss, W, V, rng, **kwargs)


class DiOpen:
    """Double integrator in open loop."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None):
        num_inputs = 1
        num_outputs = 2
        num_states = 2
        self.process = DI(num_inputs, num_outputs, num_states,
                          var_x, var_y, dt, rng)


class DiLqr(LQR):
    """Double integrator with LQR control."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        num_inputs = 1
        num_outputs = 2
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r)


class DiLqg(LQG):
    """Double integrator with LQG control."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 normalize_cost=False):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, normalize_cost)


class DiGym(gym.Env):
    """Double integrator that follows the gym interface."""

    metadata = {'render.modes': ['console']}

    def __init__(self, num_inputs, num_outputs, num_states, var_x=0., var_y=0.,
                 dt=0.1, rng=None, cost_threshold=1e-3, state_threshold=None,
                 q: Union[float, np.iterable] = 0.5, r=0.5, dtype=np.float32,
                 use_observations_in_cost=False):
        super().__init__()
        self.dt = dt
        self.dtype = dtype
        self.use_observations_in_cost = use_observations_in_cost
        self.process = DI(num_inputs, num_outputs, num_states, var_x, var_y,
                          self.dt, rng)

        self.min = -1
        self.max = 1
        self.action_space = spaces.Box(-10, 10, (1,), self.dtype)
        self.observation_space = spaces.Box(self.min, self.max, (num_outputs,),
                                            self.dtype)
        self.init_state_space = spaces.Box(self.min / 2, self.max / 2,
                                           (num_states,), self.dtype)
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

        self.states = None
        self.cost = None
        self.t = None

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def step(self, action):

        self.states = self.process.step(self.t, self.states, action)
        np.clip(self.states, self.min, self.max, self.states)

        observation = self.process.output(self.t, self.states, action)
        np.clip(observation, self.min, self.max, observation)

        x = observation if self.use_observations_in_cost else self.states
        self.cost = self.get_cost(x, action)

        done = self.is_done(action)
        # or abs(self.states[0].item()) > self.state_threshold

        reward = -self.cost + done * 10 * np.exp(-self.t / 4)

        self.t += self.dt

        return observation, reward, done, {}

    def is_done(self, u):
        cost = get_lqr_cost(self.states, u, self.Q_states, self.R, self.dt)
        return cost < self.cost_threshold

    def reset(self, state_init=None):

        self.states = state_init or self.init_state_space.sample()
        action = 0

        self.t = 0

        observation = self.process.output(self.t, self.states, action)

        x = observation if self.use_observations_in_cost else self.states
        self.cost = self.get_cost(x, action)

        return observation

    def render(self, mode='human'):
        if mode != 'console':
            raise NotImplementedError()
        logging.info("States: ", self.states, "\tCost: ", self.cost)


class PID:
    """PID Controller."""

    def __init__(self, setpoint=None, k_p=1, k_i=1, k_d=1, dt=1,
                 integral_windup_limit=None):

        self.setpoint = setpoint

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.dt = dt

        self.times = [-1]
        self.errors = [0]

        self.p = 0
        self.i = 0
        self.d = 0

        if integral_windup_limit is not None:
            assert np.iterable(integral_windup_limit)
            assert len(integral_windup_limit) == 2
        self.integral_windup_limit = integral_windup_limit

        self.control_variable = 0

    def update(self, process_value, setpoint=None, t=None):
        """Calculate PID value for given reference feedback."""

        if setpoint is not None:
            self.setpoint = setpoint

        current_time = t if t is not None else time.time()
        delta_time = current_time - self.times[-1]
        if delta_time < self.dt:
            return self.control_variable
        elif delta_time > self.dt:
            print("WARNING: PID seems to have not been updated regularly.")

        error = self.setpoint - process_value
        delta_error = error - self.errors[-1]

        self.p = error

        self.i += error * delta_time

        self.clamp_integral()

        self.d = delta_error / delta_time if delta_time else 0

        self.times.append(current_time)
        self.errors.append(error)

        self.control_variable = (self.k_p * self.p +
                                 self.k_i * self.i +
                                 self.k_d * self.d)

        return self.control_variable

    def reset(self):
        self.setpoint = None
        self.i = 0
        self.times = [-1]
        self.errors = [0]
        self.control_variable = 0

    def clamp_integral(self):
        if self.integral_windup_limit is None:
            return
        self.i = np.clip(self.i, *self.integral_windup_limit)


def get_ev_sum(gramian: np.ndarray, threshold: Optional[float] = 0.9):
    # Compute eigenvalues and eigenvectors of Gramian.
    w, v = np.linalg.eigh(gramian)
    # Sort ascending.
    w = w[::-1]
    v = v[:, ::-1]
    # Determine how many eigenvectors to use.
    fraction_explained = np.cumsum(w) / np.sum(w)
    n = np.min(np.flatnonzero(fraction_explained > threshold))
    # Add up eigenvectors weighted by corresponding eigenvalues.
    weighted_sum = np.dot(v[:, :n], w[:n])
    return np.abs(weighted_sum)


class AbstractMasker(ABC):
    """Helper class to set certain rows in the readout and stimulation matrix
    of a controller to zero."""
    def __init__(self, model, p: float, method: str):
        self.model = model
        self.p = p
        self.method = method
        self._controllability_mask = None
        self._observability_mask = None
        self.n = None

    @abstractmethod
    def apply_mask(self):
        raise NotImplementedError

    def compute_mask(self, **kwargs):
        if self.method == 'gramian':
            self._compute_mask_gramian(**kwargs)
        elif self.method == 'random':
            self._compute_mask_random(**kwargs)
        else:
            raise NotImplementedError

    def _compute_mask_random(self, rng: np.random.Generator):
        self._controllability_mask = \
            np.flatnonzero(rng.binomial(1, self.p, self.n))
        self._observability_mask = \
            np.flatnonzero(rng.binomial(1, self.p, self.n))

    def _compute_mask_gramian(self, controllability_gramian: np.ndarray,
                              observability_gramian: np.ndarray,
                              num_controls: int, num_observations: int):
        weighted_sum = get_ev_sum(controllability_gramian)
        self._controllability_mask = np.argsort(weighted_sum)[:num_controls]
        weighted_sum = get_ev_sum(observability_gramian)
        self._observability_mask = np.argsort(weighted_sum)[:num_observations]

    @property
    def num_controls(self):
        return len(self._controllability_mask)

    @property
    def num_observations(self):
        return len(self._observability_mask)

    @property
    def controllability(self):
        return 1 - self.num_controls / self.n

    @property
    def observability(self):
        return 1 - self.num_observations / self.n
