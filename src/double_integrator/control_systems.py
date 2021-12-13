import control
import mxnet as mx
import numpy as np

from src.double_integrator.utils import (get_lqr_cost, get_initial_states,
                                         get_additive_white_gaussian_noise)


class DI:
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None):
        self.dt = dt

        self.n_x_process = 2  # Number of process states
        self.n_y_process = 1  # Number of process outputs
        self.n_u_process = 1  # Number of process inputs

        # Dynamics matrix:
        self.A = np.zeros((self.n_x_process, self.n_x_process))
        self.A[0, 1] = 1

        # Input matrix:
        self.B = np.zeros((self.n_x_process, self.n_u_process))
        self.B[1, 0] = 1  # Control only second state (acceleration).

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Only observe position.
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # Process noise:
        self.W = var_x * np.eye(self.n_x_process) if var_x else None

        # Output noise:
        self.V = var_y * np.eye(self.n_y_process) if var_y else None

        ss = control.StateSpace(self.A, self.B, self.C, self.D, self.dt)
        self.system = StochasticLinearIOSystem(ss, self.W, self.V, rng=rng)

    def get_initial_states(self, mu, Sigma, n=1, rng=None):
        return get_initial_states(mu, Sigma, self.n_x_process, n, rng)

    def step(self, t, x, u):
        return self.system.dynamics(t, x, u)


class DiLqr(DI):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        super().__init__(var_x, var_y, dt, rng)

        self.n_y_process = self.n_x_process  # Number of process outputs
        self.n_y_control = self.n_u_process  # Number of control outputs

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Assume both states are perfectly observable.
        self.C[1, 1] = 1
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # State cost matrix:
        self.Q = q * np.eye(self.n_x_process)

        # Control cost matrix:
        self.R = r * np.eye(self.n_y_control)

        # Feedback gain matrix:
        self.K = self.get_feedback_gain()

    def get_feedback_gain(self):
        # Solve LQR. Returns state feedback gain K, solution S to Riccati
        # equation, and eigenvalues E of closed-loop system.
        K, S, E = control.lqr(self.A, self.B, self.Q, self.R)
        return K

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def get_control(self, x, u=None):
        return -self.K.dot(x)

    def step(self, t, x, u):
        return self.system.dynamics(t, x, self.get_control(x) + u)


class DiLqg(DiLqr):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        super().__init__(var_x, var_y, dt, rng, q, r)

        self.n_y_process = 1  # Number of process outputs

        # Output matrices:
        self.C = np.zeros((self.n_y_process, self.n_x_process))
        self.C[0, 0] = 1  # Only observe position.
        self.D = np.zeros((self.n_y_process, self.n_u_process))

        # Kalman gain matrix:
        self.L = self.get_Kalman_gain()

    def get_Kalman_gain(self):
        # Solve LQE. Returns Kalman estimator gain L, solution P to Riccati
        # equation, and eigenvalues F of estimator poles A-LC.
        L, P, F = control.lqe(self.A, np.eye(self.n_x_process), self.C, self.W,
                              self.V)
        return L

    def apply_filter(self, t, mu, Sigma, u, y, asymptotic=True):
        mu = self.system.step(t, mu, u, deterministic=True)

        if asymptotic:
            L = self.L
        else:
            Sigma = self.A @ Sigma @ self.A.T + self.W
            L = Sigma @ self.C.T @ np.linalg.inv(self.C @ Sigma @ self.C.T +
                                                 self.V)
            Sigma = (1 - L @ self.C) @ Sigma

        mu += self.dt * L @ (y - self.system.output(t, mu, u,
                                                    deterministic=True))

        return mu, Sigma


class DiMlp(DI):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 num_hidden=1, path_model=None):
        super().__init__(var_x, var_y, dt, rng)

        self.n_y_control = self.n_u_process

        # State cost matrix:
        self.Q = q * np.eye(self.n_x_process)

        # Control cost matrix:
        self.R = r * np.eye(self.n_y_control)

        self.mlp = MLPModel(num_hidden)
        # self.mlp.hybridize()
        if path_model is None:
            self.mlp.initialize()
        else:
            self.mlp.load_parameters(path_model)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def get_control(self, x):
        # Add dummy dimension for batch size.
        x = mx.nd.array(np.expand_dims(x, 0))
        u = self.mlp(x)
        return u.asnumpy()[0]

    def step(self, t, x, u):
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(y) + u)


class DiMlpLqe(DiLqg):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 num_hidden=1, path_model=None):
        super().__init__(var_x, var_y, dt, rng, q, r)

        self.model = MLPModel(num_hidden)
        # self.mlp.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_control(self, x, u=None):
        # Add dummy dimension for batch size.
        x = mx.nd.array(np.expand_dims(x, 0))
        u = self.model(x)
        return u.asnumpy()[0]

    def step(self, t, x, u):
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(y) + u)


class DiRnn(DI):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, rnn_kwargs: dict = None):
        super().__init__(var_x, var_y, dt, rng)
        # Here we use the noisy state measurements as input to the RNN.

        self.n_y_control = self.n_u_process

        # State cost matrix:
        self.Q = q * np.eye(self.n_x_process)

        # Control cost matrix:
        self.R = r * np.eye(self.n_y_control)

        self.model = RNNModel(**rnn_kwargs)
        # self.rnn.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.dt)

    def get_control(self, x, u):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = mx.nd.array(np.expand_dims(u, [0, 1]))
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x = mx.nd.array(np.reshape(x, (-1, 1, self.model.num_hidden)))
        y, x = self.model(u, x)
        return y.asnumpy().ravel(), x[0].asnumpy().ravel()

    def step(self, t, x, u):
        # Todo: The RNN hidden states need to be initialized better.
        x_rnn = np.zeros((self.model.num_layers, self.model.num_hidden))
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(x_rnn, y)[0] + u)


class DiRnnLqe(DiLqg):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, rnn_kwargs: dict = None):
        super().__init__(var_x, var_y, dt, rng, q, r)
        # Here we use the estimated states as input to the RNN. The LQR
        # controller is replaced by the RNN.

        self.model = RNNModel(**rnn_kwargs)
        # self.rnn.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_control(self, x, u=None):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = mx.nd.array(np.expand_dims(u, [0, 1]))
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x = mx.nd.array(np.reshape(x, (-1, 1, self.model.num_hidden)))
        y, x = self.model(u, x)
        return y.asnumpy().ravel(), x[0].asnumpy().ravel()

    def step(self, t, x, u):
        x_rnn = np.zeros((self.model.num_layers, self.model.num_hidden))
        y = self.system.output(t, x, u)
        return self.system.dynamics(t, x, self.get_control(x_rnn, y)[0] + u)


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

    def step(self, t, x, u, method='euler-maruyama', deterministic=False):
        dxdt = super().dynamics(t, x, u)

        if method == 'euler-maruyama':
            x_new = x + self.dt * dxdt
            if self.W is not None and not deterministic:
                dW = get_additive_white_gaussian_noise(
                    np.eye(len(x)) * np.sqrt(self.dt), rng=self.rng)
                x_new += np.dot(self.W, dW)
            return x_new
        else:
            raise NotImplementedError

    def output(self, t, x, u, deterministic=False):
        out = super().output(t, x, u)
        if self.V is not None and not deterministic:
            out += get_additive_white_gaussian_noise(self.V, rng=self.rng)
        return out


class RNNModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1,
                 activation='relu', **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden
        self.num_layers = num_layers

        with self.name_scope():
            self.rnn = mx.gluon.rnn.RNN(num_hidden, num_layers, activation)
            self.decoder = mx.gluon.nn.Dense(num_outputs, activation='tanh',
                                             in_units=num_hidden,
                                             flatten=False)

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args, **kwargs):
        output, hidden = self.rnn(x, args[0])
        decoded = self.decoder(output)
        return decoded, hidden


class MLPModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_outputs=1, **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden

        with self.name_scope():
            self.hidden = mx.gluon.nn.Dense(num_hidden, activation='relu')
            self.output = mx.gluon.nn.Dense(num_outputs, activation='tanh')

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.output(self.hidden(x))
