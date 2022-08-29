import control
import mxnet as mx
import numpy as np

from src.double_integrator.utils import (get_lqr_cost, get_initial_states,
                                         get_additive_white_gaussian_noise)
from src.ff_pid.brownian import brownian
from src.ff_pid.pid import PID


class LQR:
    def __init__(self, process, q=0.5, r=0.5, dtype='float32',
                 normalize_cost=False):

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
        # Solve LQR. Returns state feedback gain K, solution S to Riccati
        # equation, and eigenvalues E of closed-loop system.
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
    def __init__(self, process, dtype='float32'):

        self.process = process
        self.dtype = dtype

        # Kalman gain matrix:
        self.L = self.get_Kalman_gain()

    def get_Kalman_gain(self):
        # Solve LQE. Returns Kalman estimator gain L, solution P to Riccati
        # equation, and eigenvalues F of estimator poles A-LC.
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


class MLP:
    def __init__(self, process, q=0.5, r=0.5, path_model=None,
                 model_kwargs: dict = None, dtype='float32'):

        self.process = process
        self.dtype = dtype

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.model = MLPModel(**model_kwargs)
        self.model.hybridize()
        if path_model is None:
            self.model.initialize()
        else:
            self.model.load_parameters(path_model)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt)

    def get_control(self, x):
        # Add dummy dimension for batch size.
        x = mx.nd.array(np.expand_dims(x, 0))
        u = self.model(x)
        return u.asnumpy()[0]

    def step(self, t, x, y):
        u = self.get_control(y)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        c = self.get_cost(x, u)

        return x, y, u, c

    def dynamics(self, t, x, u):
        y = self.process.output(t, x, u)
        u = self.get_control(y)

        return self.process.dynamics(t, x, u)


class RNN:
    def __init__(self, process, q=0.5, r=0.5, path_model=None,
                 model_kwargs: dict = None, gpu=0, dtype='float32'):

        self.process = process
        self.dtype = dtype

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.context = mx.gpu(gpu) if mx.context.num_gpus() > 0 else mx.cpu()
        self.model = RNNModel(**model_kwargs)
        self.model.hybridize()
        if path_model is None:
            self.model.initialize(ctx=self.context)
        else:
            self.model.load_parameters(path_model, ctx=self.context)

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt)

    def get_control(self, x, u):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = mx.nd.array(np.expand_dims(u, [0, 1]), self.context)
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x = mx.nd.array(np.reshape(x, (-1, 1, self.model.num_hidden)),
                        self.context)
        y, x = self.model(u, x)
        return y.asnumpy().ravel(), x[0].asnumpy().ravel()

    def step(self, t, x, y, x_rnn):
        u, x_rnn = self.get_control(x_rnn, y)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        c = self.get_cost(x, u)

        return x, y, u, c, x_rnn

    def dynamics(self, t, x, u):
        x_rnn = np.zeros(self.model.num_hidden, self.dtype)
        y = self.process.output(t, x, u)
        u, x_rnn = self.get_control(x_rnn, y)

        return self.process.dynamics(t, x, u)


class PidRnn:
    def __init__(self, process, q=0.5, r=0.5, path_model=None,
                 model_kwargs: dict = None, gpu=0, k_p=1, k_i=1, k_d=1,
                 dtype='float32'):

        self.process = process
        self.dtype = dtype

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.context = mx.gpu(gpu) if mx.context.num_gpus() > 0 else mx.cpu()
        self.model_kwargs = model_kwargs
        self.path_model = path_model
        self.model = self.get_model()
        self.model_setpoint = self.get_model()
        self.pid = PID(k_p=k_p, k_i=k_i, k_d=k_d)

    def get_model(self):
        model = RNNModel(**self.model_kwargs)
        model.hybridize()
        if self.path_model is None:
            model.initialize(ctx=self.context)
        else:
            model.load_parameters(self.path_model, ctx=self.context)
        return model

    def get_cost(self, x, u):
        return get_lqr_cost(x, u, self.Q, self.R, self.process.dt)

    def get_control(self, x_perturbed, x_setpoint, u, t):
        u_perturbed, x_rnn_perturbed = \
            self._forward(x_perturbed, u, self.model)
        u_setpoint, x_rnn_setpoint = \
            self._forward(x_setpoint, u, self.model_setpoint)
        u = self.pid.update(u_perturbed, u_setpoint, t)
        return u_perturbed + u, x_rnn_perturbed, x_rnn_setpoint

    def _forward(self, x, u, model):
        # Add dummy dimensions for shape [num_timesteps, batch_size,
        # num_states].
        u = mx.nd.array(np.expand_dims(u, [0, 1]), self.context)
        # Add dummy dimensions for shape [num_layers, batch_size, num_states].
        x = mx.nd.array(np.reshape(x, (-1, 1, model.num_hidden)), self.context)
        y, x = model(u, x)
        return y.asnumpy().ravel(), x[0].asnumpy().ravel()

    def step(self, t, x, y, x_rnn_perturbed, x_rnn_setpoint):
        u, x_rnn_perturbed, x_rnn_setpoint = self.get_control(
            x_rnn_perturbed, x_rnn_setpoint, y, t)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        c = self.get_cost(x, u)

        return x, y, u, c, x_rnn_perturbed, x_rnn_setpoint

    def dynamics(self, t, x, u):
        x_rnn = np.zeros(self.model.num_hidden, self.dtype)
        y = self.process.output(t, x, u)
        u, _, _ = self.get_control(x_rnn, x_rnn.copy(), y, t)

        return self.process.dynamics(t, x, u)


class LqeMlp:
    def __init__(self, process, q=0.5, r=0.5, path_model=None,
                 model_kwargs=None):
        self.process = process
        self.estimator = LQE(self.process)
        self.control = MLP(self.process, q, r, path_model, model_kwargs)

    def step(self, t, x, x_est, Sigma):
        u = self.control.get_control(x_est)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        x_est, Sigma = self.estimator.step(t, x_est, Sigma, u, y)
        c = self.control.get_cost(x_est, u)

        return x, y, u, c, x_est, Sigma

    def dynamics(self, t, x, u):
        y = self.process.output(t, x, u)
        u = self.control.get_control(y)

        return self.process.dynamics(t, x, u)


class LqeRnn:
    def __init__(self, process, q=0.5, r=0.5, path_model=None,
                 model_kwargs=None, dtype='float32'):
        self.process = process
        self.dtype = dtype
        self.estimator = LQE(self.process)
        self.control = RNN(self.process, q, r, path_model, model_kwargs)

    def step(self, t, x, x_rnn, x_est, Sigma):
        u, x_rnn = self.control.get_control(x_rnn, x_est)
        x = self.process.step(t, x, u)
        y = self.process.output(t, x, u)
        x_est, Sigma = self.estimator.step(t, x_est, Sigma, u, y)
        c = self.control.get_cost(x_est, u)

        return x, y, u, c, x_rnn, x_est, Sigma

    def dynamics(self, t, x, u):
        x_rnn = np.zeros(self.control.model.num_hidden, self.dtype)
        y = self.process.output(t, x, u)
        u, x_rnn = self.control.get_control(x_rnn, y)

        return self.process.dynamics(t, x, u)


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

    def step(self, t, x, u, method='euler-maruyama', deterministic=False):
        dxdt = super().dynamics(t, x, u).astype(self.dtype)

        if method == 'euler-maruyama':
            x_new = x + self.dt * dxdt
            if self.W is not None and not deterministic:
                dW = get_additive_white_gaussian_noise(
                    np.eye(len(x), dtype=self.dtype) * np.sqrt(self.dt),
                    rng=self.rng, dtype=self.dtype)
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


class DI(StochasticLinearIOSystem):
    def __init__(self, num_inputs, num_outputs, num_states, var_x=0, var_y=0,
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
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None):
        num_inputs = 1
        num_outputs = 2
        num_states = 2
        self.process = DI(num_inputs, num_outputs, num_states,
                          var_x, var_y, dt, rng)


class DiLqr(LQR):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5):
        num_inputs = 1
        num_outputs = 2
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r)


class DiLqg(LQG):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 normalize_cost=False):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, normalize_cost)


class DiMlp(MLP):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs=None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class DiLqeMlp(LqeMlp):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs=None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class DiRnn(RNN):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None, gpu=0):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs, gpu)


class DiPidRnn(PidRnn):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None, gpu=0, k_p=1,
                 k_i=1, k_d=1):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs, gpu, k_p,
                         k_i, k_d)


class DiLqeRnn(LqeRnn):
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = DI(num_inputs, num_outputs, num_states,
                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class RNNModel(mx.gluon.HybridBlock):

    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1, input_size=1,
                 activation_rnn=None, activation_decoder=None, **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden
        self.num_layers = num_layers

        with self.name_scope():
            if self.num_layers == 1 and activation_rnn == 'linear':
                self.rnn = mx.gluon.rnn.RNNCell(num_hidden,
                                                mx.gluon.nn.LeakyReLU(1),
                                                input_size=input_size)
            else:
                self.rnn = mx.gluon.rnn.RNN(
                    num_hidden, num_layers, activation_rnn,
                    input_size=input_size, prefix='rnn_')
            self.decoder = mx.gluon.nn.Dense(
                num_outputs, activation=activation_decoder,
                in_units=num_hidden, flatten=False, prefix='decoder_')

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args):
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


# noinspection PyUnusedLocal
class StochasticLinearIOSystemMx(mx.gluon.HybridBlock):
    def __init__(self, num_inputs, num_outputs, num_states, context, dt=0.1,
                 dtype='float32', **kwargs):
        super().__init__(**kwargs)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_states = num_states
        self.context = context
        self.dt = dt
        self.dtype = dtype

        with self.name_scope():
            specs = dict(grad_req='null', init=mx.init.Zero(),
                         dtype=self.dtype, allow_deferred_init=True,
                         differentiable=False)
            self.A = self.params.get(
                'A', shape=(self.num_states, self.num_states), **specs)
            self.B = self.params.get(
                'B', shape=(self.num_states, self.num_inputs), **specs)
            self.C = self.params.get(
                'C', shape=(self.num_outputs, self.num_states), **specs)
            self.D = self.params.get(
                'D', shape=(self.num_outputs, self.num_inputs), **specs)
            self.W = None
            self.V = None
        self.initialize(mx.init.Zero(), self.context)

    def begin_state(self, batch_size, context, F):
        return F.zeros((batch_size, self.num_states), context)

    def step(self, F, x, u, method, deterministic):
        dxdt = self.dynamics(F, x, u)
        x = self.integrate(F, x, dxdt, method)
        return self.add_process_noise(F, x, deterministic)

    def dynamics(self, F, x, u):
        return F.dot(x, self.A.data().T) + F.dot(u, self.B.data().T)

    def integrate(self, F, x, dxdt, method='euler-maruyama'):
        if method == 'euler-maruyama':
            return x + self.dt * dxdt
        else:
            raise NotImplementedError

    def add_process_noise(self, F, x, deterministic):
        if self.W is None or deterministic:
            return x
        dW = self.get_additive_white_gaussian_noise(
            F.eye(self.num_states, ctx=self.context) *
            F.sqrt(F.array([self.dt], self.context)))
        return x + F.dot(self.W.data(), dW)

    def output(self, F, x, u, deterministic):
        y = F.dot(x, self.C.data().T) + F.dot(u, self.D.data().T)
        return self.add_observation_noise(F, y, deterministic)

    def add_observation_noise(self, F, y, deterministic):
        if self.V is None or deterministic:
            return y
        return y + self.get_additive_white_gaussian_noise(self.V.data())

    def get_additive_white_gaussian_noise(self, scale):
        return mx.random.normal(mx.nd.zeros(len(scale), self.context),
                                mx.nd.diag(scale), dtype=self.dtype)

    def hybrid_forward(self, F, x, *args, **kwargs):
        u = args[0]
        deterministic = kwargs.get('deterministic', False)
        x = self.step(F, x, u, kwargs.get('method', 'euler-maruyama'),
                      deterministic)
        return self.output(F, x, u, deterministic), x


class DIMx(StochasticLinearIOSystemMx):
    def __init__(self, num_inputs, num_outputs, num_states, context, var_x=0,
                 var_y=0, dt=0.1, dtype='float32', freeze=True, **kwargs):

        super().__init__(num_inputs, num_outputs, num_states, context, dt,
                         dtype, **kwargs)

        self.A.data()[0, 1] = 1
        self.B.data()[1, 0] = 1  # Control only second state (acceleration).
        self.C.data()[:] = mx.nd.eye(self.num_outputs, self.num_states,
                                     ctx=self.context, dtype=self.dtype)
        specs = dict(grad_req='null', init=mx.init.Zero(), dtype=self.dtype,
                     allow_deferred_init=False, differentiable=False)
        if var_x:
            self.W = self.params.get(
                'W', shape=(self.num_states, self.num_states), **specs)
            self.W.initialize(ctx=self.context)
            self.W.data()[:] = var_x * mx.nd.eye(self.num_states,
                                                 ctx=self.context, dtype=dtype)
        if var_y:
            self.V = self.params.get(
                'V', shape=(self.num_outputs, self.num_outputs), **specs)
            self.V.initialize(ctx=self.context)
            self.V.data()[:] = var_y * mx.nd.eye(self.num_outputs,
                                                 ctx=self.context, dtype=dtype)
        if freeze:
            self.collect_params().setattr('grad_req', 'null')


class ControlledNeuralSystem(mx.gluon.HybridBlock):

    def __init__(self, neuralsystem: RNNModel, controller: RNNModel, context,
                 batch_size, **kwargs):
        super().__init__(**kwargs)
        self.neuralsystem = neuralsystem
        self.controller = controller
        self.context = context
        self.batch_size = batch_size

    def hybrid_forward(self, F, x, **kwargs):
        neuralsystem_states, controller_states = self.begin_state(F)
        neuralsystem_outputs = []
        for neuralsystem_input in x:
            if F is mx.ndarray:
                neuralsystem_input = F.expand_dims(neuralsystem_input, 0)
            controller_output, controller_states = self.controller(
                neuralsystem_states[0], controller_states)
            neuralsystem_output, neuralsystem_states = self.neuralsystem(
                neuralsystem_input,
                [neuralsystem_states[0] + controller_output])
            neuralsystem_outputs.append(neuralsystem_output)
        return F.concat(*neuralsystem_outputs, dim=0)

    def readout(self, F, x):
        return (F.dot(x, self.controller.rnn.l0_i2h_weight.data().T)
                + self.controller.rnn.l0_i2h_bias.data())

    def readin(self, F, x):
        return (F.dot(x, self.controller.decoder.weight.data().T) +
                self.controller.decoder.bias.data())

    def begin_state(self, F):
        kwargs = {'batch_size': self.batch_size, 'func': F.zeros,
                  'ctx': self.context}
        return (self.neuralsystem.rnn.begin_state(**kwargs),
                self.controller.rnn.begin_state(**kwargs))

    def apply_drift(self, where, dt, delta, drift, rng):
        if where in (None, 'None', 'none', ''):
            return
        elif where == 'sensor':
            parameters = self.neuralsystem.rnn.l0_i2h_weight
            scale = 1
        elif where == 'processor':
            parameters = self.neuralsystem.rnn.l0_h2h_weight
            scale = 1e-1
        elif where == 'actuator':
            parameters = self.neuralsystem.decoder.weight
            scale = 1e-2
        else:
            raise NotImplementedError
        shape = parameters.shape
        w = np.ravel(parameters.data().asnumpy())
        w = brownian(w, 1, dt, delta, scale * drift, None, rng)[0]
        parameters.data()[:] = np.reshape(w, shape)

    def get_reg_weights(self):
        return [self.controller.decoder.weight,
                self.controller.rnn.l0_i2h_weight]

    def sparsify(self, atol):
        weight_list = self.get_reg_weights()
        for weights in weight_list:
            idxs = np.nonzero(weights.data().abs().asnumpy() < atol)
            if len(idxs[0]):
                weights.data()[idxs] = 0


class ClosedControlledNeuralSystem(ControlledNeuralSystem):
    def __init__(self, environment: DIMx, neuralsystem: RNNModel,
                 controller: RNNModel, context, batch_size, num_steps: int,
                 **kwargs):
        super().__init__(neuralsystem, controller, context, batch_size,
                         **kwargs)
        self.environment = environment
        self.num_steps = num_steps

    def hybrid_forward(self, F, x, **kwargs):
        _, neuralsystem_states, controller_states = self.begin_state(F)
        environment_states = x
        neuralsystem_output = \
            F.zeros((1, self.batch_size, self.environment.num_inputs),
                    self.context)
        neuralsystem_outputs = []
        environment_state_list = []
        for _ in range(self.num_steps):
            environment_output, environment_states = self.environment(
                environment_states, neuralsystem_output)
            controller_output, controller_states = self.controller(
                neuralsystem_states[0], controller_states)
            neuralsystem_output, neuralsystem_states = self.neuralsystem(
                environment_output,
                [neuralsystem_states[0] + controller_output])
            neuralsystem_outputs.append(neuralsystem_output)
            environment_state_list.append(environment_states)
        return (F.concat(*neuralsystem_outputs, dim=0),
                F.concat(*environment_state_list, dim=0))

    def begin_state(self, F):
        environment_states = self.environment.begin_state(self.batch_size,
                                                          self.context, F)
        return (environment_states,) + super().begin_state(F)
