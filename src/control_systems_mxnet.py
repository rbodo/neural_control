import os
from typing import Optional, List, Dict

import mxnet as mx
import numpy as np
from yacs.config import CfgNode

from src.empirical_gramians import emgr
from src import control_systems
from src.utils import get_lqr_cost, atleast_3d


class MlpModel(mx.gluon.HybridBlock):
    """Multi-layer perceptron."""
    def __init__(self, num_hidden=1, num_outputs=1, **kwargs):

        super().__init__(**kwargs)

        self.num_hidden = num_hidden

        with self.name_scope():
            self.hidden = mx.gluon.nn.Dense(num_hidden, activation='relu')
            self.output = mx.gluon.nn.Dense(num_outputs, activation='tanh')

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.output(self.hidden(x))


class RnnModel(mx.gluon.HybridBlock):
    """Multi-layer Elman RNN with fully-connected decoder."""
    def __init__(self, num_hidden=1, num_layers=1, num_outputs=1, num_inputs=1,
                 activation_rnn=None, activation_decoder=None, **kwargs):

        super().__init__(**kwargs)

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_outputs = num_outputs

        with self.name_scope():
            if self.num_layers == 1 and activation_rnn == 'linear':
                self.rnn = mx.gluon.rnn.RNNCell(num_hidden,
                                                mx.gluon.nn.LeakyReLU(1),
                                                input_size=num_inputs)
            else:
                self.rnn = mx.gluon.rnn.RNN(
                    num_hidden, num_layers, activation_rnn,
                    input_size=num_inputs, prefix='rnn_')
            self.decoder = mx.gluon.nn.Dense(
                num_outputs, activation=activation_decoder,
                in_units=num_hidden, flatten=False, prefix='decoder_')

    # noinspection PyUnusedLocal
    def hybrid_forward(self, F, x, *args):
        output, hidden = self.rnn(x, args[0])
        decoded = self.decoder(output)
        return decoded, hidden


class StochasticLinearIOSystem(mx.gluon.HybridBlock):
    def __init__(self, num_inputs, num_outputs, num_states, context, dt=0.1,
                 dtype='float32', **kwargs):
        super().__init__(**kwargs)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_states = num_states
        self.context = context
        self.dtype = dtype

        with self.name_scope():
            specs = dict(grad_req='null', init=mx.init.Zero(),
                         dtype=self.dtype, allow_deferred_init=True,
                         differentiable=False)
            self.dt = self.params.get('dt', shape=(1,), **specs)
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
        self.dt.data()[:] = dt

    def begin_state(self, batch_size, context, F):
        return F.zeros((1, batch_size, self.num_states), ctx=context)

    def step(self, F, x, u, **kwargs):
        dxdt = self.dynamics(F, x, u, **kwargs)
        x = self.integrate(F, x, dxdt, **kwargs)
        return self.add_process_noise(F, x, **kwargs)

    @staticmethod
    def dynamics(F, x, u, **kwargs):
        return F.elemwise_add(F.dot(x, F.transpose(kwargs['A'])),
                              F.dot(u, F.transpose(kwargs['B'])))

    def integrate(self, F, x, dxdt, **kwargs):
        method = kwargs.get('method', 'euler-maruyama')
        if method == 'euler-maruyama':  # x + dt * dx/dt
            return F.elemwise_add(x, F.broadcast_mul(kwargs['dt'], dxdt))
        else:
            raise NotImplementedError

    def add_process_noise(self, F, x, **kwargs):
        W = kwargs.pop('W', None)
        if W is None or kwargs.get('deterministic', False):
            return x
        dW = self.get_additive_white_gaussian_noise(
            F, self.num_states,
            F.broadcast_mul(F.ones(self.num_states, ctx=self.context),
                            F.sqrt(kwargs['dt'])))
        return F.broadcast_add(x, F.dot(W, dW))

    def output(self, F, x, u, **kwargs):
        y = F.elemwise_add(F.dot(x, F.transpose(kwargs['C'])),
                           F.dot(u, F.transpose(kwargs['D'])))
        return self.add_observation_noise(F, y, **kwargs)

    def add_observation_noise(self, F, y, **kwargs):
        V = kwargs.pop('V', None)
        if V is None or kwargs.get('deterministic', False):
            return y
        return F.broadcast_add(
            y, self.get_additive_white_gaussian_noise(F, self.num_outputs,
                                                      F.diag(V)))

    def get_additive_white_gaussian_noise(self, F, n, scale):
        return F.sample_normal(F.zeros(n, ctx=self.context), scale,
                               dtype=self.dtype)

    def hybrid_forward(self, F, x, *args, **kwargs):
        u = args[0]
        x = self.step(F, x, u, **kwargs)
        return self.output(F, x, u, **kwargs), x


class DI(StochasticLinearIOSystem):
    """Double integrator dynamical system."""
    def __init__(self, num_inputs, num_outputs, num_states, context, var_x=0,
                 var_y=0, dt=0.1, dtype='float32', **kwargs):

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


class MLP:
    """Dynamical system controlled by a multi-layer perceptron."""
    def __init__(self, process: control_systems.StochasticLinearIOSystem,
                 q: Optional[float] = 0.5, r: Optional[float] = 0.5,
                 path_model: Optional[str] = None,
                 model_kwargs: Optional[dict] = None,
                 dtype: Optional[str] = 'float32'):

        self.process = process
        self.dtype = dtype

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.model = MlpModel(**model_kwargs)
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
    """Dynamical system controlled by an RNN."""
    def __init__(self, process: control_systems.StochasticLinearIOSystem,
                 q: Optional[float] = 0.5, r: Optional[float] = 0.5,
                 path_model: Optional[str] = None,
                 model_kwargs: Optional[dict] = None, gpu: Optional[int] = 0,
                 dtype: Optional[str] = 'float32'):

        self.process = process
        self.dtype = dtype

        # State cost matrix:
        self.Q = q * np.eye(self.process.num_states, dtype=self.dtype)

        # Control cost matrix:
        self.R = r * np.eye(self.process.num_inputs, dtype=self.dtype)

        self.context = mx.gpu(gpu) if mx.context.num_gpus() > 0 else mx.cpu()
        self.model = RnnModel(**model_kwargs)
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


class ControlledNeuralSystem(mx.gluon.HybridBlock):
    """Perturbed neural system stabilized by an RNN controller."""
    def __init__(self, neuralsystem: RnnModel, controller: RnnModel,
                 device: mx.context.Context, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.neuralsystem = neuralsystem
        self.controller = controller
        self.context = device
        self.batch_size = batch_size
        self._weight_hashes = None

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

    def add_noise(self, where, sigma, dt, rng: np.random.Generator):
        if where in (None, 'None', 'none', ''):
            return
        elif where == 'sensor':
            parameters = self.neuralsystem.rnn.l0_i2h_weight
        elif where == 'processor':
            parameters = self.neuralsystem.rnn.l0_h2h_weight
        elif where == 'actuator':
            parameters = self.neuralsystem.decoder.weight
        else:
            raise NotImplementedError
        w = parameters.data().asnumpy()
        noise = rng.standard_normal(w.shape) * sigma * np.sqrt(dt)
        parameters.data()[:] = w + noise

    def get_reg_weights(self):
        return [self.controller.decoder.weight,
                self.controller.rnn.l0_i2h_weight]

    def sparsify(self, atol):
        weight_list = self.get_reg_weights()
        for weights in weight_list:
            idxs = np.nonzero(weights.data().abs().asnumpy() < atol)
            if len(idxs[0]):
                weights.data()[idxs] = 0

    def get_weight_hash(self) -> Dict[str, List[int]]:
        """Get the hash values of the model parameters."""
        return {'neuralsystem': to_hash(self.neuralsystem.collect_params()),
                'controller': to_hash(self.controller.collect_params())}

    def cache_weight_hash(self):
        """Store the hash values of the model parameters."""
        self._weight_hashes = self.get_weight_hash()

    def is_static(self, key: str, weight_hashes: Dict[str, List[int]]) -> bool:
        return np.array_equal(weight_hashes[key], self._weight_hashes[key])

    def assert_plasticity(self, freeze_neuralsystem: bool,
                          freeze_controller: bool):
        """Make sure only the allowed weights have been modified."""
        hashes = self.get_weight_hash()
        assert self.is_static('neuralsystem', hashes) == freeze_neuralsystem
        assert self.is_static('controller', hashes) == freeze_controller


class ClosedControlledNeuralSystem(ControlledNeuralSystem):
    """A neural system coupled with a controller and environment."""
    def __init__(self, environment: StochasticLinearIOSystem,
                 neuralsystem: RnnModel, controller: RnnModel,
                 device: mx.context.Context, batch_size: int, num_steps: int,
                 **kwargs):
        super().__init__(neuralsystem, controller, device, batch_size,
                         **kwargs)
        self.environment = environment
        self.num_steps = num_steps

    def hybrid_forward(self, F, x, **kwargs):
        _, neuralsystem_states, controller_states = self.begin_state(F)
        environment_states = x
        neuralsystem_output = \
            F.zeros((1, self.batch_size, self.environment.num_inputs),
                    ctx=self.context)
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


class PidRnn:
    """A perturbed RNN stabilized by a PID controller."""
    def __init__(self, process: control_systems.StochasticLinearIOSystem,
                 q=0.5, r=0.5, path_model=None, model_kwargs: dict = None,
                 gpu=0, k_p=1, k_i=1, k_d=1, dtype='float32'):

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
        self.pid = control_systems.PID(k_p=k_p, k_i=k_i, k_d=k_d)

    def get_model(self):
        model = RnnModel(**self.model_kwargs)
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
    """A dynamical system with Kalman filter and multi-layer perceptron
    controller."""
    def __init__(self, process: control_systems.StochasticLinearIOSystem,
                 q=0.5, r=0.5, path_model=None, model_kwargs=None):
        self.process = process
        self.estimator = control_systems.LQE(self.process)
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
    """A dynamical system with Kalman filter and RNN controller."""
    def __init__(self, process: control_systems.StochasticLinearIOSystem,
                 q=0.5, r=0.5, path_model=None, model_kwargs=None,
                 dtype='float32'):
        self.process = process
        self.dtype = dtype
        self.estimator = control_systems.LQE(self.process)
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


class DiMlp(MLP):
    """Double integrator controlled by a multi-layer perceptron."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs=None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = control_systems.DI(num_inputs, num_outputs, num_states,
                                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class DiPidRnn(PidRnn):
    """Double integrator with a perturbed RNN stabilized by a PID controller.
    """
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None, gpu=0, k_p=1,
                 k_i=1, k_d=1):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = control_systems.DI(num_inputs, num_outputs, num_states,
                                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs, gpu, k_p,
                         k_i, k_d)


class DiRnn(RNN):
    """Double integrator controlled by an RNN."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None, gpu=0):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = control_systems.DI(num_inputs, num_outputs, num_states,
                                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs, gpu)


class DiLqeMlp(LqeMlp):
    """Double integrator with Kalman filter and multi-layer perceptron
    controller."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs=None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = control_systems.DI(num_inputs, num_outputs, num_states,
                                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class DiLqeRnn(LqeRnn):
    """Double integrator with Kalman filter and RNN controller."""
    def __init__(self, var_x=0, var_y=0, dt=0.1, rng=None, q=0.5, r=0.5,
                 path_model=None, model_kwargs: dict = None):
        num_inputs = 1
        num_outputs = 1
        num_states = 2
        process = control_systems.DI(num_inputs, num_outputs, num_states,
                                     var_x, var_y, dt, rng)
        super().__init__(process, q, r, path_model, model_kwargs)


class Masker:
    """Helper class to set certain rows in the readout and stimulation matrix
    of a controller to zero."""
    def __init__(self, model: ControlledNeuralSystem, p,
                 rng: np.random.Generator):
        self.model = model
        self.p = p
        n = self.model.neuralsystem.num_hidden
        self._controllability_mask = np.nonzero(rng.binomial(1, self.p, n))
        self._observability_mask = np.nonzero(rng.binomial(1, self.p, n))
        self.controllability = 1 - len(self._controllability_mask[0]) / n
        self.observability = 1 - len(self._observability_mask[0]) / n

    def apply_mask(self):
        if self.p == 0:
            return
        self.model.controller.decoder.weight.data()[
            self._controllability_mask] = 0
        self.model.controller.rnn.l0_i2h_weight.data()[
            :, self._observability_mask] = 0


class Gramians(mx.gluon.HybridBlock):
    """Estimator for empirical controllability and observability Gramians."""
    def __init__(self, model: ControlledNeuralSystem,
                 environment: StochasticLinearIOSystem,
                 device: mx.context.Context, dt: float, T: float, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.environment = environment
        self.device = device
        self.dt = dt
        self.T = T
        self.num_inputs = self.model.controller.num_hidden
        self.num_hidden = self.model.neuralsystem.num_hidden
        self.num_outputs = self.model.controller.num_hidden
        self._return_observations = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        environment_states = self.environment.begin_state(1, self.device, F)
        neuralsystem_states = [args[0]]
        neuralsystem_output = \
            F.zeros((1, 1, self.environment.num_inputs), self.device)
        outputs = []
        for t in np.arange(0, self.T, self.dt):
            ut = F.array(atleast_3d(x(t)), self.device)
            environment_output, environment_states = self.environment(
                environment_states, neuralsystem_output)
            neuralsystem_output, neuralsystem_states = self.model.neuralsystem(
                environment_output,
                [neuralsystem_states[0] + self.model.readin(F, ut)])
            if self._return_observations:
                outputs.append(self.model.readout(F, neuralsystem_states[0]))
            else:
                outputs.append(neuralsystem_states[0])
        return F.concat(*outputs, dim=0).reshape((len(outputs), -1)).T

    # noinspection PyUnusedLocal
    def _ode(self, f, g, t, x0, u, p):
        x0 = mx.nd.array(np.expand_dims(x0, (0, 1)), self.device)
        return self.__call__(u, x0).asnumpy()

    def compute_gramian(self, kind):
        return emgr(None, 1,
                    [self.num_inputs, self.num_hidden, self.num_outputs],
                    [self.dt, self.T - self.dt], kind, ode=self._ode,
                    us=np.zeros(self.num_inputs), xs=np.zeros(self.num_hidden),
                    nf=12*[0]+[3])  # Normalize trajectories

    def compute_controllability(self):
        self._return_observations = False
        return self.compute_gramian('c')

    def compute_observability(self):
        self._return_observations = True
        return self.compute_gramian('o')


class L2L1(mx.gluon.loss.L2Loss):
    """L2 loss on activations with L1 loss on weights."""
    def __init__(self, lambda_: float, context, **kwargs):
        super().__init__(**kwargs)
        self.l1 = mx.gluon.loss.L1Loss(lambda_)
        self.context = context

    def hybrid_forward(self, F, pred, label, weight_list: list = None,
                       sample_weight=None):
        l2 = super().hybrid_forward(F, pred, label, sample_weight)
        if weight_list is not None:
            for weights in weight_list:
                l2 = l2 + self.l1(
                    weights.data(), F.zeros(weights.shape,
                                            ctx=self.context)).mean()
        return l2


class LqrLoss(mx.gluon.loss.Loss):
    """Linear Quadratic Regulator loss."""
    def __init__(self, weight=1, batch_axis=0, **kwargs):
        self.Q = kwargs.pop('Q')
        self.R = kwargs.pop('R')
        super().__init__(weight, batch_axis, **kwargs)

    # noinspection PyMethodOverriding,PyProtectedMember
    def hybrid_forward(self, F, x, u, sample_weight=None):
        loss = (F.sum(x * F.batch_dot(F.tile(self.Q, (len(x), 1, 1)), x), 1) +
                F.sum(u * F.batch_dot(F.tile(self.R, (len(x), 1, 1)), u), 1))
        loss = mx.gluon.loss._apply_weighting(F, loss, self._weight,
                                              sample_weight)
        if mx.is_np_array():
            if F is mx.nd.ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)


def get_device(config: CfgNode) -> mx.context.Context:
    """Return hardware backend to run on."""

    # Disable an irrelevant cudnn library warning.
    os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'

    return mx.gpu(int(config.GPU)) if mx.context.num_gpus() > 0 else mx.cpu()


def to_hash(params: mx.gluon.ParameterDict) -> List[int]:
    """Compute the hash values of a list of weight matrices."""
    return [hash(tuple(param.data().asnumpy().ravel()))
            for param in params.values()]
