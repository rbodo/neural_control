import numpy as np
from scipy.linalg import solve_discrete_are
from numpy.linalg import inv
from numpy.random import multivariate_normal


# Dynamical systems

# numerical integration; can be applied to np and mxnet.np


def euler(f, x, t, dt):
    """
    Euler approximation

    f: state equation function
    t: time at which the function is evaluated
    x: state vector at time k
    dt: step size

    Returns the state at time k+1
    """

    return x + f(x, t) * dt


def RK4(f, x, t, dt):
    """Apply Runge Kutta Formulas to find next value of x"""

    k1 = dt * f(x, t)
    k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * f(x + k3, t + 1.0 * dt)

    return x + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_maruyama(f, x, t, dt, L, Q):
    """
    Euler-Maruyama approximation

    f: state equation function
    x: state vector at time k
    t: evaluation time
    dt: temporal interval
    L: N x D dispersion matrix (not state/time dependent at the moment)
    Q: spectral density (identity matrix for d-dimensional Brownian motion)

    Returns the state at time k+1
    """

    [N, D] = np.shape(L)
    db = np.random.multivariate_normal(np.zeros(D), Q * dt)

    return x + f(x, t) * dt + np.matmul(L, db)


# Control theory

class LQR:
    """
        
        Discrete time linear quadratic regulator
        
        state equation:
        x_n+1 = x_n' A x_n + B u_n + e_n with e_n ~ N(0, W)

        initial state x(0) ~ N(mu0, Sigma0)

        cost function
        J = sum_n [x_n' Q x_n + u_n' R u_n]

    """

    def __init__(self, A, B, W, Q, R, mu0, Sigma0):

        self.n_state = A.shape[0]
        self.n_control = B.shape[1]

        self.A = A
        self.B = B
        self.W = W

        self.Q = Q
        self.R = R

        self.mu0 = mu0
        self.Sigma0 = Sigma0

        # solution to the DARE
        self.S = solve_discrete_are(A, B, Q, R)

        # Feedback gain matrix
        self.L = np.linalg.inv(B.T @ self.S @ B + R) @ B.T @ self.S @ A

    def initial_state(self):
        return multivariate_normal(self.mu0, self.Sigma0)

    def control(self, x):
        return -self.L @ x

    def step(self, x, u=None):
        if u:
            return multivariate_normal(self.A @ x + self.B @ u, self.W)
        else:
            return multivariate_normal(self.A @ x, self.W)

    def reward(self, x, u):
        return - x.T @ self.Q @ x - np.atleast_2d(
            u).T @ self.R @ np.atleast_2d(u)


class LQG(LQR):
    """
    Linear quadratic Gaussian regulator. Adds observation model

        y_n = C x_n + v_n with v_n ~ N(0, V)

    """

    def __init__(self, A, B, W, Q, R, mu0, Sigma0, C, V):
        super().__init__(A, B, W, Q, R, mu0, Sigma0)

        self.n_obs = C.shape[0]

        self.C = C
        self.V = V

    def observe(self, x):
        return multivariate_normal(self.C @ x, self.V)


def kalman_filter(mu, Sigma, y, A, C, W, V, uk=None, B=None):
    # KF prediction step
    muk1 = (A @ mu)[:, np.newaxis]
    if uk:
        muk1 += (B @ uk)[:, np.newaxis]
    Sigmak1 = A @ Sigma @ A.T + W

    # KF update step
    r = y[:, np.newaxis] - C @ muk1  # innovation
    K = Sigmak1 @ C.T @ inv(C @ Sigmak1 @ C.T + V)  # optimal Kalman gain

    muk = np.squeeze(muk1 + K @ r)
    Sigmak = (np.eye(A.shape[0]) - K @ C) @ Sigmak1

    return muk, Sigmak


class double_integrator(LQG):
    """

    The discrete-time (controlled, stochastic) double integrator is our standard minimal benchmark model:

        x_n+1 = A x_n + B u_n + w_n
        y_n = C x_n + v_n
   
    with
    
        A = [1 dt; 0 1]
        b = [0; dt]
        w_n ~ N([0; 0], dt*var_x*I)
        C = [1, 0]
        v_n ~ N([0; 0], var_y*I)

    """

    def __init__(self, dt=0.1, var_x=10 ** -2, var_y=0.0, q=0.5, r=0.5,
                 mu0=np.zeros(2), var0=0.0):
        """

        params:

        dt      : time step
        var_x   : variance of the process noise
        var_y   : variance of the observation noise
        q       : state cost (position)
        r       : control cost (energy expenditure)
        mu0     : initial state (position and velocity)
        var_0   : variance on initial state
        """

        self.dt = dt

        A = np.array([[1, dt], [0, 1]])
        B = np.atleast_2d(np.array([0, dt])).T
        W = dt * var_x * np.eye(2)
        C = np.atleast_2d(np.array([1, 0]))
        V = var_y * np.eye(1)

        Q = np.array([[dt * q, 0], [0, 0]])
        R = np.atleast_2d(np.array([dt * r]))

        mu0 = mu0
        Sigma0 = var0 * np.eye(2)  # P0 in text

        super().__init__(A, B, W, Q, R, mu0, Sigma0, C, V)


## Neural Networks

class Adam:

    def __init__(self, eta=10 ** -2, beta1=0.9, beta2=0.999):
        self.vP = None
        self.sP = None
        self.t = 0
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2

    def update(self, grad):
        """
        Returns new parameter update by controlling the gradient
        """

        if self.vP is None:
            self.vP = np.zeros(grad.size)
            self.sP = np.zeros(grad.size)

        self.t += 1
        self.vP = self.beta1 * self.vP + (1.0 - self.beta1) * grad
        self.sP = self.beta2 * self.sP + (1.0 - self.beta2) * grad ** 2

        vhat = self.vP / (1.0 - self.beta1 ** self.t)
        shat = self.sP / (1.0 - self.beta2 ** self.t)

        return self.eta * vhat / (np.sqrt(shat) + 10 ** -9)

    def reset(self):
        self.vP = None
        self.vS = None
        self.t = 0


class SGD:

    def __init__(self, eta=10 ** -2):
        self.eta = eta

    def update(self, grad):
        return self.eta * grad

    def reset(self):
        pass


def grad_clip(grad, clip=2.0):
    """
    Gradient clipping
    """
    if clip is None:
        return grad
    elif not np.any(grad):
        return grad
    else:
        return clip * grad / np.linalg.norm(grad.flatten())


# reinforcement learning

class TDlearn():
    """
    Temporal difference learning (finite vs infinite horizon?)
    """

    def __init__(self, theta, optimizer=Adam(), gamma=0.99, elambda=0.4):
        self.n_states = theta.size

        # Parameter initialization
        self.theta = theta

        # Discount factor; this one is tricky for the infinite horizon LQG based on undiscounted rewards (since optimal policy is controllable and hence 0? But what about noise term?)
        self.gamma = gamma  # discount factor

        # Eligibility traces
        self.elambda = elambda
        self.etrace = np.zeros(self.n_states)

        self.optimizer = optimizer

        # TD error
        self.delta = np.nan

    def value(self, xn):
        """
        Returns state value estimate for state xn
        """
        return np.dot(self.theta, xn)

    def td_error(self, xn, rn, xn1):
        """
        Returns temporal difference error
        """
        return rn + self.gamma * self.value(xn1) - self.value(xn)

    def step(self, xn, rn, xn1):
        # eligibility update
        self.etrace = self.gamma * self.elambda * self.etrace + xn

        # Compute TD error
        self.delta = self.td_error(xn, rn, xn1)

        # semi-gradient SGD parameter update
        self.theta = self.theta + self.optimizer.update(
            self.delta * self.etrace)


class Qlearning():
    """
        Q learning via neural fitted Q iteration; THIS DOES NOT YET WORK PROPERLY; FOCUS ON POLICY GRADIENTS?
    """

    def __init__(self, theta_state, theta_control, optimizer=Adam(),
                 gamma=0.99, elambda=0.4):
        self.n_states = theta_state.size
        self.n_control = theta_control.size

        self.theta = np.hstack([theta_state, theta_control])
        self.n_theta = self.theta.size

        # Discount factor; this one is tricky for the infinite horizon LQG based on undiscounted rewards (since optimal policy is controllable and hence 0? But what about noise term?)
        self.gamma = gamma  # discount factor

        # control space used for sampling controls
        self.control_space = np.linspace(-1.0, 1.0, 10)

        # eligibility traces
        self.elambda = elambda
        self.etrace = np.zeros(self.n_theta)

        self.optimizer = optimizer

        # self.qhat = None

        # TD error
        self.delta = None

    def value(self, xn, un):
        """
        Returns state action value estimate for joint state-action z = (x,u)
        """
        return np.dot(self.theta, np.hstack([xn, un]))

    def td_error(self, xn, un, rn, xn1):
        """
        Returns temporal difference error
        """

        # We must find the maximum over actions. Here we just sample the action space
        max_q = np.max(
            list(map(lambda u: self.value(xn1, u), self.control_space)))

        return rn + self.gamma * max_q - self.value(xn, un)

        # return rn + self.gamma * self.value(zn1) -  self.value(zn)

    def step(self, xn, un, rn, xn1):
        # zn = np.hstack([xn, un])

        # # # estimated cost-to-go
        # # self.qhat = np.dot(self.theta, zn)

        # eligibility update
        self.etrace = self.gamma * self.elambda * self.etrace + np.hstack(
            [xn, un])

        # Compute TD error
        self.delta = self.td_error(xn, un, rn, xn1)

        # semi-gradient SGD parameter update; NOTE: be careful with gradient descent/ascent; FORMULATE ALL AS GRADIENT DESCENT!!
        self.theta = self.theta + self.optimizer.update(
            self.delta * self.etrace)

    def control(self, xn):
        # keep random control in same space as sampled control
        # return 10**-3 * 2.0 * (np.random.rand(self.n_control) - 0.5)

        # select action which minimizes the q value
        i = np.argmin(list(
            map(lambda u: np.dot(self.theta, np.hstack([xn, u])),
                self.control_space)))

        # print(self.control_space[i])
        return np.atleast_1d(self.control_space[i])


class ActorCritic():
    """
        Actor-Critic learning; still needs to be generalized to multiple continuous actions
    """

    def __init__(self, theta_mu, theta_sigma, theta_v,
                 optimizers=[Adam(), Adam()], elambda_theta=0.4,
                 elambda_v=0.4):
        """
        :params
            theta_mu : n_states parameter vector for mean of the action
            theta_sigma : n_states parameter vector for variance of the action
            theta_v : n_states parameter vector for the value estimate
            optimizers : 2 optimizers for policy and value
            elambda_theta : decay parameter of eligibility trace for policy
            elambda_v : decay parameter of eligibility trace for value

        """

        # THIS BECOMES MUCH EASIER IN A MXNET (OR JAX) IMPLEMENTATION; THOUGH AT THE EXPENSE OF UNDERSTANDING

        self.n_states = theta_mu.size

        # Parameter initialization
        self.theta = np.hstack([theta_mu, theta_sigma])

        self.theta_v = theta_v  # w in lecture notes

        self.beta = 0.01  # reasonable value? Used in page 251 of S&B
        self.barc = 0.0

        self.elambda_theta = elambda_theta
        self.etrace_theta = np.zeros(2 * self.n_states)

        self.elambda_v = elambda_v
        self.etrace_v = np.zeros(self.n_states)

        self.optimizers = optimizers

        self.vhat = None

    def control(self, x):
        mu = np.dot(self.theta[:self.n_states], x)
        # print(x) # NOTE: This becomes really large due to the squared values; should we renormalize the inputs?
        # print(np.dot(self.theta[self.n_states:], x))
        sigma = np.exp(np.dot(self.theta[self.n_states:], x))
        # print(x)
        # print(mu)
        # print(sigma)
        return multivariate_normal(np.atleast_1d(mu), np.atleast_2d(sigma))

    def step(self, xn, un, gn, xn1):
        # TD error
        self.delta = gn - self.barc + np.dot(self.theta_v, xn1) - np.dot(
            self.theta_v, xn)

        # mean cost update
        self.barc = self.barc + self.beta * self.delta

        # gradient computation
        grad_vhat = xn

        mu = np.dot(self.theta[:self.n_states], xn)
        sigma = np.exp(np.dot(self.theta[self.n_states:], xn))

        grad_mu = (
                          un - mu) * xn / sigma ** 2  # NOTE: Previous version ommited square by mistake
        grad_sigma = (((un - mu ** 2) / sigma ** 2) - 1.0) * xn

        # eligibility updates
        self.etrace_theta = self.elambda_theta * self.etrace_theta + np.hstack(
            [grad_mu, grad_sigma])
        self.etrace_v = self.elambda_v * self.etrace_v + grad_vhat

        # semi-gradient SGD parameter update
        self.theta = self.theta - self.optimizers[0].update(
            self.delta * self.etrace_theta)
        # self.theta[self.n_states:] = 0 # keep variance fixed
        self.theta_v = self.theta_v + self.optimizers[1].update(
            self.delta * self.etrace_v)
