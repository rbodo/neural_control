import time

import numpy as np


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