import os
import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiLqr
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import plot_timeseries, plot_phase_diagram


def add_variables(monitor: Monitor):
    dtype = 'float32'
    kwargs = [
        dict(name='states', label='States', column_labels=['x', 'v'],
             dtype=dtype),
        dict(name='outputs', label='Output', column_labels=['y'], dtype=dtype),
        dict(name='control', label='Control', column_labels=['u'],
             dtype=dtype),
        dict(name='cost', label='Cost', column_labels=['c'], dtype=dtype)
    ]
    for k in kwargs:
        monitor.add_variable(**k)


def run_single(system_open, system_closed, times, monitor, inits):
    x = inits['x']

    for t in times:
        u = system_closed.get_control(x)
        x = system_open.step(t, x, u)
        y = system_open.output(t, x, u)
        c = system_closed.get_cost(x, u)

        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)


def main(config):

    label = 'lqr'
    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator with LQR feedback.
    system_closed = DiLqr(process_noise, observation_noise, dt, RNG,
                          config.controller.cost.lqr.Q,
                          config.controller.cost.lqr.R)
    system_open = system_closed.system

    # Sample some initial states.
    X0 = system_closed.get_initial_states(config.process.STATE_MEAN,
                                          config.process.STATE_COVARIANCE, 1,
                                          RNG)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with LQR control.
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        inits = {'x': x}
        run_single(system_open, system_closed, times, monitor, inits)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=system_closed.step, rng=RNG,
                           xt=config.controller.STATE_TARGET, path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqr.py')

    main(_config)

    sys.exit()
