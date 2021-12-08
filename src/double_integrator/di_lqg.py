import os
import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import plot_timeseries, plot_phase_diagram


def main(config):

    label = 'lqg'
    path_out = config.paths.PATH_OUT
    process_noise = config.process.PROCESS_NOISE
    observation_noise = config.process.OBSERVATION_NOISE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))

    # Create double integrator with LQG feedback.
    system_closed = DiLqg(process_noise, observation_noise, dt, RNG,
                          config.controller.cost.lqr.Q,
                          config.controller.cost.lqr.R)
    system_open = system_closed.system

    # Sample some initial states.
    n = 1
    X0 = system_closed.get_initial_states(mu0, Sigma0, n, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('state_estimates', 'States',
                         column_labels=[r'$\hat{x}$', r'$\hat{v}$'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])
    monitor.add_variable('control', 'Control', column_labels=['u'])
    monitor.add_variable('cost', 'Cost', column_labels=['c'])

    # Simulate the system with LQG control.
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        x_est = mu0
        Sigma = Sigma0
        for t in times:
            u = system_closed.get_control(x_est)
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            x_est, Sigma = system_closed.apply_filter(t, x_est, Sigma, u, y)
            c = system_closed.get_cost(x_est, u)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c,
                                     state_estimates=x_est)

        path = os.path.join(path_out, 'timeseries_{}_{}'.format(label, i))
        plot_timeseries(monitor.get_last_experiment(), path=path)

        path = os.path.join(path_out, 'phase_diagram_{}_{}'.format(label, i))
        plot_phase_diagram(monitor.get_last_trajectory(),
                           odefunc=system_closed.step, rng=RNG,
                           xt=config.controller.STATE_TARGET, path=path)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_lqg.py')

    main(_config)

    sys.exit()
