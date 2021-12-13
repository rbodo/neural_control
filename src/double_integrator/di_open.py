import sys

import numpy as np

from src.double_integrator.configs.config import get_config
from src.double_integrator.control_systems import DI
from src.double_integrator.utils import RNG, Monitor
from src.double_integrator.plotting import create_plots


def main(config):

    label = 'open'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator in open loop configuration.
    system_closed = DI(process_noise, observation_noise, dt, RNG)
    system_open = system_closed.system

    # Sample some initial states.
    X0 = system_closed.get_initial_states(config.process.STATE_MEAN,
                                          config.process.STATE_COVARIANCE)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('outputs', 'Output', column_labels=['y'])

    # Simulate the system without control.
    u = [0]
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        for t in times:
            x = system_open.step(t, x, u)
            y = system_open.output(t, x, u)
            monitor.update_variables(t, states=x, outputs=y)

        create_plots(monitor, config, system_closed, label, i, RNG)


if __name__ == '__main__':

    _config = get_config('/home/bodrue/PycharmProjects/neural_control/src/'
                         'double_integrator/configs/config_open.py')

    main(_config)

    sys.exit()
