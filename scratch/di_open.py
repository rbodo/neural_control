import sys

import numpy as np

from scratch import configs
from src.control_systems import DiOpen
from src.utils import RNG, Monitor, apply_config
from src.plotting import create_plots


def main(config):

    label = 'open'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator in open loop configuration.
    system = DiOpen(process_noise, observation_noise, dt, RNG)

    # Sample some initial states.
    X0 = system.process.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    monitor.add_variable('states', 'States', column_labels=['x', 'v'])
    monitor.add_variable('outputs', 'Output', column_labels=[r'$y_x$',
                                                             r'$y_v$'])

    # Simulate the system without control.
    u = [0]
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        for t in times:
            x = system.process.step(t, x, u)
            y = system.process.output(t, x, u)
            monitor.update_variables(t, states=x, outputs=y)

        create_plots(monitor, config, system.process, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_open.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
