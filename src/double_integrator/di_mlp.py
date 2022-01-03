import sys

import numpy as np

from src.double_integrator import configs
from src.double_integrator.control_systems import DiMlp
from src.double_integrator.utils import RNG, Monitor, apply_config
from src.double_integrator.plotting import create_plots


def add_variables(monitor: Monitor):
    dtype = 'float32'
    kwargs = [
        dict(name='states', label='States', column_labels=['x', 'v'],
             dtype=dtype),
        dict(name='outputs', label='Output', column_labels=[r'$y_x$'],
             dtype=dtype),
        dict(name='control', label='Control', column_labels=['u'],
             dtype=dtype),
        dict(name='cost', label='Cost', column_labels=['c'], dtype=dtype)
    ]
    for k in kwargs:
        monitor.add_variable(**k)


def main(config):

    label = 'mlp'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps

    # Create double integrator with MLP feedback.
    system = DiMlp(process_noise, observation_noise, dt, RNG,
                   config.controller.cost.lqr.Q, config.controller.cost.lqr.R,
                   config.paths.FILEPATH_MODEL,
                   {'num_hidden': config.model.NUM_HIDDEN})

    # Sample some initial states.
    X0 = system.process.get_initial_states(config.process.STATE_MEAN,
                                           config.process.STATE_COVARIANCE)

    times = np.linspace(0, T, num_steps, endpoint=False)

    monitor = Monitor()
    add_variables(monitor)

    # Simulate the system with MLP control.
    for i, x in enumerate(X0):
        monitor.update_parameters(experiment=i, process_noise=process_noise,
                                  observation_noise=observation_noise)
        y = system.process.output(0, x, 0)
        for t in times:
            x, y, u, c = system.step(t, x, y)

            monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        create_plots(monitor, config, system, label, i, RNG)


if __name__ == '__main__':

    _config = configs.config_mlp.get_config()

    apply_config(_config)

    main(_config)

    sys.exit()
