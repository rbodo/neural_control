import sys
import os

import numpy as np
from gym.wrappers import TimeLimit
from tqdm.contrib import tzip
from yacs.config import CfgNode

from src.double_integrator import configs
from src.double_integrator.control_systems import DiLqg
from src.double_integrator.di_lqg import jitter, get_grid
from src.double_integrator.di_rnn import add_variables
from src.double_integrator.plotting import plot_cost, plot_trajectories
from src.double_integrator.ppo_recurrent import MlpRnnPolicy, RecurrentPPO
from src.double_integrator.train_rnn_ppo import DoubleIntegrator, eval_rnn
from src.double_integrator.utils import apply_config, RNG, Monitor


def run_lqg(system, times, monitor, inits):
    x = inits['x']
    x_est = inits['x_est']
    Sigma = inits['Sigma']
    u = system.control.get_control(x_est)
    y = system.process.output(0, x, u)
    c = system.control.get_cost(x, u)

    for t in times:
        monitor.update_variables(t, states=x, outputs=y, control=u, cost=c)

        x, y, u, c, x_est, Sigma = system.step(t, x, x_est, Sigma)


def main(config: 'CfgNode', n: int):
    gpu = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'

    path_figures = config.paths.PATH_FIGURES
    path_model = config.paths.FILEPATH_MODEL + f'_{n}'
    filepath_output_data = config.paths.FILEPATH_OUTPUT_DATA
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    grid_size = config.simulation.GRID_SIZE
    dt = T / num_steps
    w = config.process.PROCESS_NOISES[0]
    v = config.process.OBSERVATION_NOISES[0]
    mu0 = config.process.STATE_MEAN
    Sigma0 = config.process.STATE_COVARIANCE * np.eye(len(mu0))
    q = config.controller.cost.lqr.Q
    r = config.controller.cost.lqr.R

    lqg = DiLqg(w, v, dt, RNG, q, r, normalize_cost=True)

    # Sample some initial states.
    grid = get_grid(grid_size, 0.5)

    # Initialize the state vectors at each jittered grid location.
    X0 = jitter(grid, Sigma0, RNG)
    X0_est = jitter(grid, lqg.process.W, RNG)

    times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')

    monitor = Monitor()
    add_variables(monitor)

    # Set cost threshold impossibly low so we always run for the full duration.
    env = DoubleIntegrator(w, v, dt, RNG, cost_threshold=1e-4, q=q, r=r)
    env = TimeLimit(env, num_steps)

    policy_kwargs = {'lstm_hidden_size': 50}
    model = RecurrentPPO(MlpRnnPolicy, env, verbose=1, device='cuda',
                         policy_kwargs=policy_kwargs)
    model = model.load(path_model)

    for i, (x, x_est) in enumerate(tzip(X0, X0_est, leave=False)):
        monitor.update_parameters(experiment=i)

        monitor.update_parameters(controller='lqg')
        inits = {'x': x.copy(), 'x_est': x_est, 'Sigma': Sigma0}
        run_lqg(lqg, times, monitor, inits)

        monitor.update_parameters(controller='rnn')
        eval_rnn(env, model, x, monitor)

    df = monitor.get_dataframe()
    df.to_pickle(filepath_output_data)
    print(f"Saved data to {filepath_output_data}.")

    plot_cost(df, os.path.join(path_figures, f'lqg_rnn_ppo_{n}_cost.png'))
    plot_trajectories(df, os.path.join(path_figures,
                                       f'lqg_rnn_ppo_{n}_trajectory.png'))


if __name__ == '__main__':
    for _n in [15]:  # [1, 2, 9, 13, 14, 17, 22, 23, 25, 28, 30]:
        _config = configs.config_lqg_vs_ppo.get_config()
        apply_config(_config)
        main(_config, _n)

    sys.exit()
