import numpy as np

from src.double_integrator.plotting import plot_kalman_gain_vs_noise_levels

process_noise = np.logspace(-2, -1, 5, dtype='float32')
observation_noise = np.logspace(-1, 0, 5, dtype='float32')

plot_kalman_gain_vs_noise_levels(process_noise, observation_noise)
