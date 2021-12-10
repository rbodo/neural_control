from src.double_integrator.configs.config import get_config
from src.double_integrator.plotting import plot_kalman_gain_vs_noise_levels


config = get_config(
    '/home/bodrue/PycharmProjects/neural_control/src/double_integrator/'
    'configs/config_collect_training_data.py')

process_noises = config.process.PROCESS_NOISES
observation_noises = config.process.OBSERVATION_NOISES

plot_kalman_gain_vs_noise_levels(process_noises, observation_noises)
