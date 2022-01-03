from src.double_integrator import configs
from src.double_integrator.plotting import plot_kalman_gain_vs_noise_levels
from src.double_integrator.utils import apply_config


config = configs.config_collect_training_data.get_config()

apply_config(config)

process_noises = config.process.PROCESS_NOISES
observation_noises = config.process.OBSERVATION_NOISES

plot_kalman_gain_vs_noise_levels(process_noises, observation_noises)
