from scratch import configs
from src.plotting import plot_kalman_gain_vs_noise_levels
from src.utils import apply_config


config = configs.config_collect_training_data.get_config()

apply_config(config)

process_noises = config.process.PROCESS_NOISES
observation_noises = config.process.OBSERVATION_NOISES

plot_kalman_gain_vs_noise_levels(process_noises, observation_noises)
