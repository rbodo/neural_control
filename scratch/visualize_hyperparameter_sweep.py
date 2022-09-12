import os
import sys

import optuna
import plotly.io as pio

from scratch import configs
from src.utils import apply_config

pio.renderers.default = 'png'

# config = configs.config_hyperparameter_rnn_low_noise.get_config()
config = configs.config_hyperparameter_rnn_high_noise.get_config()
apply_config(config)

study_name = config.paths.STUDY_NAME
filepath_output = config.paths.FILEPATH_OUTPUT_DATA
storage_name = f'sqlite:///{filepath_output}'
study = optuna.create_study(study_name=study_name, storage=storage_name,
                            load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

path_figures = config.paths.PATH_FIGURES
fig = optuna.visualization.plot_param_importances(study)
fig.write_image(os.path.join(path_figures, 'param_importances.png'))
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image(os.path.join(path_figures, 'optimization_history.png'))
fig = optuna.visualization.plot_slice(study)
fig.write_image(os.path.join(path_figures, 'slice.png'))

sys.exit()
