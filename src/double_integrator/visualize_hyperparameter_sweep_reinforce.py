import os
import sys

import optuna
import plotly.io as pio

from src.double_integrator import configs
from src.double_integrator.utils import apply_config

pio.renderers.default = 'png'

config = configs.config_train_rnn_lqg_reinforce.get_config('')
apply_config(config)

study_name = config.paths.STUDY_NAME
filepath_output = config.paths.FILEPATH_OUTPUT_DATA
storage_name = f'sqlite:///{filepath_output}'
study = optuna.create_study(storage_name, study_name=study_name,
                            direction='maximize', load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

path_figures = config.paths.PATH_FIGURES
fig = optuna.visualization.plot_param_importances(study)
fig.write_image(os.path.join(path_figures, 'param_importances.png'))
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image(os.path.join(path_figures, 'optimization_history.png'))
fig = optuna.visualization.plot_slice(study)
fig.write_image(os.path.join(path_figures, 'slice.png'))

sys.exit()
