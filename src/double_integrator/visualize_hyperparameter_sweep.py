import os
import sys

import optuna
import plotly.io as pio

from src.double_integrator.configs.config import get_config

pio.renderers.default = 'png'

base_path = '/home/bodrue/PycharmProjects/neural_control/src/' \
            'double_integrator/'
filepath_config = os.path.join(
    base_path, 'configs/config_hyperparameter_rnn_high_noise.py')
config = get_config(filepath_config)
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
