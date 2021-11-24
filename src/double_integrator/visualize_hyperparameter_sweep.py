import optuna
import plotly.io as pio
pio.renderers.default = 'png'

study_name = 'rnn'
storage_name = 'sqlite:///../../{}.db'.format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,
                            load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('param_importances.png')
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('optimization_history.png')
fig = optuna.visualization.plot_slice(study)
fig.write_image('slice.png')

print()
