import os
import sys

import mlflow

label = 'linear_rnn_lqr'

# Needs to be set before executing run function.
# https://github.com/mlflow/mlflow/issues/608
os.environ['MLFLOW_TRACKING_URI'] = 'file:/home/bodrue/Data/neural_control/' \
                                    f'{label}/mlruns'

mlflow.run('https://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@github.com/rbodo/'
           'neural_control.git',
           entry_point=f'examples/{label}.py',
           version='rnn_controller',  # branch
           experiment_name=label,
           env_manager='local')

sys.exit()
