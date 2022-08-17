import os
import sys

import mlflow

os.environ['MLFLOW_TRACKING_URI'] = 'file:/home/bodrue/Data/neural_control/double_integrator/rnn_controller/mlruns'
mlflow.run('https://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@github.com/rbodo/'
           'neural_control.git',
           'src/double_integrator/train_rnn_controller.py',
           'rnn_controller',
           env_manager='local')

sys.exit()
