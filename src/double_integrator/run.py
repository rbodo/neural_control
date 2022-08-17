import sys

import mlflow

mlflow.run('https://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@github.com/rbodo/'
           'neural_control.git',
           'src/double_integrator/train_rnn_controller.py',
           'rnn_controller',
           env_manager='local')

sys.exit()
