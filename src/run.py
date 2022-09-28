import argparse
import os
import sys

import mlflow

label = 'nonlinear_rnn_rl'

# Needs to be set before executing run function.
# https://github.com/mlflow/mlflow/issues/608
os.environ['MLFLOW_TRACKING_URI'] = 'file:' + os.path.expanduser(
    f'~/Data/neural_control/{label}/mlruns')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--experiment_id', type=int, required=False)
experiment_id = parser.parse_args().experiment_id

mlflow.run('https://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@github.com/rbodo/'
           'neural_control.git',
           experiment_id=experiment_id,
           entry_point=f'examples/{label}.py',
           version='debug_snellius',  # branch
           experiment_name=label,
           run_name='Main',
           env_manager='local')

sys.exit()
