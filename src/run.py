import argparse
import os
import sys

import mlflow


parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str)
parser.add_argument('--sweep_id', type=int, required=False, default=-1)
parser.add_argument('--resume_experiment', type=str, required=False,
                    default='')
args = parser.parse_args()
experiment_name = args.experiment_name
# Needs to be set before executing run function.
# https://github.com/mlflow/mlflow/issues/608
os.environ['MLFLOW_TRACKING_URI'] = 'file:' + os.path.expanduser(
    f'~/Data/neural_control/{experiment_name}/mlruns')

mlflow.run('https://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@github.com/rbodo/'
           'neural_control.git',
           entry_point=f'examples/{experiment_name}.py',
           parameters={'sweep_id': args.sweep_id,
                       'resume_experiment': args.resume_experiment},
           version='master',  # branch
           experiment_name=experiment_name,
           run_name='Main',
           env_manager='local')

sys.exit()
