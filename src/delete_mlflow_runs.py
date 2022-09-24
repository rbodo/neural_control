import os
import shutil
import sys

import mlflow

experiment_id = '1'
label = 'linear_rnn_lqr'
tag = '2022-09-21_17:29:07'
path = os.path.expanduser(f'~/Data/neural_control/{label}')
os.chdir(path)
runs = mlflow.search_runs([experiment_id])
runs_to_delete = runs.loc[runs['tags.main_start_time'] != tag]
for run_id in runs_to_delete['run_id']:
    # mlflow.delete_run(run_id)  # Only changes STATUS flag
    shutil.rmtree(os.path.join(path, 'mlruns', experiment_id, run_id))

sys.exit()
