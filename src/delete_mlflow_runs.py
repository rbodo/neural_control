import os
import shutil
import sys

import mlflow
from mlflow.entities import ViewType

experiment_id = '0'
path = '/home/bodrue/Data/neural_control/double_integrator/rnn_controller'
os.chdir(path)
for run in mlflow.list_run_infos(experiment_id, ViewType.DELETED_ONLY):
    shutil.rmtree(os.path.join(path, 'mlruns', experiment_id, run.run_id))

sys.exit()
