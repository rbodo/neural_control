import os
import sys

import numpy as np
import mxnet as mx

from src.double_integrator import configs
from src.double_integrator.control_systems_mxnet import DiRnn
from src.double_integrator.di_rnn import run_single, add_variables
from src.double_integrator.plotting import plot_phase_diagram
from src.double_integrator.train_rnn_controller import get_model
from src.double_integrator.utils import Monitor

GPU = 2
os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'
context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()
config = configs.config_train_rnn_controller.get_config()
rng = np.random.default_rng(config.SEED)
T = config.simulation.T
num_steps = config.simulation.NUM_STEPS
dt = T / num_steps
process_noise = config.process.PROCESS_NOISES[0]
observation_noise = config.process.OBSERVATION_NOISES[0]

path_model = '/home/bodrue/Data/neural_control/double_integrator/' \
             'rnn_controller/rnn.params'
model = get_model(config, context, freeze_neuralsystem=False,
                  freeze_controller=True, load_weights_from=path_model)

system = DiRnn(process_noise, observation_noise, dt, rng, model_kwargs=dict(
    activation_rnn='relu', activation_decoder='relu'), gpu=GPU)
system.model = model.neuralsystem

monitor = Monitor()
add_variables(monitor)
times = np.linspace(0, T, num_steps, endpoint=False, dtype='float32')
X0 = [(0.4, 0.1), (0.8, -0.2)]
for i, x in enumerate(X0):
    monitor.update_parameters(experiment=i)
    x_rnn = np.zeros((system.model.num_layers,
                      system.model.num_hidden))
    y = system.process.output(0, x, 0)
    inits = {'x': x, 'x_rnn': x_rnn, 'y': y}
    run_single(system, times, monitor, inits)

df = monitor.get_dataframe()

fig = plot_phase_diagram(monitor.get_last_trajectory(),
                         xt=[0, 0], show=False, xlim=[-1, 1], ylim=[-1, 1],
                         line_label='RNN')
fig.legend()
fig.show()

sys.exit()
