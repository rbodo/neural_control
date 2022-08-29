import os
import sys

import numpy as np
import mxnet as mx

from src.double_integrator import configs
from src.double_integrator.control_systems import DIMx, \
    ClosedControlledNeuralSystem
from src.double_integrator.plotting import plot_phase_diagram
from src.double_integrator.train_rnn_controller import get_model

GPU = 2
os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'
context = mx.gpu(GPU) if mx.context.num_gpus() > 0 else mx.cpu()
config = configs.config_train_rnn_controller.get_config()
T = config.simulation.T
num_steps = config.simulation.NUM_STEPS
dt = T / num_steps
process_noise = config.process.PROCESS_NOISES[0]
observation_noise = config.process.OBSERVATION_NOISES[0]
batch_size = 1
config.training.BATCH_SIZE = batch_size

path_model = '/home/bodrue/Data/neural_control/double_integrator/' \
             'rnn_controller/rnn.params'
model = get_model(config, context, freeze_neuralsystem=False,
                  freeze_controller=True, load_weights_from=path_model)
environment = DIMx(1, 1, 2, context, process_noise, observation_noise, dt,
                   prefix='environment_')
model_closed = ClosedControlledNeuralSystem(
    environment, model.neuralsystem, model.controller, context, batch_size,
    num_steps)

data = mx.ndarray.array(np.expand_dims((0.8, -0.2), (0, 1)), model.context)
neuralsystem_outputs, environment_states = model_closed(data)
fig = plot_phase_diagram({'x': environment_states[:, 0, 0].asnumpy(),
                          'v': environment_states[:, 0, 1].asnumpy()},
                         xt=[0, 0], show=False, #xlim=[-1, 1], ylim=[-1, 1],
                         line_label='RNN')
fig.legend()
fig.show()

sys.exit()
