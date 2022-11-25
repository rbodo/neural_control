import os
import sys

import mxnet as mx
import numpy as np
import pandas as pd

from examples import configs
from examples.linear_rnn_lqg import LqgPipeline
from examples.visualize_linear_rnn_lqr import (
    add_states_mx, plot_training_curve_unperturbed, get_log_path, get_runs_all,
    plot_trajectories_unperturbed, get_training_data_unperturbed, add_scalars)
from src.control_systems_mxnet import (StochasticLinearIOSystem,
                                       ClosedControlledNeuralSystem)
from src.utils import get_data


def main(experiment_id, experiment_name, tag_start_time):
    # Get path where experiment data has been saved.
    log_path = get_log_path(experiment_name, '~/Data/neural_control')
    path = os.path.join(log_path, 'mlruns', experiment_id)

    # Get all training runs.
    runs = get_runs_all(log_path, experiment_id, tag_start_time)

    # Get configuration.
    config = configs.linear_rnn_lqg.get_config()

    # Get data to produce example trajectories through phase space.
    data_dict = get_data(config, 'states')
    data_test = data_dict['data_test']

    # Get pipeline for LQG experiment.
    pipeline = LqgPipeline(config, data_dict)
    pipeline.device = pipeline.get_device()

    # Get environment to produce example trajectories through phase space.
    environment = pipeline.get_environment()

    # Get unperturbed model before training.
    model_untrained = get_model_unperturbed_untrained(pipeline, environment)

    # Get unperturbed model after training.
    model_trained = get_model_trained(pipeline, environment, runs, path)
    pipeline.model = model_trained

    # Show example trajectories of unperturbed model before and after training.
    trajectories_unperturbed = get_trajectories_unperturbed(
        data_test, model_trained, model_untrained, pipeline)
    plot_trajectories_unperturbed(trajectories_unperturbed, log_path,
                                  'B. LQG oracle (particle)')

    # Show metric vs times of unperturbed model.
    training_data_unperturbed = get_training_data_unperturbed(runs, path)
    plot_training_curve_unperturbed(training_data_unperturbed, log_path,
                                    logy=True)


def get_trajectories_unperturbed(
        data_test: mx.gluon.data.DataLoader,
        model_trained: ClosedControlledNeuralSystem,
        model_untrained: ClosedControlledNeuralSystem, pipeline: LqgPipeline
) -> pd.DataFrame:
    num_batches = len(data_test)
    test_indexes = np.arange(0, num_batches, 16)
    num_steps = model_trained.num_steps
    data = {'index': [], 'controller': [], 'x0': [], 'x1': []}
    for test_index, (lqr_states, lqr_control) in enumerate(data_test):
        # Compute baseline loss.
        lqr_states = lqr_states.as_in_context(pipeline.device)

        if test_index not in test_indexes:
            continue

        # Get initial state.
        lqr_states = mx.nd.moveaxis(lqr_states, -1, 0)
        x_init = lqr_states[:1]

        # Get trajectories of trained model.
        _, environment_states = model_trained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index,
                    controller='RNN after training')

        # Get trajectories of untrained model.
        _, environment_states = model_untrained(x_init)
        add_states_mx(data, environment_states)
        add_scalars(data, num_steps, index=test_index,
                    controller='RNN before training')

        # Store LQG trajectory.
        add_states_mx(data, lqr_states)
        add_scalars(data, num_steps, index=test_index, controller='LQG')
    return pd.DataFrame(data)


def get_model_trained(
        pipeline: LqgPipeline, environment: StochasticLinearIOSystem,
        runs: pd.DataFrame, path: str):
    run_id = runs['run_id'].iloc[0]  # First random seed.
    path_model = os.path.join(path, run_id, 'artifacts', 'models',
                              'rnn.params')
    pipeline.model = pipeline.get_model(True, True, environment, path_model)
    return ClosedControlledNeuralSystem(
        environment, pipeline.model.neuralsystem, pipeline.model.controller,
        pipeline.device, pipeline.model.batch_size,
        pipeline.config.simulation.NUM_STEPS)


def get_model_unperturbed_untrained(
        pipeline: LqgPipeline, environment: StochasticLinearIOSystem):
    pipeline.model = pipeline.get_model(True, True, environment)
    return ClosedControlledNeuralSystem(
        environment, pipeline.model.neuralsystem, pipeline.model.controller,
        pipeline.device, pipeline.model.batch_size,
        pipeline.config.simulation.NUM_STEPS)


if __name__ == '__main__':
    _experiment_id = '1'
    _experiment_name = 'linear_rnn_lqg'
    _tag_start_time = '2022-10-18_14:21:00'

    main(_experiment_id, _experiment_name, _tag_start_time)

    sys.exit()
