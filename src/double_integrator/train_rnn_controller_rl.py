import logging
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from gym.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from tqdm import tqdm

from src.double_integrator import configs
from src.double_integrator.control_systems_torch import RnnModel, DiGym, \
    Masker, Gramians
from src.double_integrator.hyperparameter_rnn_ppo import linear_schedule, \
    FigureRecorderTest, evaluate
from src.double_integrator.ppo_recurrent import RecurrentPPO, MlpRnnPolicy
from src.double_integrator.utils import get_artifact_path


def get_model(config, device, freeze_neuralsystem, freeze_controller,
              load_weights_from: str = None):
    environment_num_states = 2
    environment_num_outputs = 1
    neuralsystem_num_states = config.model.NUM_HIDDEN_NEURALSYSTEM
    neuralsystem_num_layers = config.model.NUM_LAYERS_NEURALSYSTEM
    neuralsystem_num_outputs = 1
    controller_num_states = config.model.NUM_HIDDEN_CONTROLLER
    controller_num_layers = config.model.NUM_LAYERS_CONTROLLER
    activation_rnn = config.model.ACTIVATION
    activation_decoder = None  # Defaults to 'linear'
    process_noise = config.process.PROCESS_NOISES[0]
    observation_noise = config.process.OBSERVATION_NOISES[0]
    learning_rate = config.training.LEARNING_RATE
    T = config.simulation.T
    num_steps = config.simulation.NUM_STEPS
    dt = T / num_steps
    dtype = torch.float32

    environment = DiGym(neuralsystem_num_outputs, environment_num_outputs,
                        environment_num_states, device, process_noise,
                        observation_noise, dt, cost_threshold=1e-4,
                        use_observations_in_cost=True)
    environment = TimeLimit(environment, num_steps)

    controller = RnnModel(controller_num_states, controller_num_layers,
                          neuralsystem_num_states, neuralsystem_num_states,
                          activation_rnn, activation_decoder, device, dtype)
    if load_weights_from is None:
        controller.init_zero()

    policy_kwargs = {'lstm_hidden_size': neuralsystem_num_states,
                     'n_lstm_layers': neuralsystem_num_layers,
                     'activation_fn': activation_rnn,
                     'net_arch': [],
                     'controller': controller,
                     # 'shared_lstm': True,
                     # 'enable_critic_lstm': False,
                     }
    log_dir = get_artifact_path('tensorboard_log')
    model = RecurrentPPO(
        MlpRnnPolicy, environment, verbose=0, device=device, seed=config.SEED,
        tensorboard_log=log_dir, policy_kwargs=policy_kwargs, n_epochs=10,
        learning_rate=linear_schedule(learning_rate, 0.005), n_steps=num_steps,
        batch_size=None)

    if load_weights_from is not None:
        model.set_parameters(load_weights_from)

    controller.requires_grad_(not freeze_controller)
    model.policy.action_net.requires_grad_(not freeze_neuralsystem)
    model.policy.lstm_actor.neuralsystem.requires_grad_(
        not freeze_neuralsystem)

    return model, environment


def add_noise(model, where, sigma, dt, rng: np.random.Generator):
    if where in (None, 'None', 'none', ''):
        return
    elif where == 'sensor':
        parameters = model.policy.lstm_actor.neuralsystem.weight_ih_l0
    elif where == 'processor':
        parameters = model.policy.lstm_actor.neuralsystem.weight_hh_l0
    elif where == 'actuator':
        parameters = model.policy.action_net.weight
    else:
        raise NotImplementedError
    noise = rng.standard_normal(parameters.shape) * sigma * np.sqrt(dt)
    parameters[:] = parameters + torch.tensor(noise, dtype=parameters.dtype,
                                              device=parameters.device)


class MaskingCallback(BaseCallback):
    def __init__(self, masker: Masker):
        super().__init__()
        self.masker = masker

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        self.masker.apply_mask()

    def _on_training_end(self):
        self.masker.apply_mask()


def train(config, perturbation_type, perturbation_level, dropout_probability,
          device, save_model=True):
    num_epochs = int(config.training.NUM_EPOCHS)
    num_steps = config.simulation.NUM_STEPS
    T = config.simulation.T
    dt = T / num_steps
    seed = config.SEED
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    num_test = 100

    is_perturbed = perturbation_type in ['sensor', 'actuator', 'processor']
    if is_perturbed:
        path_model = config.paths.FILEPATH_MODEL
        model, env = get_model(config, device, freeze_neuralsystem=True,
                               freeze_controller=False,
                               load_weights_from=path_model)
    else:
        model, env = get_model(config, device, freeze_neuralsystem=False,
                               freeze_controller=True)

    controlled_neuralsystem = model.policy.lstm_actor

    if is_perturbed and perturbation_level > 0:
        add_noise(model, perturbation_type, perturbation_level, dt, rng)

    logging.info("Computing baseline performances...")
    reward, length = evaluate(model, env, num_test, deterministic=False)
    mlflow.log_metric('training_reward', reward, -1)
    reward, length = evaluate(model, env, num_test, 'trajectory_-1.png')
    mlflow.log_metric('test_reward', reward, -1)

    if is_perturbed:
        # Initialize controller to non-zero weights only after computing
        # baseline accuracy of perturbed network before training. Otherwise the
        # untrained controller output will degrade accuracy further.
        controlled_neuralsystem.controller.init_nonzero()

    masker = Masker(controlled_neuralsystem, dropout_probability, rng)
    callbacks = [EveryNTimesteps(10000, FigureRecorderTest()),
                 MaskingCallback(masker)]

    logging.info("Training...")
    model.learn(num_epochs, callback=callbacks)

    if save_model:
        path_model = get_artifact_path('models/rnn.params')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        model.save(path_model)
        logging.info(f"Saved model to {path_model}.")

    if is_perturbed:
        with torch.no_grad():
            logging.info("Computing Gramians...")
            gramians = Gramians(controlled_neuralsystem, env,
                                model.policy.action_net, T)
            g_c = gramians.compute_controllability()
            g_o = gramians.compute_observability()
            g_c_eig = np.linalg.eig(g_c)
            g_o_eig = np.linalg.eig(g_o)
            mlflow.log_metrics({'controllability': np.prod(g_c_eig[0]).item(),
                                'observability': np.prod(g_o_eig[0]).item()})
    else:
        g_c = g_o = None

    training_reward = model.ep_info_buffer[-1]['r']
    test_reward, length = evaluate(model, env, num_test)
    return {'perturbation_type': perturbation_type,
            'perturbation_level': perturbation_level,
            'dropout_probability': dropout_probability,
            'training_reward': training_reward, 'test_reward': test_reward,
            'controllability': g_c, 'observability': g_o}


def main(config):
    device = torch.device('cpu' if GPU is None else 'cuda:0')
    perturbation_types = config.perturbation.PERTURBATION_TYPES
    perturbation_levels = config.perturbation.PERTURBATION_LEVELS
    dropout_probabilities = config.perturbation.DROPOUT_PROBABILITIES
    seeds = config.SEEDS

    dfs = []
    mlflow.set_tracking_uri(os.path.join('file:' + config.paths.BASE_PATH,
                                         'mlruns'))
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name='Main')
    out = train(config, None, 0, 0, device)
    dfs.append(out)
    config.defrost()
    config.paths.FILEPATH_MODEL = get_artifact_path('models/rnn.params')
    with open(get_artifact_path('config.txt'), 'w') as f:
        f.write(config.dump())
    for perturbation_type in tqdm(perturbation_types, 'perturbation_type',
                                  leave=False):
        mlflow.start_run(run_name='Perturbation type', nested=True)
        mlflow.log_param('perturbation_type', perturbation_type)
        for perturbation_level in tqdm(perturbation_levels,
                                       'perturbation_level', leave=False):
            mlflow.start_run(run_name='Perturbation level', nested=True)
            mlflow.log_param('perturbation_level', perturbation_level)
            for dropout_probability in tqdm(
                    dropout_probabilities, 'dropout_probability', leave=False):
                mlflow.start_run(run_name='Dropout probability', nested=True)
                mlflow.log_param('dropout_probability', dropout_probability)
                for seed in tqdm(seeds, 'seed', leave=False):
                    config.SEED = seed
                    mlflow.start_run(run_name='seed', nested=True)
                    mlflow.log_param('seed', seed)
                    out = train(config, perturbation_type, perturbation_level,
                                dropout_probability, device)
                    dfs.append(out)
                    mlflow.end_run()
                mlflow.end_run()
            mlflow.end_run()
        mlflow.end_run()
    df = pd.DataFrame(dfs)
    df.to_pickle(get_artifact_path('output.pkl'))
    mlflow.end_run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    GPU = 9
    if GPU is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

    EXPERIMENT_NAME = 'train_rnn_controller'

    _config = configs.config_train_rnn_controller_rl.get_config()

    main(_config)

    sys.exit()
