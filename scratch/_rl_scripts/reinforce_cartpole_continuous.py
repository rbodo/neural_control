import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Normal

from scratch._rl_scripts.cartpole_env_continuous import ContinuousCartPoleEnv

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=54, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

env = ContinuousCartPoleEnv()
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = torch.relu(x)
        return self.affine2(x)


policy = Policy().to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    out = policy(state)
    mean = torch.tanh(out[:, 0])
    var = torch.sigmoid(out[:, 1])
    m = Normal(mean, var)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return np.clip(np.array(action.cpu(), 'float32'), -1, 1)


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    show_plot = False
    if show_plot:
        plt.plot(-np.array([p.detach().cpu().numpy() for p in policy.saved_log_probs]))
        plt.plot(returns.detach().cpu().numpy(), '--')
        plt.plot(np.array([p.detach().cpu().numpy() for p in policy_loss]))
        plt.hlines(0, 0, plt.gcf().axes[0].get_xlim()[1], 'k')
        plt.show()
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'
                  ''.format(i_episode, ep_reward, running_reward))
        if running_reward > 475:#env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(
                  running_reward, t))
            break


if __name__ == '__main__':
    main()
