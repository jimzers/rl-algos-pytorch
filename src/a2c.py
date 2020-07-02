import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.distributions import Normal
from torch.optim import Adam

from utils.utils import make_mlp, init_xav_weights, init_weights


class ActorNetwork(nn.Module):
    """
    Actor thingy. Gonna try this instead of a helper fn, might be easier to understand and structure
    This actor is diagonal gaussian
    """

    def __init__(self, obs_dim, hidden_size, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def distribution(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))  # personal choice for tanh
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, state, action=None):
        """
        Forward pass, returns policy distribution.
        If action given, also gives log likelihood of action
        """
        # if there's an action
        policy_dist = self.distribution(state)
        log_prob = None
        if action:
            # TODO: fix the action here if it's not in the right format
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
        return policy_dist, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size, value_dim=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, value_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)  # personal choice for no activation fn on last layer...
        return output


class A2CAgent:
    """
    Attempt 1 of A2C algorithm..

    notes:
    By running different exploration policies in diff threads, the overall changes made to the params by multiple actor learnings applying online updates in parallel
    are less correlated in time compared to single agent applyijng online updates... so no need for a replay memory

    No replay memory
    Use onpolicy RL because no expperience replay

    For advantage function, you only need a value fn estimator.
    adv = r_(t+1) + V(S_(t+1)) - V(S_t)

    Update the actor and critic networks based on mod fn from counter.
    No need to polyak average for naive implementation.

    take action a based on policy using Q estimate



    """

    def __init__(self, env, hidden_size=64, actor_lr=0.01, critic_lr=0.01):
        """
        What needs to be here:
        initialize the actor and critic networks.
        continuous action space.

        """
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.low = env.action_space.low
        self.high = env.action_space.high

        self.actor = ActorNetwork(self.input_dim, hidden_size, self.action_dim)
        self.actor = self.actor.to(self.actor.device)
        self.actor.apply(init_xav_weights)
        self.critic = CriticNetwork(self.input_dim, hidden_size)
        self.critic = self.critic.to(self.critic.device)
        self.critic.apply(init_xav_weights)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def choose_action(self, state):
        """
        Chooses an action given an observation.
        """
        self.actor.eval()
        obs = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        # get action from actor
        dist, _ = self.actor(obs)
        action = dist.sample()
        action = action.cpu().detach().numpy()

        return np.clip(action, self.low, self.high)

    def train(self):
        """
        Steps to take here:
        grab the entire trajectory first

        """

        rewards_arr = []
        log_likelihood_arr = []
        value_arr = []

        obs = self.env.reset()
        done = False

        self.actor.eval()
        self.critic.eval()

        while not done:

            action = self.choose_action(obs)
            obs_n, r, done, info = self.env.step(action)

            # store reward
            rewards_arr.append(r)

            # store log likelihood
            _, log_likelihood = self.actor.forward(torch.tensor(obs, dtype=torch.float).to(self.actor.device),
                                                   torch.tensor(action, dtype=torch.float32).to(self.actor.device))
            log_likelihood_arr.append(log_likelihood)

            # value estimate at this current state
            value_est = self.critic(torch.tensor(obs, dtype=torch.float).to(self.critic.device))
            value_arr.append(value_est)

            # on to the next!
            obs = obs_n

        advantage_arr = np.zeros(len(value_arr))
        return_arr = np.zeros(len(value_arr))
        # for debugging
        q_arr = np.zeros(len(value_arr))

        # get advantages
        # TODO: change it so that you only get timesteps to T-1
        for t in reversed(range(len(value_arr))):
            if t == len(value_arr) - 1:
                return_arr[t] = rewards_arr[t]
            else:
                return_arr[t] = rewards_arr[t] + return_arr[t + 1]

            # adv = r_(t+1) + V(S_(t+1)) - V(S_t)
            if t == len(value_arr) - 1:
                adv = -value_arr[t]
            else:
                adv = return_arr[t + 1] + value_arr[t + 1] - value_arr[t]

            advantage_arr[t] = adv
            q_arr[t] = return_arr[t] + value_arr[t]

        self.actor.train()
        self.critic.train()

        self.actor_optimizer.zero_grad()
        actor_loss = (torch.tensor(log_likelihood_arr, dtype=torch.float32, requires_grad=True) *
                      torch.tensor(advantage_arr, dtype=torch.float32)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = torch.tensor(advantage_arr, dtype=torch.float32, requires_grad=True).pow(2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy(), rewards_arr, len(rewards_arr), q_arr, advantage_arr, value_arr


import gym

env = gym.make('InvertedPendulum-v2')
agent = A2CAgent(env)
#
# s = env.reset()
# # env.render()
# s_new, r, d, info = env.step(agent.choose_action(s))

epochs = int(1e6)
for i in range(epochs):
    poli_loss, pog, rew_arr, e_len, q_log_arr, adv_log_arr, val_log_arr = agent.train()
    # print(sum(r_arr))
    if i % 1000 == 0:
        print('epoch: %3d \t policy loss: %.3f \t value fn loss: %.3f \t return: %.3f \t avg_ep_len: %.3f' %
              (i, poli_loss, pog, sum(rew_arr), e_len))
        print('q arr:')
        print(q_log_arr)
        print('adv arr:')
        print(adv_log_arr)
        print('val arr:')
        print(val_log_arr)

