import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.distributions import Normal

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

    def __init__(self, env, hidden_size=64):
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

        """
        ...

import gym
env = gym.make('InvertedPendulum-v2')
agent = A2CAgent(env)

s = env.reset()
# env.render()
s_new, r, d, info = env.step(agent.choose_action(s))