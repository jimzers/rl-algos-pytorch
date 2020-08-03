import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.distributions import Normal
from torch.optim import Adam

from utils.utils import init_xav_weights

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
        if action is not None:
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


    