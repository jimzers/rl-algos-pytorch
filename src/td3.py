import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.memory_counter = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.memory_counter % self.mem_size  # wrap around
        self.state_mem[idx] = state
        self.new_state_mem[idx] = new_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = 1 - done

    def sample(self, batch_size):
        real_batch_size = min(self.memory_counter, batch_size)  # just in case memory too small
        batch_indices = np.random.choice(len(self.state_mem), size=real_batch_size, replace=False)

        states = self.state_mem[batch_indices]
        actions = self.action_mem[batch_indices]
        rewards = self.reward_mem[batch_indices]
        new_states = self.new_state_mem[batch_indices]
        done_vals = self.terminal_mem[batch_indices]

        return states, actions, rewards, new_states, done_vals



class ActorNetwork(nn.Module):
    """
    Deterministic actor.
    Note: make sure to add batch normalization!
    """
    def __init__(self, obs_dim, hidden_size, action_dim, act_noise):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.bnorm1 = nn.BatchNorm1d(hidden_size)
        self.bnorm2 = nn.BatchNorm1d(hidden_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.bnorm1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bnorm2(x)
        x = self.fc3(x)  # no non-linearities / normalization after this
        res = torch.tanh(x)  # to ensure result is bound within -1, 1
        # TODO: IF THERE'S AN ISSUE WITH LOWER / UPPER LIMITS OF ACTION SPACE (E.G. ACTOR NOT GOING TO ANY EXTREME LIMITS),
        # THEN THE ACTOR NEEDS A DIFFERENT WAY TO BOUND (torch clamp? multiplication scale?)

        return res


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
