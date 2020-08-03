from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



class ReplayBuffer:
    """
    please work god
    taken from spinning up implementation of replay buffer
    """
    
    def __init__(self, max_size, input_shape, action_shape):
        self.max_size = max_size
        self.memory_counter = 0
        self.curr_size = 0

        self.state_mem = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.next_state_mem = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.action_mem = np.zeros((self.max_size, *action_shape), dtype=np.float32)
        self.reward_mem = np.zeros(self.max_size, dtype=np.float32)
        self.done_mem = np.zeros(self.max_size, dtype=np.float32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def store(self, state, action, reward, next_state, done):
        self.state_mem[self.memory_counter] = state
        self.next_state_mem[self.memory_counter] = next_state
        self.action_mem[self.memory_counter] = action
        self.reward_mem[self.memory_counter] = reward
        self.done_mem[self.memory_counter] = done

        self.memory_counter = (self.memory_counter + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size):
        batch_indices = np.random.randint(0, self.curr_size, size=batch_size)
        ## use a dictionary this time for "organization"
        batch = {
            'state': self.state_mem[batch_indices],
            'action': self.action_mem[batch_indices],
            'next_state': self.next_state_mem[batch_indices],
            'reward': self.reward_mem[batch_indices],
            'done': self.done_mem[batch_indices]
        }
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}


class ActorNetwork(nn.Module):
    """
    Deterministic actor.
    Note: make sure to add batch normalization!
    """
    def __init__(self, obs_dim, hidden_size, action_dim, filename='ddpg_actor'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filename = filename

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.lnorm1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.lnorm2(x)
        x = self.fc3(x)  # no non-linearities / normalization after this
        res = torch.tanh(x)  # to ensure result is bound within -1, 1

        return res

    def save_checkpoint(self):
        # print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        # print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))


class CriticNetwork(nn.Module):
    def __init__(self, combined_dim, hidden_size, value_dim=1, filename='ddpg_critic'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, value_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filename = filename

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state, action), dim=-1)))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)  # personal choice for no activation fn on last layer...
        return output

    def save_checkpoint(self):
        # print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        # print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))


