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



