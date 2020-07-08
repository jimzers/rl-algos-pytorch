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


class A2CAgentV2:
    """
    Revised A2C agent. Now refits the value fn as the baseline estimate. Also has discount factor

    notes:
    Get rewards to go...
    Get advntage estimates....

    Loss of the value function will be based on the rewards to go, rather than the advantage.
    """

    def __init__(self, env, hidden_size=64, gamma=0.99, actor_lr=0.01, critic_lr=0.01):
        """
        init actor and critic networks
        init weights of actor and critic networks
        continuous action space

        """

        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.low = env.action_space.low
        self.high = env.action_space.high

        self.gamma = gamma

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
        Choose an action given observation
        """
        # turn off gradient tracking
        self.actor.eval()
        obs = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        # get action from actor
        dist, _ = self.actor(obs)
        action = dist.sample()
        action = action.cpu().detach().numpy()

        # reactivate gradient tracking
        self.actor.train()

        return np.clip(action, self.low, self.high)

    def train(self):
        """
        Steps to take:

        Still need rewards array, log likelihoods, values

        Refit the baseline fn (value fn) on the returns, not the advantage. Stronger learning signal!

        PLEASE STORE STUFF IN LISTS! IF YOU STORE IN NUMPY ARRAYS THE GRADIENTS WILL GET WIPED OUT

        """

        rewards_arr = []
        log_prob_arr = []
        value_arr = []

        obs = self.env.reset()
        done = False

        while not done:

            action = self.choose_action(obs)
            obs_n, r, done, info = self.env.step(action)

            # store reward
            rewards_arr.append(r)

            # store log prob
            _, log_prob = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.actor.device),
                                                   torch.tensor(action, dtype=torch.float32).to(self.actor.device))

            log_prob_arr.append(log_prob)

            # value estimate at current state
            value_est = self.critic(torch.tensor(obs, dtype=torch.float32).to(self.critic.device))

            value_arr.append(value_est)

            # the cycle repeats
            obs = obs_n

        # advantage array
        adv_arr = []
        # rewards to go
        return_arr = []

        total_return = 0

        # for debugging
        q_arr = []

        # check shapes of rewards and value arrays for broadcasting errors
        # print('REWARDS ARRAY Single ELem')
        # print(rewards_arr[0])
        # print('VALUE ARRAY SHAPE')
        # print(value_arr[0].shape)
        # get rewards to go and advantages

        # TODO: deal with the problem of the last episode having terrible performance
        for t in reversed(range(len(value_arr))):

            # todo: add gamma here if you want. make sure gamma is based on num of timesteps surpassed
            # print('RETURN VALUE')
            # print(rewards_arr[t])
            # print('BRUH')
            total_return = rewards_arr[t] + self.gamma * total_return
            return_arr.insert(0, total_return)

            if t == len(value_arr) - 1:
                adv = -value_arr[t]

                q_arr.insert(0, torch.zeros_like(value_arr[0]))
            else:
                # is it total_return or rewards_arr[t] ??? it
                adv = rewards_arr[t] + (self.gamma**(t+1)) * value_arr[t + 1] - (self.gamma ** t) * value_arr[t]

                q_arr.insert(0, rewards_arr[t] + value_arr[t + 1])

            adv_arr.insert(0, adv)

        self.actor_optimizer.zero_grad()
        # print('DEBUG MORE STUFGF')
        # print(torch.stack(log_prob_arr).shape)
        # print(torch.stack(adv_arr).squeeze().detach().shape)
        # print('POGCHMPION STUFF')
        # print(torch.stack(log_prob_arr))
        # print((torch.stack(log_prob_arr) * torch.stack(adv_arr).squeeze().detach()).shape)
        actor_loss = -(torch.stack(log_prob_arr) * torch.stack(adv_arr).squeeze().detach()).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # TODO: print shapes of rewards to go and value array
        # print('VALUE FN STACKED')
        # print(torch.stack(value_arr).shape)
        # print('REWARDS TO GO')
        # print(torch.tensor(return_arr).shape)

        #TODO : see shapes here again. the critic loss is struggling here
        self.critic_optimizer.zero_grad()
        # print('POGCHAMP')
        # print((torch.stack(value_arr).squeeze() - torch.tensor(return_arr).to(self.critic.device)).shape)
        # print(((torch.stack(value_arr).squeeze() - torch.tensor(return_arr).to(self.critic.device)) ** 2).shape)
        critic_loss = ((torch.stack(value_arr).squeeze() - torch.tensor(return_arr).to(self.critic.device)) ** 2).mean()
        # lol same thing
        # critic_loss = ((torch.tensor(return_arr).to(self.critic.device) - torch.stack(value_arr).squeeze()) ** 2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy(), rewards_arr, len(rewards_arr), q_arr, adv_arr, value_arr

import gym

env = gym.make('InvertedPendulum-v2')
agent = A2CAgentV2(env)
#
# s = env.reset()
# # env.render()
# s_new, r, d, info = env.step(agent.choose_action(s))

epochs = int(1e6)
for i in range(epochs):
    poli_loss, pog, rew_arr, e_len, q_log_arr, adv_log_arr, val_log_arr = agent.train()
    # print(sum(r_arr))
    if i % 100 == 0:
        print('epoch: %3d \t policy loss: %.3f \t value fn loss: %.3f \t return: %.3f \t avg_ep_len: %.3f' %
              (i, poli_loss, pog, sum(rew_arr), e_len))
        # print('q arr:')
        # print(q_log_arr)
        # print('adv arr:')
        # print(adv_log_arr)
        # print('val arr:')
        # print(val_log_arr)