import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

import gym

import numpy as np
from utils.utils import make_mlp


class PolicyGradient:
    """
    Try again with a clean slate. This time only discrete.
    """

    def __init__(self, env, episode_limit=5000,
                 policy_layers=(32,), policy_lr=0.01, gamma=1):
        """
        Constructor for Policy Gradient Class.
        """
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.episode_limit = episode_limit  # number of timesteps to collect per step
        self.gamma = gamma

        layer_arr = [self.input_dim] + list(policy_layers) + [self.action_dim]
        self.policy_network = make_mlp(layer_arr, nn.Tanh, nn.Softmax)
        self.policy_lr = policy_lr

        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=policy_lr)

    def policy_dist(self, obs):
        """
        Get the policy's distribution given a state
        """
        logits = self.policy_network(obs)
        # VERY IMPORTANT: MAKE SURE YOU ADD THE LOGITS KEYWORD... OTHERWISE IT DEFAULTS TO 'PROBS='
        dist = Categorical(logits=logits)
        return dist

    def act(self, obs):
        """
        Get the action from a policy given a state
        """
        # print(obs)
        # print(type(obs))
        dist = self.policy_dist(torch.as_tensor(obs, dtype=torch.float32))
        action = dist.sample()
        return action.item()

    def get_likelihood(self, obs, action):
        log_li = self.policy_dist(obs).log_prob(action)
        return log_li

    def train(self):
        """
        Train a single epoch of the policy gradient.

        TODO:
        way to store anything needed in a single episode

        loop episodes until enough to fill the batch is complete
        make sure to grab weight about each episode
        since you're doing GAE, have to loop backwards in order to capture the rewards for each timestep
        remember that the objective function is log pi(a|s) * A

        make a step with optimizer (zero the grads, take a step... whatever ya gotta do)
        """
        state_arr = []
        action_arr = []
        done_arr = []
        reward_arr = []

        eps_len = []

        return_arr = []

        s = self.env.reset()
        d = False
        start_timestep = 0
        timestep = 0

        while not d:
            # render this env
            # self.env.render()

            # get action, take action
            a = self.act(s)
            s_n, r, d, info = self.env.step(a)

            state_arr.append(s)
            action_arr.append(a)
            reward_arr.append(r)
            done_arr.append(d)
            timestep += 1
            s = s_n

            if d:

                rewards_to_go = np.arange(timestep - start_timestep, dtype=np.float32)
                for t in reversed(range(start_timestep, timestep)):
                    if t == timestep - 1:
                        # print(type(reward_arr[t]))
                        # print(reward_arr[t])
                        rewards_to_go[t - start_timestep] = reward_arr[t]  # + 0
                    else:
                        # print('DONE ARRAY')
                        # print((1 - done_arr[t]))
                        # print((1 - done_arr[t]) * self.gamma * \
                        #                                     rewards_to_go[t + 1 - start_timestep])
                        rewards_to_go[t - start_timestep] = reward_arr[t] + (1 - done_arr[t]) * self.gamma * \
                                                            rewards_to_go[t + 1 - start_timestep]
                        # print(rewards_to_go[t - start_timestep])

                    # weights_arr[t] = rewards_to_go[t] * self.get_likelihood(state_arr[t], action_arr[t])

                return_arr += list(rewards_to_go)

                eps_len.append(timestep - start_timestep)

                if len(action_arr) > self.episode_limit:
                    break

                s = self.env.reset()
                d = False
                start_timestep = timestep

        # rewards_to_go = np.zeros_like(action_arr)
        # for t in reversed(range(len(action_arr))):
        #     if t == len(action_arr) - 1:
        #         rewards_to_go[t] = reward_arr[t]  # + 0
        #     else:
        #         # (1 - done_arr[t]) *  not needed bc runs only one episode
        #         rewards_to_go[t] = reward_arr[t] + self.gamma * rewards_to_go[t + 1]

        # print(torch.as_tensor(state_arr, dtype=torch.float32))
        # print(action_arr)
        # print(type(action_arr))
        # print(torch.as_tensor(action_arr, dtype=torch.int32).shape)

        # print('REWARDS TO GO')
        # print(torch.as_tensor(rewards_to_go, dtype=torch.float32))
        # print(torch.as_tensor(rewards_to_go, dtype=torch.float32).shape)
        # print('LIKELIHOODS')

        cost = torch.as_tensor(return_arr, dtype=torch.float32) * \
               self.get_likelihood(
                   torch.as_tensor(state_arr, dtype=torch.float32),
                   torch.as_tensor(action_arr, dtype=torch.int32)
               )

        self.policy_optimizer.zero_grad()
        policy_loss = -cost.mean()

        policy_loss.backward()
        self.policy_optimizer.step()
        # print(return_arr)
        # print('POLICY LOSS')
        # print(policy_loss)
        # print(state_arr)

        return policy_loss.detach().numpy().squeeze(), return_arr, eps_len


# Run the Policy Gradient here for now
env = gym.make('CartPole-v1')
pg = PolicyGradient(env, 5000)

epochs = int(1e6)
for i in range(epochs):
    poli_loss, r_arr, e_len = pg.train()
    # print(sum(r_arr))
    print('epoch: %3d \t policy loss: %.3f \t return: %.3f \t avg_ep_len: %.3f' %
          (i, poli_loss, sum(r_arr), sum(e_len) / len(e_len)))

env.close()
