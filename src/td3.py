import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import gym

from utils.utils import init_xav_weights


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.memory_counter = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.done_mem = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.memory_counter % self.mem_size  # wrap around
        self.state_mem[idx] = state
        self.new_state_mem[idx] = new_state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.done_mem[idx] = done
        self.memory_counter += 1

    def sample(self, batch_size):
        real_batch_size = min(self.memory_counter, batch_size)  # just in case memory too small
        batch_indices = np.random.choice(len(self.state_mem), size=real_batch_size, replace=False)

        states = self.state_mem[batch_indices]
        actions = self.action_mem[batch_indices]
        rewards = self.reward_mem[batch_indices]
        new_states = self.new_state_mem[batch_indices]
        done_vals = self.done_mem[batch_indices]

        return states, actions, rewards, new_states, done_vals



class ActorNetwork(nn.Module):
    """
    Deterministic actor.
    Note: make sure to add batch normalization!
    """
    def __init__(self, obs_dim, hidden_size, action_dim, filename='td3_actor'):
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
        # TODO: IF THERE'S AN ISSUE WITH LOWER / UPPER LIMITS OF ACTION SPACE (E.G. ACTOR NOT GOING TO ANY EXTREME LIMITS),
        # THEN THE ACTOR NEEDS A DIFFERENT WAY TO BOUND (torch clamp? multiplication scale?)

        return res

    def save_checkpoint(self):
        # print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        # print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))


class CriticNetwork(nn.Module):
    def __init__(self, combined_dim, hidden_size, value_dim=1, filename='td3_critic'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size, value_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filename = filename

    def forward(self, state, action):
        # print("CRITIC NETWORK FORWARD")
        # print(state.shape)
        # print(action.shape)
        # print('bruh')
        # print(torch.cat((state, action), dim=-1).shape)
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


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # lets you call on the class
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * \
                self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class TD3Agent:
    """
    Attempt 1 of TD3 Agent.
    Important things to note:
    Double Q learning (use the smaller of two Q values to form targets)
    Delayed policy updates (1 policy update per 2 Q fn updates)
    Add noise to target action

    """
    def __init__(self, env, policy_update_delay=2, batch_size=64, hidden_size=64, tau=0.95, gamma=0.99, actor_lr=0.01, critic_lr=0.01):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # add clipping values for high and low
        self.low = env.action_space.low
        self.high = env.action_space.high

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.memory = ReplayBuffer(1000000, [self.input_dim], self.action_dim)


        self.actor = ActorNetwork(self.input_dim, hidden_size, self.action_dim, filename='td3_actor')
        self.actor = self.actor.to(self.actor.device)
        self.actor.apply(init_xav_weights)

        self.target_actor = ActorNetwork(self.input_dim, hidden_size, self.action_dim, filename='td3_target_actor')
        self.target_actor = self.target_actor.to(self.target_actor.device)
        self.target_actor.apply(init_xav_weights)

        self.critic = CriticNetwork(self.input_dim + self.action_dim, hidden_size, filename='td3_critic')
        self.critic = self.critic.to(self.critic.device)
        self.critic.apply(init_xav_weights)

        self.target_critic = CriticNetwork(self.input_dim + self.action_dim, hidden_size, filename='td3_target_critic')
        self.target_critic = self.target_critic.to(self.target_critic.device)
        self.target_critic.apply(init_xav_weights)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.target_actor_opt = Adam(self.target_actor.parameters(), lr=actor_lr)

        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        self.noise = OUActionNoise(mu=np.zeros(self.action_dim))

    def choose_action(self, state):
        self.actor.eval()
        observation = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        # get action from actor
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise()).to(self.actor.device)

        self.actor.train()
        mu_prime = mu_prime.cpu().detach().numpy()

        return np.clip(mu_prime, self.low, self.high)

    def store(self, state, action, rew, new_state, done):
        self.memory.store_transition(state, action, rew, new_state, done)

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            # print('skipped!')
            # print(self.memory.memory_counter)
            # print(self.batch_size)
            return  # don't learn if the replay buffer doesn't have enough memories

        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        # send off the stuff to the gpu
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

        # print("replay buffer stuff ------------------------------------------------")
        # print(state.shape)  # (bsize, obs_dims)
        # print(action.shape)  # (bsize, 1)
        # print(reward.shape)  # (bsize,)
        # print(new_state.shape)  # (bsize, obs_dims)
        # print(done.shape)  # (bsize,) wut????

        # set networks to eval mode
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # get target actions to use on target critic network
        target_actions = self.target_actor.forward(new_state)
        # TODO: probably have to squeeze these networks
        critic_val_new = self.target_critic.forward(new_state, target_actions)
        critic_val = self.critic.forward(state, action)

        # todo: see shape
        target_reward = reward + self.gamma * (1 - done) * critic_val_new.squeeze()
        # print('target_reward')
        # print(target_reward.shape)  # (bsize, bsize)
        # print((1 - done).shape)
        # print('bruh')
        # print(1 - done)

        self.critic.train()
        self.critic_opt.zero_grad()
        # print(((target_reward - critic_val.squeeze())**2).shape)
        critic_loss = torch.mean((target_reward - critic_val.squeeze())**2)
        critic_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        # print(self.critic.forward(state, mu).squeeze().shape)
        actor_loss = -torch.mean(self.critic.forward(state, mu).squeeze())
        actor_loss.backward()
        self.actor_opt.step()

        self.update_network_parameters()

        return actor_loss, critic_loss, reward, target_reward, critic_val, critic_val_new

    def train(self):
        done = False
        score = 0
        state = self.env.reset()
        learn_res = None

        while not done:
            action = self.choose_action(state)
            next_state, rew, done, info = self.env.step(action)
            self.store(state, action, rew, next_state, int(done))
            learn_res = self.learn()
            if learn_res != None:
                actor_loss, critic_loss, rew, target_reward, critic_val, critic_val_new = learn_res
            score += rew
            state = next_state
        if learn_res is not None:
            self.save_models()
            return actor_loss, critic_loss, rew, target_reward, critic_val, critic_val_new, score, self.tau
        else:
            print('one training step done')
            return


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


env = gym.make('InvertedPendulum-v2')
agent = TD3Agent(env)
#
# s = env.reset()
# # env.render()
# s_new, r, d, info = env.step(agent.choose_action(s))

epochs = int(1e6)
logging_interval = 100

actor_loss_arr = []
critic_loss_arr = []
reward_arr = []
target_reward_arr = []
critic_val_arr = []
critic_val_new_arr = []
eps_score_arr = []
tau_arr = []

for i in range(epochs):
    # print('epoch run')
    learn_res = agent.train()
    if learn_res is not None:
        actor_lossz, critic_lossz, rewardz, target_rewardz, critic_valz, critic_val_newz, eps_scorez, tau_valz = learn_res
        # print(sum(r_arr))
        actor_loss_arr.append(actor_lossz.cpu().detach().item())
        critic_loss_arr.append(critic_lossz.cpu().detach().item())
        reward_arr.append(rewardz.cpu().detach().numpy())
        target_reward_arr.append(target_rewardz.cpu().detach().numpy())
        critic_val_arr.append(critic_valz.cpu().detach().numpy())
        critic_val_new_arr.append(critic_val_newz.cpu().detach().numpy())
        eps_score_arr.append(sum(eps_scorez).cpu().detach())
        tau_arr.append(tau_valz)

        if i % logging_interval == 0:
            # print(np.average(actor_loss_arr))
            # print(np.average(critic_loss_arr))
            # print(type(eps_score_arr[0]))
            # print(np.average(eps_score_arr))
            print('epoch: %3d \t avg actor loss %.3f \t avg critic loss: %.3f \t avg reward: %.3f' %
                  (i, np.average(actor_loss_arr), np.average(critic_loss_arr), np.average(eps_score_arr)))
            # print('actor loss:')
            # print(actor_lossz)
            # print('critic loss:')
            # print(critic_lossz)
            # print('reward:')
            # print(rewardz)
            # print('target reward:')
            # print(target_rewardz)
            # print('critic value:')
            # print(critic_valz)
            # print('target critic value:')
            # print(critic_val_newz)
            # print('episode score:')
            # print(eps_scorez)
            # print('tau:')
            # print(tau_valz)

            actor_loss_arr = []
            critic_loss_arr = []
            reward_arr = []
            target_reward_arr = []
            critic_val_arr = []
            critic_val_new_arr = []
            eps_score_arr = []
            tau_arr = []
