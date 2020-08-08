from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.utils import init_xav_weights


class ReplayBuffer:
    """
    please work god
    taken from spinning up implementation of replay buffer
    """
    
    def __init__(self, max_size, state_shape, action_shape):
        self.max_size = max_size
        self.memory_counter = 0
        self.curr_size = 0

        self.state_mem = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.next_state_mem = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.action_mem = np.zeros((self.max_size, *action_shape), dtype=np.float32)
        self.reward_mem = np.zeros(self.max_size, dtype=np.float32)
        self.done_mem = np.zeros(self.max_size, dtype=np.float32)

        print("memory shapes:")
        print(self.state_mem.shape)
        print(self.next_state_mem.shape)
        print(self.action_mem.shape)
        print(self.reward_mem.shape)
        print(self.done_mem.shape)

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
        # print("batch sample shapes")
        # print("memory shapes:")
        # print(self.state_mem[batch_indices].shape)
        # print(self.next_state_mem[batch_indices].shape)
        # print(self.action_mem[batch_indices].shape)
        # print(self.reward_mem[batch_indices].shape)
        # print(self.done_mem[batch_indices].shape)
        # print("batch samples:")
        # print(batch_indices)
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
        # print("getting states and actions")
        # print(state.shape)
        # print(action.shape)
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


class DDPGAgent:
    """
    Attempt 2 of DDPG. Pleeeeease work?


    """

    def __init__(self, env, starting_epoch=0, epochs=100, epoch_len=4000, replay_size=int(1e6), gamma=0.99, tau=0.80, hidden_size=100, actor_lr=0.01, critic_lr=0.01, actor_noise=0.1, batch_size=100, start_steps=10000, max_eps_len=1000, num_test_episodes=10, update_epoch=1000, update_freq=50, num_updates=50):

        self.env = env
        self.input_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape[0]
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

        self.max_eps_len = max_eps_len
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.epochs = epochs
        self.epoch_len = epoch_len
        self.curr_epoch = starting_epoch
        self.update_epoch = update_epoch
        self.update_freq = update_freq
        self.num_updates = num_updates

        self.num_test_episodes = num_test_episodes

        self.gamma = gamma
        self.tau = tau
        self.actor_noise = actor_noise

        self.replay_buffer = ReplayBuffer(replay_size, list(self.input_dim), [self.action_dim])

        self.actor = ActorNetwork(self.input_dim[0], hidden_size, self.action_dim, filename='ddpg_actor')
        self.actor = self.actor.to(self.actor.device)
        self.actor.apply(init_xav_weights)

        with torch.no_grad():
            self.target_actor = ActorNetwork(self.input_dim[0], hidden_size, self.action_dim, filename='ddpg_target_actor')
            self.target_actor = self.target_actor.to(self.target_actor.device)
            self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size, filename='ddpg_critic')
        self.critic = self.critic.to(self.critic.device)
        self.critic.apply(init_xav_weights)

        with torch.no_grad():
            self.target_critic = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size, filename='ddpg_target_critic')
            self.target_critic = self.target_critic.to(self.target_critic.device)
            self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        self.logging_dict = {
            'critic_loss': [],
            'actor_loss': [],
            'q_vals': [],
            'episode_len': [],
            'episode_return': []
        }

    def choose_action(self, state, noise_scale):
        """
        normal noise.
        """
        with torch.no_grad():
            obs = torch.as_tensor(state, dtype=torch.float32).to(self.actor.device)
            action = self.actor(obs)
        noise = noise_scale * np.random.randn(self.action_dim)
        noisy_action = action.cpu().detach() + noise

        return np.clip(noisy_action, self.low, self.high)

    def compute_critic_loss(self, batch):
        """
        compute critic loss.
        """
        state, action, reward, next_state, done = batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done']

        # print("stuff")
        # print(next_state)
        # print(next_state.shape)
        critic_val = self.critic(state, action).squeeze()

        with torch.no_grad():
            target_action = self.target_actor(next_state)
            target_critic_val = self.target_critic(next_state, target_action)
            target_reward = reward + self.gamma * (1 - done) * target_critic_val.squeeze()

        # print("critic loss shape")
        # print(((critic_val - target_reward) ** 2).shape)
        # print("end critic loss shape")
        critic_loss = torch.mean((critic_val - target_reward) ** 2)

        q_vals = critic_val.cpu().detach().numpy()

        return critic_loss, q_vals

    def compute_actor_loss(self, batch):
        """
        compute actor loss.
        """
        state = batch['state']
        mu = self.actor(state)
        actor_loss = -torch.mean(self.critic(state, mu).squeeze())

        return actor_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

    def learn(self, batch):
        """
        single learning step :)
        """
        # Critic training
        self.critic_opt.zero_grad()
        critic_loss, q_vals = self.compute_critic_loss(batch)
        critic_loss.backward()
        self.critic_opt.step()

        # turn off critic grad tracking
        self.critic.eval()

        # Actor training
        self.actor_opt.zero_grad()
        actor_loss = self.compute_actor_loss(batch)
        actor_loss.backward()
        self.actor_opt.step()

        # reenable critic grad tracking
        self.critic.train()

        # consolidate changes to network
        with torch.no_grad():
            self.update_network_parameters()

        # add stuff to logging output
        self.logging_dict['critic_loss'].append(critic_loss.cpu().detach().numpy())
        self.logging_dict['actor_loss'].append(actor_loss.cpu().detach().numpy())
        self.logging_dict['q_vals'].append(q_vals)


    def test(self):
        """
        new idea taken from spinning up implementation: test the deterministic policy!
        """
        for t in range(self.num_test_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            episode_len = 0
            while not(done or episode_len == self.max_eps_len):
                action = self.choose_action(state, 0)
                next_state, reward, done, info = self.env.step(action)
                episode_return += reward
                episode_len += 1

            self.logging_dict['episode_len'].append(episode_len)
            self.logging_dict['episode_return'].append(episode_return)


    # def log_stuff(self):
    #     print("logging output:")
    #
    #     print("logging finished")

    def train_loop(self):
        """
        sample random stuff until you get enough random stuff

        and then do one step of the learning per one episode rollout
        """

        state = self.env.reset()
        episode_return = 0
        episode_len = 0

        starting_timestep = self.curr_epoch * self.epoch_len

        for t in range(starting_timestep, self.epochs * self.epoch_len):
            if self.replay_buffer.curr_size < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.choose_action(state, self.actor_noise)

            next_state, reward, done, info = self.env.step(action)
            episode_return += reward
            episode_len += 1

            # if your env hits a done state because the env hit the step limit, that's not really a done state
            if episode_len == self.max_eps_len:
                done = False

            self.replay_buffer.store(state, action, reward, next_state, done)

            # advance state
            state = next_state

            # reset the episode if done or hit time limit
            if done or (episode_len == self.max_eps_len):
                state = self.env.reset()
                episode_return = 0
                episode_len = 0

            if t > self.update_epoch and t % self.update_freq == 0:
                # update loop
                for i in range(self.num_updates):
                    batch_data = self.replay_buffer.sample(self.batch_size)
                    self.learn(batch_data)

            # end of epoch logic
            if (t+1) % self.epoch_len == 0:  # t+1 helps not trigger this on first step.
                self.curr_epoch = (t+1) // self.epoch_len

                self.test()  # this is causing actor instabillity???

                print("current epoch: " + str(self.curr_epoch))
                print("average critic loss: {0:.2f}".format(np.mean(self.logging_dict['critic_loss'][-self.num_updates:])))
                print("average actor loss: {0:.2f}".format(np.mean(self.logging_dict['actor_loss'][-self.num_updates:])))
                print("average episode length: {0:.7f}".format(np.mean(self.logging_dict['episode_len'][-self.num_test_episodes:])))
                print("average episode return: {0:.7f}".format(np.mean(self.logging_dict['episode_return'][-self.num_test_episodes:])))
                print_q_vals = True
                if print_q_vals:
                    print(self.logging_dict['q_vals'][-self.num_updates:])

                # need to log ep return


import gym
env = gym.make('InvertedPendulum-v2')
# env = gym.make('Swimmer-v2')
agent = DDPGAgent(env)

from torchsummary import summary

# summary(agent.actor, [1], 100)
# summary()

agent.train_loop()