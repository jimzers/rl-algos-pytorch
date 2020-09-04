from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

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
    Actor thingy. Gonna try this instead of a helper fn, might be easier to understand and structure
    This actor is diagonal gaussian
    """

    def __init__(self, obs_dim, hidden_size, action_dim, filename='sac_actor'):
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
        return output.squeeze()

    def save_checkpoint(self):
        # print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        # print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))


class SACAgent:
    """
    oh god nothing works any more
    """

    def __init__(self, env, alpha=0.2, starting_epoch=0, epochs=100, epoch_len=4000, replay_size=int(1e6), gamma=0.99, tau=0.80,
                 hidden_size=100, actor_lr=0.01, critic_lr=0.01, actor_noise=0.1, batch_size=100, start_steps=10000,
                 max_eps_len=1000, num_test_episodes=10, update_epoch=1000, update_freq=50, num_updates=50):

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

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.actor_noise = actor_noise

        self.replay_buffer = ReplayBuffer(replay_size, list(self.input_dim), [self.action_dim])

        self.actor = ActorNetwork(self.input_dim[0], hidden_size, self.action_dim, filename='sac_actor')
        self.actor = self.actor.to(self.actor.device)
        self.actor.apply(init_xav_weights)

        # with torch.no_grad():
        self.target_actor = ActorNetwork(self.input_dim[0], hidden_size, self.action_dim,
                                         filename='sac_target_actor')
        self.target_actor = self.target_actor.to(self.target_actor.device)
        self.target_actor.load_state_dict(self.actor.state_dict())


        self.critic1 = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size, filename='sac_critic1')
        self.critic1 = self.critic1.to(self.critic1.device)
        self.critic1.apply(init_xav_weights)

        self.target_critic1 = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size,
                                            filename='sac_target_critic1')
        self.target_critic1 = self.target_critic1.to(self.target_critic1.device)
        self.target_critic1.apply(init_xav_weights)

        self.critic2 = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size, filename='sac_critic2')
        self.critic2 = self.critic2.to(self.critic2.device)
        self.critic2.apply(init_xav_weights)

        self.target_critic2 = CriticNetwork(self.input_dim[0] + self.action_dim, hidden_size,
                                            filename='sac_target_critic2')
        self.target_critic2 = self.target_critic2.to(self.target_critic2.device)
        self.target_critic2.apply(init_xav_weights)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_opt = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

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
            dist, _ = self.actor(obs)
            action = dist.sample()
        noise = noise_scale * np.random.randn(self.action_dim)
        noisy_action = action.cpu().detach() + noise

        return np.clip(noisy_action, self.low, self.high)

    def compute_critic_loss(self, batch):
        """
        compute critic loss.
        """
        state, action, reward, next_state, done = batch['state'], batch['action'], batch['reward'], batch['next_state'], \
                                                  batch['done']

        # print("stuff")
        # print(next_state)
        # print(next_state.shape)
        critic_val1 = self.critic1(state, action).squeeze()
        critic_val2 = self.critic2(state, action).squeeze()

        #
        with torch.no_grad():
            target_action_distribution, _ = self.target_actor(next_state)
            target_action = target_action_distribution.sample()
            log_prob = self.target_actor.distribution(next_state).log_prob(target_action)
            target_critic_val1 = self.target_critic1(next_state, target_action)
            target_critic_val2 = self.target_critic2(next_state, target_action)

            target_critic_val = torch.min(target_critic_val1, target_critic_val2)

            target_reward = reward + self.gamma * (1 - done) * (target_critic_val - self.alpha * log_prob)


        # print("critic loss shape")
        # print(((critic_val - target_reward) ** 2).shape)
        # print("end critic loss shape")
        critic_loss1 = torch.mean((critic_val1 - target_reward) ** 2)
        critic_loss2 = torch.mean((critic_val2 - target_reward) ** 2)
        critic_loss = critic_loss1 + critic_loss2

        q_vals1 = critic_val1.cpu().detach().numpy()
        q_vals2 = critic_val2.cpu().detach().numpy()
        q_vals = [q_vals1, q_vals2]

        return critic_loss, q_vals

    def compute_actor_loss(self, batch):
        """
        compute actor loss.
        """
        state = batch['state']
        dist, _ = self.actor(state)
        mu = dist.sample()
        # print()
        log_prob = self.actor.distribution(state).log_prob(mu)

        q1 = self.critic1(state, mu).squeeze()
        q2 = self.critic2(state, mu).squeeze()

        min_critic = torch.min(q1, q2)

        actor_loss = -torch.mean(min_critic - (self.alpha * log_prob))

        return actor_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_((1.0 - tau) * param.data + tau * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
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
        self.critic1.eval()
        self.critic2.eval()

        # Actor training
        self.actor_opt.zero_grad()
        actor_loss = self.compute_actor_loss(batch)
        actor_loss.backward()
        self.actor_opt.step()

        # reenable critic grad tracking
        self.critic1.train()
        self.critic2.train()

        # consolidate changes to network
        # with torch.no_grad():
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
            while not (done or episode_len == self.max_eps_len):
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
            if (t + 1) % self.epoch_len == 0:  # t+1 helps not trigger this on first step.
                self.curr_epoch = (t + 1) // self.epoch_len

                self.test()  # this is causing actor instabillity???

                print("current epoch: " + str(self.curr_epoch))
                print("average critic loss: {0:.2f}".format(
                    np.mean(self.logging_dict['critic_loss'][-self.num_updates:])))
                print(
                    "average actor loss: {0:.2f}".format(np.mean(self.logging_dict['actor_loss'][-self.num_updates:])))
                print("average episode length: {0:.7f}".format(
                    np.mean(self.logging_dict['episode_len'][-self.num_test_episodes:])))
                print("average episode return: {0:.7f}".format(
                    np.mean(self.logging_dict['episode_return'][-self.num_test_episodes:])))
                print_q_vals = True
                if print_q_vals:
                    print(self.logging_dict['q_vals'][-self.num_updates:])

                # need to log ep return


import gym

env = gym.make('InvertedPendulum-v2')
# env = gym.make('Swimmer-v2')
agent = SACAgent(env)

from torchsummary import summary

# summary(agent.actor, [1], 100)
# summary()

agent.train_loop()