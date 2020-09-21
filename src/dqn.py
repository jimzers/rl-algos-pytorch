import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import gym


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
        self.action_mem = np.zeros(self.max_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.max_size, dtype=np.float32)
        self.done_mem = np.zeros(self.max_size, dtype=np.float32)

        # print("memory shapes:")
        # print(self.state_mem.shape)
        # print(self.next_state_mem.shape)
        # print(self.action_mem.shape)
        # print(self.reward_mem.shape)
        # print(self.done_mem.shape)

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

# global counter
#
# counter = 0

class CriticNetwork(nn.Module):
    """
    This is a value fn approximator.
    """

    def __init__(self, state_dim, hidden_size1, hidden_size2, action_dim, filename='dqn_critic'):
        super(CriticNetwork, self).__init__()
        # print('STATE DIMENIONS')
        # print(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # keep it simple just 3 fully connected layers
        self.fc3 = nn.Linear(hidden_size2, action_dim)
        self.filename = filename

    def forward(self, state):
        # global counter
        # print("forwarddd")
        # print(state)
        # counter += 1
        # print(counter)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)  # personal choice for no activation fn on last layer...

        return output

    def save_checkpoint(self):
        # print('----- saving checkpoint -------')
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        # print('-------- loading checkpoint --------')
        self.load_state_dict(torch.load(self.filename))


class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, lr=0.003, batch_size=64, n_epochs=500,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.env = env
        self.input_dim = env.observation_space.shape[0]

        # since discrete
        self.action_dim = env.action_space.n

        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.replay_buffer = ReplayBuffer(max_mem_size, [self.input_dim], [self.action_dim])

        self.value_fn = CriticNetwork(self.input_dim, 256, 256, self.action_dim).to(self.device)
        self.value_optim = optim.Adam(self.value_fn.parameters(), lr=self.lr)

        self.n_epochs = n_epochs

    def choose_action(self, obs):
        # e greedy
        if np.random.random() > self.epsilon:
            # print("experimental states")
            # print(torch.tensor(obs).to(self.device))
            # print(torch.tensor(obs).to(self.device).shape)
            # print(torch.tensor([obs]).to(self.device))
            # print(torch.tensor([obs]).to(self.device).shape)
            # print("========================================")

            with torch.no_grad():

                state = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                value_arr = self.value_fn(state)
                # see if we can get away with no .item()
                # print("No .item()")
                # print(torch.argmax(value_arr))
                # print("choose action shape")
                # print(value_arr.shape)
            action = torch.argmax(value_arr).item()
        else:
            action = self.env.action_space.sample()

        return action

    def compute_critic_loss(self, batch):
        state, action, reward, next_state, done = batch['state'], batch['action'].to(torch.long), batch['reward'], batch['next_state'], \
                                                  batch['done']

        critic_val = self.value_fn(state)

        # greedy
        # print('value of actual action')
        # print(action)
        # print(critic_val.gather(-1, action.unsqueeze(-1)))
        # print("critic val")
        # print(critic_val)
        # print(action)
        # print(critic_val[action])
        # print(critic_val.gather(-1, action.unsqueeze(-1)))

        critic_val = critic_val.gather(-1, action.unsqueeze(-1)).squeeze()
        # critic_val = torch.argmax(critic_val, dim=-1)

        # print("critic value estimate")
        # print(torch.argmax(critic_val, dim=-1))
        # print(critic_val.shape)

        # with torch.no_grad():

        next_critic_val = self.value_fn(next_state)

        # print("next critic val before argmax")
        # print(next_critic_val.shape)
        # print("regular critic value")
        # print(critic_val)
        # print("next_critic_val, argmax")
        # print(torch.argmax(next_critic_val, dim=-1))

        next_critic_val = torch.max(next_critic_val, dim=-1)

        # print("next_critic_val after max")
        # print(next_critic_val)
        # # torch.max on multidimensional arrays also outputs an "indices" tensor.
        # # we use "[0]" to only get the values tensor
        # print(next_critic_val[0])

            # print("next critic val after")
            # print(next_critic_val.shape)

        target_reward = reward + (self.gamma * (1 - done) * next_critic_val[0])

        # print('target_reward')
        # print(target_reward)

        # check done value
        # print("checking done value")
        # print((1 - done))

        # check for broadcasting errors
        # print("target reward shape:")
        # print(target_reward.shape)

        # print("target reward")
        # print(target_reward.shape)

        loss = torch.mean((critic_val - target_reward) ** 2)

        return loss

    def learn(self):
        if self.replay_buffer.curr_size < self.batch_size:
            return

        self.value_optim.zero_grad()
        sampled_batch = self.replay_buffer.sample(self.batch_size)

        critic_loss = self.compute_critic_loss(sampled_batch)

        critic_loss.backward()
        self.value_optim.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end \
            else self.eps_end

    def plot_curve(self, x, scores, epsilons):
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2")

        # this is the epsilon plot
        ax.plot(x, epsilons)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Epsilon")
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

        ax2.scatter(x, running_avg, color='C4')
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        ax2.set_ylabel('Score', color='C3')
        plt.show(fig)

    def training_loop(self):
        scores, eps_history = [], []

        for i in range(self.n_epochs):
            score = 0
            done = False
            state = self.env.reset()
            while not done:
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)
                score += reward
                self.replay_buffer.store(state, action, reward, new_state, done)
                self.learn()
                state = new_state
            scores.append(score)
            eps_history.append(self.epsilon)

            avg_score = np.mean(scores[-100:])

            print('episode ' + str(i) + " | " + 'score %.2f' % score +
                  " | " + 'average score %.2f' % avg_score +
                  " | " + 'epsilon %.2f' % self.epsilon)

        x = [i + 1 for i in range(self.n_epochs)]
        self.plot_curve(x, scores, eps_history)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    agent = DQNAgent(env)
    agent.training_loop()
