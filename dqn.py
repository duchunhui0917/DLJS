import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import copy
import random

INTERVAL = 10
UPDATE_FREQ = 100
BUFFER_SIZE = 2000
BATCH_SIZE = 32
N_STATES = 4
N_ACTIONS = 2


class Net(nn.Module):
    def __init__(self, state_dim, n_action):
        super(Net, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, n_action)

    def forward(self, state):
        x1 = torch.relu(self.l1(state))
        x2 = torch.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3


class ReplayBuffer:
    def __init__(self, state_dim, action_dim):
        self.state = np.zeros((BUFFER_SIZE, state_dim))
        self.action = np.zeros((BUFFER_SIZE, action_dim))
        self.reward = np.zeros((BUFFER_SIZE, 1))
        self.next_state = np.zeros((BUFFER_SIZE, state_dim))
        self.counter = 0
        self.is_full = False

    def store(self, transition):
        state, action, reward, next_state = transition
        self.state[self.counter] = state
        self.action[self.counter] = action
        self.reward[self.counter] = reward
        self.next_state[self.counter] = next_state
        self.counter = (self.counter + 1) % BUFFER_SIZE

        if not self.is_full and self.counter == BUFFER_SIZE - 1:
            self.is_full = True

    def sample(self):
        index = np.random.choice(BUFFER_SIZE, BATCH_SIZE)
        return (
            torch.from_numpy(self.state[index].astype(np.float32)),
            torch.from_numpy(self.action[index].astype(np.int)),
            torch.from_numpy(self.reward[index].astype(np.float32)),
            torch.from_numpy(self.next_state[index].astype(np.float32))
        )


class DQN:
    def __init__(self, state_dim, action_dim, n_action, lr=0.01, gamma=0.9, epsilon=0.8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_action = n_action
        self.q = Net(state_dim, n_action)
        self.target_q = Net(state_dim, n_action)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        self.lf = nn.MSELoss()
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_freq = 0
        self.ar = 0
        self.loss = 0

    def select_action(self, state):
        state = torch.from_numpy(state)
        if random.random() < self.epsilon:
            action = int(torch.max(self.q(state), 1)[1].numpy())
            return action
        else:
            return random.randint(0, self.n_action - 1)

    def update(self):
        if self.update_freq % UPDATE_FREQ == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        self.update_freq += 1

        state, action, reward, next_state = self.replay_buffer.sample()

        q = self.q(state).gather(1, action)
        target_q = self.target_q(next_state).max(1)[0].view(BATCH_SIZE, 1)
        target_q = reward + self.gamma * target_q.detach()

        loss = self.lf(q, target_q)
        self.loss = loss
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def run(self, env, i_episode):
        state = env.reset()
        transitions = []
        cr = 0
        while True:
            env.render()
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)

            # x, x_dot, theta, theta_dot = next_state
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # reward = r1 + r2

            transitions.append([state, action, reward, next_state])
            state = next_state
            cr += reward
            if self.replay_buffer.is_full:
                self.update()
            if done:
                for transition in transitions:
                    transition[2] = cr
                    self.replay_buffer.store(transition)
                self.ar += cr

                break

        if i_episode != 0 and i_episode % INTERVAL == 0:
            print('i_episode:%d average_reward:%.2f current_reward:%.2f current_loss:%.5f'
                  % (i_episode, self.ar / INTERVAL, cr, self.loss))
            self.ar = 0
