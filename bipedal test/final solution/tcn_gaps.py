import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils import weight_norm
import gym
import math
import numpy as np

env = gym.make('BipedalWalker-v3', render_mode="human")

steps_back = 5
LEARNING_RATE = 0.01
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu2 = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu2(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.fc(y1[:, :, -1])
        return o

class actor_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.shape
        self.std = 0.5

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, *self.action_space)

        self.log_std = nn.Parameter(torch.ones(1, *self.action_space) * self.std)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


    def net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Use tanh for the last layer

        return x
    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist

    
class critic_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95

        self.actor = actor_network()
        self.critic = critic_network()

        self.actor.load_state_dict(torch.load('agent_actor.pth'))

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)
        
        # directly assign the Normal distribution object to dist
        dist = self.actor(state)
        
        mu = dist.mean
        sigma = dist.stddev
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic.forward(state)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.item()
        value = torch.squeeze(value).item()

        return action, log_prob, value

agent = Agent(n_actions=env.action_space, input_dims=env.observation_space.shape)
model = TCN(input_size=steps_back, output_size=10, num_channels=[25, 25], kernel_size=2, dropout=0.2)
# 14 numbers given from environment, we want to figure out 10 lidar measurements

EPISODES = 500

tcn_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

average_loss = 0
for i in range(EPISODES):
    observation, info = env.reset()
    score = 0
    done = False
    step = 0
    data_history = []
    index = 20

    if i % index == 0:
        print(average_loss/index)

    while not done:
        env.render()
        data_history.append(observation[:len(observation)-10])

        if step >= steps_back:
            data = torch.Tensor(data_history[len(data_history)-steps_back:]).unsqueeze(0)
            lidar_points = model.forward(data).detach().numpy().squeeze(0)

            
            reconstructed_observation = np.concatenate((
                observation[:len(observation)-10],
                lidar_points
            ))

            action_true, _, _ = agent.choose_action(reconstructed_observation)
            action_reconstructed, _, _ = agent.choose_action(reconstructed_observation)

            loss = F.mse_loss(torch.Tensor(action_true), torch.Tensor(action_reconstructed))
            loss.requires_grad = True

            tcn_optimizer.zero_grad()
            loss.backward()
            tcn_optimizer.step()

            average_loss += loss

            observation = reconstructed_observation



        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        step += 1
        observation = observation_




    
    
    



example_data = torch.Tensor(1, steps_back, 6) # batch size, time steps back, # data points

# print(example_data)
# print("--")
# print(model.forward(example_data))
