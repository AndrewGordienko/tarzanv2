import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym

env = gym.make('BipedalWalker-v3', render_mode="human")
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001

class actor_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.shape[0]

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_space))

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

        self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))

        # Clamp the raw log std coming from the network before exponentiating
        clamped_log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(clamped_log_std).expand_as(mu)
        
        # print("Actor output:", mu, std)
        return Normal(mu, std)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



class critic_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

        self._initialize_weights()
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

