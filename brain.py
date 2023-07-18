import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import numpy as np

LEARNING_RATE = 0.0001
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
MEM_SIZE = 10000
BATCH_SIZE = 64
INPUT_SHAPE = 17 * 2
ACTION_SPACE = 4

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_SHAPE, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, ACTION_SPACE)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self):
        self.network = Network()
        self.memory = ReplayBuffer()


    def choose_action(self, observation, direction):
        input_tensor = list(observation)
        input_tensor += [direction] * 17
        input_tensor = torch.Tensor(input_tensor)
        output = self.network.forward(input_tensor)

        return output
    
class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, INPUT_SHAPE),dtype=np.float32)
        self.actions = np.zeros((MEM_SIZE, ACTION_SPACE),dtype=np.float32)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, INPUT_SHAPE),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones


    
