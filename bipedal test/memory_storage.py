import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gymnasium as gym
import numpy as np

env = gym.make('BipedalWalker-v3', render_mode="human")

DEVICE = torch.device("cpu")
MEM_SIZE = 1000000
BATCH_SIZE = 64

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0

        # Determine the action dimension from the action space
        action_dim = env.action_space.shape[0]

        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((MEM_SIZE, action_dim), dtype=np.float32)  # Updated to handle multi-dimensional action
        self.log_probs = np.zeros((MEM_SIZE, action_dim), dtype=np.float32)  # Updated to handle multi-dimensional log_prob
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool_)

    def add(self, state, action, log_prob, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE

        self.states[mem_index] = state
        self.actions[mem_index] = action  # Will store multi-dimensional action
        self.log_probs[mem_index] = log_prob  # Will store multi-dimensional log_prob
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=False)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        log_probs = self.log_probs[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        return states, actions, log_probs, rewards, states_, dones
