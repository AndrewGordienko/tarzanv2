import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy 
from torch.distributions.categorical import Categorical

#env = gym.make('CartPole-v0')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')
# env = gym.make('LunarLander-v2', render_mode="human")
# env = gym.make('LunarLanderContinuous-v2', render_mode="human")
env = gym.make('BipedalWalker-v3', render_mode="human")

EPISODES = 501
MEM_SIZE = 1000000
BATCH_SIZE = 5
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
LEARNING_RATE = 0.0003
FC1_DIMS = 1024
FC2_DIMS = 512
ENTROPY_BETA = 0.02  # This is the hyperparameter that controls the strength of the entropy regularization. You might need to tune it.
DEVICE = torch.device("cpu")

best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []

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

class MLP_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = 10
        self.fc1 = nn.Linear(self.input_shape, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95

        self.actor = actor_network()
        self.critic = critic_network()
        self.encoder = MLP_encoder()
        self.memory = PPOMemory(BATCH_SIZE)

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


    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage).to(DEVICE)
            values = torch.tensor(values).to(DEVICE)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                dist = self.actor(states)  # Get the policy distribution
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probabilities = dist.log_prob(actions).sum(axis=-1)

                probability_ratio = new_probabilities.exp() / old_probs.exp()

                weighted_probabilities = advantage[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_probabilities, weighted_clipped_probabilities).mean() - ENTROPY_BETA * entropy

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()    

agent = Agent(n_actions=env.action_space, input_dims=env.observation_space.shape)
torch.save(agent.encoder.state_dict(), 'agent_encoder.pth')
max_steps = 0

for i in range(1, EPISODES):
    observation, info = env.reset()
    score = 0
    done = False
    step = 0

    while not done:
        env.render()
        step += 1

        observation = np.concatenate((
            observation[:len(observation)-10], 
            agent.encoder(torch.Tensor(observation[len(observation)-10:])).detach().numpy()
        ))

        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        
        if step % 20 == 0:
            agent.learn()
        

        observation = observation_

        
        if done:
            torch.save(agent.actor.state_dict(), 'agent_actor.pth')
            
            if score > best_reward:
                best_reward = score
                # torch.save(agent.actor.state_dict(), 'agent_actor.pth')
            
            """
            if step > max_steps:
                torch.save(agent.actor.state_dict(), 'agent_actor.pth')
                max_steps = step
            """

            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(i, average_reward/i, best_reward, score))
            break
            
        episode_number.append(i)
        average_reward_number.append(average_reward/i)
    
    """
    agent.learn()
    agent.memory = PPOMemory(BATCH_SIZE)
    """

plt.plot(episode_number, average_reward_number)
plt.show()