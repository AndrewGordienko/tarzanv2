import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control import viewer

# env = gym.make("CartPole-v0", render_mode="human")
#env = gym.make('MountainCarContinuous-v0', render_mode="human")
# env = gym.make('LunarLanderContinuous-v2', render_mode="human")

# Environment Wrapper
class DMControlWrapper:
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.action_spec = self.env.action_spec()
        obs_spec = self.env.observation_spec()
        self.observation_space = sum([int(np.prod(obs_spec[key].shape)) for key in obs_spec])
        self.action_space = self.action_spec.shape[0]

    def step(self, action):
        ts = self.env.step(action)
        reward = ts.reward
        observation = np.concatenate([ts.observation[key].astype(np.float32).flatten() for key in ts.observation])
        return observation, reward, ts.last(), {}

    def reset(self):
        ts = self.env.reset()
        observation = np.concatenate([ts.observation[key].astype(np.float32).flatten() for key in ts.observation])
        return observation, {}

    def get_physics(self):
        return self.env.physics

env = DMControlWrapper(domain_name="humanoid", task_name="walk")

# env.action_space.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 64
EPISODES = 5000
FC1_DIMS = 512
FC2_DIMS = 256
LEARNING_RATE = 0.0001
DEVICE = torch.device("cpu")

# n_actions = env.action_space.n
# input_dims = env.observation_space.shape

input_dims = env.observation_space
n_actions = env.action_space

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.mu = nn.Linear(FC2_DIMS, n_actions)
        self.sigma = nn.Linear(FC2_DIMS, n_actions)

        self.optimizer = optim.Adam(self.parameters(), LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.99)
        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-3

        dist = torch.distributions.Normal(mu, sigma)
        return dist
    
class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.99)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = BATCH_SIZE

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
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.memory = PPOMemory()
        
        self.gamma = 0.95
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)
        dist = self.actor(state)
        value = self.critic(state)
        
        action = dist.sample()
        # Clamp each component of the action to the respective action space limits
        min_action = torch.tensor(env.action_spec.minimum, dtype=torch.float).to(DEVICE)
        max_action = torch.tensor(env.action_spec.maximum, dtype=torch.float).to(DEVICE)
        action = torch.clamp(action, min_action, max_action)
        
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum the log probabilities for multi-dimensional actions
        return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def learn(self):
        torch.autograd.set_detect_anomaly(True)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            # Convert these arrays to tensors
            state_arr = torch.tensor(state_arr, dtype=torch.float).to(DEVICE)
            action_arr = torch.tensor(action_arr).to(DEVICE)
            old_prob_arr = torch.tensor(old_prob_arr).to(DEVICE)
            reward_arr = torch.tensor(reward_arr).to(DEVICE)
            dones_arr = torch.tensor(dones_arr, dtype=torch.float).to(DEVICE)
            vals_arr = torch.tensor(vals_arr).to(DEVICE)

            # Precompute next value for advantage calculation
            next_value = 0
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            gae = 0

            for t in reversed(range(len(reward_arr))):
                if t == len(reward_arr) - 1:
                    delta = reward_arr[t] - vals_arr[t]
                    gae = delta
                else:
                    delta = reward_arr[t] + self.gamma * vals_arr[t+1] * (1 - dones_arr[t]) - vals_arr[t]
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones_arr[t]) * gae

                advantages[t] = gae

            advantages = torch.tensor(advantages).to(DEVICE)

            # Normalize the advantages
            std_adv = torch.std(advantages)
            if std_adv.item() > 1e-5:  # To prevent division by near-zero number
                advantages = (advantages - advantages.mean()) / std_adv
            else:
                advantages -= advantages.mean()

            # Compute the returns
            returns = advantages + vals_arr

            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                # Clipping value function updates
                value_preds_clipped = vals_arr[batch] + (critic_value - vals_arr[batch]).clamp(-self.policy_clip, self.policy_clip)
                value_loss = (returns[batch] - critic_value).pow(2)
                value_loss_clipped = (returns[batch] - value_preds_clipped).pow(2)

                critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()

                new_probabilities = dist.log_prob(actions).sum(dim=-1)
                #print("--")
                #print(new_probabilities.exp(), old_probs.exp())
                # probability_ratio = new_probabilities.exp() / old_probs.exp()
                probability_ratio = torch.exp(new_probabilities - old_probs)

                weighted_probabilities = advantages[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]

                actor_loss = -torch.mean(torch.min(weighted_probabilities, weighted_clipped_probabilities))
                #critic_loss = (returns[batch] - critic_value) ** 2
                #critic_loss = critic_loss.mean()

                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1)

                self.actor.optimizer.step()
                self.critic.optimizer.step()
                #self.actor.scheduler.step()
                #self.critic.scheduler.step()

        self.memory.clear_memory()



best_average_reward = -np.inf  # start with a very low value
total_reward = 0
agent = Agent()
avg_rewards = []

def render_with_agent_policy(agent, env):
    def policy(time_step):
        observation = np.concatenate([time_step.observation[key].astype(np.float32).flatten() for key in time_step.observation])
        action, _, _ = agent.choose_action(observation)
        return action

    viewer.launch(env.env, policy=policy)

for i in range(1, EPISODES + 1):  # Start from 1 for correct averaging

    observation, info = env.reset()
    terminated = False
    score = 0
    step = 0

    while not terminated:
        action, probs, value = agent.choose_action(observation)
        observation_, reward, terminated, truncated = env.step(action)
        
        agent.memory.store_memory(observation, action, probs, value, reward, terminated)
        observation = observation_
        
        # Check if memory is full
        if len(agent.memory.states) >= BATCH_SIZE:
            agent.learn()
            agent.memory.clear_memory()

        score += reward
        step += 1
        
        if terminated or truncated:
            total_reward += score
            agent.learn()
            break
    
    avg_reward = total_reward / i
    print(f"Episode {i} - avg reward: {avg_reward}, score: {score}")
    avg_rewards.append(avg_reward)


render_with_agent_policy(agent, env)

plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward over Episodes')
plt.show()
env.close()
