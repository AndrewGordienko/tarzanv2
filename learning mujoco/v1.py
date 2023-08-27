import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dm_control import suite
from dm_control import viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image as PILImage

# Hyperparameters
EPISODES = 5001
BATCH_SIZE = 40
GAMMA = 0.99
LEARNING_RATE = 3e-4
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")

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

class actor_network(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.input_shape = (input_dims,)
        self.action_space = env.action_space
        self.std = 0.5

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.log_std = nn.Parameter(torch.zeros(1, self.action_space) * self.std)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -2  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


    def net(self, x):
        # print("")
        # print("input:", x)
        x = F.relu(self.fc1(x))
        # print("After fc1:", x)
        x = F.relu(self.fc2(x))
        #print("After fc2:", x)
        x = self.fc3(x)
        #print("After fc3:", x)
        return x

    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = (log_std + 1e-8).exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)

        return policy_dist
    
class critic_network(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_shape = (input_dims,)

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

        self.actor = actor_network(input_dims)
        self.critic = critic_network(input_dims)
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
        noise = torch.normal(mean=0., std=0.1, size=action.shape).to(DEVICE)

        min_val = torch.tensor(env.action_spec.minimum, dtype=torch.float).to(DEVICE)
        max_val = torch.tensor(env.action_spec.maximum, dtype=torch.float).to(DEVICE)

        #action = (action + noise).clamp(min=min_val, max=max_val)

        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic.forward(state)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.item()
        value = torch.squeeze(value).item()

        return action, log_prob, value


    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            delta = 0
            discount = 1

            values = vals_arr # This is where you define 'values' before accessing it

            for t in reversed(range(len(reward_arr) - 1)):
                delta = reward_arr[t] + self.gamma * values[t + 1] * (1 - int(dones_arr[t])) - values[t]
                advantage[t] = delta + discount * self.gae_lambda * advantage[t + 1]

                discount *= self.gamma * self.gae_lambda

            advantage = torch.tensor(advantage).to(DEVICE)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

            values = torch.tensor(values).to(DEVICE)


            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                dist = self.actor(states)  # Get the policy distribution
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probabilities = dist.log_prob(actions).sum(axis=-1)

                # print("")
                # print("new_probabilities: ", new_probabilities)
                # print("--")
                # print("old probs: ", old_probs)

                epsilon = 1e-8
                probability_ratio = (new_probabilities.exp() + epsilon) / (old_probs.exp() + epsilon)
                #probability_ratio = torch.clamp(probability_ratio, 0.1, 10)

                weighted_probabilities = advantage[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                
                actor_loss = -torch.min(weighted_probabilities, weighted_clipped_probabilities).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                entropy = dist.entropy().mean()
                total_loss = total_loss - entropy * 0.05 #(entropy beta)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()   

# Main Loop
agent = Agent(n_actions=env.action_space, input_dims=env.observation_space)
step = 0
best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []

def render_with_agent_policy(agent, env):
    def policy(time_step):
        observation = np.concatenate([time_step.observation[key].astype(np.float32).flatten() for key in time_step.observation])
        action, _, _ = agent.choose_action(observation)
        return action

    viewer.launch(env.env, policy=policy)


for i in range(1, EPISODES):
    observation, _ = env.reset()
    score = 0
    done = False

    # if i % 50 == 0:
        # render_with_agent_policy(agent, env)

    while not done:
        step += 1
        action, prob, val = agent.choose_action(observation)


        observation_, reward, done, _ = env.step(action)

        if reward != None:
            score += reward
            
        agent.remember(observation, action, prob, val, reward, done)
        
        observation = observation_
        
        if done:
            if score > best_reward:
                torch.save(agent.actor.state_dict(), 'agent_actor.pth')
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(i, average_reward/i, best_reward, score))
            episode_number.append(i)
            average_reward_number.append(average_reward/i)

            break
    
    agent.learn()

    # frames[0].save('ppo_training.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)

    


    
render_with_agent_policy(agent, env)

# Plotting the results
plt.plot(episode_number, average_reward_number)
plt.show()






