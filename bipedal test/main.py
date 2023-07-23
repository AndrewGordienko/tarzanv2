import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from brain import Agent

env = gym.make('BipedalWalker-v3', render_mode="human")
agent = Agent()

best_reward = float("-inf")
average_reward = 0
EPISODES = 100

for i in range(1, EPISODES):
    score = 0
    state, info = env.reset(seed=42)

    while True:
        env.render()
        # action = env.action_space.sample()
        action = agent.choose_action(state)
        state_, reward, done, truncated, info = env.step(action)

        agent.learn()
        agent.memory.add(state, action, reward, state_, done)

        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(i, average_reward/i, best_reward, score))
            break
