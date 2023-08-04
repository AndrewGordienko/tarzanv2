import gym
import torch
from brain import Agent
from memory_storage import ReplayBuffer
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v3', render_mode="human")

agent = Agent()

best_reward = float("-inf")
average_reward = 0
EPISODES = 5000
EPISODE_LENGTH = 1000  # Define a maximum episode length
average_rewards = []
best_rewards = []

for i in range(1, EPISODES):
    step = 0
    score = 0
    state, info = env.reset(seed=42)

    while True:
        env.render()
        action, log_prob = agent.choose_action(state)
        state_, reward, done, truncated, info = env.step(action)

        agent.memory.add(state, action, log_prob, reward, state_, done)

        state = state_
        score += reward
        step += 1

        if done or step >= EPISODE_LENGTH:
            if score > best_reward:
                torch.save(agent.actor.state_dict(), 'agent_actor.pth')

                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(i, average_reward/i, best_reward, score))
            break
    
    average_rewards.append(average_reward/i)
    best_rewards.append(best_reward)
    
    for _ in range(5):
        agent.learn()

    # Clear the memory buffer for the next episode
    agent.memory = ReplayBuffer()

plt.plot(average_rewards, label='Average Reward')
plt.plot(best_rewards, label='Best Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.legend()
plt.show()





