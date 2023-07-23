import torch
import numpy as np
from models import actor_network, critic_network
from memory_storage import ReplayBuffer

BATCH_SIZE = 64
LEARNING_RATE = 0.0001

class Agent:
    def __init__(self, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95
        self.entropy_weight = 0.001

        self.actor = actor_network()
        self.critic = critic_network()
        self.memory = ReplayBuffer()
        self.device = torch.device("cpu")

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=LEARNING_RATE)

    def choose_action(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            policy_dist = self.actor(observation)
            action = policy_dist.sample()
            return action[0]
            
    def compute_advantages(self, rewards, dones, values, next_values):
        gae = 0
        advantages = torch.zeros_like(rewards).to(self.device)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return

        states, actions, rewards, states_, dones = self.memory.sample()

        values = self.critic(states).detach()
        values_ = self.critic(states_).detach()
        advantages = self.compute_advantages(rewards, dones, values, values_)

        for _ in range(self.n_epochs):
            old_log_probs = self.actor(states).log_prob(actions)

            self.optimizer.zero_grad()

            # Compute critic loss
            critic_values = self.critic(states)
            critic_loss = (rewards + self.gamma * values_ * (1 - dones) - critic_values).pow(2).mean()

            # Compute actor loss
            new_log_probs = self.actor(states).log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages.unsqueeze(1)
            actor_loss = - torch.min(surr1, surr2).mean()

            # Compute entropy loss
            entropy_loss = - self.actor(states).entropy().mean()

            # Compute total loss
            loss = critic_loss + actor_loss + self.entropy_weight * entropy_loss

            loss.backward()
            self.optimizer.step()
