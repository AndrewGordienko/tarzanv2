import torch
import numpy as np
from models import actor_network, critic_network
from memory_storage import ReplayBuffer

BATCH_SIZE = 64
LEARNING_RATE = 0.001
torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 4
        self.gae_lambda = 0.95
        self.entropy_weight = 0.01

        self.actor = actor_network()
        self.critic = critic_network()
        self.memory = ReplayBuffer()
        self.device = torch.device("cpu")

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001)

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        policy_dist = self.actor(observation)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]


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

        for _ in range(self.n_epochs):
            states, actions, log_probs, rewards, states_, dones = self.memory.sample()

            values = self.critic(states).squeeze()
            next_values = self.critic(states_).squeeze()

            advantages = self.compute_advantages(rewards, dones, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            new_log_probs = self.actor(states).log_prob(actions).squeeze()
            ratio = torch.exp(new_log_probs - log_probs)

            advantages = advantages.view(-1, 1)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages
            actor_loss = - torch.min(surr1, surr2).mean()

            returns = rewards + self.gamma * next_values * (1 - dones)
            critic_loss = ((returns - values)**2).mean()

            entropy_loss = - self.actor(states).entropy().mean()

            #loss = actor_loss + 0.5 * critic_loss - self.entropy_weight * entropy_loss
            loss = 0.5 * critic_loss + actor_loss - self.entropy_weight * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.optimizer.step()
        
