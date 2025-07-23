import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_agent import BaseAgent
from models.actor_critic import Actor
from models.coma_critic import COMACritic

class COMAAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, agent_id, config):
        super().__init__(obs_dim, act_dim, agent_id, config)
        self.actor = Actor(obs_dim, act_dim, config.hidden_dim)
        self.critic = COMACritic(config.n_agents, obs_dim, act_dim, config.hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.gamma = config.gamma
        self.tau = config.tau
        
    def select_action(self, obs, explore=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = F.softmax(self.actor(obs), dim=-1)
        
        if explore:
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(probs).item()
        return action
    
    def update(self, batch, agents=None):
        obs, actions, rewards, next_obs, dones = batch
        agent_idx = self.agent_id
        
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute counterfactual baseline
        baseline = self.critic.counterfactual_baseline(obs, actions)
        
        # Compute Q-values
        q_values = self.critic(obs, actions)
        
        # Compute advantages
        advantages = q_values - baseline.detach()
        
        # Actor loss (policy gradient)
        log_probs = F.log_softmax(self.actor(obs[:, agent_idx]), dim=-1)
        selected_log_probs = log_probs.gather(1, actions[:, agent_idx].unsqueeze(1))
        actor_loss = -torch.mean(selected_log_probs * advantages[:, agent_idx].unsqueeze(1))
        
        # Critic loss
        with torch.no_grad():
            next_q = self.critic(next_obs, actions)
            target_q = rewards[:, agent_idx] + self.gamma * (1 - dones[:, agent_idx]) * next_q
        
        critic_loss = F.mse_loss(q_values, target_q)
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.policy_version += 1
        self.train_step += 1
        
        return actor_loss.item(), critic_loss.item()