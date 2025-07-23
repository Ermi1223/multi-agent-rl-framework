import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from models.actor_critic import Actor, Critic

class MADDPGAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, agent_id, config):
        super().__init__(obs_dim, act_dim, agent_id, config)
        self.actor = Actor(obs_dim, act_dim, config.hidden_dim)
        self.critic = Critic(config.n_agents * obs_dim, config.n_agents * act_dim, config.hidden_dim)
        
        self.target_actor = Actor(obs_dim, act_dim, config.hidden_dim)
        self.target_critic = Critic(config.n_agents * obs_dim, config.n_agents * act_dim, config.hidden_dim)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.gamma = config.gamma
        self.tau = config.tau
        
    def select_action(self, obs, explore=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(obs).squeeze(0).detach().numpy()
        if explore:
            noise = np.random.normal(0, 0.1, size=self.act_dim)
            action = np.clip(action + noise, -1, 1)
        return action
    
    def update(self, batch, agents):
        obs, actions, rewards, next_obs, dones = batch
        agent_idx = self.agent_id
        
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Extract agent-specific values
        agent_rewards = rewards[:, agent_idx].unsqueeze(1)
        agent_dones = dones[:, agent_idx].unsqueeze(1)
        
        # Compute next actions from target actor
        next_actions = []
        for i in range(self.config.n_agents):
            next_actions.append(agents[i].target_actor(next_obs[:, i]))
        next_actions = torch.cat(next_actions, dim=1)
        
        # Compute target Q-value
        next_q = self.target_critic(
            next_obs.reshape(next_obs.shape[0], -1), 
            next_actions
        )
        target_q = agent_rewards + self.gamma * (1 - agent_dones) * next_q
        
        # Compute current Q-value
        current_q = self.critic(
            obs.reshape(obs.shape[0], -1), 
            actions.reshape(actions.shape[0], -1)
        )
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        pred_actions = []
        for i in range(self.config.n_agents):
            if i == agent_idx:
                pred_actions.append(self.actor(obs[:, i]))
            else:
                pred_actions.append(agents[i].actor(obs[:, i]).detach())
        pred_actions = torch.cat(pred_actions, dim=1)
        
        actor_loss = -self.critic(
            obs.reshape(obs.shape[0], -1), 
            pred_actions
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        self.policy_version += 1
        self.train_step += 1
        
        return critic_loss.item(), actor_loss.item()