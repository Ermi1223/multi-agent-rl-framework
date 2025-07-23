import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from models.q_networks import QNetwork

class IQLAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, agent_id, config):
        super().__init__(obs_dim, act_dim, agent_id, config)
        self.q_net = QNetwork(obs_dim, act_dim, config.hidden_dim)
        self.target_q_net = QNetwork(obs_dim, act_dim, config.hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        
    def select_action(self, obs, explore=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_net(obs).detach().numpy().flatten()
        
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        return np.argmax(q_values)
    
    def update(self, batch, agents=None):
        obs, actions, rewards, next_obs, dones = batch
        agent_idx = self.agent_id
        
        # Convert to tensors
        obs = torch.tensor(obs[:, agent_idx], dtype=torch.float32)
        actions = torch.tensor(actions[:, agent_idx], dtype=torch.long)
        rewards = torch.tensor(rewards[:, agent_idx], dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(next_obs[:, agent_idx], dtype=torch.float32)
        dones = torch.tensor(dones[:, agent_idx], dtype=torch.float32).unsqueeze(1)
        
        # Compute Q-values
        current_q = self.q_net(obs).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_q_net(next_obs).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.policy_version += 1
        self.train_step += 1
        
        return loss.item()