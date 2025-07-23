import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from models.q_networks import QNetwork
from models.mixer import QMixer

class QMixAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, agent_id, config):
        super().__init__(obs_dim, act_dim, agent_id, config)
        self.q_net = QNetwork(obs_dim, act_dim, config.hidden_dim)
        self.target_q_net = QNetwork(obs_dim, act_dim, config.hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.mixer = QMixer(config.n_agents, config.state_dim, config.hidden_dim)
        self.target_mixer = QMixer(config.n_agents, config.state_dim, config.hidden_dim)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()), 
            lr=config.learning_rate
        )
        
        self.gamma = config.gamma
        self.tau = config.tau
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

    def save(self, path):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'mixer_state_dict': self.mixer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_version': self.policy_version
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_version = checkpoint['policy_version']

        
    def select_action(self, obs, explore=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_net(obs).detach().numpy().flatten()
        
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        return np.argmax(q_values)
    
    def update(self, batch, agents):
        obs = batch['obs']        # (batch_size, n_agents, obs_dim)
        actions = batch['actions']  # (batch_size, n_agents)
        rewards = batch['rewards']  # (batch_size, n_agents)
        next_obs = batch['next_obs']  # (batch_size, n_agents, obs_dim)
        dones = batch['dones']      # (batch_size, n_agents)
        states = batch.get('states', None)           # (batch_size, state_dim)
        next_states = batch.get('next_states', None) # (batch_size, state_dim)

        batch_size = obs.shape[0]
        n_agents = obs.shape[1]

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)  # (batch_size, n_agents, 1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        states = torch.tensor(states, dtype=torch.float32) if states is not None else None
        next_states = torch.tensor(next_states, dtype=torch.float32) if next_states is not None else None

        # Current Q-values for all agents
        current_qs = []
        for i, agent in enumerate(agents):
            q_values = agent.q_net(obs[:, i, :])
            chosen_q = q_values.gather(1, actions[:, i].unsqueeze(1))
            current_qs.append(chosen_q)
        current_q = torch.stack(current_qs, dim=1)  # (batch_size, n_agents, 1)

        # Target Q-values for all agents
        with torch.no_grad():
            target_qs = []
            for i, agent in enumerate(agents):
                target_q_values = agent.target_q_net(next_obs[:, i, :])
                max_target_q = target_q_values.max(dim=1)[0].unsqueeze(1)
                target_qs.append(max_target_q)
            target_q = torch.stack(target_qs, dim=1)  # (batch_size, n_agents, 1)

            target_q_total = self.target_mixer(target_q, next_states)  # (batch_size, 1)

            # Aggregate rewards and dones across agents for team reward and done
            total_rewards = rewards.sum(dim=1)  # (batch_size, 1)
            done_flags = dones.max(dim=1)[0]   # (batch_size, 1)

            target_q_total = total_rewards + self.gamma * (1 - done_flags) * target_q_total

        current_q_total = self.mixer(current_q, states)  # (batch_size, 1)

        loss = nn.MSELoss()(current_q_total, target_q_total)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update all agents' target q networks
        for agent in agents:
            for target_param, param in zip(agent.target_q_net.parameters(), agent.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.policy_version += 1
        self.train_step += 1

        return loss.item()
