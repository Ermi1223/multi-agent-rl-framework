import torch
import torch.nn as nn

class COMACritic(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    def forward(self, obs, actions):
        # obs: [batch_size, n_agents, obs_dim]
        # actions: [batch_size, n_agents] (discrete actions)
        batch_size = obs.shape[0]
        flat_obs = obs.view(batch_size, -1)
        q_values = self.net(flat_obs)
        return q_values
    
    def counterfactual_baseline(self, obs, actions):
        batch_size = obs.shape[0]
        n_agents = self.n_agents
        
        # Compute baseline: average Q over possible actions
        all_actions = torch.arange(self.act_dim).repeat(batch_size, n_agents, 1)
        expanded_obs = obs.unsqueeze(2).repeat(1, 1, self.act_dim, 1)
        
        # Compute Q-values for all possible actions
        flat_obs = expanded_obs.view(batch_size, n_agents * self.act_dim, -1)
        q_values = self.net(flat_obs)
        q_values = q_values.view(batch_size, n_agents, self.act_dim)
        
        # Average over actions
        baseline = q_values.mean(dim=2)
        return baseline