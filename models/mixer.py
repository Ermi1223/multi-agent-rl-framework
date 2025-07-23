import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(states))  # (batch_size, n_agents * hidden_dim)
        b1 = self.hyper_b1(states)              # (batch_size, hidden_dim)

        w1 = w1.view(batch_size, self.n_agents, -1)  # (batch_size, n_agents, hidden_dim)
        b1 = b1.view(batch_size, 1, -1)               # (batch_size, 1, hidden_dim)

        agent_qs_t = agent_qs.transpose(1, 2)         # (batch_size, 1, n_agents)

        hidden = F.elu(torch.bmm(agent_qs_t, w1) + b1)  # (batch_size, 1, hidden_dim)

        w2 = torch.abs(self.hyper_w2(states))        # (batch_size, hidden_dim)
        w2 = w2.view(batch_size, -1, 1)               # (batch_size, hidden_dim, 1)

        b2 = self.hyper_b2(states).view(batch_size, 1, 1)  # (batch_size, 1, 1)

        y = torch.bmm(hidden, w2) + b2                # (batch_size, 1, 1)

        return y.view(batch_size, -1)                  # (batch_size, 1)

