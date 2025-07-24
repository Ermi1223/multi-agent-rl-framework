import torch
import torch.nn as nn

class AttentionCommModule(nn.Module):
    def __init__(self, input_dim, comm_dim, num_heads=4):
        super().__init__()
        self.comm_dim = comm_dim
        self.num_heads = num_heads
        self.head_dim = comm_dim // num_heads
        
        self.query = nn.Linear(input_dim, comm_dim)
        self.key = nn.Linear(input_dim, comm_dim)
        self.value = nn.Linear(input_dim, comm_dim)
        self.fc_out = nn.Linear(comm_dim, comm_dim)
        
    def forward(self, agent_obs, messages):
        # agent_obs: [batch_size, input_dim]
        # messages: list of [batch_size, comm_dim] from other agents
        
        batch_size = agent_obs.size(0)
        num_agents = len(messages) + 1  # Including self
        
        # Create full message set including self
        all_messages = torch.stack([agent_obs] + messages, dim=1)  # [batch, num_agents, input_dim]
        
        # Compute Q, K, V
        Q = self.query(agent_obs).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(all_messages).view(batch_size, num_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(all_messages).view(batch_size, num_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # Weighted sum
        weighted = torch.matmul(attn, V).squeeze(1)
        weighted = weighted.transpose(1, 2).contiguous().view(batch_size, -1)
        
        return self.fc_out(weighted)