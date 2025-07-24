import numpy as np
import random
from collections import defaultdict

class FingerprintedReplayBuffer:
    def __init__(self, capacity, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = []
        self.policy_versions = defaultdict(lambda: np.zeros(n_agents))
        self.idx = 0

    def add(self, obs, actions, rewards, next_obs, dones, agent_versions):
        transition = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'versions': agent_versions
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx % self.capacity] = transition
        self.idx += 1
        
        # Update policy tracking
        for i, version in enumerate(agent_versions):
            self.policy_versions[version][i] += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            'obs': np.array([t['obs'] for t in batch]),
            'actions': np.array([t['actions'] for t in batch]),
            'rewards': np.array([t['rewards'] for t in batch]),
            'next_obs': np.array([t['next_obs'] for t in batch]),
            'dones': np.array([t['dones'] for t in batch]),
            'versions': np.array([t['versions'] for t in batch])
        }
    
    def get_stationarity_metric(self):
        # Assuming self.policy_versions is dict of numpy arrays
        return np.mean([np.std(v) for v in self.policy_versions.values()])

    
    def __len__(self):
        return len(self.buffer)