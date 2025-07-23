import torch
import numpy as np

class BaseAgent:
    def __init__(self, obs_dim, act_dim, agent_id, config):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.policy_version = 0
        self.train_step = 0
        
    def select_action(self, obs, explore=False):
        raise NotImplementedError
        
    def update(self, batch, agents=None):
        raise NotImplementedError
        
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_version': self.policy_version
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_version = checkpoint['policy_version']
        
    def get_action_probs(self, num_samples=1000):
        """Estimate action distribution for specialization analysis"""
        probs = np.zeros(self.act_dim)
        for _ in range(num_samples):
            obs = np.random.randn(self.obs_dim)
            action = self.select_action(obs, explore=False)
            probs[action] += 1
        return probs / np.sum(probs)
    
    def policy_entropy(self):
        """Calculate current policy entropy"""
        probs = self.get_action_probs()
        return -np.sum(probs * np.log(probs + 1e-10))