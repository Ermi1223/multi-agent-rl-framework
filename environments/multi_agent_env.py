import numpy as np
from pettingzoo.mpe import simple_spread_v3, simple_adversary_v3, simple_tag_v3

class MultiAgentEnv:
    def __init__(self, env_name="cooperative", n_agents=3, continuous=False):
        self.env_name = env_name
        self.n_agents = n_agents
        self.continuous = continuous
        
        if "cooperative" in env_name:
            self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=continuous)
        elif "mixed" in env_name:
            self.env = simple_adversary_v3.parallel_env(N=n_agents, continuous_actions=continuous)
        elif "competitive" in env_name:
            self.env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=n_agents-1, continuous_actions=continuous)
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
            
        self.agents = self.env.possible_agents
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        return self.env.step(actions)
    
    def render(self, mode='human'):
        # Just call render without any argument
        return self.env.render()

    
    def get_global_state(self):
        # Concatenate all agent observations for centralized training
        obs = self.env.observe(self.env.agents)
        return np.concatenate([obs[agent] for agent in self.env.agents])
    
    def close(self):
        self.env.close()
        
    def get_state_dim(self):
        return len(self.get_global_state())
    
    def get_agent_obs_dim(self, agent_id):
        return self.observation_spaces[agent_id].shape[0]
    
    def get_agent_action_dim(self, agent_id):
        if self.continuous:
            return self.action_spaces[agent_id].shape[0]
        return self.action_spaces[agent_id].n