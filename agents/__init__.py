from .base_agent import BaseAgent
from .iql_agent import IQLAgent
from .maddpg_agent import MADDPGAgent
from .qmix_agent import QMixAgent
from .coma_agent import COMAAgent

def agent_factory(algorithm, obs_dim, act_dim, agent_id, config):
    """Factory function to create agent instances based on algorithm"""
    agent_classes = {
        'IQL': IQLAgent,
        'MADDPG': MADDPGAgent,
        'QMIX': QMixAgent,
        'COMA': COMAAgent
    }
    
    if algorithm not in agent_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent_classes[algorithm](obs_dim, act_dim, agent_id, config)