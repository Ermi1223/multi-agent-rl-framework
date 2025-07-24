import numpy as np

def calculate_specialization(agents):
    """Calculate role specialization based on action entropy"""
    role_scores = []
    for agent in agents:
        action_probs = agent.get_action_probs()
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        max_entropy = np.log(len(action_probs))
        role_scores.append(1 - entropy / max_entropy)
    return np.mean(role_scores)

def calculate_recovery(prev_reward, failure_reward, post_reward):
    """Calculate recovery performance after failure"""
    if prev_reward <= 0 or failure_reward <= 0:
        return 0
    recovery = (post_reward - failure_reward) / prev_reward
    return max(0, min(1, recovery))

def calculate_communication_efficiency(episode_reward, no_comm_reward):
    """Calculate communication efficiency gain"""
    if no_comm_reward <= 0:
        return 0
    return max(0, (episode_reward - no_comm_reward) / no_comm_reward)