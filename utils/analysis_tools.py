import numpy as np
from scipy.stats import entropy

def calculate_role_consistency(agents, num_episodes=10):
    """Measure consistency of agent roles over multiple episodes"""
    role_assignments = []

    for _ in range(num_episodes):
        actions = []
        for agent in agents:
            obs = np.random.randn(agent.obs_dim)
            action = agent.select_action(obs, explore=False)

            # Handle continuous or array-like actions
            if isinstance(action, (np.ndarray, float)):
                action = int(np.round(action))  # discretize
            actions.append(action)

        if len(set(actions)) == 0:
            continue

        dominant_action = np.argmax(np.bincount(actions))
        dominant_idx = actions.index(dominant_action)
        role_assignments.append(dominant_idx)

    if not role_assignments:
        return 0.0

    counts = np.bincount(role_assignments)
    return 1 - (entropy(counts) / (np.log(len(agents)) + 1e-6))


def calculate_specialization(agents, num_trials=20):
    """Estimate specialization by measuring action entropy across agents"""
    action_histories = [[] for _ in agents]

    for _ in range(num_trials):
        for i, agent in enumerate(agents):
            obs = np.random.randn(agent.obs_dim)
            action = agent.select_action(obs, explore=False)

            if isinstance(action, (np.ndarray, float)):
                action = int(np.round(action))  # Discretize
            action_histories[i].append(action)

    # Entropy per agent
    entropies = []
    for actions in action_histories:
        if len(actions) == 0:
            entropies.append(0)
        else:
            entropies.append(entropy(np.bincount(actions) + 1e-6))

    max_entropy = np.log2(max(len(set(a)) for a in action_histories) + 1e-6)
    specialization_score = 1 - (np.mean(entropies) / (max_entropy + 1e-6))
    return specialization_score


def calculate_non_stationarity(buffer):
    """Quantify environmental non-stationarity by comparing reward drift"""
    if len(buffer) < 10:
        return 0.0

    recent = buffer.sample(min(100, len(buffer)//2))
    older = buffer.sample(min(100, len(buffer)//2))

    # Handle both dict or tuple-like samples
    def get_rewards(samples):
        if isinstance(samples, dict):
            return samples['rewards']
        elif isinstance(samples[0], (tuple, list)):
            return [s[2] for s in samples]  # assuming (obs, action, reward, next_obs)
        return []

    recent_rewards = get_rewards(recent)
    older_rewards = get_rewards(older)

    if not recent_rewards or not older_rewards:
        return 0.0

    reward_diff = np.mean(recent_rewards) - np.mean(older_rewards)
    return abs(reward_diff)


def analyze_coordination(episode_history):
    """Analyze coordination patterns from episode history"""
    coordination_scores = []
    for episode in episode_history:
        rewards = episode.get('rewards', [])
        if not rewards:
            continue
        reward_var = np.var(rewards)
        coordination_scores.append(1 / (1 + reward_var))

    if not coordination_scores:
        return 0.0

    return np.mean(coordination_scores)
