import yaml
import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from environments.multi_agent_env import MultiAgentEnv
from utils.analysis_tools import calculate_role_consistency, calculate_specialization
from agents import agent_factory
from configs import load_config

def flatten_obs(obs_dict):
    """Flatten dict observation into a 1D numpy array"""
    flat_list = []
    for v in obs_dict.values():
        if isinstance(v, (list, np.ndarray)):
            flat_list.extend(np.array(v).flatten())
        else:
            flat_list.append(v)
    return np.array(flat_list, dtype=np.float32)

def save_results_to_csv(results, filepath="evaluation_results.csv"):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as csvfile:
        fieldnames = ['episode', 'average_reward', 'specialization', 'role_consistency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # write header only once

        writer.writerow({
            'episode': results.get('episode', 'N/A'),
            'average_reward': results['reward'],
            'specialization': results['specialization'],
            'role_consistency': results['role_consistency']
        })

def evaluate(config, checkpoint_episode=1800, num_episodes=10):
    env = MultiAgentEnv(config.env_name, config.n_agents, config.continuous)
    
    # Reset environment once to get sample obs for inferring obs_dim per agent
    sample_obs = env.reset()
    # If reset returns tuple, extract first element (obs dict)
    if isinstance(sample_obs, tuple):
        sample_obs = sample_obs[0]
    
    agents = []
    for i, agent_id in enumerate(env.agents):
        obs_i = sample_obs[agent_id]
        if isinstance(obs_i, dict):
            obs_i = flatten_obs(obs_i)
        obs_dim = len(obs_i)
        print(f"Agent {i} obs_dim used for agent creation: {obs_dim}")
        
        agent = agent_factory(
            config.algorithm,
            obs_dim,
            env.get_agent_action_dim(agent_id),
            i,
            config
        )
        
        path = os.path.join(config.checkpoint_dir, f"agent_{i}_ep_{checkpoint_episode}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        agent.load(path)
        agents.append(agent)

    total_rewards = []
    specialization_scores = []
    role_consistencies = []

    for _ in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract observation dict

        episode_rewards = np.zeros(config.n_agents)
        episode_actions = []

        for _ in range(config.max_steps):
            actions_dict = {}
            for i, agent_id in enumerate(env.agents):
                obs_i = obs[agent_id]
                if isinstance(obs_i, dict):
                    obs_i = flatten_obs(obs_i)
                action = agents[i].select_action(obs_i, explore=False)
                actions_dict[agent_id] = action
                episode_actions.append(action)

            step_result = env.step(actions_dict)
            # env.step might return a tuple: (obs, rewards, dones, infos)
            if isinstance(step_result, tuple) and len(step_result) >= 4:
                next_obs, rewards, dones, infos = step_result[:4]
            else:
                next_obs = step_result
                # You might need to adjust if your env.step returns differently

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            # Adjust competitive rewards if needed
            if "competitive" in config.env_name:
                for agent_id in env.agents:
                    if "adversary" in agent_id:
                        rewards[agent_id] = -rewards[agent_id]

            for i, agent_id in enumerate(env.agents):
                episode_rewards[i] += rewards[agent_id]

            obs = next_obs

            if all(dones.values()):
                break

        total_rewards.append(np.sum(episode_rewards))
        specialization_scores.append(calculate_specialization(agents))
        role_consistencies.append(calculate_role_consistency(agents))

    avg_reward = np.mean(total_rewards)
    avg_specialization = np.mean(specialization_scores)
    avg_role_consistency = np.mean(role_consistencies)

    print(f"\nEvaluation Results (Episode {checkpoint_episode}):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Specialization: {avg_specialization:.2f}")
    print(f"  Role Consistency: {avg_role_consistency:.2f}")

    env.close()

    results = {
        'episode': checkpoint_episode,
        'reward': avg_reward,
        'specialization': avg_specialization,
        'role_consistency': avg_role_consistency
    }

    save_results_to_csv(results)

    return results


if __name__ == "__main__":
    config = load_config("configs/cooperative.yaml")
    evaluate(config, checkpoint_episode=1800, num_episodes=10)
