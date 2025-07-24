import yaml
import os
import time
import numpy as np
import torch
from tqdm import trange
from datetime import datetime

from environments.multi_agent_env import MultiAgentEnv
from environments.failure_simulator import FailureSimulator
from utils.replay_buffer import FingerprintedReplayBuffer
from utils.logger import Logger
from utils.coordinator import calculate_specialization, calculate_recovery
from agents import agent_factory

def load_config(config_path):
    """Load and merge configurations"""
    with open("configs/base.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    
    with open(config_path, "r") as f:
        env_config = yaml.safe_load(f)
    
    # Merge configurations (env overrides base)
    config = {**base_config, **env_config}
    return type('Config', (object,), config)()

def create_agents(env, config):
    agents = []
    for i, agent_id in enumerate(env.agents):
        obs_dim = env.get_agent_obs_dim(agent_id)
        act_dim = env.get_agent_action_dim(agent_id)
        agents.append(agent_factory(config.algorithm, obs_dim, act_dim, i, config))
    return agents

def train(config):
    # Initialize environment
    env = MultiAgentEnv(config.env_name, config.n_agents, config.continuous)
    logger = Logger(config)
    failure_simulator = FailureSimulator(config.failure_prob, config.failure_type)
    
    # Initialize agents
    agents = create_agents(env, config)
    buffer = FingerprintedReplayBuffer(config.buffer_size, config.n_agents)
    
    # Training loop
    for episode in trange(config.total_episodes):
        obs = env.reset()
        # Unpack obs if tuple, else use as is
        if isinstance(obs, tuple):
            obs_dict, _ = obs
        else:
            obs_dict = obs
        
        episode_rewards = np.zeros(config.n_agents)
        agent_versions = [agent.policy_version for agent in agents]
        failure_occurred = False
        failure_step = None
        prev_reward = 0
        failure_reward = 0
        
        for step in range(config.max_steps):
            actions_dict = {}

            for i, agent_id in enumerate(env.agents):
                action = agents[i].select_action(obs_dict[agent_id], explore=True)

                if config.failure_simulation and not failure_occurred:
                    action = failure_simulator.simulate_failure(agent_id, action, step)
                    if failure_simulator.is_agent_failed(agent_id):
                        failure_occurred = True
                        failure_step = step
                        prev_reward = np.sum(episode_rewards)

                actions_dict[agent_id] = action

            # Step environment
            next_obs = env.step(actions_dict)
            # Unpack next_obs if tuple, else use directly
            if isinstance(next_obs, tuple):
                next_obs_dict = next_obs[0]
                rewards = next_obs[1]
                terminations = next_obs[2]
                truncations = next_obs[3]
                infos = next_obs[4]
            else:
                next_obs_dict = next_obs
                # You might need to get rewards, terminations, truncations, infos differently here if step returns only obs dict

            dones = {
                agent: terminations[agent] or truncations[agent]
                for agent in terminations
            }

            # Adjust rewards for competitive envs
            if "competitive" in config.env_name:
                for agent_id in env.agents:
                    if "adversary" in agent_id:
                        rewards[agent_id] = -rewards[agent_id]

            # Store transition in replay buffer
            buffer.add(
                obs=np.array([obs_dict[agent] for agent in env.agents]),
                actions=np.array([actions_dict[agent] for agent in env.agents]),
                rewards=np.array([rewards[agent] for agent in env.agents]),
                next_obs=np.array([next_obs_dict[agent] for agent in env.agents]),
                dones=np.array([dones[agent] for agent in env.agents]),
                agent_versions=agent_versions
            )

            # Update episode rewards
            for i, agent_id in enumerate(env.agents):
                episode_rewards[i] += rewards[agent_id]

            if failure_occurred and failure_step == step:
                failure_reward = np.sum(episode_rewards) - prev_reward

            if episode % config.save_interval == 0:
                frame = env.render(mode='rgb_array')
                if frame is not None and len(frame.shape) >= 2:
                    logger.add_frame(frame)
                else:
                    print(f"Warning: Skipping invalid frame at episode {episode}, step {step}")



            obs_dict = next_obs_dict  # update for next step

            # Train agents if enough samples in buffer
            if len(buffer) > config.batch_size:
                batch = buffer.sample(config.batch_size)
                if config.algorithm == "QMIX":
                    batch['states'] = np.random.randn(config.batch_size, config.state_dim)
                    batch['next_states'] = np.random.randn(config.batch_size, config.state_dim)
                for agent in agents:
                    agent.update(batch, agents)
                agent_versions = [agent.policy_version for agent in agents]

            if all(dones.values()):
                break

        # Calculate specialization, stationarity, recovery
        specialization = calculate_specialization(agents) if config.use_specialization else None
        stationarity = buffer.get_stationarity_metric()
        recovery = None
        if failure_occurred:
            post_reward = np.sum(episode_rewards) - prev_reward - failure_reward
            recovery = calculate_recovery(prev_reward, failure_reward, post_reward)

        logger.log_episode(
            episode,
            episode_rewards,
            specialization=specialization,
            stationarity=stationarity,
            recovery=recovery
        )

        if episode % config.save_interval == 0:
            logger.save_models(agents, episode)
            logger.save_video(episode)

    logger.generate_plots()
    env.close()


if __name__ == "__main__":
    config = load_config("configs/cooperative.yaml")
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train(config)