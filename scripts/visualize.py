import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import imageio
from environments.multi_agent_env import MultiAgentEnv
from agents import agent_factory
from configs import load_config

def flatten_obs(obs_dict):
    flat_list = []
    for v in obs_dict.values():
        if isinstance(v, (list, np.ndarray)):
            flat_list.extend(np.array(v).flatten())
        else:
            flat_list.append(v)
    return np.array(flat_list, dtype=np.float32)

def save_video(frames, filename, fps=10):
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            if frame is not None and frame.ndim == 3:  # Ensure the frame is valid
                writer.append_data(frame)
            else:
                print("Skipping invalid frame")

def visualize_behavior(config, checkpoint_episode=10000, num_episodes=3):
    env = MultiAgentEnv(config.env_name, config.n_agents, config.continuous)
    agents = []
    
    # Load trained agents
    for i in range(config.n_agents):
        agent = agent_factory(config.algorithm, 
                             env.get_agent_obs_dim(f"agent_{i}"),
                             env.get_agent_action_dim(f"agent_{i}"),
                             i, config)
        path = os.path.join(config.checkpoint_dir, f"agent_{i}_ep_{checkpoint_episode}.pt")
        agent.load(path)
        agents.append(agent)
    
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # If it's a tuple, unpack it
            obs_dict, _ = obs
        else:
            obs_dict = obs
        frames = []
        action_log = {agent_id: [] for agent_id in env.agents}
        
        for step in range(config.max_steps):
            actions_dict = {}
            for i, agent_id in enumerate(env.agents):
                action = agents[i].select_action(obs_dict[agent_id], explore=False)
                actions_dict[agent_id] = action
                action_log[agent_id].append(action)
            
            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            dones = {agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}

            if isinstance(next_obs, tuple):  # If it's a tuple, unpack it
                next_obs_dict, _ = next_obs
            else:
                next_obs_dict = next_obs
            
            # Capture valid frame
            frame = env.render(mode='rgb_array')
            if frame is None:
                print(f"Render returned None at step {step} of episode {ep}")
            elif frame.ndim != 3:
                print(f"Invalid frame at step {step} of episode {ep} - shape: {frame.shape}")
            else:
                frames.append(frame)
            
            obs_dict = next_obs_dict
            
            if all(dones.values()):
                break
        
        # Save video
        save_video(frames, f"episode_{ep}.mp4")
        
        # Plot action distributions
        plt.figure(figsize=(12, 8))
        for i, agent_id in enumerate(env.agents):
            actions = action_log[agent_id]
            plt.subplot(len(env.agents), 1, i+1)
            plt.hist(actions, bins=20)
            plt.title(f"Agent {agent_id} Action Distribution")
        plt.tight_layout()
        plt.savefig(f"action_distribution_ep_{ep}.png")
        plt.close()
    
    env.close()



if __name__ == "__main__":
    config = load_config("configs/cooperative.yaml")
    visualize_behavior(config, checkpoint_episode=1800)
