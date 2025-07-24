import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import imageio

class Logger:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(config.log_dir)
        self.episode_rewards = []
        self.specialization_scores = []
        self.stationarity_metrics = []
        self.communication_metrics = []
        self.failure_recovery = []
        self.frames = []
    
    def log_episode(self, episode, rewards, specialization=None, 
                    stationarity=None, communication=None, recovery=None):
        total_reward = np.sum(rewards)
        self.episode_rewards.append(total_reward)
        
        # Log to tensorboard
        self.writer.add_scalar('Reward/Total', total_reward, episode)
        for i, r in enumerate(rewards):
            self.writer.add_scalar(f'Reward/Agent_{i}', r, episode)
        
        if specialization is not None:
            self.specialization_scores.append(specialization)
            self.writer.add_scalar('Metrics/Specialization', specialization, episode)
        
        if stationarity is not None:
            self.stationarity_metrics.append(stationarity)
            self.writer.add_scalar('Metrics/Stationarity', stationarity, episode)
        
        if communication is not None:
            self.communication_metrics.append(communication)
            self.writer.add_scalar('Metrics/Communication', communication, episode)
        
        if recovery is not None:
            self.failure_recovery.append(recovery)
            self.writer.add_scalar('Metrics/Recovery', recovery, episode)
    
    def save_models(self, agents, episode):
        for agent in agents:
            path = os.path.join(self.config.checkpoint_dir, f"agent_{agent.agent_id}_ep_{episode}.pt")
            agent.save(path)
    
    def add_frame(self, frame):
        # Only add frames that look valid (at least 2D image)
        if frame is not None and len(frame.shape) >= 2:
            self.frames.append(frame)
        else:
            print(f"Warning: Skipped adding invalid frame with shape {getattr(frame, 'shape', None)}")
    
    def save_video(self, episode):
        # Filter valid frames before saving
        valid_frames = [f for f in self.frames if f is not None and len(f.shape) >= 2]
        if not valid_frames:
            print(f"Warning: No valid frames to save video for episode {episode}. Skipping.")
            return
        
        path = os.path.join(self.config.log_dir, f"episode_{episode}.mp4")
        imageio.mimsave(path, valid_frames, fps=10)
        self.frames = []
    
    def generate_plots(self):
        # Plot reward curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Curve')
        plt.savefig(os.path.join(self.config.log_dir, 'reward_curve.png'))
        plt.close()
        
        # Plot specialization
        if self.specialization_scores:
            plt.figure(figsize=(12, 6))
            plt.plot(self.specialization_scores)
            plt.xlabel('Episode')
            plt.ylabel('Specialization Score')
            plt.title('Agent Specialization Over Time')
            plt.savefig(os.path.join(self.config.log_dir, 'specialization.png'))
            plt.close()
        
        # Plot stationarity
        if self.stationarity_metrics:
            plt.figure(figsize=(12, 6))
            plt.plot(self.stationarity_metrics)
            plt.xlabel('Episode')
            plt.ylabel('Stationarity Metric')
            plt.title('Non-stationarity Handling')
            plt.savefig(os.path.join(self.config.log_dir, 'stationarity.png'))
            plt.close()
        
        # Plot communication
        if self.communication_metrics:
            plt.figure(figsize=(12, 6))
            plt.plot(self.communication_metrics)
            plt.xlabel('Episode')
            plt.ylabel('Communication Efficiency')
            plt.title('Communication Protocol Effectiveness')
            plt.savefig(os.path.join(self.config.log_dir, 'communication.png'))
            plt.close()
        
        # Plot failure recovery
        if self.failure_recovery:
            plt.figure(figsize=(12, 6))
            plt.plot(self.failure_recovery)
            plt.xlabel('Episode')
            plt.ylabel('Recovery Rate')
            plt.title('Failure Recovery Performance')
            plt.savefig(os.path.join(self.config.log_dir, 'recovery.png'))
            plt.close()
        
        self.writer.close()
