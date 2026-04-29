import pygame
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

from game.snake_game import SnakeGame
from agent.agent import DQNAgent

class TrainingPipeline:
    """
    Training pipeline for DQN agent with visualization
    """
    
    def __init__(self, num_episodes=1000, save_freq=100, plot_freq=50, 
                 model_dir='models', plot_dir='plots'):
        """
        Initialize training pipeline
        
        Args:
            num_episodes: Number of training episodes
            save_freq: Frequency of model saving
            plot_freq: Frequency of plot updates
            model_dir: Directory to save models
            plot_dir: Directory to save plots
        """
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        
        # Initialize environment and agent
        self.env = SnakeGame(speed=50)  # Moderate speed for training
        self.agent = DQNAgent(
            state_size=11,
            action_size=3,
            lr=0.001,
            gamma=0.9,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32,
            target_update=100,
            use_prioritized_replay=False
        )
        
        # Training statistics
        self.scores = []
        self.mean_scores = []
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_history = []
        
        # Best score tracking
        self.best_score = 0
        self.best_model_path = os.path.join(model_dir, 'best_model.pth')
        
        # Initialize plots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('DQN Training Progress', fontsize=16)
        
        print("Training Pipeline Initialized")
        print(f"  Episodes: {num_episodes}")
        print(f"  Save frequency: {save_freq}")
        print(f"  Plot frequency: {plot_freq}")
        print(f"  Model directory: {model_dir}")
        print(f"  Plot directory: {plot_dir}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        for episode in range(1, self.num_episodes + 1):
            # Train one episode
            score, total_reward, losses = self.agent.train_episode(self.env)
            
            # Store statistics
            self.scores.append(score)
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.agent.epsilon)
            
            if losses:
                self.episode_losses.append(np.mean(losses))
            else:
                self.episode_losses.append(0)
            
            # Calculate moving average
            if len(self.scores) >= 10:
                mean_score = np.mean(self.scores[-10:])
                self.mean_scores.append(mean_score)
            else:
                mean_score = np.mean(self.scores)
                self.mean_scores.append(mean_score)
            
            # Print progress
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                stats = self.agent.get_stats()
                
                print(f"Episode {episode:4d}/{self.num_episodes} | "
                      f"Score: {score:3d} | "
                      f"Mean Score: {mean_score:5.2f} | "
                      f"Reward: {total_reward:6.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Loss: {self.episode_losses[-1]:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Save best model
            if score > self.best_score:
                self.best_score = score
                self.agent.save_model(self.best_model_path)
                print(f"  *** NEW BEST SCORE: {score} ***")
            
            # Update plots
            if episode % self.plot_freq == 0:
                self.update_plots(episode)
            
            # Save checkpoint
            if episode % self.save_freq == 0:
                checkpoint_path = os.path.join(self.model_dir, f'checkpoint_episode_{episode}.pth')
                self.agent.save_model(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Final save and plot
        self.agent.save_model(os.path.join(self.model_dir, 'final_model.pth'))
        self.update_plots(self.num_episodes, save=True)
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best score: {self.best_score}")
        print(f"Final mean score: {self.mean_scores[-1]:.2f}")
        print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print(f"{'='*50}")
        
        # Close pygame
        pygame.quit()
    
    def update_plots(self, episode, save=False):
        """Update training progress plots"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Scores over time
        self.ax1.plot(self.scores, alpha=0.6, color='blue', label='Episode Score')
        self.ax1.plot(self.mean_scores, color='red', linewidth=2, label='Mean Score (10 episodes)')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Score')
        self.ax1.set_title(f'Score Progress (Episode {episode})')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rewards and Loss
        if len(self.episode_rewards) > 0:
            ax2_twin = self.ax2.twinx()
            
            # Plot rewards
            self.ax2.plot(self.episode_rewards, color='green', alpha=0.7, label='Total Reward')
            self.ax2.set_xlabel('Episode')
            self.ax2.set_ylabel('Total Reward', color='green')
            self.ax2.tick_params(axis='y', labelcolor='green')
            
            # Plot loss on secondary axis
            if len(self.episode_losses) > 0:
                # Smooth the loss curve
                window = min(50, len(self.episode_losses))
                smoothed_losses = np.convolve(self.episode_losses, 
                                             np.ones(window)/window, mode='valid')
                ax2_twin.plot(smoothed_losses, color='orange', alpha=0.7, label='Loss (smoothed)')
                ax2_twin.set_ylabel('Loss', color='orange')
                ax2_twin.tick_params(axis='y', labelcolor='orange')
            
            self.ax2.set_title(f'Rewards and Loss (Episode {episode})')
            self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Epsilon decay
        self.ax3.plot(self.epsilon_history, color='purple', linewidth=2)
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Epsilon')
        self.ax3.set_title(f'Exploration Rate (Episode {episode})')
        self.ax3.grid(True, alpha=0.3)
        
        # Adjust layout and show
        plt.tight_layout()
        plt.pause(0.01)  # Brief pause to update the plot
        
        # Save plot if requested
        if save:
            plot_path = os.path.join(self.plot_dir, f'training_progress_episode_{episode}.png')
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved: {plot_path}")
    
    def evaluate(self, num_episodes=10, model_path=None):
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: Number of evaluation episodes
            model_path: Path to model file (if None, use current model)
        """
        if model_path:
            self.agent.load_model(model_path)
            print(f"Loaded model: {model_path}")
        
        print(f"\n{'='*30}")
        print("EVALUATION MODE")
        print(f"{'='*30}")
        
        eval_scores = []
        eval_rewards = []
        
        for episode in range(1, num_episodes + 1):
            # Create environment with rendering enabled
            eval_env = SnakeGame(speed=100)  # Slower speed for viewing
            
            # Play episode
            score, total_reward = self.agent.play_episode(eval_env, render=True)
            
            eval_scores.append(score)
            eval_rewards.append(total_reward)
            
            print(f"Eval Episode {episode}: Score = {score}, Reward = {total_reward:.2f}")
            
            # Brief pause between episodes
            time.sleep(1)
        
        # Print evaluation summary
        print(f"\nEvaluation Summary:")
        print(f"  Mean Score: {np.mean(eval_scores):.2f} ± {np.std(eval_scores):.2f}")
        print(f"  Max Score: {np.max(eval_scores)}")
        print(f"  Min Score: {np.min(eval_scores)}")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        
        pygame.quit()

def main():
    """Main function to run training"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create training pipeline
    trainer = TrainingPipeline(
        num_episodes=500,
        save_freq=100,
        plot_freq=25,
        model_dir='models',
        plot_dir='plots'
    )
    
    # Start training
    trainer.train()
    
    # Optional: Evaluate after training
    # trainer.evaluate(num_episodes=5, model_path='models/best_model.pth')

if __name__ == '__main__':
    main()
