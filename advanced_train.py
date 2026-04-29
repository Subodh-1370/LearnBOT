import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
import argparse

from game.enhanced_state_game import EnhancedStateGame
from agent.advanced_agent import AdvancedAgent

class AdvancedTrainingPipeline:
    """
    Advanced training pipeline with multiple DQN architectures and reward functions
    """
    
    def __init__(self, num_episodes=1000, save_freq=100, plot_freq=25, 
                 model_dir='models', plot_dir='plots', dqn_type='double', 
                 reward_function='standard', use_enhanced_state=True):
        """
        Initialize advanced training pipeline
        
        Args:
            num_episodes: Number of training episodes
            save_freq: Frequency of model saving
            plot_freq: Frequency of plot updates
            model_dir: Directory to save models
            plot_dir: Directory to save plots
            dqn_type: Type of DQN ('double', 'dueling', 'noisy', 'standard')
            reward_function: Type of reward function ('standard', 'shaped', 'sparse', 'dense')
            use_enhanced_state: Use enhanced state representation
        """
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.dqn_type = dqn_type
        self.reward_function = reward_function
        self.use_enhanced_state = use_enhanced_state
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Initialize environment and agent
        if use_enhanced_state:
            self.env = EnhancedStateGame(speed=50, headless=True, enable_animations=False)
            state_size = 20  # Enhanced state has 20 features
        else:
            from game.enhanced_snake_game import EnhancedSnakeGame
            self.env = EnhancedSnakeGame(speed=50, headless=True, enable_animations=False)
            state_size = 11  # Standard state has 11 features
        
        self.agent = AdvancedAgent(
            state_size=state_size,
            action_size=3,
            lr=0.0005,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=50000,
            batch_size=64,
            target_update=100,
            dqn_type=dqn_type,
            use_prioritized_replay=True
        )
        
        # Training statistics
        self.scores = []
        self.mean_scores = []
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_history = []
        self.episode_steps = []
        
        # Initialize plots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Advanced Training - {dqn_type.upper()} DQN - {reward_function.upper()} Rewards', fontsize=16)
        
        print(f"Advanced Training Pipeline initialized:")
        print(f"  Episodes: {num_episodes}")
        print(f"  DQN Type: {dqn_type}")
        print(f"  Reward Function: {reward_function}")
        print(f"  Enhanced State: {use_enhanced_state}")
        print(f"  State Size: {state_size}")
    
    def calculate_reward(self, old_score, new_score, steps_since_last_food, distance_to_food, done):
        """
        Calculate reward based on selected reward function
        
        Args:
            old_score: Previous score
            new_score: Current score
            steps_since_last_food: Steps since last food
            distance_to_food: Distance to food
            done: Game over flag
            
        Returns:
            reward: Calculated reward
        """
        if self.reward_function == 'standard':
            # Standard reward function
            if done:
                return -10
            elif new_score > old_score:
                return 10
            else:
                return -0.1
                
        elif self.reward_function == 'shaped':
            # Shaped reward function with distance-based rewards
            if done:
                return -10
            elif new_score > old_score:
                return 10
            else:
                # Reward for getting closer to food
                max_distance = self.env.w + self.env.h
                distance_reward = -0.01 * (distance_to_food / max_distance)
                
                # Penalty for taking too long
                time_penalty = -0.001 * steps_since_last_food
                
                return distance_reward + time_penalty
                
        elif self.reward_function == 'sparse':
            # Sparse reward function - only reward for food
            if done:
                return -10
            elif new_score > old_score:
                return 10
            else:
                return 0  # No penalty for movement
                
        elif self.reward_function == 'dense':
            # Dense reward function with many feedback signals
            if done:
                return -10
            elif new_score > old_score:
                return 10
            else:
                # Small positive reward for survival
                survival_reward = 0.01
                
                # Distance-based reward
                max_distance = self.env.w + self.env.h
                distance_reward = -0.005 * (distance_to_food / max_distance)
                
                # Time-based penalty
                time_penalty = -0.0005 * steps_since_last_food
                
                return survival_reward + distance_reward + time_penalty
        
        return 0
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"STARTING ADVANCED TRAINING")
        print(f"{'='*60}")
        print(f"DQN Type: {self.dqn_type}")
        print(f"Reward Function: {self.reward_function}")
        print(f"Enhanced State: {self.use_enhanced_state}")
        print(f"{'='*60}")
        
        start_time = time.time()
        best_score = 0
        
        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            old_score = 0
            total_reward = 0
            episode_losses = []
            steps = 0
            
            while True:
                # Get action
                action_idx = self.agent.get_action(state, training=True)
                action = [0, 0, 0]
                action[action_idx] = 1
                
                # Calculate distance to food before move
                head = self.env.head
                food = self.env.food
                distance_to_food = abs(head[0] - food[0]) + abs(head[1] - food[1])
                
                # Take action
                done, base_reward, new_score = self.env.play_step(action)
                
                # Calculate custom reward
                reward = self.calculate_reward(
                    old_score, new_score, self.env.steps_since_last_food, 
                    distance_to_food, done
                )
                
                next_state = self.env.get_enhanced_state() if self.use_enhanced_state else self.env.get_state()
                
                # Store experience
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                total_reward += reward
                old_score = new_score
                steps += 1
                
                if done:
                    break
            
            # Update statistics
            self.scores.append(new_score)
            self.episode_rewards.append(total_reward)
            self.episode_losses.extend(episode_losses)
            self.epsilon_history.append(self.agent.epsilon)
            self.episode_steps.append(steps)
            
            # Calculate mean score
            if len(self.scores) >= 10:
                mean_score = np.mean(self.scores[-10:])
            else:
                mean_score = np.mean(self.scores)
            
            self.mean_scores.append(mean_score)
            
            # Update best score
            if new_score > best_score:
                best_score = new_score
                self.agent.save_model(os.path.join(self.model_dir, 'best_model.pth'))
                print(f"  *** NEW BEST SCORE: {best_score} ***")
            
            # Print progress
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode:4d}/{self.num_episodes:4d} | "
                  f"Score: {new_score:3d} | "
                  f"Mean Score: {mean_score:6.2f} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Epsilon: {self.agent.epsilon:6.3f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Time: {elapsed_time:5.1f}s")
            
            # Update plots
            if episode % self.plot_freq == 0:
                self.update_plots(episode)
            
            # Save model
            if episode % self.save_freq == 0:
                model_path = os.path.join(self.model_dir, f'checkpoint_episode_{episode}.pth')
                self.agent.save_model(model_path)
                print(f"  Checkpoint saved: {model_path}")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        # Final plot update
        self.update_plots(self.num_episodes)
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best score: {best_score}")
        print(f"Final mean score: {self.mean_scores[-1]:.2f}")
        print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print(f"{'='*60}")
        
        return self.scores, self.mean_scores
    
    def update_plots(self, episode):
        """Update training plots"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Score progression
        self.ax1.plot(self.scores, alpha=0.6, label='Episode Score')
        self.ax1.plot(self.mean_scores, 'r-', linewidth=2, label='Mean Score (10 episodes)')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Score')
        self.ax1.set_title('Score Progression')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot 2: Reward progression
        if len(self.episode_rewards) > 0:
            # Calculate moving average
            window_size = min(50, len(self.episode_rewards))
            if len(self.episode_rewards) >= window_size:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
                self.ax2.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
                self.ax2.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            else:
                self.ax2.plot(self.episode_rewards, label='Episode Reward')
        
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Total Reward')
        self.ax2.set_title('Reward Progression')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Plot 3: Epsilon decay
        self.ax3.plot(self.epsilon_history, 'g-', linewidth=2)
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Epsilon')
        self.ax3.set_title('Exploration Rate (Epsilon)')
        self.ax3.grid(True)
        
        # Plot 4: Loss progression
        if len(self.episode_losses) > 0:
            # Sample losses for plotting (too many to plot all)
            sample_rate = max(1, len(self.episode_losses) // 1000)
            sampled_losses = self.episode_losses[::sample_rate]
            self.ax4.plot(sampled_losses, alpha=0.6, label='Training Loss')
            
            # Calculate moving average
            window_size = min(100, len(sampled_losses))
            if len(sampled_losses) >= window_size:
                moving_avg = np.convolve(sampled_losses, np.ones(window_size)/window_size, mode='valid')
                self.ax4.plot(range(window_size-1, len(sampled_losses)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
        
        self.ax4.set_xlabel('Training Step')
        self.ax4.set_ylabel('Loss')
        self.ax4.set_title('Training Loss')
        self.ax4.legend()
        self.ax4.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        
        plot_path = os.path.join(self.plot_dir, f'advanced_training_{self.dqn_type}_{self.reward_function}_episode_{episode}.png')
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if episode % self.plot_freq == 0:
            print(f"Training plot saved: {plot_path}")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Snake Game Training')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--dqn-type', type=str, default='double', 
                       choices=['double', 'dueling', 'noisy', 'standard'],
                       help='Type of DQN architecture')
    parser.add_argument('--reward-function', type=str, default='standard',
                       choices=['standard', 'shaped', 'sparse', 'dense'],
                       help='Type of reward function')
    parser.add_argument('--enhanced-state', action='store_true', help='Use enhanced state representation')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Plot directory')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create training pipeline
    trainer = AdvancedTrainingPipeline(
        num_episodes=args.episodes,
        save_freq=100,
        plot_freq=25,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        dqn_type=args.dqn_type,
        reward_function=args.reward_function,
        use_enhanced_state=args.enhanced_state
    )
    
    # Start training
    scores, mean_scores = trainer.train()
    
    print(f"\nTraining completed successfully!")
    print(f"Final mean score: {mean_scores[-1]:.2f}")
    print(f"Best score achieved: {max(scores)}")

if __name__ == '__main__':
    main()
