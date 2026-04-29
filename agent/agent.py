import numpy as np
import random
from collections import deque

from model.dqn_model import DQN, TargetDQN
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    """
    Deep Q-Learning Agent for Snake Game
    
    Implements epsilon-greedy exploration, experience replay, and target network
    """
    
    def __init__(self, state_size=11, action_size=3, lr=0.001, gamma=0.9, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=32, target_update=100,
                 use_prioritized_replay=False):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state vector (default: 11)
            action_size: Number of possible actions (default: 3)
            lr: Learning rate (default: 0.001)
            gamma: Discount factor (default: 0.9)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Epsilon decay rate (default: 0.995)
            epsilon_min: Minimum epsilon value (default: 0.01)
            memory_size: Replay buffer capacity (default: 10000)
            batch_size: Training batch size (default: 32)
            target_update: Frequency of target network updates (default: 100)
            use_prioritized_replay: Use prioritized experience replay (default: False)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Initialize neural networks
        self.q_network = DQN(state_size, 128, action_size, lr)
        self.target_network = TargetDQN(self.q_network)
        
        # Initialize replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
            self.use_prioritized_replay = True
        else:
            self.memory = ReplayBuffer(memory_size)
            self.use_prioritized_replay = False
        
        # Training statistics
        self.training_step = 0
        self.losses = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.scores = deque(maxlen=1000)
        
        print(f"DQN Agent initialized:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Memory size: {memory_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target update: {target_update}")
        print(f"  Prioritized replay: {use_prioritized_replay}")
    
    def get_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether agent is training (affects exploration)
            
        Returns:
            action: Chosen action
        """
        if training and random.random() <= self.epsilon:
            # Explore: random action
            return random.choice([0, 1, 2])
        else:
            # Exploit: best action based on Q-values
            state = np.array(state)
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train the agent using experience replay
        
        Returns:
            loss: Training loss (None if not enough experiences)
        """
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = np.ones(len(states))
        
        # Convert actions to indices
        action_indices = np.array([np.argmax(a) if isinstance(a, np.ndarray) else a for a in actions])
        
        # Train the network
        loss = self.q_network.train_step(states, action_indices, rewards, next_states, dones, self.gamma)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.update(self.q_network)
            print(f"Target network updated at step {self.training_step}")
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and loss is not None:
            # Calculate TD-errors for priority updates
            with torch.no_grad():
                current_q = self.q_network.q_network(torch.FloatTensor(states))
                next_q = self.target_network.q_network(torch.FloatTensor(next_states))
                target_q = rewards + self.gamma * next_q.max(1)[0] * (~dones)
                td_errors = target_q - current_q.gather(1, torch.LongTensor(action_indices).unsqueeze(1)).squeeze()
                
            self.memory.update_priorities(indices, td_errors.numpy())
            self.memory.anneal_beta()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store training statistics
        if loss is not None:
            self.losses.append(loss)
        
        return loss
    
    def train_episode(self, env, max_steps=10000):
        """
        Train the agent for one episode
        
        Args:
            env: Game environment
            max_steps: Maximum steps per episode
            
        Returns:
            score: Episode score
            total_reward: Total reward for the episode
            losses: List of training losses
        """
        state = env.reset()
        total_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Choose action
            action_idx = self.get_action(state, training=True)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # Take action
            done, reward, score = env.play_step(action)
            next_state = env.get_state()
            
            # Remember experience
            self.remember(state, action_idx, reward, next_state, done)
            
            # Train the agent
            loss = self.train()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Store episode statistics
        self.rewards.append(total_reward)
        self.scores.append(score)
        
        return score, total_reward, losses
    
    def play_episode(self, env, max_steps=10000, render=True):
        """
        Play one episode without training (evaluation mode)
        
        Args:
            env: Game environment
            max_steps: Maximum steps per episode
            render: Whether to render the game
            
        Returns:
            score: Episode score
            total_reward: Total reward for the episode
        """
        state = env.reset()
        total_reward = 0
        
        if not render:
            # Disable rendering for faster evaluation
            original_speed = env.speed
            env.speed = 1000000  # Very high FPS
        
        for step in range(max_steps):
            # Choose action (no exploration)
            action_idx = self.get_action(state, training=False)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # Take action
            done, reward, score = env.play_step(action)
            next_state = env.get_state()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        if not render:
            env.speed = original_speed
        
        return score, total_reward
    
    def get_stats(self):
        """Get training statistics"""
        stats = {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'memory_stats': self.memory.get_stats(),
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'total_episodes': len(self.scores)
        }
        return stats
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.q_network.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.q_network.load(filepath)
        self.target_network.update(self.q_network)

# Import torch for prioritized replay
try:
    import torch
except ImportError:
    print("Warning: PyTorch not installed. Prioritized replay will not work properly.")
    torch = None

if __name__ == '__main__':
    # Test the agent
    from game.snake_game import SnakeGame
    
    # Create environment and agent
    env = SnakeGame(speed=1000)  # Fast speed for testing
    agent = DQNAgent()
    
    print("Testing agent...")
    
    # Test action selection
    state = env.get_state()
    action = agent.get_action(state, training=True)
    print(f"Selected action: {action}")
    
    # Test training episode
    print("Running training episode...")
    score, total_reward, losses = agent.train_episode(env, max_steps=100)
    print(f"Episode completed - Score: {score}, Total Reward: {total_reward}, Losses: {len(losses)}")
    
    # Test statistics
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")
