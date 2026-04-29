import numpy as np
import random
import sys
import os
import torch
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.advanced_dqn import create_dqn
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class AdvancedAgent:
    """
    Advanced RL Agent with multiple DQN architectures and prioritized experience replay
    """
    
    def __init__(self, state_size=11, action_size=3, lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=50000, batch_size=64, target_update=100,
                 dqn_type='double', use_prioritized_replay=True,
                 double_dqn_update_freq=1000):
        """
        Initialize Advanced Agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            memory_size: Replay buffer capacity
            batch_size: Training batch size
            target_update: Frequency of target network updates
            dqn_type: Type of DQN ('double', 'dueling', 'noisy', 'standard')
            use_prioritized_replay: Use prioritized experience replay
            double_dqn_update_freq: Frequency for Double DQN target updates
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
        self.dqn_type = dqn_type
        self.use_prioritized_replay = use_prioritized_replay
        self.double_dqn_update_freq = double_dqn_update_freq
        
        # Initialize neural networks
        self.q_network = create_dqn(
            dqn_type=dqn_type,
            input_size=state_size,
            hidden_size=128,
            output_size=action_size,
            learning_rate=lr
        )
        
        # Initialize replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)
        
        # Training statistics
        self.training_step = 0
        self.losses = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.scores = deque(maxlen=1000)
        
        # For Double DQN
        if dqn_type == 'double':
            self.last_target_update = 0
        
        print(f"Advanced Agent initialized:")
        print(f"  DQN Type: {dqn_type}")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Memory size: {memory_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Prioritized replay: {use_prioritized_replay}")
        print(f"  Target update: {target_update}")
    
    def get_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether agent is training
            
        Returns:
            action: Chosen action
        """
        if training and random.random() <= self.epsilon:
            # Explore: random action
            return random.choice([0, 1, 2])
        else:
            # Exploit: best action based on Q-values
            state = np.array(state)
            q_values = self.predict(state)
            return np.argmax(q_values[0])
    
    def predict(self, state):
        """
        Predict Q-values for given state
        
        Args:
            state: Current state
            
        Returns:
            q_values: Predicted Q-values
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experience replay"""
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
        if self.dqn_type == 'double':
            loss = self._train_double_dqn(states, action_indices, rewards, next_states, dones, weights)
        elif self.dqn_type == 'noisy':
            loss = self._train_noisy_dqn(states, action_indices, rewards, next_states, dones, weights)
        else:
            loss = self._train_standard_dqn(states, action_indices, rewards, next_states, dones, weights)
        
        # Update target network
        self.training_step += 1
        if self.dqn_type == 'double' and self.training_step % self.double_dqn_update_freq == 0:
            self.q_network.update_target_network()
            print(f"Double DQN target network updated at step {self.training_step}")
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and loss is not None:
            self._update_priorities(indices, states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store training statistics
        if loss is not None:
            self.losses.append(loss)
        
        return loss
    
    def _train_standard_dqn(self, states, actions, rewards, next_states, dones, weights):
        """Train standard DQN"""
        return self.q_network.train_step(states, actions, rewards, next_states, dones, self.gamma)
    
    def _train_double_dqn(self, states, actions, rewards, next_states, dones, weights):
        """Train Double DQN"""
        return self.q_network.train_step(states, actions, rewards, next_states, dones, self.gamma)
    
    def _train_noisy_dqn(self, states, actions, rewards, next_states, dones, weights):
        """Train Noisy DQN"""
        # Reset noise for exploration
        self.q_network.reset_noise()
        
        # Train with standard DQN loss
        loss = self.q_network.train_step(states, actions, rewards, next_states, dones, self.gamma)
        
        return loss
    
    def _update_priorities(self, indices, states, actions, rewards, next_states, dones):
        """Update priorities for prioritized experience replay"""
        if not self.use_prioritized_replay:
            return
        
        # Calculate TD-errors for priority updates
        with torch.no_grad():
            # Convert to tensors
            states_tensor = torch.FloatTensor(states)
            next_states_tensor = torch.FloatTensor(next_states)
            rewards_tensor = torch.FloatTensor(rewards)
            dones_tensor = torch.BoolTensor(dones)
            actions_tensor = torch.LongTensor(actions)
            
            # Use the correct network reference based on DQN type
            if self.dqn_type == 'double':
                current_q = self.q_network(states_tensor)
                next_q = self.q_network(next_states_tensor)
            else:
                current_q = self.q_network(states_tensor)
                next_q = self.q_network(next_states_tensor)
            
            target_q = rewards_tensor + self.gamma * next_q.max(1)[0] * (~dones_tensor)
            td_errors = target_q - current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            
        self.memory.update_priorities(indices, td_errors.numpy())
        self.memory.anneal_beta()
    
    def train_episode(self, env, max_steps=10000):
        """Train the agent for one episode"""
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
        """Play one episode without training"""
        state = env.reset()
        total_reward = 0
        
        if not render:
            original_speed = env.speed
            env.speed = 1000000
        
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
            'dqn_type': self.dqn_type,
            'memory_stats': self.memory.get_stats(),
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'total_episodes': len(self.scores)
        }
        return stats
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.q_network.device))
        self.q_network.eval()

# Factory function for creating different agent types
def create_agent(agent_type='advanced', **kwargs):
    """
    Factory function to create different types of agents
    
    Args:
        agent_type: Type of agent ('advanced', 'standard')
        **kwargs: Additional parameters for agent initialization
    
    Returns:
        Agent instance
    """
    if agent_type == 'advanced':
        return AdvancedAgent(**kwargs)
    else:
        # Return standard agent
        from agent.agent import DQNAgent
        return DQNAgent(**kwargs)

if __name__ == '__main__':
    # Test different agent types
    print("Testing different agent types...")
    
    # Test Advanced Agent with Double DQN
    agent1 = AdvancedAgent(dqn_type='double', use_prioritized_replay=True)
    print(f"Advanced Agent (Double DQN) created")
    
    # Test Advanced Agent with Dueling DQN
    agent2 = AdvancedAgent(dqn_type='dueling', use_prioritized_replay=True)
    print(f"Advanced Agent (Dueling DQN) created")
    
    # Test Advanced Agent with Noisy DQN
    agent3 = AdvancedAgent(dqn_type='noisy', use_prioritized_replay=False)
    print(f"Advanced Agent (Noisy DQN) created")
    
    print("All agent types working correctly!")
