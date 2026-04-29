import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training
    
    Stores and samples experiences for training the neural network
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store (default: 10000)
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Unzip experiences
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough experiences to sample"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all experiences from the buffer"""
        self.buffer.clear()
    
    def get_stats(self):
        """Get buffer statistics"""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'fullness': len(self.buffer) / self.capacity
        }

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples experiences based on their TD-error for more efficient learning
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Rate at which beta approaches 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add new experience with maximum priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample experiences based on priorities
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Unzip experiences
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD-errors
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD-errors for those experiences
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Add small constant to avoid zero priority
            self.priorities[idx] = priority
    
    def anneal_beta(self):
        """Gradually increase beta towards 1"""
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size):
        return self.size >= batch_size
    
    def get_stats(self):
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fullness': self.size / self.capacity,
            'alpha': self.alpha,
            'beta': self.beta
        }

if __name__ == '__main__':
    # Test the replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy experiences
    for i in range(50):
        state = np.random.randn(11)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(11)
        done = np.random.choice([True, False])
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Sample a batch
    if buffer.is_ready(32):
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"Sampled batch shapes:")
        print(f"States: {states.shape}")
        print(f"Actions: {actions.shape}")
        print(f"Rewards: {rewards.shape}")
        print(f"Next states: {next_states.shape}")
        print(f"Dones: {dones.shape}")
