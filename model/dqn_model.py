import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """
    Deep Q Network for Snake Game
    
    Architecture:
    - Input: State vector (11 features)
    - Hidden Layer 1: 128 neurons with ReLU
    - Hidden Layer 2: 128 neurons with ReLU  
    - Output: Q-values for 3 actions [Straight, Left, Right]
    """
    
    def __init__(self, input_size=11, hidden_size=128, output_size=3, learning_rate=0.001):
        """
        Initialize DQN
        
        Args:
            input_size: Size of state vector (default: 11)
            hidden_size: Number of neurons in hidden layers (default: 128)
            output_size: Number of possible actions (default: 3)
            learning_rate: Learning rate for optimizer (default: 0.001)
        """
        super(DQN, self).__init__()
        
        # Define network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, state):
        """
        Get Q-values for a given state
        
        Args:
            state: Input state (numpy array or tensor)
            
        Returns:
            Q-values as numpy array
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.forward(state)
            
        return q_values.cpu().numpy()
    
    def train_step(self, states, actions, rewards, next_states, dones, gamma=0.9):
        """
        Train the network using a batch of experiences
        
        Args:
            states: Batch of current states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of terminal flags
            gamma: Discount factor (default: 0.9)
            
        Returns:
            loss: Training loss for this batch
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Get current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q-values (target network would be better, but using same network for simplicity)
        next_q_values = self.forward(next_states).max(1)[0].detach()
        
        # Calculate target Q-values using Bellman equation
        target_q_values = rewards + (gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath):
        """
        Save model state
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model state
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

class TargetDQN(DQN):
    """
    Target Network for more stable training
    This network is updated less frequently than the main network
    """
    
    def __init__(self, main_network):
        """
        Initialize target network with same architecture as main network
        
        Args:
            main_network: Main DQN network to copy parameters from
        """
        super(TargetDQN, self).__init__()
        # Copy architecture from main network
        self.load_state_dict(main_network.state_dict())
        
        # Freeze target network parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def update(self, main_network):
        """
        Update target network with main network parameters
        
        Args:
            main_network: Main DQN network to copy parameters from
        """
        self.load_state_dict(main_network.state_dict())

if __name__ == '__main__':
    # Test the DQN model
    model = DQN()
    
    # Create dummy state (11 features)
    dummy_state = torch.randn(1, 11)
    
    # Forward pass
    q_values = model(dummy_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    
    # Test prediction
    dummy_state_np = dummy_state.numpy()
    predictions = model.predict(dummy_state_np)
    print(f"Predictions: {predictions}")
