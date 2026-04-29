import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DoubleDQN(nn.Module):
    """
    Double Deep Q-Network for more stable Q-value estimation
    
    Double DQN reduces overestimation bias by using separate networks
    for action selection and value evaluation.
    """
    
    def __init__(self, input_size=11, hidden_size=128, output_size=3, learning_rate=0.001):
        super(DoubleDQN, self).__init__()
        
        # Main network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Target network
        self.target_fc1 = nn.Linear(input_size, hidden_size)
        self.target_fc2 = nn.Linear(hidden_size, hidden_size)
        self.target_fc3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Initialize target network
        self.update_target_network()
    
    def forward(self, x):
        """Forward pass through main network"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def target_forward(self, x):
        """Forward pass through target network"""
        x = self.relu(self.target_fc1(x))
        x = self.relu(self.target_fc2(x))
        x = self.target_fc3(x)
        return x
    
    def update_target_network(self):
        """Copy main network weights to target network"""
        self.target_fc1.load_state_dict(self.fc1.state_dict())
        self.target_fc2.load_state_dict(self.fc2.state_dict())
        self.target_fc3.load_state_dict(self.fc3.state_dict())
    
    def train_step(self, states, actions, rewards, next_states, dones, gamma=0.95):
        """
        Train using Double DQN algorithm
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Terminal flags
            gamma: Discount factor
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target calculation
        with torch.no_grad():
            # Use main network to select best actions
            next_q_values_main = self.forward(next_states)
            next_actions = next_q_values_main.max(1)[1]
            
            # Use target network to evaluate selected actions
            next_q_values_target = self.target_forward(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            # Calculate target Q-values
            target_q_values = rewards + (gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture
    
    Separates state value and advantage estimation for better learning efficiency.
    """
    
    def __init__(self, input_size=11, hidden_size=128, output_size=3, learning_rate=0.001):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass through dueling architecture"""
        # Shared layers
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))
        
        # Value stream
        value = self.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = self.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class NoisyDQN(nn.Module):
    """
    Noisy Deep Q-Network with exploration through noisy linear layers
    
    Replaces epsilon-greedy exploration with learned exploration through noise.
    """
    
    def __init__(self, input_size=11, hidden_size=128, output_size=3, learning_rate=0.001):
        super(NoisyDQN, self).__init__()
        
        # Noisy linear layers
        self.fc1 = NoisyLinear(input_size, hidden_size)
        self.fc2 = NoisyLinear(hidden_size, hidden_size)
        self.fc3 = NoisyLinear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass through noisy network"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Register buffer for noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise parameters"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """Scale noise for stable training"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return torch.nn.functional.linear(x, weight, bias)

# Factory function for creating different DQN types
def create_dqn(dqn_type='double', **kwargs):
    """
    Factory function to create different types of DQN networks
    
    Args:
        dqn_type: Type of DQN ('double', 'dueling', 'noisy', 'standard')
        **kwargs: Additional parameters for network initialization
    
    Returns:
        DQN network instance
    """
    if dqn_type == 'double':
        return DoubleDQN(**kwargs)
    elif dqn_type == 'dueling':
        return DuelingDQN(**kwargs)
    elif dqn_type == 'noisy':
        return NoisyDQN(**kwargs)
    else:
        # Return standard DQN
        from model.dqn_model import DQN
        return DQN(**kwargs)

if __name__ == '__main__':
    # Test different DQN types
    input_size = 11
    output_size = 3
    
    print("Testing different DQN architectures...")
    
    # Test Double DQN
    double_dqn = DoubleDQN(input_size, 128, output_size)
    print(f"Double DQN created with {sum(p.numel() for p in double_dqn.parameters())} parameters")
    
    # Test Dueling DQN
    dueling_dqn = DuelingDQN(input_size, 128, output_size)
    print(f"Dueling DQN created with {sum(p.numel() for p in dueling_dqn.parameters())} parameters")
    
    # Test Noisy DQN
    noisy_dqn = NoisyDQN(input_size, 128, output_size)
    print(f"Noisy DQN created with {sum(p.numel() for p in noisy_dqn.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, input_size)
    
    with torch.no_grad():
        double_output = double_dqn(dummy_input)
        dueling_output = dueling_dqn(dummy_input)
        noisy_output = noisy_dqn(dummy_input)
        
        print(f"Double DQN output shape: {double_output.shape}")
        print(f"Dueling DQN output shape: {dueling_output.shape}")
        print(f"Noisy DQN output shape: {noisy_output.shape}")
    
    print("All DQN types working correctly!")
