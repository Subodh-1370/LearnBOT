# LearnBot - Reinforcement Learning Game Agent

An AI agent that learns to play the Snake game using **Deep Q-Learning (DQN)** without hardcoded logic.

## [![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![PyGame](https://img.shields.io/badge/PyGame-2.1%2B-green.svg)](https://www.pygame.org/)

## Overview

LearnBot demonstrates the power of reinforcement learning by training an AI agent to master the classic Snake game. The agent uses engineered state features and Deep Q-Networks to learn optimal playing strategies through trial and error.

### Key Features

- **Pure DQN Implementation**: No external RL libraries - everything built from scratch
- **Engineered State Representation**: Smart feature extraction instead of raw pixels
- **Experience Replay**: Stable training through memory replay
- **Target Network**: Improved learning stability
- **Real-time Visualization**: Live training progress monitoring
- **Modular Architecture**: Clean, maintainable codebase

## Project Structure

```
LearnBot/
|
|--- game/
|    |--- snake_game.py          # Snake game environment
|
|--- model/
|    |--- dqn_model.py           # Deep Q-Network implementation
|
|--- agent/
|    |--- agent.py               # DQN agent with training logic
|
|--- utils/
|    |--- replay_buffer.py       # Experience replay buffer
|
|--- train.py                   # Training pipeline with visualization
|--- play.py                    # Play mode for trained models
|--- requirements.txt           # Project dependencies
|--- README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project**
   ```bash
   # Navigate to project directory
   cd LearnBOT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch, pygame, numpy, matplotlib; print('All dependencies installed successfully!')"
   ```

## Quick Start

### 1. Train the Agent

Start training the AI agent from scratch:

```bash
python train.py
```

The training will:
- Run for 500 episodes by default
- Display real-time progress plots
- Save the best model automatically
- Create checkpoints every 100 episodes

### 2. Watch the Trained Agent Play

After training, watch your AI play:

```bash
python play.py --model models/best_model.pth
```

### 3. Benchmark Performance

Test the agent's performance:

```bash
python play.py --model models/best_model.pth --benchmark --num-games 100
```

## Detailed Usage

### Training Options

Customize training parameters:

```bash
python train.py
```

Training parameters can be modified in `train.py`:
- `num_episodes`: Number of training episodes (default: 500)
- `save_freq`: Model saving frequency (default: 100)
- `plot_freq`: Plot update frequency (default: 25)

### Play Mode Options

```bash
python play.py [OPTIONS]
```

Options:
- `--model PATH`: Path to trained model (default: `models/best_model.pth`)
- `--speed FPS`: Game speed (default: 50)
- `--no-info`: Disable additional information display
- `--benchmark`: Run benchmark mode
- `--num-games N`: Number of games for benchmark (default: 100)
- `--games N`: Number of games to play in continuous mode

Examples:
```bash
# Play with best model at normal speed
python play.py

# Play faster without extra info
python play.py --speed 100 --no-info

# Run benchmark on 1000 games
python play.py --benchmark --num-games 1000

# Play exactly 10 games then quit
python play.py --games 10
```

## How It Works

### 1. Game Environment

The Snake game is implemented using PyGame with:
- Grid-based movement (32x24 grid)
- Collision detection (walls and self)
- Score tracking
- Food spawning mechanics

### 2. State Representation

Instead of raw pixels, we use engineered features:

```
State Vector (11 features):
- [0-2] Danger detection (straight, left, right)
- [3-6] Current direction (left, right, up, down)
- [7-10] Food position relative to head (left, right, up, down)
```

### 3. Action Space

Three discrete actions:
- **[1, 0, 0]**: Go straight
- **[0, 1, 0]**: Turn right
- **[0, 0, 1]**: Turn left

### 4. Reward Function

- **+10**: Eating food
- **-10**: Collision (game over)
- **-0.1**: Each step (encourages efficiency)

### 5. Deep Q-Network Architecture

```
Input Layer (11 neurons)
    |
    v
Hidden Layer 1 (128 neurons, ReLU)
    |
    v
Hidden Layer 2 (128 neurons, ReLU)
    |
    v
Output Layer (3 neurons - Q-values for each action)
```

### 6. Training Algorithm

The agent uses Deep Q-Learning with:

- **Experience Replay**: Stores and samples past experiences
- **Target Network**: Stabilizes training
- **Epsilon-Greedy Policy**: Balances exploration and exploitation
- **Bellman Equation**: Q(s,a) = r + × max(Q(s',a'))

## Training Process

### Phase 1: Random Exploration
- Epsilon = 1.0 (100% random actions)
- Agent explores the environment
- Experiences stored in replay buffer

### Phase 2: Learning Phase
- Epsilon gradually decays to 0.01
- Agent learns from replay buffer
- Q-values converge to optimal policy

### Phase 3: Exploitation
- Epsilon = 0.01 (mostly greedy actions)
- Agent uses learned policy
- Performance stabilizes

## Performance Metrics

During training, you'll see:
- **Episode Score**: Points earned in each game
- **Mean Score**: Rolling average (10 episodes)
- **Total Reward**: Cumulative reward per episode
- **Epsilon**: Current exploration rate
- **Loss**: Training loss

## Expected Results

With proper training, the agent should achieve:
- **Average Score**: 15-25 points
- **Best Score**: 40+ points
- **Learning Time**: 30-60 minutes (500 episodes)

## Troubleshooting

### Common Issues

1. **PyGame Display Issues**
   ```bash
   # On Linux, install SDL dependencies
   sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
   ```

2. **CUDA Issues with PyTorch**
   ```bash
   # Install CPU-only PyTorch if CUDA issues occur
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Slow Training**
   - Reduce `num_episodes` for testing
   - Increase game speed in `train.py`
   - Use smaller `batch_size`

4. **Poor Performance**
   - Increase training episodes
   - Adjust learning rate (try 0.0005)
   - Modify reward function
   - Increase replay buffer size

### Performance Tips

- **GPU Training**: Install CUDA-enabled PyTorch for faster training
- **Batch Size**: Larger batches (64-128) can improve stability
- **Learning Rate**: Try 0.0005 for more stable learning
- **Network Size**: Increase hidden layers to 256 for complex strategies

## Advanced Features

### Target Network Updates

The agent uses a target network that updates every 100 steps for stable training.

### Prioritized Experience Replay

Optional feature for more efficient learning (can be enabled in `agent.py`).

### Custom Reward Functions

Modify the reward function in `snake_game.py` to experiment with different training objectives.

## Contributing

Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share training results

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Deep Q-Learning algorithm by DeepMind
- PyGame for game development
- PyTorch for deep learning
- OpenAI Gym inspiration for environment design

---

**Happy Training!** Watch your AI evolve from random movements to intelligent Snake-playing strategies!
