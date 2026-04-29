import pygame
import numpy as np
import os
import time
import argparse

from game.enhanced_snake_game import EnhancedSnakeGame
from agent.agent import DQNAgent

class PlayMode:
    """
    Play mode for watching trained AI agent play Snake
    """
    
    def __init__(self, model_path=None, speed=50, display_info=True):
        """
        Initialize play mode
        
        Args:
            model_path: Path to trained model file
            speed: Game speed (FPS)
            display_info: Whether to display additional information
        """
        self.model_path = model_path
        self.speed = speed
        self.display_info = display_info
        
        # Initialize environment
        self.env = EnhancedSnakeGame(speed=speed, headless=False, enable_animations=True)  # Full visuals for play mode
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=11,
            action_size=3,
            lr=0.001,
            gamma=0.9,
            epsilon=0.0,  # No exploration in play mode
            epsilon_decay=1.0,
            epsilon_min=0.0,
            memory_size=1000,
            batch_size=32,
            target_update=100
        )
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
            print(f"Loaded model: {model_path}")
        else:
            print("Warning: No model loaded. Using random policy.")
        
        # Statistics
        self.total_games = 0
        self.total_score = 0
        self.best_score = 0
        self.current_streak = 0
        self.best_streak = 0
        
        # Font for additional info
        if display_info:
            pygame.font.init()
            self.info_font = pygame.font.SysFont('arial', 20)
    
    def play_single_game(self, max_steps=10000):
        """
        Play a single game
        
        Args:
            max_steps: Maximum steps per game
            
        Returns:
            score: Game score
            steps: Number of steps taken
        """
        state = self.env.reset()
        steps = 0
        
        while steps < max_steps:
            # Get action from agent
            action_idx = self.agent.get_action(state, training=False)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # Take action
            done, reward, score = self.env.play_step(action)
            next_state = self.env.get_state()
            
            state = next_state
            steps += 1
            
            # Display additional info if enabled
            if self.display_info:
                self._display_info(score, steps, action_idx)
            
            if done:
                break
        
        return score, steps
    
    def _display_info(self, score, steps, action_idx):
        """Display additional game information"""
        info_texts = [
            f"Steps: {steps}",
            f"Action: {['Straight', 'Left', 'Right'][action_idx]}",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Games: {self.total_games}",
            f"Best Score: {self.best_score}"
        ]
        
        y_offset = 30
        for text in info_texts:
            surface = self.info_font.render(text, True, (255, 255, 255))
            self.env.display.blit(surface, (10, y_offset))
            y_offset += 25
    
    def play_continuous(self, num_games=None):
        """
        Play continuously until user quits
        
        Args:
            num_games: Number of games to play (None for infinite)
        """
        print(f"\n{'='*40}")
        print("PLAY MODE - Press ESC or close window to quit")
        print(f"{'='*40}")
        print(f"Model: {self.model_path or 'Random'}")
        print(f"Speed: {self.speed} FPS")
        print(f"Display Info: {self.display_info}")
        print(f"{'='*40}")
        
        game_count = 0
        
        try:
            while True:
                # Check if we should stop
                if num_games and game_count >= num_games:
                    break
                
                # Play one game
                score, steps = self.play_single_game()
                
                # Update statistics
                self.total_games += 1
                self.total_score += score
                
                if score > self.best_score:
                    self.best_score = score
                    self.current_streak = 1
                    print(f"  *** NEW BEST SCORE: {score} ***")
                elif score == self.best_score:
                    self.current_streak += 1
                else:
                    self.current_streak = 0
                
                if self.current_streak > self.best_streak:
                    self.best_streak = self.current_streak
                
                game_count += 1
                
                # Print game summary
                avg_score = self.total_score / self.total_games
                print(f"Game {game_count}: Score = {score}, Steps = {steps}, "
                      f"Avg Score = {avg_score:.2f}, Best = {self.best_score}")
                
                # Brief pause between games
                time.sleep(1)
                
                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print("User quit")
                        return
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        # Final statistics
        print(f"\n{'='*40}")
        print("SESSION SUMMARY")
        print(f"{'='*40}")
        print(f"Total Games: {self.total_games}")
        print(f"Average Score: {self.total_score / self.total_games:.2f}")
        print(f"Best Score: {self.best_score}")
        print(f"Best Streak: {self.best_streak}")
        print(f"{'='*40}")
    
    def benchmark(self, num_games=100):
        """
        Benchmark the agent's performance
        
        Args:
            num_games: Number of games to benchmark
        """
        print(f"\n{'='*40}")
        print(f"BENCHMARK MODE - {num_games} games")
        print(f"{'='*40}")
        
        scores = []
        steps_list = []
        
        # Disable display for faster benchmarking
        original_speed = self.env.speed
        self.env.speed = 1000000  # Very high FPS
        
        for game in range(1, num_games + 1):
            score, steps = self.play_single_game()
            scores.append(score)
            steps_list.append(steps)
            
            if game % 10 == 0:
                avg_score = np.mean(scores)
                print(f"Game {game:3d}: Score = {score:3d}, Avg = {avg_score:.2f}")
        
        # Restore original speed
        self.env.speed = original_speed
        
        # Calculate statistics
        scores = np.array(scores)
        steps_list = np.array(steps_list)
        
        print(f"\n{'='*40}")
        print("BENCHMARK RESULTS")
        print(f"{'='*40}")
        print(f"Games: {num_games}")
        print(f"Score Statistics:")
        print(f"  Mean:     {np.mean(scores):.2f}")
        print(f"  Std Dev:  {np.std(scores):.2f}")
        print(f"  Min:      {np.min(scores)}")
        print(f"  Max:      {np.max(scores)}")
        print(f"  Median:   {np.median(scores):.2f}")
        print(f"Steps Statistics:")
        print(f"  Mean:     {np.mean(steps_list):.1f}")
        print(f"  Std Dev:  {np.std(steps_list):.1f}")
        print(f"  Min:      {np.min(steps_list)}")
        print(f"  Max:      {np.max(steps_list)}")
        
        # Score distribution
        score_counts = {}
        for score in scores:
            score_counts[score] = score_counts.get(score, 0) + 1
        
        print(f"\nScore Distribution:")
        for score in sorted(score_counts.keys()):
            count = score_counts[score]
            percentage = (count / num_games) * 100
            print(f"  {score:2d}: {count:3d} games ({percentage:5.1f}%)")
        
        print(f"{'='*40}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Play with trained DQN agent')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--speed', type=int, default=50,
                       help='Game speed (FPS)')
    parser.add_argument('--no-info', action='store_true',
                       help='Disable additional info display')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games for benchmark')
    parser.add_argument('--games', type=int, default=None,
                       help='Number of games to play in continuous mode')
    
    args = parser.parse_args()
    
    # Create play mode
    player = PlayMode(
        model_path=args.model,
        speed=args.speed,
        display_info=not args.no_info
    )
    
    if args.benchmark:
        # Run benchmark
        player.benchmark(num_games=args.num_games)
    else:
        # Play continuously
        player.play_continuous(num_games=args.games)

if __name__ == '__main__':
    main()
