import numpy as np
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from game.enhanced_snake_game import EnhancedSnakeGame
from agent.agent import DQNAgent

class ModelBenchmark:
    """
    Comprehensive benchmark system for comparing different models and configurations
    """
    
    def __init__(self, results_dir='benchmark_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}
        
    def benchmark_model(self, model_path, num_games=100, game_speed=1000):
        """
        Benchmark a single model
        
        Args:
            model_path: Path to the model file
            num_games: Number of games to test
            game_speed: Game speed for faster testing
            
        Returns:
            dict: Benchmark results
        """
        print(f"Benchmarking {model_path}...")
        
        # Initialize environment and agent
        env = EnhancedSnakeGame(speed=game_speed, headless=True, enable_animations=False)
        agent = DQNAgent()
        
        # Load model
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"Loaded model: {model_path}")
        else:
            print(f"Model not found: {model_path}")
            return None
        
        # Run benchmark
        scores = []
        steps = []
        rewards = []
        game_times = []
        
        for game in range(num_games):
            start_time = time.time()
            
            # Play one game
            score, total_reward = agent.play_episode(env, render=False)
            
            end_time = time.time()
            game_time = end_time - start_time
            
            scores.append(score)
            steps.append(100)  # Approximate steps (would need to track actual steps)
            rewards.append(total_reward)
            game_times.append(game_time)
            
            if (game + 1) % 20 == 0:
                print(f"  Game {game+1}/{num_games}: Score = {score}, Avg = {np.mean(scores):.2f}")
        
        # Calculate statistics
        results = {
            'model_path': model_path,
            'num_games': num_games,
            'timestamp': datetime.now().isoformat(),
            'scores': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'all_scores': scores
            },
            'steps': {
                'mean': np.mean(steps),
                'std': np.std(steps),
                'min': np.min(steps),
                'max': np.max(steps)
            },
            'rewards': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'performance': {
                'avg_game_time': np.mean(game_times),
                'total_time': np.sum(game_times),
                'games_per_second': num_games / np.sum(game_times)
            }
        }
        
        return results
    
    def compare_models(self, model_paths, num_games=100):
        """
        Compare multiple models
        
        Args:
            model_paths: List of model file paths
            num_games: Number of games per model
        """
        print(f"Comparing {len(model_paths)} models...")
        
        comparison_results = {}
        
        for model_path in model_paths:
            results = self.benchmark_model(model_path, num_games)
            if results:
                comparison_results[model_path] = results
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_path, result in comparison_results.items():
            json_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    json_result[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (np.integer, np.floating)):
                            json_result[key][sub_key] = sub_value.item()
                        elif isinstance(sub_value, list):
                            json_result[key][sub_key] = [x.item() if hasattr(x, 'item') else x for x in sub_value]
                        else:
                            json_result[key][sub_key] = sub_value
                else:
                    if hasattr(value, 'item'):
                        json_result[key] = value.item()
                    else:
                        json_result[key] = value
            json_results[model_path] = json_result
        
        # Save comparison results
        comparison_file = os.path.join(self.results_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(comparison_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Comparison results saved to: {comparison_file}")
        
        # Generate comparison report
        self.generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def generate_comparison_report(self, results):
        """
        Generate visual comparison report
        
        Args:
            results: Benchmark results dictionary
        """
        if not results:
            print("No results to report")
            return
        
        # Extract data for plotting
        model_names = []
        mean_scores = []
        max_scores = []
        std_scores = []
        
        for model_path, result in results.items():
            model_name = os.path.basename(model_path)
            model_names.append(model_name)
            mean_scores.append(result['scores']['mean'])
            max_scores.append(result['scores']['max'])
            std_scores.append(result['scores']['std'])
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison Report', fontsize=16)
        
        # 1. Mean Scores Comparison
        bars1 = ax1.bar(model_names, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
        ax1.set_title('Mean Scores Comparison')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, mean_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. Max Scores Comparison
        bars2 = ax2.bar(model_names, max_scores, alpha=0.7, color='orange')
        ax2.set_title('Maximum Scores Comparison')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars2, max_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score}', ha='center', va='bottom')
        
        # 3. Score Distribution (Box Plot)
        all_scores = []
        labels = []
        for model_path, result in results.items():
            all_scores.append(result['scores']['all_scores'])
            labels.append(os.path.basename(model_path))
        
        ax3.boxplot(all_scores, labels=labels)
        ax3.set_title('Score Distribution')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Performance Metrics
        avg_times = [result['performance']['avg_game_time'] for result in results.values()]
        games_per_sec = [result['performance']['games_per_second'] for result in results.values()]
        
        ax4_twin = ax4.twinx()
        
        bars3 = ax4.bar(model_names, avg_times, alpha=0.7, color='green', label='Avg Game Time')
        ax4.set_ylabel('Average Game Time (s)', color='green')
        ax4.tick_params(axis='y', labelcolor='green')
        
        line1 = ax4_twin.plot(model_names, games_per_sec, 'ro-', label='Games/sec')
        ax4_twin.set_ylabel('Games per Second', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        
        ax4.set_title('Performance Metrics')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f'comparison_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_file}")
        
        # Print summary table
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'Mean Score':<12} {'Max Score':<12} {'Std Dev':<12} {'Games/sec':<12}")
        print("-"*80)
        
        for model_path, result in results.items():
            model_name = os.path.basename(model_path)
            print(f"{model_name:<20} {result['scores']['mean']:<12.2f} "
                  f"{result['scores']['max']:<12} {result['scores']['std']:<12.2f} "
                  f"{result['performance']['games_per_second']:<12.2f}")
        
        print("="*80)

def main():
    """Main benchmark function"""
    # Find all model files
    models_dir = 'models'
    model_files = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("No model files found in 'models' directory")
        return
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {model_file}")
    
    # Create benchmark
    benchmark = ModelBenchmark()
    
    # Compare all models
    results = benchmark.compare_models(model_files, num_games=50)
    
    print("\nBenchmark completed!")

if __name__ == '__main__':
    main()
