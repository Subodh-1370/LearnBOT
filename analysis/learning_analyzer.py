import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pickle
from collections import defaultdict

class LearningAnalyzer:
    """
    Comprehensive learning behavior analysis system for AI agents
    """
    
    def __init__(self, results_dir='analysis_results'):
        """
        Initialize learning analyzer
        
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Data storage
        self.training_data = defaultdict(list)
        self.episode_data = []
        self.action_data = []
        self.state_data = []
        self.reward_data = []
        
        # Analysis settings
        self.plot_style = 'seaborn-v0_8'
        self.figure_size = (15, 10)
        self.dpi = 150
        
        # Set plotting style
        plt.style.use(self.plot_style)
        sns.set_palette("husl")
    
    def record_episode(self, episode_num: int, score: int, reward: float, 
                      steps: int, epsilon: float, loss: float, 
                      actions_taken: List[int], states_seen: List[np.ndarray],
                      additional_metrics: Dict = None):
        """
        Record episode data for analysis
        
        Args:
            episode_num: Episode number
            score: Episode score
            reward: Total reward
            steps: Number of steps
            epsilon: Current epsilon value
            loss: Training loss
            actions_taken: Actions taken during episode
            states_seen: States encountered during episode
            additional_metrics: Additional metrics to record
        """
        episode_record = {
            'episode': episode_num,
            'score': score,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'action_distribution': self._analyze_action_distribution(actions_taken),
            'state_complexity': self._analyze_state_complexity(states_seen),
            'learning_progress': self._calculate_learning_progress(episode_num, score),
            **(additional_metrics or {})
        }
        
        self.episode_data.append(episode_record)
        
        # Store action data
        for i, action in enumerate(actions_taken):
            self.action_data.append({
                'episode': episode_num,
                'step': i,
                'action': action,
                'epsilon': epsilon
            })
        
        # Store state data (sample to avoid memory issues)
        if len(states_seen) > 100:
            sample_indices = np.random.choice(len(states_seen), 100, replace=False)
            sampled_states = [states_seen[i] for i in sample_indices]
        else:
            sampled_states = states_seen
        
        for i, state in enumerate(sampled_states):
            self.state_data.append({
                'episode': episode_num,
                'step': i,
                'state_features': state.tolist(),
                'epsilon': epsilon
            })
    
    def _analyze_action_distribution(self, actions: List[int]) -> Dict:
        """Analyze distribution of actions taken"""
        if len(actions) == 0:
            return {'straight': 0, 'left': 0, 'right': 0, 'entropy': 0}
        
        action_counts = np.bincount(actions, minlength=3)
        total = len(actions)
        
        distribution = {
            'straight': action_counts[0] / total,
            'left': action_counts[1] / total,
            'right': action_counts[2] / total
        }
        
        # Calculate entropy (measure of randomness)
        probs = list(distribution.values())
        entropy = -sum(p * np.log2(p + 1e-8) for p in probs if p > 0)
        distribution['entropy'] = entropy
        
        return distribution
    
    def _analyze_state_complexity(self, states: List[np.ndarray]) -> Dict:
        """Analyze complexity of states encountered"""
        if not states:
            return {'mean_variance': 0, 'feature_diversity': 0}
        
        states_array = np.array(states)
        
        # Calculate variance across features
        feature_variance = np.var(states_array, axis=0)
        mean_variance = np.mean(feature_variance)
        
        # Calculate feature diversity (unique states / total states)
        unique_states = len(set(tuple(s) for s in states))
        feature_diversity = unique_states / len(states)
        
        return {
            'mean_variance': mean_variance,
            'feature_diversity': feature_diversity,
            'feature_variance_std': np.std(feature_variance)
        }
    
    def _calculate_learning_progress(self, episode: int, score: int) -> Dict:
        """Calculate learning progress metrics"""
        if len(self.episode_data) < 10:
            return {'recent_avg': score, 'improvement_rate': 0, 'stability': 0}
        
        recent_episodes = self.episode_data[-10:]
        recent_scores = [ep['score'] for ep in recent_episodes]
        recent_avg = np.mean(recent_scores)
        
        # Calculate improvement rate
        if len(recent_episodes) >= 5:
            early_avg = np.mean(recent_scores[:5])
            late_avg = np.mean(recent_scores[-5:])
            improvement_rate = (late_avg - early_avg) / (early_avg + 1e-8)
        else:
            improvement_rate = 0
        
        # Calculate stability (inverse of variance)
        stability = 1 / (np.var(recent_scores) + 1)
        
        return {
            'recent_avg': recent_avg,
            'improvement_rate': improvement_rate,
            'stability': stability
        }
    
    def generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive learning analysis"""
        if not self.episode_data:
            return {'error': 'No episode data available'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.episode_data)
        
        analysis = {
            'summary_stats': self._calculate_summary_stats(df),
            'learning_curves': self._analyze_learning_curves(df),
            'action_patterns': self._analyze_action_patterns(),
            'state_analysis': self._analyze_state_patterns(),
            'convergence_analysis': self._analyze_convergence(df),
            'performance_metrics': self._calculate_performance_metrics(df)
        }
        
        return analysis
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        return {
            'total_episodes': len(df),
            'avg_score': df['score'].mean(),
            'max_score': df['score'].max(),
            'score_std': df['score'].std(),
            'avg_reward': df['reward'].mean(),
            'max_reward': df['reward'].max(),
            'avg_steps': df['steps'].mean(),
            'final_epsilon': df['epsilon'].iloc[-1],
            'training_efficiency': self._calculate_training_efficiency(df)
        }
    
    def _analyze_learning_curves(self, df: pd.DataFrame) -> Dict:
        """Analyze learning curves and trends"""
        # Calculate moving averages
        window_sizes = [5, 10, 25, 50]
        curves = {}
        
        for window in window_sizes:
            if len(df) >= window:
                curves[f'moving_avg_{window}'] = df['score'].rolling(window=window).mean().tolist()
        
        # Calculate trend
        if len(df) >= 10:
            recent_slope = np.polyfit(range(10), df['score'].tail(10), 1)[0]
            overall_slope = np.polyfit(range(len(df)), df['score'], 1)[0]
        else:
            recent_slope = overall_slope = 0
        
        return {
            'moving_averages': curves,
            'recent_trend': recent_slope,
            'overall_trend': overall_slope,
            'plateau_episodes': self._detect_plateaus(df['score'])
        }
    
    def _detect_plateaus(self, scores: pd.Series, window: int = 20, threshold: float = 0.1) -> List[int]:
        """Detect learning plateaus"""
        plateau_episodes = []
        
        if len(scores) < window:
            return plateau_episodes
        
        rolling_std = scores.rolling(window=window).std()
        plateau_mask = rolling_std < threshold
        
        for i in range(window, len(scores)):
            if plateau_mask.iloc[i]:
                plateau_episodes.append(i)
        
        return plateau_episodes
    
    def _analyze_action_patterns(self) -> Dict:
        """Analyze action selection patterns"""
        if not self.action_data:
            return {}
        
        action_df = pd.DataFrame(self.action_data)
        
        # Action distribution over time
        action_evolution = action_df.groupby('episode')['action'].value_counts(normalize=True).unstack(fill_value=0)
        
        # Entropy evolution
        entropy_by_episode = []
        for episode in action_df['episode'].unique():
            episode_actions = action_df[action_df['episode'] == episode]['action']
            action_counts = episode_actions.value_counts(normalize=True)
            entropy = -sum(p * np.log2(p + 1e-8) for p in action_counts)
            entropy_by_episode.append(entropy)
        
        return {
            'action_evolution': action_evolution.to_dict(),
            'entropy_evolution': entropy_by_episode,
            'final_action_distribution': action_evolution.iloc[-1].to_dict() if len(action_evolution) > 0 else {}
        }
    
    def _analyze_state_patterns(self) -> Dict:
        """Analyze state visitation patterns"""
        if not self.state_data:
            return {}
        
        state_df = pd.DataFrame(self.state_data)
        
        # Feature statistics
        all_features = np.array([s['state_features'] for s in self.state_data])
        feature_stats = {
            'mean': np.mean(all_features, axis=0).tolist(),
            'std': np.std(all_features, axis=0).tolist(),
            'min': np.min(all_features, axis=0).tolist(),
            'max': np.max(all_features, axis=0).tolist()
        }
        
        # State diversity over time
        diversity_by_episode = []
        for episode in state_df['episode'].unique():
            episode_states = state_df[state_df['episode'] == episode]['state_features']
            unique_states = len(set(tuple(s) for s in episode_states))
            diversity = unique_states / len(episode_states)
            diversity_by_episode.append(diversity)
        
        return {
            'feature_statistics': feature_stats,
            'diversity_evolution': diversity_by_episode,
            'state_space_coverage': len(set(tuple(s['state_features']) for s in self.state_data))
        }
    
    def _analyze_convergence(self, df: pd.DataFrame) -> Dict:
        """Analyze convergence behavior"""
        if len(df) < 50:
            return {'converged': False, 'reason': 'Insufficient data'}
        
        # Check if score has stabilized
        recent_scores = df['score'].tail(20)
        score_variance = recent_scores.var()
        
        # Check if epsilon has reached minimum
        final_epsilon = df['epsilon'].iloc[-1]
        
        # Check if learning rate (improvement) has decreased
        if len(df) >= 100:
            early_improvement = np.polyfit(range(50), df['score'].head(50), 1)[0]
            late_improvement = np.polyfit(range(50), df['score'].tail(50), 1)[0]
            improvement_decreased = late_improvement < early_improvement * 0.5
        else:
            improvement_decreased = False
        
        converged = (score_variance < 1.0 and final_epsilon < 0.05 and improvement_decreased)
        
        return {
            'converged': converged,
            'score_variance': score_variance,
            'final_epsilon': final_epsilon,
            'improvement_decreased': improvement_decreased,
            'convergence_episode': self._find_convergence_episode(df) if converged else None
        }
    
    def _find_convergence_episode(self, df: pd.DataFrame) -> Optional[int]:
        """Find the episode where convergence likely occurred"""
        if len(df) < 50:
            return None
        
        # Look for point where score variance becomes consistently low
        window_size = 20
        threshold = 1.0
        
        for i in range(window_size, len(df) - window_size):
            window_variance = df['score'].iloc[i-window_size:i].var()
            next_window_variance = df['score'].iloc[i:i+window_size].var()
            
            if window_variance < threshold and next_window_variance < threshold:
                return i
        
        return None
    
    def _calculate_training_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate training efficiency (score per episode)"""
        if len(df) < 10:
            return 0.0
        
        # Efficiency measured as score improvement per episode
        first_10_avg = df['score'].head(10).mean()
        last_10_avg = df['score'].tail(10).mean()
        episodes_used = len(df)
        
        efficiency = (last_10_avg - first_10_avg) / episodes_used
        return max(0, efficiency)  # Non-negative efficiency
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed performance metrics"""
        return {
            'consistency': 1 / (df['score'].std() + 1),
            'peak_performance': df['score'].max(),
            'average_performance': df['score'].mean(),
            'performance_growth': (df['score'].tail(10).mean() - df['score'].head(10).mean()) / (df['score'].head(10).mean() + 1e-8),
            'survival_rate': (df['score'] > 0).mean(),
            'high_score_rate': (df['score'] >= 10).mean()
        }
    
    def create_visualization_report(self, save_plots: bool = True) -> str:
        """Create comprehensive visualization report"""
        if not self.episode_data:
            return "No data available for visualization"
        
        df = pd.DataFrame(self.episode_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=self.figure_size)
        fig.suptitle('Comprehensive Learning Analysis Report', fontsize=16)
        
        # 1. Score progression
        axes[0, 0].plot(df['episode'], df['score'], alpha=0.6, label='Episode Score')
        if len(df) >= 10:
            axes[0, 0].plot(df['episode'], df['score'].rolling(10).mean(), 'r-', linewidth=2, label='10-episode MA')
        axes[0, 0].set_title('Score Progression')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Reward progression
        axes[0, 1].plot(df['episode'], df['reward'], alpha=0.6, color='green')
        if len(df) >= 10:
            axes[0, 1].plot(df['episode'], df['reward'].rolling(10).mean(), 'r-', linewidth=2)
        axes[0, 1].set_title('Reward Progression')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].grid(True)
        
        # 3. Epsilon decay
        axes[0, 2].plot(df['episode'], df['epsilon'], 'b-', linewidth=2)
        axes[0, 2].set_title('Exploration Rate (Epsilon)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Epsilon')
        axes[0, 2].grid(True)
        
        # 4. Score distribution
        axes[1, 0].hist(df['score'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 5. Steps per episode
        axes[1, 1].plot(df['episode'], df['steps'], alpha=0.6, color='orange')
        if len(df) >= 10:
            axes[1, 1].plot(df['episode'], df['steps'].rolling(10).mean(), 'r-', linewidth=2)
        axes[1, 1].set_title('Steps per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)
        
        # 6. Loss progression
        if 'loss' in df.columns and df['loss'].notna().any():
            axes[1, 2].plot(df['episode'], df['loss'], alpha=0.6, color='red')
            if len(df) >= 10:
                axes[1, 2].plot(df['episode'], df['loss'].rolling(10).mean(), 'b-', linewidth=2)
            axes[1, 2].set_title('Training Loss')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Training Loss')
        
        # 7. Action entropy evolution
        if self.action_data:
            action_df = pd.DataFrame(self.action_data)
            entropy_by_episode = []
            for episode in action_df['episode'].unique():
                episode_actions = action_df[action_df['episode'] == episode]['action']
                action_counts = episode_actions.value_counts(normalize=True)
                entropy = -sum(p * np.log2(p + 1e-8) for p in action_counts)
                entropy_by_episode.append(entropy)
            
            axes[2, 0].plot(range(len(entropy_by_episode)), entropy_by_episode, 'g-', linewidth=2)
            axes[2, 0].set_title('Action Entropy Evolution')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Entropy')
            axes[2, 0].grid(True)
        else:
            axes[2, 0].text(0.5, 0.5, 'No Action Data', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Action Entropy Evolution')
        
        # 8. Learning progress heatmap
        if len(df) >= 20:
            # Create heatmap of recent performance
            recent_data = df.tail(20)
            performance_matrix = recent_data[['score', 'reward', 'steps']].values.T
            im = axes[2, 1].imshow(performance_matrix, cmap='RdYlBu', aspect='auto')
            axes[2, 1].set_title('Recent Performance Heatmap')
            axes[2, 1].set_yticks([0, 1, 2])
            axes[2, 1].set_yticklabels(['Score', 'Reward', 'Steps'])
            axes[2, 1].set_xlabel('Recent Episodes')
            plt.colorbar(im, ax=axes[2, 1])
        else:
            axes[2, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Recent Performance Heatmap')
        
        # 9. Performance metrics summary
        metrics = self._calculate_performance_metrics(df)
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        axes[2, 2].text(0.1, 0.9, metrics_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2, 2].set_title('Performance Metrics Summary')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f'learning_analysis_{timestamp}.png')
            fig.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return plot_path
        else:
            plt.show()
            return "Plot displayed"
    
    def save_analysis_data(self, filename: str = None) -> str:
        """Save all analysis data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'learning_analysis_{timestamp}.json'
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for JSON serialization
        analysis_data = {
            'episode_data': self.episode_data,
            'analysis_results': self.generate_comprehensive_analysis(),
            'metadata': {
                'total_episodes': len(self.episode_data),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_types': ['episode_data', 'action_data', 'state_data']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        return filepath
    
    def load_analysis_data(self, filepath: str) -> bool:
        """Load analysis data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.episode_data = data.get('episode_data', [])
            return True
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            return False

if __name__ == '__main__':
    # Test the learning analyzer
    analyzer = LearningAnalyzer()
    
    # Generate some test data
    for episode in range(100):
        score = np.random.poisson(episode // 10 + 1)
        reward = score * 10 - np.random.exponential(5)
        steps = np.random.randint(20, 200)
        epsilon = max(0.01, 1.0 - episode * 0.01)
        loss = max(0.01, 1.0 - episode * 0.005)
        
        actions = np.random.choice(3, steps)
        states = [np.random.rand(11) for _ in range(min(steps, 100))]
        
        analyzer.record_episode(episode, score, reward, steps, epsilon, loss, actions, states)
    
    # Generate analysis
    analysis = analyzer.generate_comprehensive_analysis()
    print("Analysis Summary:")
    print(f"Total Episodes: {analysis['summary_stats']['total_episodes']}")
    print(f"Average Score: {analysis['summary_stats']['avg_score']:.2f}")
    print(f"Max Score: {analysis['summary_stats']['max_score']}")
    print(f"Training Efficiency: {analysis['summary_stats']['training_efficiency']:.4f}")
    
    # Create visualization
    plot_path = analyzer.create_visualization_report(save_plots=True)
    print(f"Visualization saved to: {plot_path}")
    
    # Save data
    data_path = analyzer.save_analysis_data()
    print(f"Analysis data saved to: {data_path}")
