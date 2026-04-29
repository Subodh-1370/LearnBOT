import pygame
import random
import numpy as np
import math
import os
import sys
from enum import Enum
from typing import Optional, Tuple, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.asset_manager import asset_manager

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SpeedSnakeGame:
    """
    Snake Game with Speed Mode - increasing difficulty over time
    """
    
    def __init__(self, width=800, height=600, block_size=20, base_speed=40, 
                 headless=False, enable_animations=True, speed_increment=5,
                 max_speed=200, difficulty_mode='progressive'):
        """
        Initialize Speed Snake Game
        
        Args:
            width: Game window width
            height: Game window height  
            block_size: Size of each snake segment and food
            base_speed: Initial game speed (FPS)
            headless: Disable rendering for training performance
            enable_animations: Enable smooth animations
            speed_increment: Speed increase per food eaten
            max_speed: Maximum game speed limit
            difficulty_mode: 'progressive', 'exponential', 'step'
        """
        self.width = width
        self.height = height
        self.block_size = block_size
        self.base_speed = base_speed
        self.speed = base_speed
        self.headless = headless
        self.enable_animations = enable_animations and not headless
        self.speed_increment = speed_increment
        self.max_speed = max_speed
        self.difficulty_mode = difficulty_mode
        
        # Calculate grid dimensions
        self.w = width // block_size
        self.h = height // block_size
        
        # Animation parameters
        self.animation_speed = 0.15
        self.food_pulse_time = 0
        self.food_pulse_speed = 0.1
        
        # Speed mode parameters
        self.speed_level = 1
        self.last_speed_increase_score = 0
        self.speed_boost_timer = 0
        self.speed_boost_active = False
        
        # Visual settings
        self.bg_color = (20, 25, 40)
        self.grid_color = (30, 35, 50)
        self.ui_bg_color = (15, 20, 35)
        self.speed_color = (255, 100, 100)
        
        # Initialize pygame
        pygame.init()
        
        # Initialize display and assets only if not headless
        if not self.headless:
            self._init_display()
            self._init_assets()
            self._init_fonts()
        
        self.reset()
    
    def _init_display(self):
        """Initialize pygame display"""
        pygame.display.set_caption('Speed Snake Game - LearnBot')
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.background = self._create_background()
    
    def _init_assets(self):
        """Initialize game assets"""
        self.snake_head_sprites = {
            Direction.RIGHT: asset_manager.load_sprite("snake_head_right", (self.block_size, self.block_size), (0, 200, 0)),
            Direction.LEFT: asset_manager.load_sprite("snake_head_left", (self.block_size, self.block_size), (0, 200, 0)),
            Direction.UP: asset_manager.load_sprite("snake_head_up", (self.block_size, self.block_size), (0, 200, 0)),
            Direction.DOWN: asset_manager.load_sprite("snake_head_down", (self.block_size, self.block_size), (0, 200, 0))
        }
        
        self.snake_body_sprite = asset_manager.load_sprite("snake_body", (self.block_size, self.block_size), (0, 150, 0))
        self.food_sprites = asset_manager.create_animated_sprites("food", (self.block_size, self.block_size), 4)
        self.particles = []
    
    def _init_fonts(self):
        """Initialize fonts"""
        # Use system fonts directly to avoid pygame font issues
        self.score_font = pygame.font.SysFont("arial", 28)
        self.ui_font = pygame.font.SysFont("arial", 20)
        self.game_over_font = pygame.font.SysFont("arial", 48)
    
    def _create_background(self) -> pygame.Surface:
        """Create textured background with grid"""
        background = pygame.Surface((self.width, self.height))
        background.fill(self.bg_color)
        
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(background, self.grid_color, (x, 0), (x, self.height), 1)
        
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(background, self.grid_color, (0, y), (self.width, y), 1)
        
        pygame.draw.rect(background, (40, 45, 60), (0, 0, self.width, self.height), 3)
        
        return background
    
    def reset(self):
        """Reset game to initial state"""
        self.head = [self.w // 2, self.h // 2]
        self.snake = [self.head,
                     [self.head[0] - 1, self.head[1]],
                     [self.head[0] - 2, self.head[1]]]
        
        self.direction = Direction.RIGHT
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self.total_steps = 0
        
        # Reset speed parameters
        self.speed = self.base_speed
        self.speed_level = 1
        self.last_speed_increase_score = 0
        self.speed_boost_timer = 0
        self.speed_boost_active = False
        
        # Animation state
        self.smooth_positions = [[float(x), float(y)] for x, y in self.snake]
        self.target_positions = [[float(x), float(y)] for x, y in self.snake]
        
        # UI state
        self.show_game_over = False
        self.game_over_timer = 0
        
        self._place_food()
        
        return self.get_state()
    
    def _place_food(self):
        """Place food at random position not occupied by snake"""
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break
    
    def _update_speed(self):
        """Update game speed based on score and difficulty mode"""
        if self.difficulty_mode == 'progressive':
            # Linear progression
            if self.score > self.last_speed_increase_score + 2:  # Increase every 2 foods
                self.speed = min(self.speed + self.speed_increment, self.max_speed)
                self.speed_level += 1
                self.last_speed_increase_score = self.score
                self._create_speed_particles()
        
        elif self.difficulty_mode == 'exponential':
            # Exponential progression
            if self.score > self.last_speed_increase_score + 1:  # Increase every food
                speed_increase = int(self.speed_increment * (1.2 ** self.speed_level))
                self.speed = min(self.speed + speed_increase, self.max_speed)
                self.speed_level += 1
                self.last_speed_increase_score = self.score
                self._create_speed_particles()
        
        elif self.difficulty_mode == 'step':
            # Step progression - increase at specific score thresholds
            thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
            if self.score in thresholds and self.score > self.last_speed_increase_score:
                self.speed = min(self.speed + self.speed_increment * 2, self.max_speed)
                self.speed_level += 1
                self.last_speed_increase_score = self.score
                self._create_speed_particles()
    
    def _create_speed_particles(self):
        """Create particle effects for speed increase"""
        if not self.headless:
            for _ in range(15):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 5)
                self.particles.append({
                    'x': self.head[0] * self.block_size + self.block_size // 2,
                    'y': self.head[1] * self.block_size + self.block_size // 2,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': 40,
                    'color': (255, random.randint(150, 255), random.randint(0, 100))
                })
    
    def play_step(self, action):
        """Execute one game step"""
        self.frame_iteration += 1
        self.total_steps += 1
        self.steps_since_last_food += 1
        
        # Update speed boost timer
        if self.speed_boost_active:
            self.speed_boost_timer -= 1
            if self.speed_boost_timer <= 0:
                self.speed_boost_active = False
        
        # Collect user input
        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # Move
        if not self.game_over:
            self._move(action)
            self._update_animations()
        
        # Check if game over
        reward = 0
        done = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            self.game_over = True
            self.show_game_over = True
            self.game_over_timer = pygame.time.get_ticks() if not self.headless else 0
            return done, reward, self.score
        
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._create_food_particles()
            self._place_food()
            self.steps_since_last_food = 0
            self._update_speed()
        else:
            self.snake.pop()
            if len(self.smooth_positions) > len(self.snake):
                self.smooth_positions.pop()
        
        # Update UI and Clock
        if not self.headless:
            self._update_ui()
            self.clock.tick(self.speed)
        
        return done, reward, self.score
    
    def _update_animations(self):
        """Update smooth animations"""
        if self.enable_animations:
            for i, (smooth_pos, target_pos) in enumerate(zip(self.smooth_positions, self.target_positions)):
                smooth_pos[0] += (target_pos[0] - smooth_pos[0]) * self.animation_speed
                smooth_pos[1] += (target_pos[1] - smooth_pos[1]) * self.animation_speed
            
            self.food_pulse_time += self.food_pulse_speed
            
            self.particles = [p for p in self.particles if p['life'] > 0]
            for particle in self.particles:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['life'] -= 1
    
    def _create_food_particles(self):
        """Create particle effects when food is eaten"""
        if not self.headless:
            food_x = self.food[0] * self.block_size + self.block_size // 2
            food_y = self.food[1] * self.block_size + self.block_size // 2
            
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.particles.append({
                    'x': food_x,
                    'y': food_y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': 30,
                    'color': (255, random.randint(100, 200), 0)
                })
    
    def is_collision(self, pt=None):
        """Check if there's a collision"""
        if pt is None:
            pt = self.head
        
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        """Update game display"""
        if self.headless:
            return
        
        self.display.blit(self.background, (0, 0))
        self._draw_ui_panel()
        
        for particle in self.particles:
            alpha = particle['life'] / 30.0
            size = int(3 * alpha)
            if size > 0:
                pygame.draw.circle(self.display, particle['color'], 
                                 (int(particle['x']), int(particle['y'])), size)
        
        self._draw_snake()
        self._draw_food()
        
        if self.show_game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_ui_panel(self):
        """Draw UI panel with speed information"""
        panel_height = 60
        panel_surface = pygame.Surface((self.width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill(self.ui_bg_color)
        self.display.blit(panel_surface, (0, 0))
        
        score_text = self.score_font.render(f"Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(topleft=(20, 15))
        self.display.blit(score_text, score_rect)
        
        length_text = self.ui_font.render(f"Length: {len(self.snake)}", True, (200, 200, 200))
        length_rect = length_text.get_rect(topleft=(20, 40))
        self.display.blit(length_text, length_rect)
        
        # Speed information
        speed_color = self.speed_color if self.speed > self.base_speed + 20 else (255, 255, 255)
        speed_text = self.ui_font.render(f"Speed: {self.speed} FPS", True, speed_color)
        speed_rect = speed_text.get_rect(topleft=(200, 15))
        self.display.blit(speed_text, speed_rect)
        
        level_text = self.ui_font.render(f"Level: {self.speed_level}", True, speed_color)
        level_rect = level_text.get_rect(topleft=(200, 40))
        self.display.blit(level_text, level_rect)
        
        # Speed bar
        bar_width = 200
        bar_height = 10
        bar_x = 400
        bar_y = 20
        pygame.draw.rect(self.display, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        speed_ratio = (self.speed - self.base_speed) / (self.max_speed - self.base_speed)
        speed_ratio = min(speed_ratio, 1.0)
        filled_width = int(bar_width * speed_ratio)
        
        bar_color = (255, 100, 100) if speed_ratio > 0.5 else (255, 200, 100) if speed_ratio > 0.25 else (100, 255, 100)
        pygame.draw.rect(self.display, bar_color, (bar_x, bar_y, filled_width, bar_height))
        pygame.draw.rect(self.display, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        
        if self.game_over:
            restart_text = self.ui_font.render("Press R to restart", True, (255, 200, 100))
            restart_rect = restart_text.get_rect(topright=(self.width - 20, 20))
            self.display.blit(restart_text, restart_rect)
    
    def _draw_snake(self):
        """Draw snake with sprites"""
        if self.enable_animations and len(self.smooth_positions) == len(self.snake):
            positions = self.smooth_positions
        else:
            positions = [[float(x), float(y)] for x, y in self.snake]
        
        for i, pos in enumerate(positions[1:], 1):
            x = int(pos[0] * self.block_size)
            y = int(pos[1] * self.block_size)
            self.display.blit(self.snake_body_sprite, (x, y))
        
        if positions:
            head_pos = positions[0]
            head_sprite = self.snake_head_sprites[self.direction]
            x = int(head_pos[0] * self.block_size)
            y = int(head_pos[1] * self.block_size)
            self.display.blit(head_sprite, (x, y))
    
    def _draw_food(self):
        """Draw food with animation"""
        if self.food:
            pulse_scale = 1.0 + 0.1 * math.sin(self.food_pulse_time)
            frame_index = int(self.food_pulse_time) % len(self.food_sprites)
            food_sprite = self.food_sprites[frame_index]
            
            if pulse_scale != 1.0:
                original_size = food_sprite.get_size()
                new_size = (int(original_size[0] * pulse_scale), 
                           int(original_size[1] * pulse_scale))
                food_sprite = pygame.transform.scale(food_sprite, new_size)
            
            x = self.food[0] * self.block_size + (self.block_size - food_sprite.get_width()) // 2
            y = self.food[1] * self.block_size + (self.block_size - food_sprite.get_height()) // 2
            
            self.display.blit(food_sprite, (x, y))
    
    def _draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (0, 0))
        
        game_over_text = self.game_over_font.render("GAME OVER", True, (255, 100, 100))
        game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.display.blit(game_over_text, game_over_rect)
        
        score_text = self.score_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.display.blit(score_text, score_rect)
        
        speed_text = self.ui_font.render(f"Max Speed: {self.speed} FPS (Level {self.speed_level})", True, (255, 200, 100))
        speed_rect = speed_text.get_rect(center=(self.width // 2, self.height // 2 + 60))
        self.display.blit(speed_text, speed_rect)
    
    def _move(self, action):
        """Move snake based on action"""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        x, y = self.head[0], self.head[1]
        
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        
        self.head = [x, y]
        self.snake.insert(0, self.head)
        self.target_positions.insert(0, [float(x), float(y)])
    
    def get_state(self):
        """
        Get enhanced state representation with speed information
        
        Returns:
            numpy array: Enhanced state vector with speed features
        """
        head = self.snake[0]
        
        # Basic danger detection (3 features)
        danger_straight = self.is_collision([head[0] + (head[0] - self.snake[1][0]), 
                                            head[1] + (head[1] - self.snake[1][1])])
        
        if self.direction == Direction.RIGHT:
            danger_left = self.is_collision([head[0], head[1] - 1])
            danger_right = self.is_collision([head[0], head[1] + 1])
        elif self.direction == Direction.LEFT:
            danger_left = self.is_collision([head[0], head[1] + 1])
            danger_right = self.is_collision([head[0], head[1] - 1])
        elif self.direction == Direction.UP:
            danger_left = self.is_collision([head[0] - 1, head[1]])
            danger_right = self.is_collision([head[0] + 1, head[1]])
        elif self.direction == Direction.DOWN:
            danger_left = self.is_collision([head[0] + 1, head[1]])
            danger_right = self.is_collision([head[0] - 1, head[1]])
        
        # Current direction (4 features)
        direction_left = 1 if self.direction == Direction.LEFT else 0
        direction_right = 1 if self.direction == Direction.RIGHT else 0
        direction_up = 1 if self.direction == Direction.UP else 0
        direction_down = 1 if self.direction == Direction.DOWN else 0
        
        # Food position (4 features)
        food_left = 1 if self.food[0] < head[0] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_up = 1 if self.food[1] < head[1] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        
        # Speed features (3 features)
        speed_normalized = (self.speed - self.base_speed) / (self.max_speed - self.base_speed)
        speed_level_normalized = min(self.speed_level / 10, 1.0)  # Normalize to 0-1
        time_pressure = min(self.steps_since_last_food / 50, 1.0)  # Time pressure indicator
        
        # Combine all features
        state = np.array([
            # Basic features (11)
            danger_straight, danger_left, danger_right,
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down,
            # Speed features (3)
            speed_normalized, speed_level_normalized, time_pressure
        ])
        
        return state

if __name__ == '__main__':
    game = SpeedSnakeGame(base_speed=10, difficulty_mode='progressive')
    
    while True:
        done, reward, score = game.play_step([1, 0, 0])
        
        if done:
            print(f'Game Over! Score: {score}, Max Speed: {game.speed} FPS')
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            game.reset()
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            quit()
    
    pygame.quit()
