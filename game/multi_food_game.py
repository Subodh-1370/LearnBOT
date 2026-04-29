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

class MultiFoodSnakeGame:
    """
    Snake Game with Multi-food Mode - multiple food items with different values
    """
    
    def __init__(self, width=800, height=600, block_size=20, speed=40, 
                 headless=False, enable_animations=True, num_foods=3,
                 food_types=['normal', 'bonus', 'super']):
        """
        Initialize Multi-food Snake Game
        
        Args:
            width: Game window width
            height: Game window height  
            block_size: Size of each snake segment and food
            speed: Game speed (FPS)
            headless: Disable rendering for training performance
            enable_animations: Enable smooth animations
            num_foods: Number of food items on screen
            food_types: Types of food available
        """
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.headless = headless
        self.enable_animations = enable_animations and not headless
        self.num_foods = num_foods
        self.food_types = food_types
        
        # Calculate grid dimensions
        self.w = width // block_size
        self.h = height // block_size
        
        # Animation parameters
        self.animation_speed = 0.15
        self.food_pulse_time = 0
        self.food_pulse_speed = 0.1
        
        # Food type properties
        self.food_properties = {
            'normal': {'color': (220, 20, 60), 'points': 10, 'spawn_chance': 0.7},
            'bonus': {'color': (255, 165, 0), 'points': 25, 'spawn_chance': 0.25},
            'super': {'color': (255, 215, 0), 'points': 50, 'spawn_chance': 0.05}
        }
        
        # Visual settings
        self.bg_color = (20, 25, 40)
        self.grid_color = (30, 35, 50)
        self.ui_bg_color = (15, 20, 35)
        
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
        pygame.display.set_caption('Multi-food Snake Game - LearnBot')
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
        self.foods = []
        self.frame_iteration = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self.total_steps = 0
        self.food_eaten_this_episode = 0
        
        # Animation state
        self.smooth_positions = [[float(x), float(y)] for x, y in self.snake]
        self.target_positions = [[float(x), float(y)] for x, y in self.snake]
        
        # UI state
        self.show_game_over = False
        self.game_over_timer = 0
        
        self._place_foods()
        
        return self.get_state()
    
    def _place_foods(self):
        """Place multiple food items on the board"""
        self.foods = []
        
        for _ in range(self.num_foods):
            self._place_single_food()
    
    def _place_single_food(self):
        """Place a single food item"""
        # Determine food type based on spawn chances
        rand = random.random()
        cumulative_chance = 0
        food_type = 'normal'
        
        for f_type in self.food_types:
            cumulative_chance += self.food_properties[f_type]['spawn_chance']
            if rand <= cumulative_chance:
                food_type = f_type
                break
        
        # Find valid position
        valid_positions = []
        
        for x in range(1, self.w - 1):
            for y in range(1, self.h - 1):
                if [x, y] not in self.snake and [x, y] not in [f['pos'] for f in self.foods]:
                    valid_positions.append([x, y])
        
        if valid_positions:
            food_pos = random.choice(valid_positions)
            self.foods.append({
                'pos': food_pos,
                'type': food_type,
                'value': self.food_properties[food_type]['points'],
                'color': self.food_properties[food_type]['color'],
                'spawn_time': self.frame_iteration
            })
    
    def play_step(self, action):
        """Execute one game step"""
        self.frame_iteration += 1
        self.total_steps += 1
        self.steps_since_last_food += 1
        
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
        
        # Check food collision
        food_eaten = False
        for i, food in enumerate(self.foods):
            if self.head == food['pos']:
                self.score += food['value']
                reward = food['value']
                self._create_food_particles(food['color'])
                self.foods.pop(i)
                self._place_single_food()  # Replace eaten food
                self.steps_since_last_food = 0
                self.food_eaten_this_episode += 1
                food_eaten = True
                break
        
        if not food_eaten:
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
    
    def _create_food_particles(self, color):
        """Create particle effects when food is eaten"""
        if not self.headless:
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.particles.append({
                    'x': self.head[0] * self.block_size + self.block_size // 2,
                    'y': self.head[1] * self.block_size + self.block_size // 2,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': 30,
                    'color': color
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
        self._draw_foods()
        
        if self.show_game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_ui_panel(self):
        """Draw UI panel with multi-food information"""
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
        
        # Food information
        food_count_text = self.ui_font.render(f"Foods: {len(self.foods)}", True, (255, 200, 100))
        food_count_rect = food_count_text.get_rect(topleft=(200, 15))
        self.display.blit(food_count_text, food_count_rect)
        
        eaten_text = self.ui_font.render(f"Eaten: {self.food_eaten_this_episode}", True, (200, 200, 200))
        eaten_rect = eaten_text.get_rect(topleft=(200, 40))
        self.display.blit(eaten_text, eaten_rect)
        
        # Food type legend
        legend_x = 350
        for i, food_type in enumerate(self.food_types):
            color = self.food_properties[food_type]['color']
            points = self.food_properties[food_type]['points']
            
            # Draw food indicator
            pygame.draw.circle(self.display, color, (legend_x + i * 80, 25), 8)
            
            # Draw food label
            label_text = self.ui_font.render(f"{food_type[0].upper()}: {points}", True, (200, 200, 200))
            label_rect = label_text.get_rect(topleft=(legend_x + i * 80 + 15, 18))
            self.display.blit(label_text, label_rect)
        
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
    
    def _draw_foods(self):
        """Draw multiple food items with different colors"""
        for food in self.foods:
            pulse_scale = 1.0 + 0.1 * math.sin(self.food_pulse_time + food['spawn_time'] * 0.1)
            frame_index = int(self.food_pulse_time) % len(self.food_sprites)
            food_sprite = self.food_sprites[frame_index]
            
            # Recolor food based on type
            food_sprite = food_sprite.copy()
            food_sprite.fill(food['color'], special_flags=pygame.BLEND_MULT)
            
            if pulse_scale != 1.0:
                original_size = food_sprite.get_size()
                new_size = (int(original_size[0] * pulse_scale), 
                           int(original_size[1] * pulse_scale))
                food_sprite = pygame.transform.scale(food_sprite, new_size)
            
            x = food['pos'][0] * self.block_size + (self.block_size - food_sprite.get_width()) // 2
            y = food['pos'][1] * self.block_size + (self.block_size - food_sprite.get_height()) // 2
            
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
        
        eaten_text = self.ui_font.render(f"Food Eaten: {self.food_eaten_this_episode}", True, (255, 200, 100))
        eaten_rect = eaten_text.get_rect(center=(self.width // 2, self.height // 2 + 60))
        self.display.blit(eaten_text, eaten_rect)
    
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
        Get enhanced state representation with multi-food information
        
        Returns:
            numpy array: Enhanced state vector with multi-food features
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
        
        # Find closest food
        if self.foods:
            closest_food = min(self.foods, key=lambda f: abs(head[0] - f['pos'][0]) + abs(head[1] - f['pos'][1]))
            food_pos = closest_food['pos']
            food_value = closest_food['value']
            food_type = closest_food['type']
        else:
            food_pos = [0, 0]
            food_value = 10
            food_type = 'normal'
        
        # Food position (4 features)
        food_left = 1 if food_pos[0] < head[0] else 0
        food_right = 1 if food_pos[0] > head[0] else 0
        food_up = 1 if food_pos[1] < head[1] else 0
        food_down = 1 if food_pos[1] > head[1] else 0
        
        # Multi-food features (3 features)
        food_count_normalized = len(self.foods) / self.num_foods
        food_value_normalized = food_value / 50  # Normalize to max value
        food_type_bonus = 1.0 if food_type == 'super' else 0.5 if food_type == 'bonus' else 0.0
        
        # Combine all features
        state = np.array([
            # Basic features (11)
            danger_straight, danger_left, danger_right,
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down,
            # Multi-food features (3)
            food_count_normalized, food_value_normalized, food_type_bonus
        ])
        
        return state

if __name__ == '__main__':
    game = MultiFoodSnakeGame(speed=10, num_foods=3)
    
    while True:
        done, reward, score = game.play_step([1, 0, 0])
        
        if done:
            print(f'Game Over! Score: {score}, Food Eaten: {game.food_eaten_this_episode}')
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
