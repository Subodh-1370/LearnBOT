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

class MazeSnakeGame:
    """
    Snake Game with Maze Mode - obstacles and walls for increased difficulty
    """
    
    def __init__(self, width=800, height=600, block_size=20, speed=40, 
                 headless=False, enable_animations=True, maze_complexity='medium'):
        """
        Initialize Maze Snake Game
        
        Args:
            width: Game window width
            height: Game window height  
            block_size: Size of each snake segment and food
            speed: Game speed (FPS)
            headless: Disable rendering for training performance
            enable_animations: Enable smooth animations
            maze_complexity: 'easy', 'medium', 'hard'
        """
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.headless = headless
        self.enable_animations = enable_animations and not headless
        self.maze_complexity = maze_complexity
        
        # Calculate grid dimensions
        self.w = width // block_size
        self.h = height // block_size
        
        # Animation parameters
        self.animation_speed = 0.15
        self.food_pulse_time = 0
        self.food_pulse_speed = 0.1
        
        # Visual settings
        self.bg_color = (20, 25, 40)
        self.grid_color = (30, 35, 50)
        self.wall_color = (100, 100, 120)
        self.obstacle_color = (80, 60, 60)
        self.ui_bg_color = (15, 20, 35)
        
        # Initialize pygame only if not headless
        pygame.init()
        
        # Initialize display and assets only if not headless
        if not self.headless:
            self._init_display()
            self._init_assets()
            self._init_fonts()
        
        self.reset()
    
    def _init_display(self):
        """Initialize pygame display"""
        pygame.display.set_caption('Maze Snake Game - LearnBot')
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
    
    def _generate_maze(self):
        """Generate maze obstacles based on complexity"""
        self.obstacles = set()
        
        if self.maze_complexity == 'easy':
            num_obstacles = random.randint(5, 10)
            max_line_length = 3
        elif self.maze_complexity == 'medium':
            num_obstacles = random.randint(10, 20)
            max_line_length = 5
        else:  # hard
            num_obstacles = random.randint(20, 30)
            max_line_length = 7
        
        # Generate random obstacles
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 50:  # Limit attempts to avoid infinite loops
                if random.random() < 0.7:  # 70% chance for line obstacles
                    # Create line obstacle
                    x = random.randint(2, self.w - 3)
                    y = random.randint(2, self.h - 3)
                    length = random.randint(2, max_line_length)
                    horizontal = random.random() < 0.5
                    
                    line_obstacles = []
                    if horizontal:
                        for i in range(length):
                            if x + i < self.w - 1:
                                line_obstacles.append((x + i, y))
                    else:
                        for i in range(length):
                            if y + i < self.h - 1:
                                line_obstacles.append((x, y + i))
                    
                    # Check if obstacles don't interfere with snake starting position
                    valid = True
                    for obs in line_obstacles:
                        if (abs(obs[0] - self.w // 2) < 3 and 
                            abs(obs[1] - self.h // 2) < 3):
                            valid = False
                            break
                    
                    if valid:
                        self.obstacles.update(line_obstacles)
                        break
                else:  # 30% chance for single obstacles
                    x = random.randint(1, self.w - 2)
                    y = random.randint(1, self.h - 2)
                    
                    # Check if obstacle doesn't interfere with snake starting position
                    if (abs(x - self.w // 2) >= 3 or 
                        abs(y - self.h // 2) >= 3):
                        self.obstacles.add((x, y))
                        break
                
                attempts += 1
    
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
        
        # Generate maze
        self._generate_maze()
        
        # Animation state
        self.smooth_positions = [[float(x), float(y)] for x, y in self.snake]
        self.target_positions = [[float(x), float(y)] for x, y in self.snake]
        
        # UI state
        self.show_game_over = False
        self.game_over_timer = 0
        
        self._place_food()
        
        return self.get_state()
    
    def _place_food(self):
        """Place food at random position not occupied by snake or obstacles"""
        valid_positions = []
        
        for x in range(1, self.w - 1):
            for y in range(1, self.h - 1):
                if [x, y] not in self.snake and (x, y) not in self.obstacles:
                    valid_positions.append([x, y])
        
        if valid_positions:
            self.food = random.choice(valid_positions)
        else:
            # If no valid positions, regenerate maze
            self._generate_maze()
            self._place_food()
    
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
        
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._create_food_particles()
            self._place_food()
            self.steps_since_last_food = 0
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
        
        # Check wall collision
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        
        # Check self collision
        if pt in self.snake[1:]:
            return True
        
        # Check obstacle collision
        if (pt[0], pt[1]) in self.obstacles:
            return True
        
        return False
    
    def _update_ui(self):
        """Update game display"""
        if self.headless:
            return
        
        self.display.blit(self.background, (0, 0))
        self._draw_ui_panel()
        self._draw_obstacles()
        
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
    
    def _draw_obstacles(self):
        """Draw maze obstacles"""
        for obstacle in self.obstacles:
            x, y = obstacle
            rect = pygame.Rect(x * self.block_size, y * self.block_size, 
                            self.block_size, self.block_size)
            pygame.draw.rect(self.display, self.obstacle_color, rect)
            pygame.draw.rect(self.display, self.wall_color, rect, 2)
    
    def _draw_ui_panel(self):
        """Draw UI panel"""
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
        
        maze_text = self.ui_font.render(f"Maze: {self.maze_complexity}", True, (255, 200, 100))
        maze_rect = maze_text.get_rect(topleft=(200, 15))
        self.display.blit(maze_text, maze_rect)
        
        obstacles_text = self.ui_font.render(f"Obstacles: {len(self.obstacles)}", True, (200, 200, 200))
        obstacles_rect = obstacles_text.get_rect(topleft=(200, 40))
        self.display.blit(obstacles_text, obstacles_rect)
        
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
        Get enhanced state representation with maze information
        
        Returns:
            numpy array: Enhanced state vector with maze features
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
        
        # Maze features (4 features)
        # Obstacle proximity in each direction
        obstacle_left = 0
        obstacle_right = 0
        obstacle_up = 0
        obstacle_down = 0
        
        # Check for obstacles in each direction
        for i in range(1, 6):  # Check up to 5 blocks away
            if (head[0] - i, head[1]) in self.obstacles:
                obstacle_left = 1
            if (head[0] + i, head[1]) in self.obstacles:
                obstacle_right = 1
            if (head[0], head[1] - i) in self.obstacles:
                obstacle_up = 1
            if (head[0], head[1] + i) in self.obstacles:
                obstacle_down = 1
        
        # Combine all features
        state = np.array([
            # Basic features (11)
            danger_straight, danger_left, danger_right,
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down,
            # Maze features (4)
            obstacle_left, obstacle_right, obstacle_up, obstacle_down
        ])
        
        return state

if __name__ == '__main__':
    game = MazeSnakeGame(speed=10, maze_complexity='medium')
    
    while True:
        done, reward, score = game.play_step([1, 0, 0])
        
        if done:
            print(f'Game Over! Score: {score}')
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
