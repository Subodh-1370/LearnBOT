import pygame
import random
import numpy as np
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    def __init__(self, width=640, height=480, block_size=20, speed=40):
        """
        Initialize Snake Game
        
        Args:
            width: Game window width
            height: Game window height  
            block_size: Size of each snake segment and food
            speed: Game speed (FPS)
        """
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        
        # Calculate grid dimensions
        self.w = width // block_size
        self.h = height // block_size
        
        # Initialize pygame
        pygame.init()
        
        # Initialize display
        pygame.display.set_caption('Snake Game - LearnBot')
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (200, 0, 0)
        self.GREEN1 = (0, 255, 0)
        self.GREEN2 = (0, 200, 0)
        
        # Font for score
        try:
            self.font = pygame.font.SysFont('arial', 25)
        except:
            # Fallback to default font if Arial is not available
            self.font = pygame.font.Font(None, 36)
        
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Start snake in the center
        self.head = [self.w // 2, self.h // 2]
        self.snake = [self.head,
                     [self.head[0] - 1, self.head[1]],
                     [self.head[0] - 2, self.head[1]]]
        
        # Random initial direction
        self.direction = Direction.RIGHT
        self.score = 0
        self.food = None
        self.frame_iteration = 0
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
    
    def play_step(self, action):
        """
        Execute one game step
        
        Args:
            action: [0, 1, 2] -> [Straight, Left Turn, Right Turn]
            
        Returns:
            done: bool - game over flag
            reward: float - reward for this step
            score: int - current score
        """
        self.frame_iteration += 1
        
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action)
        
        # 3. Check if game over
        reward = 0
        done = False
        
        # Check collision
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return done, reward, self.score
        
        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and Clock
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 6. Return game over and score
        return done, reward, self.score
    
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
        
        return False
    
    def _update_ui(self):
        """Update the game display"""
        self.display.fill(self.BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, self.GREEN1, 
                           pygame.Rect(pt[0] * self.block_size, pt[1] * self.block_size, 
                                     self.block_size, self.block_size))
            pygame.draw.rect(self.display, self.GREEN2, 
                           pygame.Rect(pt[0] * self.block_size + 4, pt[1] * self.block_size + 4, 
                                     12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, self.RED, 
                        pygame.Rect(self.food[0] * self.block_size, self.food[1] * self.block_size, 
                                  self.block_size, self.block_size))
        
        # Draw score
        text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
    
    def _move(self, action):
        """Move snake based on action"""
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Left turn [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        # Move head
        x = self.head[0]
        y = self.head[1]
        
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
    
    def get_state(self):
        """
        Get current state representation
        
        Returns:
            numpy array: [danger_straight, danger_left, danger_right, 
                         direction_left, direction_right, direction_up, direction_down,
                         food_left, food_right, food_up, food_down]
        """
        head = self.snake[0]
        
        # Danger detection
        danger_straight = self.is_collision([head[0] + (head[0] - self.snake[1][0]), 
                                            head[1] + (head[1] - self.snake[1][1])])
        
        # Calculate positions for left and right turns
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
        
        # Current direction
        direction_left = 1 if self.direction == Direction.LEFT else 0
        direction_right = 1 if self.direction == Direction.RIGHT else 0
        direction_up = 1 if self.direction == Direction.UP else 0
        direction_down = 1 if self.direction == Direction.DOWN else 0
        
        # Food position relative to head
        food_left = 1 if self.food[0] < head[0] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_up = 1 if self.food[1] < head[1] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        
        state = np.array([danger_straight, danger_left, danger_right,
                          direction_left, direction_right, direction_up, direction_down,
                          food_left, food_right, food_up, food_down])
        
        return state

if __name__ == '__main__':
    # Test the game
    game = SnakeGame()
    
    while True:
        done, reward, score = game.play_step([1, 0, 0])  # Go straight
        
        if done:
            print(f'Game Over! Score: {score}')
            break
    
    pygame.quit()
