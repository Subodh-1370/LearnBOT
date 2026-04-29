import pygame
import os
import math

def create_custom_sprites():
    """Create custom sprites for the enhanced Snake game"""
    
    # Initialize pygame
    pygame.init()
    
    # Create sprites directory if it doesn't exist
    sprites_dir = "assets/sprites"
    os.makedirs(sprites_dir, exist_ok=True)
    
    # Sprite size
    size = 20
    
    # Colors
    snake_head_color = (0, 180, 0)
    snake_body_color = (0, 120, 0)
    snake_eye_color = (255, 255, 255)
    snake_pupil_color = (0, 0, 0)
    food_color = (220, 20, 60)
    food_highlight = (255, 100, 100)
    food_stem = (0, 100, 0)
    
    def create_snake_head(direction):
        """Create snake head sprite with direction"""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw head (rounded rectangle)
        pygame.draw.rect(surface, snake_head_color, (2, 2, size-4, size-4), border_radius=6)
        
        # Draw darker inner area
        pygame.draw.rect(surface, (0, 140, 0), (4, 4, size-8, size-8), border_radius=4)
        
        # Draw eyes based on direction
        eye_size = 3
        pupil_size = 1
        
        if direction == "right":
            # Eyes looking right
            pygame.draw.circle(surface, snake_eye_color, (size-6, 6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (size-5, 6), pupil_size)
            pygame.draw.circle(surface, snake_eye_color, (size-6, size-6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (size-5, size-6), pupil_size)
        elif direction == "left":
            # Eyes looking left
            pygame.draw.circle(surface, snake_eye_color, (6, 6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (5, 6), pupil_size)
            pygame.draw.circle(surface, snake_eye_color, (6, size-6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (5, size-6), pupil_size)
        elif direction == "up":
            # Eyes looking up
            pygame.draw.circle(surface, snake_eye_color, (6, 6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (6, 5), pupil_size)
            pygame.draw.circle(surface, snake_eye_color, (size-6, 6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (size-6, 5), pupil_size)
        elif direction == "down":
            # Eyes looking down
            pygame.draw.circle(surface, snake_eye_color, (6, size-6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (6, size-5), pupil_size)
            pygame.draw.circle(surface, snake_eye_color, (size-6, size-6), eye_size)
            pygame.draw.circle(surface, snake_pupil_color, (size-6, size-5), pupil_size)
        
        return surface
    
    def create_snake_body():
        """Create snake body sprite"""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw body (rounded rectangle)
        pygame.draw.rect(surface, snake_body_color, (2, 2, size-4, size-4), border_radius=4)
        
        # Draw pattern
        for i in range(2, size-2, 4):
            for j in range(2, size-2, 4):
                pygame.draw.rect(surface, (0, 80, 0), (i, j, 2, 2))
        
        return surface
    
    def create_food():
        """Create food (apple) sprite"""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw apple (circle)
        center = size // 2
        radius = size // 2 - 2
        pygame.draw.circle(surface, food_color, (center, center), radius)
        
        # Draw highlight
        highlight_offset = radius // 3
        highlight_radius = radius // 4
        pygame.draw.circle(surface, food_highlight, 
                         (center - highlight_offset, center - highlight_offset), 
                         highlight_radius)
        
        # Draw stem
        pygame.draw.rect(surface, food_stem, (center-1, 0, 2, 4))
        
        return surface
    
    # Create all sprites
    sprites = {
        'snake_head_right': create_snake_head('right'),
        'snake_head_left': create_snake_head('left'),
        'snake_head_up': create_snake_head('up'),
        'snake_head_down': create_snake_head('down'),
        'snake_body': create_snake_body(),
        'food': create_food()
    }
    
    # Save sprites
    for name, sprite in sprites.items():
        pygame.image.save(sprite, os.path.join(sprites_dir, f"{name}.png"))
        print(f"Created sprite: {name}.png")
    
    print(f"All sprites created in {sprites_dir}/")
    pygame.quit()

def create_custom_fonts():
    """Create font placeholders"""
    fonts_dir = "assets/fonts"
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Create placeholder font files (these would be actual TTF files in a real project)
    font_files = ['score.ttf', 'ui.ttf', 'game_over.ttf']
    
    for font_file in font_files:
        # Create empty placeholder files
        placeholder_path = os.path.join(fonts_dir, font_file)
        with open(placeholder_path, 'w') as f:
            f.write(f"# Placeholder for {font_file}")
        print(f"Created font placeholder: {font_file}")
    
    print(f"Font placeholders created in {fonts_dir}/")

if __name__ == "__main__":
    print("Creating custom sprites and fonts...")
    create_custom_sprites()
    create_custom_fonts()
    print("Done!")
