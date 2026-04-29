import pygame
import os
import sys
from typing import Optional, Tuple, Dict, Any

class AssetManager:
    """
    Asset loading system with fallback to generated graphics
    Ensures the game works even if sprite files are missing
    """
    
    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = assets_dir
        self.sprites_dir = os.path.join(assets_dir, "sprites")
        self.fonts_dir = os.path.join(assets_dir, "fonts")
        self.sounds_dir = os.path.join(assets_dir, "sounds")
        
        # Cache loaded assets
        self._sprite_cache: Dict[str, pygame.Surface] = {}
        self._font_cache: Dict[str, pygame.font.Font] = {}
        
        # Ensure directories exist
        os.makedirs(self.sprites_dir, exist_ok=True)
        os.makedirs(self.fonts_dir, exist_ok=True)
        os.makedirs(self.sounds_dir, exist_ok=True)
    
    def load_sprite(self, name: str, size: Tuple[int, int] = (20, 20), 
                   fallback_color: Tuple[int, int, int] = (0, 255, 0)) -> pygame.Surface:
        """
        Load sprite from file or generate fallback
        
        Args:
            name: Sprite name (without extension)
            size: Target size for the sprite
            fallback_color: Color for generated fallback
            
        Returns:
            pygame.Surface: Loaded or generated sprite
        """
        cache_key = f"{name}_{size[0]}x{size[1]}"
        
        if cache_key in self._sprite_cache:
            return self._sprite_cache[cache_key]
        
        # Try to load from file
        sprite_path = os.path.join(self.sprites_dir, f"{name}.png")
        
        if os.path.exists(sprite_path):
            try:
                sprite = pygame.image.load(sprite_path).convert_alpha()
                sprite = pygame.transform.scale(sprite, size)
                self._sprite_cache[cache_key] = sprite
                return sprite
            except Exception as e:
                print(f"Warning: Failed to load sprite {name}: {e}")
        
        # Generate fallback sprite
        sprite = self._generate_fallback_sprite(name, size, fallback_color)
        self._sprite_cache[cache_key] = sprite
        return sprite
    
    def _generate_fallback_sprite(self, name: str, size: Tuple[int, int], 
                                 color: Tuple[int, int, int]) -> pygame.Surface:
        """Generate fallback sprite based on name"""
        sprite = pygame.Surface(size, pygame.SRCALPHA)
        
        if "head" in name.lower():
            # Snake head - rounded rectangle with eyes
            pygame.draw.rect(sprite, color, (0, 0, size[0], size[1]), border_radius=size[0]//3)
            
            # Add eyes
            eye_size = max(2, size[0]//8)
            eye_color = (255, 255, 255)
            pupil_color = (0, 0, 0)
            
            # Left eye
            pygame.draw.circle(sprite, eye_color, (size[0]//3, size[1]//3), eye_size)
            pygame.draw.circle(sprite, pupil_color, (size[0]//3, size[1]//3), eye_size//2)
            
            # Right eye
            pygame.draw.circle(sprite, eye_color, (2*size[0]//3, size[1]//3), eye_size)
            pygame.draw.circle(sprite, pupil_color, (2*size[0]//3, size[1]//3), eye_size//2)
            
        elif "body" in name.lower():
            # Snake body - rounded rectangle
            inner_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(sprite, color, (0, 0, size[0], size[1]), border_radius=size[0]//4)
            pygame.draw.rect(sprite, inner_color, (2, 2, size[0]-4, size[1]-4), border_radius=size[0]//4)
            
        elif "food" in name.lower() or "apple" in name.lower():
            # Food/apple - circle with highlight
            pygame.draw.circle(sprite, (255, 0, 0), (size[0]//2, size[1]//2), size[0]//2)
            
            # Add highlight
            highlight_size = max(2, size[0]//6)
            pygame.draw.circle(sprite, (255, 100, 100), 
                             (size[0]//3, size[1]//3), highlight_size)
            
            # Add stem
            stem_color = (0, 100, 0)
            pygame.draw.rect(sprite, stem_color, 
                            (size[0]//2 - 1, 0, 2, size[1]//4))
            
        else:
            # Default - simple rectangle
            pygame.draw.rect(sprite, color, (0, 0, size[0], size[1]))
        
        return sprite
    
    def load_font(self, name: str, size: int = 25) -> pygame.font.Font:
        """
        Load font from file or use system fallback
        
        Args:
            name: Font name
            size: Font size
            
        Returns:
            pygame.font.Font: Loaded or fallback font
        """
        cache_key = f"{name}_{size}"
        
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        
        # Try to load from file
        font_path = os.path.join(self.fonts_dir, f"{name}.ttf")
        
        if os.path.exists(font_path):
            try:
                font = pygame.font.Font(font_path, size)
                self._font_cache[cache_key] = font
                return font
            except Exception as e:
                print(f"Warning: Failed to load font {name}: {e}")
        
        # Use system font fallback
        try:
            font = pygame.font.SysFont("arial", size)
        except:
            font = pygame.font.Font(None, size)
        
        self._font_cache[cache_key] = font
        return font
    
    def create_animated_sprites(self, base_name: str, size: Tuple[int, int], 
                               frames: int = 4) -> list:
        """
        Create animated sprite frames
        
        Args:
            base_name: Base sprite name
            size: Frame size
            frames: Number of animation frames
            
        Returns:
            list: List of sprite surfaces
        """
        animated_sprites = []
        
        for frame in range(frames):
            # Load base sprite
            base_sprite = self.load_sprite(base_name, size)
            
            # Create animated version (simple pulse effect for food)
            if "food" in base_name.lower() or "apple" in base_name.lower():
                # Pulse effect
                scale = 1.0 + 0.1 * pygame.math.Vector2(1, 1).rotate(frame * 90).x
                new_size = (int(size[0] * scale), int(size[1] * scale))
                animated_sprite = pygame.transform.scale(base_sprite, new_size)
                # Center the scaled sprite
                offset = ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2)
                final_sprite = pygame.Surface(size, pygame.SRCALPHA)
                final_sprite.blit(animated_sprite, offset)
                animated_sprites.append(final_sprite)
            else:
                animated_sprites.append(base_sprite.copy())
        
        return animated_sprites
    
    def clear_cache(self):
        """Clear all cached assets"""
        self._sprite_cache.clear()
        self._font_cache.clear()

# Global asset manager instance
asset_manager = AssetManager()
