import pygame
import numpy as np
import os
import math
from typing import Optional, Dict, List

class SoundManager:
    """
    Sound effects manager for the Snake game
    """
    
    def __init__(self, enabled=True, sound_volume=0.7, music_volume=0.5):
        """
        Initialize sound manager
        
        Args:
            enabled: Enable/disable sound
            sound_volume: Volume for sound effects (0.0-1.0)
            music_volume: Volume for background music (0.0-1.0)
        """
        self.enabled = enabled
        self.sound_volume = sound_volume
        self.music_volume = music_volume
        self.sounds = {}
        self.sound_cache = {}
        
        if self.enabled:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._create_sounds()
    
    def _create_sounds(self):
        """Create procedural sound effects"""
        if not self.enabled:
            return
        
        # Create sound effects using numpy arrays
        self.sounds = {
            'eat': self._create_eat_sound(),
            'game_over': self._create_game_over_sound(),
            'move': self._create_move_sound(),
            'bonus': self._create_bonus_sound(),
            'level_up': self._create_level_up_sound(),
            'collision': self._create_collision_sound()
        }
    
    def _create_eat_sound(self) -> pygame.mixer.Sound:
        """Create food eating sound"""
        sample_rate = 22050
        duration = 0.2
        samples = int(sample_rate * duration)
        
        # Create a pleasant eating sound
        t = np.linspace(0, duration, samples)
        frequency = 800  # Base frequency
        envelope = np.exp(-t * 10)  # Quick decay
        
        # Add harmonics for richer sound
        wave = (np.sin(2 * np.pi * frequency * t) * 0.5 +
                np.sin(2 * np.pi * frequency * 2 * t) * 0.3 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.2)
        
        # Apply envelope and normalize
        wave = wave * envelope
        wave = np.int16(wave * 32767 * 0.3)  # Lower volume
        
        # Create stereo sound
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def _create_game_over_sound(self) -> pygame.mixer.Sound:
        """Create game over sound"""
        sample_rate = 22050
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        
        # Descending tone for game over
        start_freq = 600
        end_freq = 200
        frequency = np.linspace(start_freq, end_freq, samples)
        
        # Create descending wave
        wave = np.sin(2 * np.pi * frequency * t)
        envelope = np.exp(-t * 2)  # Slow decay
        
        # Add some noise for dramatic effect
        noise = np.random.normal(0, 0.1, samples)
        wave = (wave + noise) * envelope
        
        wave = np.int16(wave * 32767 * 0.4)
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def _create_move_sound(self) -> pygame.mixer.Sound:
        """Create movement sound (very subtle)"""
        sample_rate = 22050
        duration = 0.05
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        frequency = 200  # Low frequency
        envelope = np.exp(-t * 50)  # Very quick decay
        
        wave = np.sin(2 * np.pi * frequency * t) * envelope
        wave = np.int16(wave * 32767 * 0.1)  # Very low volume
        
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def _create_bonus_sound(self) -> pygame.mixer.Sound:
        """Create bonus food sound"""
        sample_rate = 22050
        duration = 0.3
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        
        # Rising tone for bonus
        start_freq = 400
        end_freq = 1200
        frequency = np.linspace(start_freq, end_freq, samples)
        
        wave = np.sin(2 * np.pi * frequency * t)
        envelope = np.exp(-t * 5)
        
        # Add sparkle effect
        sparkle = np.sin(2 * np.pi * frequency * 4 * t) * 0.3
        wave = (wave + sparkle) * envelope
        
        wave = np.int16(wave * 32767 * 0.5)
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def _create_level_up_sound(self) -> pygame.mixer.Sound:
        """Create level up sound"""
        sample_rate = 22050
        duration = 0.5
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        
        # Tri-tone ascending sound
        frequencies = [523, 659, 784]  # C, E, G
        wave = np.zeros(samples)
        
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            
            mask = (t >= start_time) & (t < end_time)
            wave[mask] = np.sin(2 * np.pi * freq * t[mask])
        
        envelope = np.ones(samples)
        envelope[int(samples * 0.8):] = np.linspace(1, 0, samples - int(samples * 0.8))
        
        wave = wave * envelope
        wave = np.int16(wave * 32767 * 0.4)
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def _create_collision_sound(self) -> pygame.mixer.Sound:
        """Create collision sound"""
        sample_rate = 22050
        duration = 0.1
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        
        # Sharp impact sound
        wave = np.random.normal(0, 0.3, samples)  # White noise
        envelope = np.exp(-t * 30)  # Very quick decay
        
        wave = wave * envelope
        wave = np.int16(wave * 32767 * 0.3)
        stereo_wave = np.array([wave, wave]).T
        
        return pygame.mixer.Sound(stereo_wave)
    
    def play_sound(self, sound_name: str, volume_multiplier: float = 1.0):
        """
        Play a sound effect
        
        Args:
            sound_name: Name of the sound to play
            volume_multiplier: Volume multiplier (0.0-1.0)
        """
        if not self.enabled or sound_name not in self.sounds:
            return
        
        try:
            sound = self.sounds[sound_name]
            sound.set_volume(self.sound_volume * volume_multiplier)
            sound.play()
        except Exception as e:
            print(f"Error playing sound {sound_name}: {e}")
    
    def play_eat_sound(self, food_value: int = 10):
        """Play appropriate eating sound based on food value"""
        if food_value >= 50:
            self.play_sound('bonus', 1.0)
        elif food_value >= 25:
            self.play_sound('bonus', 0.7)
        else:
            self.play_sound('eat', 0.8)
    
    def play_move_sound(self, speed_level: int = 1):
        """Play movement sound (subtle, only at higher speeds)"""
        if speed_level > 3:  # Only play at higher speeds
            volume = min(0.2, speed_level * 0.05)
            self.play_sound('move', volume)
    
    def set_sound_volume(self, volume: float):
        """Set sound volume (0.0-1.0)"""
        self.sound_volume = max(0.0, min(1.0, volume))
    
    def set_music_volume(self, volume: float):
        """Set music volume (0.0-1.0)"""
        self.music_volume = max(0.0, min(1.0, volume))
        if self.enabled:
            pygame.mixer.music.set_volume(self.music_volume)
    
    def enable_sounds(self, enabled: bool):
        """Enable or disable sounds"""
        self.enabled = enabled
        if enabled and not pygame.mixer.get_init():
            pygame.mixer.init()
            self._create_sounds()
    
    def create_background_music(self):
        """Create simple background music"""
        if not self.enabled:
            return
        
        sample_rate = 22050
        duration = 10.0  # 10 second loop
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples)
        
        # Create simple melody
        melody_pattern = [262, 294, 330, 349, 392, 349, 330, 294]  # C major scale
        melody = []
        
        for i in range(4):  # Repeat pattern 4 times
            for freq in melody_pattern:
                note_duration = 0.25  # Quarter notes
                note_samples = int(sample_rate * note_duration)
                note_wave = np.sin(2 * np.pi * freq * t[:note_samples])
                melody.extend(note_wave)
        
        # Trim to correct length
        melody = np.array(melody[:samples])
        
        # Add bass line
        bass_freq = 131  # C3 (one octave lower)
        bass = np.sin(2 * np.pi * bass_freq * t) * 0.3
        
        # Combine melody and bass
        music = (melody * 0.5 + bass) * 0.3
        
        # Apply envelope to avoid clicks
        envelope = np.ones(samples)
        fade_samples = int(sample_rate * 0.1)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        music = music * envelope
        music = np.int16(music * 32767)
        stereo_music = np.array([music, music]).T
        
        # Save to temporary file
        temp_file = os.path.join(os.path.dirname(__file__), 'temp_music.wav')
        try:
            import wave
            with wave.open(temp_file, 'w') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(stereo_music.tobytes())
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.set_volume(self.music_volume)
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except ImportError:
            # If wave module is not available, skip background music
            pass
    
    def play_background_music(self, loop: bool = True):
        """Play background music"""
        if not self.enabled:
            return
        
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1 if loop else 0)
        except Exception as e:
            print(f"Error playing background music: {e}")
    
    def stop_background_music(self):
        """Stop background music"""
        if self.enabled:
            pygame.mixer.music.stop()
    
    def pause_background_music(self):
        """Pause background music"""
        if self.enabled:
            pygame.mixer.music.pause()
    
    def unpause_background_music(self):
        """Unpause background music"""
        if self.enabled:
            pygame.mixer.music.unpause()

# Global sound manager instance
sound_manager = SoundManager()

def initialize_sounds(enabled=True, sound_volume=0.7, music_volume=0.5):
    """Initialize the global sound manager"""
    global sound_manager
    sound_manager = SoundManager(enabled, sound_volume, music_volume)
    return sound_manager

def get_sound_manager() -> SoundManager:
    """Get the global sound manager instance"""
    return sound_manager

if __name__ == '__main__':
    # Test sound system
    print("Testing sound system...")
    
    # Initialize sounds
    manager = SoundManager(enabled=True)
    
    # Test each sound
    print("Playing test sounds...")
    
    import time
    time.sleep(0.5)
    manager.play_sound('eat')
    time.sleep(0.5)
    
    manager.play_sound('bonus')
    time.sleep(0.5)
    
    manager.play_sound('level_up')
    time.sleep(0.5)
    
    manager.play_sound('game_over')
    time.sleep(1.0)
    
    print("Sound test completed!")
