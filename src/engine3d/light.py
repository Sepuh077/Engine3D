"""
Light3D - Lighting for 3D scenes.
"""
import numpy as np
from typing import Tuple
from .color import ColorType


class Light3D:
    """
    Directional light for 3D scenes.
    
    Example:
        light = Light3D(direction=(0.5, -1, -0.5))
        light.color = Color.WHITE
    """
    
    def __init__(self, 
                 direction: Tuple[float, float, float] = (0.3, -0.7, -0.5),
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 ambient: float = 0.2):
        """
        Initialize light.
        
        Args:
            direction: Light direction vector (will be normalized)
            color: Light color (RGB 0-1)
            intensity: Light intensity multiplier
            ambient: Ambient light level (0-1)
        """
        self._direction = np.array(direction, dtype=np.float32)
        self._normalize_direction()
        
        self.color = color
        self.intensity = intensity
        self.ambient = ambient
    
    def _normalize_direction(self):
        """Normalize the direction vector."""
        norm = np.linalg.norm(self._direction)
        if norm > 0:
            self._direction /= norm
    
    @property
    def direction(self) -> np.ndarray:
        """Get light direction (normalized)."""
        return self._direction.copy()
    
    @direction.setter
    def direction(self, value: Tuple[float, float, float]):
        """Set light direction."""
        self._direction = np.array(value, dtype=np.float32)
        self._normalize_direction()
    
    def point_from(self, position: Tuple[float, float, float], 
                   target: Tuple[float, float, float] = (0, 0, 0)):
        """
        Set light to point from a position towards a target.
        
        Args:
            position: Where the light is coming from
            target: Where the light is pointing to
        """
        pos = np.array(position, dtype=np.float32)
        tgt = np.array(target, dtype=np.float32)
        self._direction = tgt - pos
        self._normalize_direction()


class PointLight3D:
    """
    Point light that emits in all directions from a position.
    (For future implementation with more advanced shaders)
    """
    
    def __init__(self,
                 position: Tuple[float, float, float] = (0, 10, 0),
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 range: float = 50.0):
        """
        Initialize point light.
        
        Args:
            position: Light position
            color: Light color (RGB 0-1)  
            intensity: Light intensity
            range: Maximum light range
        """
        self._position = np.array(position, dtype=np.float32)
        self.color = color
        self.intensity = intensity
        self.range = range
    
    @property
    def position(self) -> np.ndarray:
        return self._position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        self._position = np.array(value, dtype=np.float32)
    
    @property
    def x(self) -> float:
        return float(self._position[0])
    
    @x.setter
    def x(self, value: float):
        self._position[0] = value
    
    @property
    def y(self) -> float:
        return float(self._position[1])
    
    @y.setter
    def y(self, value: float):
        self._position[1] = value
    
    @property
    def z(self) -> float:
        return float(self._position[2])
    
    @z.setter
    def z(self, value: float):
        self._position[2] = value
