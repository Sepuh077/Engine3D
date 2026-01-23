"""
Color utilities and predefined colors.
Colors are RGB tuples with values 0-1 for GPU compatibility.
"""
from typing import Tuple, Union
import random


# Type alias for color
ColorType = Tuple[float, float, float]


class Color:
    """Predefined colors and color utilities."""
    
    # Basic colors
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN = (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    
    # Grays
    GRAY = (0.5, 0.5, 0.5)
    DARK_GRAY = (0.25, 0.25, 0.25)
    LIGHT_GRAY = (0.75, 0.75, 0.75)
    
    # Common colors
    ORANGE = (1.0, 0.5, 0.0)
    PINK = (1.0, 0.4, 0.7)
    PURPLE = (0.5, 0.0, 0.5)
    BROWN = (0.6, 0.3, 0.0)
    GOLD = (1.0, 0.84, 0.0)
    SILVER = (0.75, 0.75, 0.75)
    
    # Sky/Nature
    SKY_BLUE = (0.53, 0.81, 0.92)
    FOREST_GREEN = (0.13, 0.55, 0.13)
    OCEAN_BLUE = (0.0, 0.47, 0.75)
    SAND = (0.76, 0.7, 0.5)
    
    @staticmethod
    def from_rgb(r: int, g: int, b: int) -> ColorType:
        """Create color from RGB values (0-255)."""
        return (r / 255.0, g / 255.0, b / 255.0)
    
    @staticmethod
    def from_hex(hex_color: str) -> ColorType:
        """Create color from hex string (e.g., '#FF5500' or 'FF5500')."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    
    @staticmethod
    def random() -> ColorType:
        """Generate a random color."""
        return (random.random(), random.random(), random.random())
    
    @staticmethod
    def random_bright() -> ColorType:
        """Generate a random bright color."""
        return (
            0.3 + 0.7 * random.random(),
            0.3 + 0.7 * random.random(),
            0.3 + 0.7 * random.random()
        )
    
    @staticmethod
    def lerp(color1: ColorType, color2: ColorType, t: float) -> ColorType:
        """Linearly interpolate between two colors."""
        t = max(0, min(1, t))
        return (
            color1[0] + (color2[0] - color1[0]) * t,
            color1[1] + (color2[1] - color1[1]) * t,
            color1[2] + (color2[2] - color1[2]) * t
        )
