import numpy as np
from typing import Tuple, Optional, Union
from .color import Color, ColorType

class Material:
    """Base class for all materials."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        self.color = color
        self.alpha = alpha

    @property
    def color_vec4(self) -> np.ndarray:
        c = np.array(self.color, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            return np.append(c, self.alpha)
        return c

class UnlitMaterial(Material):
    """Material that ignores lighting and is always visible with its color."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        super().__init__(color, alpha)

class LitMaterial(Material):
    """Lambert material with diffuse lighting."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        super().__init__(color, alpha)

class SpecularMaterial(Material):
    """Phong / Blinn-Phong material for metal and plastic."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0, 
                 specular_color: ColorType = Color.WHITE, shininess: float = 32.0):
        super().__init__(color, alpha)
        self.specular_color = specular_color
        self.shininess = shininess

    @property
    def specular_vec3(self) -> np.ndarray:
        c = np.array(self.specular_color, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        return c[:3]

class EmissiveMaterial(Material):
    """Material that glows and ignores lights around it."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0, intensity: float = 1.0):
        super().__init__(color, alpha)
        self.intensity = intensity

class TransparentMaterial(Material):
    """Material with explicit alpha transparency."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 0.5):
        super().__init__(color, alpha)
