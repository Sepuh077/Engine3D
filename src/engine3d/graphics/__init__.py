"""Graphics utilities and types."""
# Re-export from types module for backward compatibility
from src.engine3d.graphics.material import LitMaterial, UnlitMaterial, EmissiveMaterial, SpecularMaterial, TransparentMaterial, Material

__all__ = ["LitMaterial", "UnlitMaterial", "EmissiveMaterial", "SpecularMaterial", "TransparentMaterial", "Material"]
