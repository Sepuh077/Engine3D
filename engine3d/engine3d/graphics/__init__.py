"""Graphics utilities and types."""
# Re-export from types module for backward compatibility
from .material import LitMaterial, UnlitMaterial, EmissiveMaterial, SpecularMaterial, TransparentMaterial, Material

__all__ = ["LitMaterial", "UnlitMaterial", "EmissiveMaterial", "SpecularMaterial", "TransparentMaterial", "Material"]
