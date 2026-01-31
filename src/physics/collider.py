import numpy as np
from typing import Tuple, Optional
from .types import ColliderType

class Collider:
    def __init__(self, collider_type: ColliderType = ColliderType.CUBE):
        self.type = collider_type
        # Cached world bounds
        # (center, radius)
        self.sphere: Optional[Tuple[np.ndarray, float]] = None
        # (center, rotation_matrix, extents)
        self.obb: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        # (min_point, max_point)
        self.aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # (center, radius, half_height)
        self.cylinder: Optional[Tuple[np.ndarray, float, float]] = None
        # (vertices, faces, model_matrix)
        self.mesh_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def update(self, sphere: Tuple[np.ndarray, float], 
               obb: Tuple[np.ndarray, np.ndarray, np.ndarray],
               aabb: Tuple[np.ndarray, np.ndarray],
               cylinder: Tuple[np.ndarray, float, float],
               mesh_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        """Update cached world bounds."""
        self.sphere = sphere
        self.obb = obb
        self.aabb = aabb
        self.cylinder = cylinder
        self.mesh_data = mesh_data

    def get_world_sphere(self):
        return self.sphere

    def get_world_obb(self):
        return self.obb
    
    def get_world_aabb(self):
        return self.aabb

    def get_world_cylinder(self):
        return self.cylinder
    
    def get_mesh_data(self):
        return self.mesh_data
