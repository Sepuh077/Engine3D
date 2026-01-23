"""
Optimized Entity class with performance improvements.
Key optimizations:
1. Avoid sorting every frame - cache sorted order
2. Batch coordinate transformations
3. Reduce numpy allocations
4. Use numba for JIT compilation (optional)
"""
from typing import List
import numpy as np

from src.polygon import Polygon
from src.camera import Camera
from src.helper import rotate_points_batch


class EntityOptimized:
    def __init__(self, filename: str, position: tuple = (0, 0, 0), scale: float = 1):
        self._load_obj(filename)
        self._position = np.array(position, dtype=np.float32)  # Use float32 for speed
        self._scale = scale
        self._angle = np.zeros(3, dtype=np.float32)
        self._updated = True
        self._camera_points = None
        self._valid_points = None
        self._sorted_indices = None  # Cache sorted polygon order
        
        # Pre-compute polygon vertex indices as contiguous array for batch operations
        self._build_index_arrays()

    def _build_index_arrays(self):
        """Pre-build arrays for batch rendering."""
        # Flatten all polygon indices for batch operations
        self._poly_lengths = np.array([len(p.indexes) for p in self.polygons], dtype=np.int32)
        self._flat_indices = np.concatenate([np.array(p.indexes) for p in self.polygons])
        self._poly_offsets = np.cumsum(np.concatenate([[0], self._poly_lengths[:-1]]))
        
        # Pre-allocate colors array
        self._colors = np.array([p.color for p in self.polygons], dtype=np.uint8)

    def _load_obj(self, filename):
        vertices = []
        self.polygons: List[Polygon] = []
        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append(
                        list(map(float, line.split()[1:]))
                    )
                elif line.startswith("f "):
                    self.polygons.append(
                        Polygon(
                            self,
                            [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                        )
                    )

        self.vertices = np.array(vertices, dtype=np.float32)  # float32 is faster
        self._original_vertices = self.vertices.copy()  # Keep original for transforms
        self._find_position()

    def _find_position(self):
        self._position = self.vertices.mean(axis=0)
        self._scale = 1.0

    # Remove np.copy() - return views when safe, or cache
    @property
    def position(self):
        return self._position  # Return view, not copy
    
    @position.setter
    def position(self, value):
        value = np.asarray(value, dtype=np.float32)
        if np.array_equal(self._position, value):
            return
        diff = value - self._position
        self.vertices += diff
        self._position = value
        self._updated = True
        self._sorted_indices = None  # Invalidate sort cache

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if self._scale == value:
            return
        # More efficient scaling from center
        self.vertices = self._position + (self.vertices - self._position) * (value / self._scale)
        self._scale = value
        self._updated = True
        self._sorted_indices = None

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value: tuple):
        value = np.asarray(value, dtype=np.float32)
        if np.array_equal(self._angle, value):
            return
        diff_angle = value - self._angle
        self.vertices = rotate_points_batch(self.vertices, self._position, diff_angle)
        self._angle = value
        self._updated = True
        self._sorted_indices = None

    @property
    def camera_points(self):
        return self._camera_points

    @property
    def valid_points(self):
        return self._valid_points

    def set_camera_points(self, camera: Camera):
        if not self._updated:
            return
        self._camera_points, self._valid_points = camera.world_to_cam(self.vertices)
        self._updated = False

    def _get_sorted_indices(self):
        """Cache sorted polygon indices - only re-sort when geometry changes."""
        if self._sorted_indices is None:
            # Compute max Z for each polygon using batch operations
            max_z_per_poly = np.array([
                self.vertices[p.indexes][:, 2].max() 
                for p in self.polygons
            ])
            self._sorted_indices = np.argsort(max_z_per_poly)
        return self._sorted_indices

    def draw(self, camera: Camera):
        self.set_camera_points(camera)
        
        # Use cached sorted order instead of sorting every frame
        sorted_indices = self._get_sorted_indices()
        
        for i in sorted_indices:
            self.polygons[i].draw(camera)

    def draw_batch(self, camera: Camera, surface):
        """
        Batch draw using pygame.draw.polygon with pre-sorted indices.
        Further optimization: use pygame.surfarray for direct pixel manipulation.
        """
        import pygame
        
        self.set_camera_points(camera)
        sorted_indices = self._get_sorted_indices()
        
        # Batch draw - collect visible polygons first
        visible_polys = []
        for i in sorted_indices:
            poly = self.polygons[i]
            if np.all(self._valid_points[poly.indexes]):
                visible_polys.append((
                    poly.color,
                    self._camera_points[poly.indexes].tolist()
                ))
        
        # Draw all visible polygons
        for color, points in visible_polys:
            pygame.draw.polygon(surface, color, points)
