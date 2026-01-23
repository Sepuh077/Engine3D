from typing import List
import numpy as np

from src.polygon import Polygon
from src.camera import Camera
from src.helper import rotate_points_batch


class Entity:
    """
    3D Entity that loads OBJ files and manages geometry.
    
    Supports both software rendering (pygame) and GPU rendering (ModernGL).
    For GPU rendering, use `original_vertices` and `get_model_matrix()`.
    For software rendering, use `vertices` which includes all transformations.
    """
    
    # OPTIMIZATION: __slots__ reduces memory and speeds up attribute access by ~20-30%
    __slots__ = ['vertices', '_original_vertices', '_model_center', 'polygons', 
                 '_position', '_scale', '_angle', 
                 '_updated', '_camera_points', '_valid_points', 
                 '_sorted_indices', '_geometry_dirty', '_transform_dirty']
    
    def __init__(self, filename: str, position: tuple = (0, 0, 0), scale: float = 1):
        # Initialize state tracking
        self._sorted_indices = None   # Cache for sorted polygon order
        self._geometry_dirty = True   # Track when re-sorting is needed
        self._transform_dirty = True  # Track when transform matrix needs update (for GPU)
        self._updated = True
        self._camera_points = None
        self._valid_points = None
        
        # Load the OBJ file
        self._load_obj(filename)
        
        # Store initial transform values
        self._scale = float(scale)
        self._angle = np.zeros(3, dtype=np.float32)
        
        # Apply initial scale if not 1
        if scale != 1:
            center = self.vertices.mean(axis=0)
            self.vertices = center + (self.vertices - center) * scale
        
        # Apply initial position offset
        pos = np.array(position, dtype=np.float32)
        if not np.allclose(pos, 0):
            self.vertices = self.vertices + pos
        
        # Update position to reflect actual center
        self._position = self.vertices.mean(axis=0)

    def _load_obj(self, filename: str):
        """Load OBJ file vertices and faces."""
        vertices = []
        self.polygons: List[Polygon] = []
        
        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    # Parse vertex: "v x y z"
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    # Parse face: "f v1/vt1/vn1 v2/vt2/vn2 ..."
                    self.polygons.append(
                        Polygon(
                            self,
                            [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                        )
                    )

        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Store original vertices centered at origin (for GPU rendering)
        center = self.vertices.mean(axis=0)
        self._original_vertices = (self.vertices - center).astype(np.float32)
        self._model_center = center.copy()
        self._position = center.copy()

    @property
    def original_vertices(self) -> np.ndarray:
        """
        Get original vertices centered at origin.
        Use this for GPU rendering with model matrix.
        """
        return self._original_vertices
    
    @property
    def model_center(self) -> np.ndarray:
        """Get the original model center (before transformations)."""
        return self._model_center

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        value = float(value)
        if self._scale == value:
            return
        # Apply scale relative to entity center
        scale_factor = value / self._scale
        self.vertices = self._position + (self.vertices - self._position) * scale_factor
        self._scale = value
        self._updated = True
        self._geometry_dirty = True
        self._transform_dirty = True

    @property
    def angle(self) -> np.ndarray:
        """Get rotation angles (radians) for X, Y, Z axes."""
        return self._angle

    @angle.setter
    def angle(self, value):
        value = np.asarray(value, dtype=np.float32)
        if np.array_equal(self._angle, value):
            return
        # Apply incremental rotation
        diff_angle = value - self._angle
        self.vertices = rotate_points_batch(
            self.vertices,
            self._position,
            diff_angle
        )
        self._angle = value
        self._updated = True
        self._geometry_dirty = True
        self._transform_dirty = True

    @property
    def position(self) -> np.ndarray:
        """Get entity position (center point)."""
        return self._position

    @position.setter
    def position(self, value):
        value = np.asarray(value, dtype=np.float32)
        if np.array_equal(self._position, value):
            return
        diff = value - self._position
        self.vertices = self.vertices + diff
        self._position = value
        self._updated = True
        self._geometry_dirty = True
        self._transform_dirty = True

    @property
    def transform_dirty(self) -> bool:
        """Check if transform has changed (for GPU buffer updates)."""
        return self._transform_dirty
    
    def clear_transform_dirty(self):
        """Clear the transform dirty flag after GPU update."""
        self._transform_dirty = False

    @property
    def camera_points(self):
        return self._camera_points

    @property
    def valid_points(self):
        return self._valid_points

    def set_camera_points(self, camera: Camera):
        """Transform vertices to camera/screen space."""
        if not self._updated:
            return self._camera_points
        self._camera_points, self._valid_points = camera.world_to_cam(self.vertices)
        self._updated = False

    def _get_sorted_indices(self):
        """Get polygon indices sorted by depth (painter's algorithm)."""
        if self._sorted_indices is None or self._geometry_dirty:
            # Compute max Z for each polygon
            z_values = np.array([self.vertices[p.indexes][:, 2].max() for p in self.polygons])
            self._sorted_indices = np.argsort(z_values)
            self._geometry_dirty = False
        return self._sorted_indices

    def draw(self, camera: Camera):
        """Draw entity using software rendering (pygame)."""
        self.set_camera_points(camera)
        
        # Use cached sorted order
        sorted_indices = self._get_sorted_indices()
        
        for i in sorted_indices:
            self.polygons[i].draw(camera)
    
    def get_triangulated_faces(self) -> np.ndarray:
        """
        Get all faces as triangles (for GPU rendering).
        Returns array of shape (num_triangles, 3) with vertex indices.
        """
        triangles = []
        for polygon in self.polygons:
            indices = polygon.indexes
            # Fan triangulation
            for i in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])
        return np.array(triangles, dtype=np.int32)
