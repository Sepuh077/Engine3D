"""
Object3D - 3D object that can be loaded, positioned, rotated, and scaled.
"""
import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING

from .color import ColorType, Color

if TYPE_CHECKING:
    import moderngl


class Object3D:
    """
    A 3D object in the scene.
    
    Load from OBJ file or create primitives. Supports position, rotation, scale.
    
    Example:
        obj = Object3D("model.obj")
        obj.position = (0, 1, 0)
        obj.rotation_y = 45  # degrees
        obj.scale = 2.0
    """
    
    def __init__(self, 
                 filename: Optional[str] = None,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 scale: float = 1.0,
                 color: Optional[ColorType] = None):
        """
        Initialize 3D object.
        
        Args:
            filename: Path to OBJ file (optional, can load later)
            position: Initial position (x, y, z)
            scale: Initial scale factor
            color: Object color (RGB 0-1), random if None
        """
        # Transform properties
        self._position = np.array(position, dtype=np.float32)
        self._rotation = np.array([0, 0, 0], dtype=np.float32)  # Euler angles in radians
        self._scale = np.array([scale, scale, scale], dtype=np.float32)
        
        # Appearance
        self._color = np.array(color if color else Color.random_bright(), dtype=np.float32)
        self._visible = True
        
        # Geometry data (loaded from file)
        self._vertices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._faces: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None
        
        # GPU resources (set by renderer)
        self._vao = None
        self._vbo = None
        self._gpu_initialized = False
        
        # Metadata
        self.name = ""
        self.tag = ""
        
        # Load file if provided
        if filename:
            self.load(filename)
    
    def load(self, filename: str):
        """
        Load 3D model from OBJ file.
        
        Args:
            filename: Path to OBJ file
        """
        vertices = []
        faces = []
        
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    face_indices = []
                    for part in line.split()[1:]:
                        # Handle "v", "v/vt", "v/vt/vn", "v//vn" formats
                        idx = int(part.split("/")[0]) - 1
                        face_indices.append(idx)
                    # Triangulate faces with more than 3 vertices
                    for i in range(1, len(face_indices) - 1):
                        faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
        
        self._vertices = np.array(vertices, dtype=np.float32)
        self._faces = np.array(faces, dtype=np.int32)
        
        # Center the model
        self._center = self._vertices.mean(axis=0)
        self._vertices = self._vertices - self._center
        
        # Compute normals
        self._compute_normals()
        
        # Mark GPU as needing update
        self._gpu_initialized = False
        
        # Set name from filename
        self.name = filename.split("/")[-1].split("\\")[-1].replace(".obj", "")
    
    def _compute_normals(self):
        """Compute per-vertex normals for lighting."""
        normals = np.zeros_like(self._vertices)
        
        for face in self._faces:
            v0, v1, v2 = self._vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal /= norm
            normals[face] += normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1
        self._normals = (normals / norms).astype(np.float32)
    
    # =========================================================================
    # Position properties
    # =========================================================================
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Get position as tuple."""
        return tuple(self._position)
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set position."""
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
    
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move object by offset."""
        self._position += np.array([dx, dy, dz], dtype=np.float32)
    
    # =========================================================================
    # Rotation properties (in degrees for user convenience)
    # =========================================================================
    
    @property
    def rotation(self) -> Tuple[float, float, float]:
        """Get rotation as tuple (degrees)."""
        return tuple(np.degrees(self._rotation))
    
    @rotation.setter
    def rotation(self, value: Tuple[float, float, float]):
        """Set rotation (degrees)."""
        self._rotation = np.radians(value).astype(np.float32)
    
    @property
    def rotation_x(self) -> float:
        """Rotation around X axis in degrees."""
        return float(np.degrees(self._rotation[0]))
    
    @rotation_x.setter
    def rotation_x(self, value: float):
        self._rotation[0] = np.radians(value)
    
    @property
    def rotation_y(self) -> float:
        """Rotation around Y axis in degrees."""
        return float(np.degrees(self._rotation[1]))
    
    @rotation_y.setter
    def rotation_y(self, value: float):
        self._rotation[1] = np.radians(value)
    
    @property
    def rotation_z(self) -> float:
        """Rotation around Z axis in degrees."""
        return float(np.degrees(self._rotation[2]))
    
    @rotation_z.setter
    def rotation_z(self, value: float):
        self._rotation[2] = np.radians(value)
    
    def rotate(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Rotate object by offset (degrees)."""
        self._rotation += np.radians([dx, dy, dz]).astype(np.float32)
    
    # =========================================================================
    # Scale properties
    # =========================================================================
    
    @property
    def scale(self) -> float:
        """Get uniform scale."""
        return float(self._scale[0])
    
    @scale.setter
    def scale(self, value: float):
        """Set uniform scale."""
        self._scale = np.array([value, value, value], dtype=np.float32)
    
    @property
    def scale_xyz(self) -> Tuple[float, float, float]:
        """Get non-uniform scale."""
        return tuple(self._scale)
    
    @scale_xyz.setter
    def scale_xyz(self, value: Tuple[float, float, float]):
        """Set non-uniform scale."""
        self._scale = np.array(value, dtype=np.float32)
    
    # =========================================================================
    # Appearance properties
    # =========================================================================
    
    @property
    def color(self) -> Tuple[float, float, float]:
        """Get color as tuple."""
        return tuple(self._color)
    
    @color.setter
    def color(self, value: ColorType):
        """Set color."""
        self._color = np.array(value, dtype=np.float32)
    
    @property
    def visible(self) -> bool:
        """Is object visible?"""
        return self._visible
    
    @visible.setter
    def visible(self, value: bool):
        """Set visibility."""
        self._visible = value
    
    def show(self):
        """Make object visible."""
        self._visible = True
    
    def hide(self):
        """Make object invisible."""
        self._visible = False
    
    # =========================================================================
    # Model matrix
    # =========================================================================
    
    def get_model_matrix(self) -> np.ndarray:
        """
        Get the 4x4 model transformation matrix.
        Order: Scale -> Rotate -> Translate
        """
        # Scale matrix
        sx, sy, sz = self._scale
        scale = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation matrices
        rx, ry, rz = self._rotation
        
        cx, sx_r = np.cos(rx), np.sin(rx)
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx_r, 0],
            [0, sx_r, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        cy, sy_r = np.cos(ry), np.sin(ry)
        rot_y = np.array([
            [cy, 0, sy_r, 0],
            [0, 1, 0, 0],
            [-sy_r, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        cz, sz_r = np.cos(rz), np.sin(rz)
        rot_z = np.array([
            [cz, -sz_r, 0, 0],
            [sz_r, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        rotation = rot_x @ rot_y @ rot_z
        
        # Translation matrix
        tx, ty, tz = self._position
        translation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1]
        ], dtype=np.float32)
        
        return scale @ rotation @ translation
    
    # =========================================================================
    # GPU methods (called by renderer)
    # =========================================================================
    
    def _init_gpu(self, ctx: 'moderngl.Context', program: 'moderngl.Program'):
        """Initialize GPU resources. Called by renderer."""
        if self._vertices is None:
            raise RuntimeError("Object has no geometry loaded")
        
        # Flatten vertices for rendering
        flat_vertices = self._vertices[self._faces.flatten()]
        flat_normals = self._normals[self._faces.flatten()]
        
        # Interleave data
        vertex_data = np.hstack([flat_vertices, flat_normals]).astype(np.float32)
        
        # Create GPU buffers
        self._vbo = ctx.buffer(vertex_data.tobytes())
        self._vao = ctx.vertex_array(
            program,
            [(self._vbo, '3f 3f', 'in_position', 'in_normal')]
        )
        self._gpu_initialized = True
    
    def _release_gpu(self):
        """Release GPU resources."""
        if self._vao:
            self._vao.release()
            self._vao = None
        if self._vbo:
            self._vbo.release()
            self._vbo = None
        self._gpu_initialized = False
    
    def __repr__(self):
        return f"Object3D(name='{self.name}', position={self.position}, scale={self.scale})"


# =============================================================================
# Primitive factory functions
# =============================================================================

def create_cube(size: float = 1.0, 
                position: Tuple[float, float, float] = (0, 0, 0),
                color: Optional[ColorType] = None) -> Object3D:
    """
    Create a cube primitive.
    
    Args:
        size: Cube size
        position: Cube position
        color: Cube color
    
    Returns:
        Object3D cube
    """
    obj = Object3D(position=position, color=color)
    
    s = size / 2
    vertices = np.array([
        # Front face
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        # Back face
        [-s, -s, -s], [-s,  s, -s], [ s,  s, -s], [ s, -s, -s],
        # Top face
        [-s,  s, -s], [-s,  s,  s], [ s,  s,  s], [ s,  s, -s],
        # Bottom face
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
        # Right face
        [ s, -s, -s], [ s,  s, -s], [ s,  s,  s], [ s, -s,  s],
        # Left face
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],       # Front
        [4, 5, 6], [4, 6, 7],       # Back
        [8, 9, 10], [8, 10, 11],    # Top
        [12, 13, 14], [12, 14, 15], # Bottom
        [16, 17, 18], [16, 18, 19], # Right
        [20, 21, 22], [20, 22, 23], # Left
    ], dtype=np.int32)
    
    obj._vertices = vertices
    obj._faces = faces
    obj._center = np.zeros(3, dtype=np.float32)
    obj._compute_normals()
    obj.name = "cube"
    
    return obj


def create_plane(width: float = 10.0, 
                 height: float = 10.0,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 color: Optional[ColorType] = None) -> Object3D:
    """
    Create a horizontal plane primitive.
    
    Args:
        width: Plane width (X axis)
        height: Plane height (Z axis)
        position: Plane position
        color: Plane color
    
    Returns:
        Object3D plane
    """
    obj = Object3D(position=position, color=color)
    
    w, h = width / 2, height / 2
    vertices = np.array([
        [-w, 0, -h],
        [ w, 0, -h],
        [ w, 0,  h],
        [-w, 0,  h],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 2, 1],  # Facing up
        [0, 3, 2],
    ], dtype=np.int32)
    
    obj._vertices = vertices
    obj._faces = faces
    obj._center = np.zeros(3, dtype=np.float32)
    obj._compute_normals()
    obj.name = "plane"
    
    return obj
