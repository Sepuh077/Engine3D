"""
GPU-accelerated 3D rendering using ModernGL.
This approach is 100-1000x faster than software rendering.

Uses the Entity class for OBJ loading and transformation management.

Install: pip install moderngl pygame numpy
"""
import pygame
import moderngl
import numpy as np
from typing import List, Optional

# Import Entity class for OBJ loading
from src.entity import Entity


# =============================================================================
# Matrix Utilities
# =============================================================================

def perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0]
    ], dtype='f4')


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Create view matrix using look-at."""
    if up is None:
        up = np.array([0, 1, 0], dtype='f4')
    
    eye = np.asarray(eye, dtype='f4')
    target = np.asarray(target, dtype='f4')
    up = np.asarray(up, dtype='f4')
    
    # Forward vector (from target to eye, because we look at -Z)
    f = target - eye
    f = f / np.linalg.norm(f)
    
    # Right vector
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    
    # Recalculate up
    u = np.cross(r, f)
    
    # Create rotation matrix
    rotation = np.array([
        [r[0], u[0], -f[0], 0],
        [r[1], u[1], -f[1], 0],
        [r[2], u[2], -f[2], 0],
        [0, 0, 0, 1]
    ], dtype='f4')
    
    # Create translation matrix
    translation = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-eye[0], -eye[1], -eye[2], 1]
    ], dtype='f4')
    
    return translation @ rotation


def rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Create rotation matrix from Euler angles (radians)."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    
    rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype='f4')
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
    rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
    
    return rx @ ry @ rz


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Create translation matrix."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1]
    ], dtype='f4')


def scale_matrix(sx: float, sy: float = None, sz: float = None) -> np.ndarray:
    """Create scale matrix."""
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype='f4')


# =============================================================================
# GPU Mesh - Wraps Entity for GPU Rendering
# =============================================================================

class GPUMesh:
    """
    GPU-accelerated mesh that uses Entity for OBJ loading.
    
    Uses Entity's original_vertices (centered at origin) for the GPU buffer,
    and applies position/rotation/scale via the model matrix. This is more
    efficient as we don't need to re-upload geometry when transforms change.
    """
    
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, entity: Entity, color: tuple = None):
        """
        Create GPU mesh from Entity.
        
        Args:
            ctx: ModernGL context
            program: Shader program
            entity: Entity instance with loaded OBJ data
            color: RGB tuple (0-1 range), random if None
        """
        self.ctx = ctx
        self.program = program
        self.entity = entity
        
        # Set color
        if color is None:
            color = (np.random.random(), np.random.random(), np.random.random())
        self.color = np.array(color, dtype='f4')
        
        # Create GPU resources (only needs to happen once!)
        self._create_gpu_buffers()
    
    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals for lighting."""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            # Cross product gives face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal /= norm
            # Accumulate to vertex normals
            normals[face] += normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1
        normals /= norms
        
        return normals.astype('f4')
    
    def _create_gpu_buffers(self):
        """
        Create VBO and VAO from entity's ORIGINAL vertices (centered at origin).
        Transformations are applied via the model matrix, not by modifying vertices.
        """
        # Use original vertices (centered at origin) - this never changes!
        vertices = self.entity.original_vertices.astype('f4')
        
        # Get triangulated faces
        faces = self.entity.get_triangulated_faces()
        
        # Compute normals
        normals = self._compute_normals(vertices, faces)
        
        # Flatten for rendering (each triangle vertex separately)
        flat_vertices = vertices[faces.flatten()]
        flat_normals = normals[faces.flatten()]
        
        # Interleave: [x, y, z, nx, ny, nz] per vertex
        vertex_data = np.hstack([flat_vertices, flat_normals]).astype('f4')
        
        # Create GPU buffer
        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f', 'in_position', 'in_normal')]
        )
        self.num_vertices = len(flat_vertices)
    
    def get_model_matrix(self) -> np.ndarray:
        """
        Get model transformation matrix from entity.
        
        This combines translation, rotation, and scale into one matrix.
        Order: Scale -> Rotate -> Translate
        """
        # Start with identity
        model = np.eye(4, dtype='f4')
        
        # Apply scale
        s = self.entity.scale
        if s != 1.0:
            model = model @ scale_matrix(s)
        
        # Apply rotation (around origin, which is model center)
        angle = self.entity.angle
        if np.any(angle != 0):
            rot = rotation_matrix(angle[0], angle[1], angle[2])
            model = model @ rot
        
        # Apply translation to final position
        pos = self.entity.position
        model = model @ translation_matrix(pos[0], pos[1], pos[2])
        
        return model
    
    def release(self):
        """Release GPU resources."""
        self.vao.release()
        self.vbo.release()


# =============================================================================
# GPU Renderer
# =============================================================================

class GPURenderer:
    """High-performance GPU-based 3D renderer using ModernGL."""
    
    VERTEX_SHADER = '''
    #version 330 core
    
    in vec3 in_position;
    in vec3 in_normal;
    
    uniform mat4 mvp;
    uniform mat4 model;
    
    out vec3 frag_normal;
    out vec3 frag_position;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        frag_normal = mat3(model) * in_normal;
        frag_position = vec3(model * vec4(in_position, 1.0));
    }
    '''
    
    FRAGMENT_SHADER = '''
    #version 330 core
    
    in vec3 frag_normal;
    in vec3 frag_position;
    
    uniform vec3 light_dir;
    uniform vec3 base_color;
    
    out vec4 frag_color;
    
    void main() {
        // Two-sided lighting
        vec3 normal = normalize(frag_normal);
        vec3 light = normalize(-light_dir);
        
        // Use abs for two-sided lighting (helps with incorrect normals)
        float diffuse = abs(dot(normal, light));
        float ambient = 0.2;
        
        vec3 color = base_color * (ambient + diffuse * 0.8);
        frag_color = vec4(color, 1.0);
    }
    '''
    
    def __init__(self, width: int, height: int):
        """Initialize renderer with pygame and ModernGL."""
        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("3D Engine - ModernGL")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Disable face culling initially (helps debug visibility issues)
        # self.ctx.enable(moderngl.CULL_FACE)
        
        # Compile shaders
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        
        self.width = width
        self.height = height
        
        # Projection matrix
        self.projection = perspective_matrix(
            fov=60,
            aspect=width / height,
            near=0.1,
            far=1000
        )
        
        # Camera
        self.camera_pos = np.array([0, 0, 10], dtype='f4')
        self.camera_target = np.array([0, 0, 0], dtype='f4')
        
        # Light direction (normalized)
        self.light_dir = np.array([0.3, -0.7, -0.5], dtype='f4')
        self.light_dir /= np.linalg.norm(self.light_dir)
        
        # Mesh storage
        self.meshes: List[GPUMesh] = []
        self.entities: List[Entity] = []
    
    def add_entity(self, entity: Entity, color: tuple = None) -> int:
        """
        Add an Entity to the renderer.
        
        Args:
            entity: Entity instance with loaded OBJ
            color: RGB tuple (0-1), random if None
            
        Returns:
            Index of the mesh
        """
        mesh = GPUMesh(self.ctx, self.program, entity, color)
        self.meshes.append(mesh)
        self.entities.append(entity)
        return len(self.meshes) - 1
    
    def load_entity(self, filename: str, position: tuple = (0, 0, 0), 
                    scale: float = 1.0, color: tuple = None) -> int:
        """
        Load OBJ file and add to renderer.
        
        Args:
            filename: Path to OBJ file
            position: Initial position (x, y, z)
            scale: Scale factor
            color: RGB tuple (0-1), random if None
            
        Returns:
            Index of the mesh
        """
        entity = Entity(filename, position, scale)
        return self.add_entity(entity, color)
    
    def set_camera(self, position: tuple, target: tuple = (0, 0, 0)):
        """Set camera position and target."""
        self.camera_pos = np.array(position, dtype='f4')
        self.camera_target = np.array(target, dtype='f4')
    
    def render(self):
        """Render all meshes."""
        # Clear screen (dark gray background)
        self.ctx.clear(0.15, 0.15, 0.18)
        
        # Compute view matrix
        view = look_at_matrix(self.camera_pos, self.camera_target)
        
        # Set light uniform
        self.program['light_dir'].value = tuple(self.light_dir)
        
        # Render each mesh
        for mesh in self.meshes:
            # Get model matrix from entity (includes position, rotation, scale)
            model = mesh.get_model_matrix()
            
            # Compute MVP (Model-View-Projection)
            mvp = model @ view @ self.projection
            
            # Set uniforms
            self.program['mvp'].write(mvp.astype('f4').tobytes())
            self.program['model'].write(model.astype('f4').tobytes())
            self.program['base_color'].value = tuple(mesh.color)
            
            # Draw
            mesh.vao.render(moderngl.TRIANGLES)
        
        # Swap buffers
        pygame.display.flip()
    
    def cleanup(self):
        """Release all GPU resources."""
        for mesh in self.meshes:
            mesh.release()
        self.program.release()
        self.ctx.release()


# =============================================================================
# Demo Application
# =============================================================================

def main():
    """Demo: Render multiple objects using Entity class."""
    SCREEN_SIZE = (800, 600)
    NUM_OBJECTS = 100
    
    # Create renderer
    renderer = GPURenderer(*SCREEN_SIZE)
    clock = pygame.time.Clock()
    
    print(f"Loading {NUM_OBJECTS} objects...")
    
    # Grid layout for objects
    grid_size = int(np.ceil(np.sqrt(NUM_OBJECTS)))
    spacing = 5.0
    
    for i in range(NUM_OBJECTS):
        # Grid position
        row = i // grid_size
        col = i % grid_size
        
        # Center the grid
        x = (col - grid_size / 2) * spacing
        y = (row - grid_size / 2) * spacing
        z = 0
        
        # Random color
        color = (
            0.3 + 0.7 * np.random.random(),
            0.3 + 0.7 * np.random.random(),
            0.3 + 0.7 * np.random.random()
        )
        
        # Load entity - scale down since the OBJ is small
        renderer.load_entity(
            "example/stairs_modular_right.obj",
            position=(x, y, z),
            scale=0.5,
            color=color
        )
    
    print(f"Done! Loaded {len(renderer.meshes)} meshes")
    
    # Camera setup - position it to see the grid
    camera_distance = grid_size * spacing * 0.8
    camera_angle = 0
    
    running = True
    rotation_speed = 0.5
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get delta time
        dt = clock.tick(60) / 1000.0
        
        # Orbit camera around the scene
        camera_angle += dt * 0.3
        cam_x = np.sin(camera_angle) * camera_distance
        cam_z = np.cos(camera_angle) * camera_distance
        cam_y = camera_distance * 0.5  # Look from above
        
        renderer.set_camera(
            position=(cam_x, cam_y, cam_z),
            target=(0, 0, 0)
        )
        
        # Rotate each entity
        # Note: We only update the angle property - the GPU uses the model matrix
        # so no need to re-upload vertex data!
        for i, entity in enumerate(renderer.entities):
            # Each object rotates at slightly different speed
            # Directly modify the internal angle array for efficiency
            entity._angle[1] += dt * rotation_speed * (1 + i * 0.01)
        
        # Render
        renderer.render()
        
        # Update title with FPS
        fps = clock.get_fps()
        pygame.display.set_caption(f"3D Engine - {NUM_OBJECTS} objects - {fps:.1f} FPS")
    
    renderer.cleanup()
    pygame.quit()
    print("Done!")


def main_simple():
    """Simple demo with single object for debugging."""
    SCREEN_SIZE = (800, 600)
    
    renderer = GPURenderer(*SCREEN_SIZE)
    clock = pygame.time.Clock()
    
    # Load single entity at origin
    print("Loading single object...")
    renderer.load_entity(
        "example/stairs_modular_right.obj",
        position=(0, 0, 0),
        scale=1.0,
        color=(0.8, 0.4, 0.2)
    )
    print(f"Entity position: {renderer.entities[0].position}")
    print(f"Entity vertices range: {renderer.entities[0].vertices.min(axis=0)} to {renderer.entities[0].vertices.max(axis=0)}")
    
    # Camera looking at origin from front
    renderer.set_camera(position=(0, 5, 15), target=(0, 0, 0))
    
    angle = 0
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        dt = clock.tick(60) / 1000.0
        
        # Rotate the entity - directly set angle (GPU uses model matrix)
        angle += dt * 0.5
        renderer.entities[0]._angle[1] = angle
        
        renderer.render()
        
        fps = clock.get_fps()
        pygame.display.set_caption(f"3D Engine - Single Object - {fps:.1f} FPS")
    
    renderer.cleanup()
    pygame.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        main_simple()
    else:
        main()
