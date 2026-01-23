# 3D Engine Performance Optimization Guide

## Current Performance Issues

Your engine is slow because of these main bottlenecks:

| Issue | Impact | Location |
|-------|--------|----------|
| Sorting polygons every frame | HIGH | `entity.py:101` |
| Individual polygon draw calls | CRITICAL | `polygon.py:31` |
| Numpy array copies in properties | MEDIUM | `entity.py:57,74` |
| No frustum culling | MEDIUM | All visible polys drawn |
| Python loop overhead | HIGH | `entity.py:102-103` |

## Solutions by Performance Gain

### 🚀 TIER 1: GPU Acceleration (10-100x faster)

**Use OpenGL via ModernGL or PyOpenGL**

This is the definitive solution. Modern GPUs are designed for this workload.

```python
# Example with ModernGL
import moderngl
import numpy as np

ctx = moderngl.create_standalone_context()

# Vertex shader (runs on GPU, massively parallel)
vertex_shader = '''
#version 330
in vec3 in_position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
'''

# Create VBO once, GPU handles all transforms
vbo = ctx.buffer(vertices.astype('f4').tobytes())
```

**Install:** `pip install moderngl moderngl-window`

### 🔥 TIER 2: Numba JIT Compilation (5-20x faster)

Compile Python to machine code:

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True, cache=True)
def transform_vertices_fast(vertices, position, scale, angles):
    """JIT-compiled vertex transformation - runs at C speed."""
    result = np.empty_like(vertices)
    n = vertices.shape[0]
    
    # Parallel loop over vertices
    for i in prange(n):
        # Transform each vertex
        x = (vertices[i, 0] - position[0]) * scale + position[0]
        y = (vertices[i, 1] - position[1]) * scale + position[1]
        z = (vertices[i, 2] - position[2]) * scale + position[2]
        result[i] = (x, y, z)
    
    return result

@jit(nopython=True, cache=True)
def sort_polygons_by_z_fast(vertices, polygon_indices, index_offsets, index_lengths):
    """Fast polygon sorting using JIT."""
    n_polys = len(index_offsets)
    max_z = np.empty(n_polys, dtype=np.float32)
    
    for i in range(n_polys):
        start = index_offsets[i]
        length = index_lengths[i]
        max_val = -np.inf
        for j in range(length):
            idx = polygon_indices[start + j]
            if vertices[idx, 2] > max_val:
                max_val = vertices[idx, 2]
        max_z[i] = max_val
    
    return np.argsort(max_z)
```

**Install:** `pip install numba`

### ⚡ TIER 3: Algorithmic Optimizations (2-5x faster)

#### 3a. Cache Sorted Order
```python
# BAD: Sorting every frame
def draw(self, camera):
    self.polygons = sorted(self.polygons, key=lambda p: p.world_z)  # O(n log n)
    
# GOOD: Cache and invalidate only when needed
def draw(self, camera):
    if self._geometry_changed:
        self._sorted_indices = np.argsort([p.world_z for p in self.polygons])
        self._geometry_changed = False
    
    for i in self._sorted_indices:
        self.polygons[i].draw(camera)
```

#### 3b. Batch Coordinate Transforms
```python
# BAD: Transform in world_to_cam called per polygon
points = camera.world_to_cam(self.entity.camera_points[self.indexes])

# GOOD: Transform ALL vertices once, index into result
self._all_screen_coords = camera.world_to_cam(self.vertices)
# Then in polygon.draw():
points = self._all_screen_coords[self.indexes]  # Just indexing, no transform
```

#### 3c. Use Float32 Instead of Float64
```python
# Float32 is 2x faster for SIMD operations
vertices = np.array(vertices, dtype=np.float32)
```

#### 3d. Frustum Culling with Bounding Boxes
```python
def _compute_bounds(self):
    """Pre-compute axis-aligned bounding box."""
    self._min_bounds = self.vertices.min(axis=0)
    self._max_bounds = self.vertices.max(axis=0)

def is_visible(self, camera):
    """Quick AABB vs frustum test."""
    return (self._max_bounds[0] >= camera.left and 
            self._min_bounds[0] <= camera.right and
            self._max_bounds[1] >= camera.bottom and 
            self._min_bounds[1] <= camera.top)
```

### 💪 TIER 4: Reduce Python Overhead (1.5-3x faster)

#### 4a. Avoid Property Copies
```python
# BAD
@property
def position(self):
    return np.copy(self._position)  # Allocates new array EVERY call

# GOOD - Return view, document that it shouldn't be modified
@property  
def position(self):
    return self._position  # No allocation
```

#### 4b. Use __slots__ for Classes
```python
class Polygon:
    __slots__ = ['entity', 'indexes', 'color']  # 20-30% faster attribute access
    
    def __init__(self, entity, indexes):
        self.entity = entity
        self.indexes = indexes
        self.color = (...)
```

#### 4c. Pre-allocate Arrays
```python
# BAD: Allocating every frame
screen = np.column_stack((x_coords, y_coords))

# GOOD: Pre-allocate and reuse
if self._screen_buffer is None:
    self._screen_buffer = np.empty((len(vertices), 2), dtype=np.float32)
self._screen_buffer[:, 0] = x_coords
self._screen_buffer[:, 1] = y_coords
```

## Quick Wins - Apply These Now

### 1. Fix the sorting (entity.py)
```python
def draw(self, camera: Camera):
    self.set_camera_points(camera)
    
    # Only sort when geometry changes
    if self._needs_sort:
        self._sorted_order = sorted(range(len(self.polygons)), 
                                    key=lambda i: self.polygons[i].world_z)
        self._needs_sort = False
    
    for i in self._sorted_order:
        self.polygons[i].draw(camera)
```

### 2. Remove np.copy() from properties (entity.py)
```python
@property
def position(self):
    return self._position  # Return view, not copy
```

### 3. Use float32 everywhere
```python
self.vertices = np.array(vertices, dtype=np.float32)
```

## Benchmark Results (Expected)

| Optimization | 100 Objects FPS |
|--------------|-----------------|
| Current code | ~0.3 FPS |
| + Cached sorting | ~1 FPS |
| + float32 + no copies | ~2 FPS |
| + Numba JIT | ~15 FPS |
| + ModernGL (GPU) | ~200+ FPS |

## Recommended Architecture

For a production 3D engine in Python, use this stack:

```
┌─────────────────────────────────────────┐
│           Your Game Logic               │
├─────────────────────────────────────────┤
│   ModernGL / PyOpenGL (GPU Rendering)   │
├─────────────────────────────────────────┤
│    Pygame/Pyglet (Window & Input)       │
├─────────────────────────────────────────┤
│   NumPy + Numba (Fast Math)             │
└─────────────────────────────────────────┘
```

## Example: ModernGL Integration

See `main_moderngl.py` for a complete example using GPU acceleration.
