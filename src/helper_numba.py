"""
Numba JIT-compiled helper functions for 5-20x speedup.

To use this module:
1. pip install numba
2. Replace `from src.helper import ...` with `from src.helper_numba import ...`

The first call will be slow (compilation), subsequent calls are blazing fast.
"""
import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed. Falling back to pure numpy.")
    print("Install with: pip install numba")


if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def rotate_points_batch(points, center, angles):
        """
        JIT-compiled rotation of 3D points around a center.
        Runs at near-C speed after first compilation.
        
        points : np.ndarray of shape (N, 3), dtype float32
        center : array (cx, cy, cz)
        angles : tuple of radians (a1, a2, a3) for rotation around X, Y, Z
        
        Returns: np.ndarray of shape (N, 3)
        """
        a1, a2, a3 = angles[0], angles[1], angles[2]
        cosx, sinx = np.cos(a1), np.sin(a1)
        cosy, siny = np.cos(a2), np.sin(a2)
        cosz, sinz = np.cos(a3), np.sin(a3)
        
        cx, cy, cz = center[0], center[1], center[2]
        
        n = points.shape[0]
        result = np.empty((n, 3), dtype=np.float32)
        
        for i in range(n):
            # Translate to origin
            x = points[i, 0] - cx
            y = points[i, 1] - cy
            z = points[i, 2] - cz
            
            # Rotate around X
            y_new = y * cosx - z * sinx
            z_new = y * sinx + z * cosx
            y, z = y_new, z_new
            
            # Rotate around Y
            x_new = x * cosy + z * siny
            z_new = -x * siny + z * cosy
            x, z = x_new, z_new
            
            # Rotate around Z
            x_new = x * cosz - y * sinz
            y_new = x * sinz + y * cosz
            x, y = x_new, y_new
            
            # Translate back
            result[i, 0] = x + cx
            result[i, 1] = y + cy
            result[i, 2] = z + cz
        
        return result

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def rotate_points_batch_parallel(points, center, angles):
        """
        Parallel JIT-compiled rotation using all CPU cores.
        Best for large point clouds (>10000 points).
        """
        a1, a2, a3 = angles[0], angles[1], angles[2]
        cosx, sinx = np.cos(a1), np.sin(a1)
        cosy, siny = np.cos(a2), np.sin(a2)
        cosz, sinz = np.cos(a3), np.sin(a3)
        
        cx, cy, cz = center[0], center[1], center[2]
        
        n = points.shape[0]
        result = np.empty((n, 3), dtype=np.float32)
        
        for i in prange(n):  # Parallel loop
            # Translate to origin
            x = points[i, 0] - cx
            y = points[i, 1] - cy
            z = points[i, 2] - cz
            
            # Rotate around X
            y_new = y * cosx - z * sinx
            z_new = y * sinx + z * cosx
            y, z = y_new, z_new
            
            # Rotate around Y
            x_new = x * cosy + z * siny
            z_new = -x * siny + z * cosy
            x, z = x_new, z_new
            
            # Rotate around Z
            x_new = x * cosz - y * sinz
            y_new = x * sinz + y * cosz
            x, y = x_new, y_new
            
            # Translate back
            result[i, 0] = x + cx
            result[i, 1] = y + cy
            result[i, 2] = z + cz
        
        return result

    @jit(nopython=True, cache=True, fastmath=True)
    def world_to_screen_fast(points, cam_pos, spread, width, height):
        """
        JIT-compiled world to screen coordinate transformation.
        
        points: (N, 3) float32 array
        cam_pos: (3,) camera position
        spread: float, field of view spread
        width, height: screen dimensions
        
        Returns: screen_coords (N, 2), valid_mask (N,)
        """
        n = points.shape[0]
        screen = np.empty((n, 2), dtype=np.float32)
        valid = np.empty(n, dtype=np.bool_)
        
        near = 1e-6
        tan_spread = np.tan(spread)
        
        for i in range(n):
            depth = points[i, 2] - cam_pos[2]
            valid[i] = depth > near
            
            if valid[i]:
                diff = depth * tan_spread
                h = height + 2 * diff
                w = width + 2 * diff
                
                screen[i, 0] = (points[i, 0] - cam_pos[0] + diff) / w * width
                screen[i, 1] = (points[i, 1] - cam_pos[1] + diff) / h * height
            else:
                screen[i, 0] = 0
                screen[i, 1] = 0
        
        return screen, valid

    @jit(nopython=True, cache=True)
    def compute_polygon_z_values(vertices, poly_indices, poly_offsets, poly_lengths):
        """
        Fast computation of max Z value for each polygon for depth sorting.
        
        vertices: (N, 3) vertex array
        poly_indices: flat array of all polygon vertex indices
        poly_offsets: starting offset for each polygon in poly_indices
        poly_lengths: number of vertices in each polygon
        
        Returns: (num_polys,) array of max Z values
        """
        n_polys = len(poly_offsets)
        max_z = np.empty(n_polys, dtype=np.float32)
        
        for i in range(n_polys):
            start = poly_offsets[i]
            length = poly_lengths[i]
            max_val = -np.inf
            
            for j in range(length):
                idx = poly_indices[start + j]
                if vertices[idx, 2] > max_val:
                    max_val = vertices[idx, 2]
            
            max_z[i] = max_val
        
        return max_z

else:
    # Fallback to numpy implementation
    from .helper import rotate_points_batch
    
    def rotate_points_batch_parallel(points, center, angles):
        return rotate_points_batch(points, center, angles)
    
    def world_to_screen_fast(points, cam_pos, spread, width, height):
        import math
        depth = points[:, 2] - cam_pos[2]
        near = 1e-6
        valid = depth > near
        
        diff = depth * math.tan(spread)
        h = height + 2 * diff
        w = width + 2 * diff
        
        screen = np.column_stack((
            (points[:, 0] - cam_pos[0] + diff) / w * width,
            (points[:, 1] - cam_pos[1] + diff) / h * height
        ))
        
        return screen.astype(np.float32), valid
    
    def compute_polygon_z_values(vertices, poly_indices, poly_offsets, poly_lengths):
        n_polys = len(poly_offsets)
        max_z = np.empty(n_polys, dtype=np.float32)
        
        for i in range(n_polys):
            start = poly_offsets[i]
            length = poly_lengths[i]
            indices = poly_indices[start:start + length]
            max_z[i] = vertices[indices, 2].max()
        
        return max_z


def rotate_point(point, center, angles):
    """Single point rotation (convenience wrapper)."""
    return rotate_points_batch(np.array([point], dtype=np.float32), center, angles)[0]


# Pre-compile functions on import (warm-up)
if HAS_NUMBA:
    _dummy = np.zeros((3, 3), dtype=np.float32)
    _center = np.zeros(3, dtype=np.float32)
    _angles = np.zeros(3, dtype=np.float32)
    rotate_points_batch(_dummy, _center, _angles)  # Trigger compilation
