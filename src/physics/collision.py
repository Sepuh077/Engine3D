import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from src.physics.collider import Collider
from .types import ColliderType


@dataclass
class CollisionManifold:
    normal: np.ndarray  # Normal pointing from B to A
    depth: float        # Penetration depth

def sphere_vs_sphere_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    ca, ra = a.get_world_sphere()
    cb, rb = b.get_world_sphere()
    diff = ca - cb
    dist_sq = diff.dot(diff)
    radius_sum = ra + rb
    
    if dist_sq > radius_sum ** 2:
        return None

    dist = np.sqrt(dist_sq)
    if dist < 1e-6:
        # Centers are coincident, choose arbitrary normal
        normal = np.array([0, 1, 0], dtype=np.float32)
        depth = radius_sum
    else:
        normal = diff / dist
        depth = radius_sum - dist
        
    return CollisionManifold(normal, depth)

def _obb_manifold(Ca, Aa, Ea, Cb, Ab, Eb) -> Optional[CollisionManifold]:
    # Translation vector from B to A (in world space)
    t = Ca - Cb
    
    # We need to express t in B's model space and A's model space for projections
    # But SAT is easier if we test axes in World space or a common frame.
    # The standard efficient implementation rotates axes into the other's frame.
    
    # Rotation matrix from B to A
    # R_AB = A.T @ B  (maps vector in B to A) - wait, let's stick to world axes projection
    
    # Axes to test: 3 from A, 3 from B, 9 cross products
    axes = []
    
    # A's local axes in world space are the columns of Aa
    for i in range(3):
        axes.append(Aa[:, i])
        
    # B's local axes in world space
    for i in range(3):
        axes.append(Ab[:, i])
        
    # Cross products
    for i in range(3):
        for j in range(3):
            axis = np.cross(Aa[:, i], Ab[:, j])
            if np.dot(axis, axis) > 1e-6: # Skip near-zero axes
                axes.append(axis / np.linalg.norm(axis))

    min_overlap = float('inf')
    best_axis = np.zeros(3)
    
    for axis in axes:
        # Project center distance
        # t is vector from B to A
        proj_t = abs(np.dot(t, axis))
        
        # Project extents of A
        # Radius of A along axis = sum of dot(axis, A_axis_i) * extent_i
        ra = sum(abs(np.dot(axis, Aa[:, i])) * Ea[i] for i in range(3))
        
        # Project extents of B
        rb = sum(abs(np.dot(axis, Ab[:, i])) * Eb[i] for i in range(3))
        
        overlap = (ra + rb) - proj_t
        
        if overlap < 0:
            return None # Separating axis found
            
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis

    # Ensure normal points from B to A
    if np.dot(best_axis, t) < 0:
        best_axis = -best_axis
        
    return CollisionManifold(best_axis, min_overlap)

def obb_vs_obb_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    Ca, Aa, Ea = a.get_world_obb()
    Cb, Ab, Eb = b.get_world_obb()
    return _obb_manifold(Ca, Aa, Ea, Cb, Ab, Eb)

def sphere_vs_obb_manifold(sphere_obj: Collider, obb_obj: Collider) -> Optional[CollisionManifold]:
    cs, rs = sphere_obj.get_world_sphere()
    Cb, Ab, Eb = obb_obj.get_world_obb()

    # Find closest point on OBB to sphere center
    d = cs - Cb
    local = Ab.T @ d
    closest_local = np.clip(local, -Eb, Eb)
    closest_world = Cb + Ab @ closest_local
    
    diff = cs - closest_world
    dist_sq = diff.dot(diff)
    
    if dist_sq > rs ** 2:
        return None
        
    dist = np.sqrt(dist_sq)
    
    if dist < 1e-6:
        # Center inside OBB? Or exactly on surface
        # For simplicity, if center is inside, we need a different strategy to find exit normal,
        # but here we might just use the vector from box center?
        # A proper implementation for inside point:
        # Find minimum distance to any face
        pass # TODO: Handle deep penetration better
        normal = (cs - Cb) 
        normal /= np.linalg.norm(normal)
        depth = rs
    else:
        normal = diff / dist
        depth = rs - dist
        
    return CollisionManifold(normal, depth)

# ... (For brevity, other mixed types could return simple boolean or approx manifold)
# We will fallback to OBB for others or implement as needed. 
# The user prioritized normal check.

def cylinder_vs_sphere_manifold(cyl: Collider, sph: Collider) -> Optional[CollisionManifold]:
    Cc, rc, hc = cyl.get_world_cylinder()
    cs, rs = sph.get_world_sphere()

    # 1. Clamp sphere center to cylinder's vertical range (segment)
    # Cylinder is Y-aligned in world space for now (based on get_world_cylinder implementation)
    # Cc is center. range [Cc.y - hc, Cc.y + hc]
    
    dy = cs[1] - Cc[1]
    clamped_y = np.clip(dy, -hc, hc)
    closest_point_on_axis = np.array([Cc[0], Cc[1] + clamped_y, Cc[2]], dtype=np.float32)
    
    # 2. Find closest point on cylinder surface to sphere center
    # Vector from axis point to sphere center
    d = cs - closest_point_on_axis
    d_len_sq = d.dot(d)
    
    if d_len_sq < 1e-6:
        # Sphere center is exactly on axis. Push out horizontally?
        # Or if strictly inside, push out via Y if closer?
        # Default to pushing out horizontally X
        normal = np.array([1, 0, 0], dtype=np.float32)
        depth = rs + rc # Full overlap approx
        # Check if vertical push is shorter
        if hc - abs(dy) < rc:
             # Closer to top/bottom cap
             normal = np.array([0, np.sign(dy), 0], dtype=np.float32)
             depth = (hc + rs) - abs(dy)
    else:
        d_len = np.sqrt(d_len_sq)
        
        # Check if sphere is touching/intersecting
        # Distance from axis is d_len. 
        # Collision if d_len < rc + rs.
        if d_len >= rc + rs:
            return None
            
        normal = d / d_len
        depth = (rc + rs) - d_len
        
    # Correct normal direction: Points B (Sphere) -> A (Cylinder)
    # Above we calculated normal = sphere - axis_point (Away from cylinder axis).
    # So normal points Cylinder -> Sphere.
    # Manifold expects B -> A. So Sphere -> Cylinder.
    # We need to invert it.
    
    return CollisionManifold(-normal, depth)

def cylinder_vs_cylinder_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    Ca, ra, ha = a.get_world_cylinder()
    Cb, rb, hb = b.get_world_cylinder()
    
    # 1. Vertical Check (Y-axis SAT)
    dy = Ca[1] - Cb[1]
    y_overlap = (ha + hb) - abs(dy)
    if y_overlap < 0:
        return None
        
    # 2. Horizontal Check (Circle-Circle)
    dx = Ca[0] - Cb[0]
    dz = Ca[2] - Cb[2]
    dist_sq = dx*dx + dz*dz
    r_sum = ra + rb
    
    if dist_sq >= r_sum * r_sum:
        return None
        
    dist = np.sqrt(dist_sq)
    horizontal_overlap = r_sum - dist
    
    # Determine Separating Axis (Smallest overlap)
    if y_overlap < horizontal_overlap:
        # Vertical collision
        normal = np.array([0, np.sign(dy), 0], dtype=np.float32)
        depth = y_overlap
    else:
        # Horizontal collision
        if dist < 1e-6:
            normal = np.array([1, 0, 0], dtype=np.float32)
        else:
            normal = np.array([dx, 0, dz], dtype=np.float32) / dist
        depth = horizontal_overlap

    # Normal B -> A
    # We calculated vector A - B (Ca - Cb). So it points B -> A. Correct.
    return CollisionManifold(normal, depth)

def cylinder_vs_obb_manifold(cyl: Collider, obb: Collider) -> Optional[CollisionManifold]:
    # SAT Implementation
    # Cylinder A: Center Cc, Radius rc, HalfHeight hc. Axis Y (0,1,0).
    # OBB B: Center Cb, Axes Ab, Extents Eb.
    
    Cc, rc, hc = cyl.get_world_cylinder()
    Cb, Ab, Eb = obb.get_world_obb()
    
    # Axes to test:
    # 1. OBB Axes (3)
    # 2. Cylinder Axis (1) -> (0,1,0)
    # 3. Cross products (Cylinder Axis x OBB Axes) -> (0,1,0) x Ab[i]
    
    cyl_axis = np.array([0, 1, 0], dtype=np.float32)
    
    axes = []
    # OBB axes
    for i in range(3):
        axes.append(Ab[:, i])
    # Cylinder axis
    axes.append(cyl_axis)
    # Cross products
    for i in range(3):
        axis = np.cross(cyl_axis, Ab[:, i])
        if np.dot(axis, axis) > 1e-6:
            axes.append(axis / np.linalg.norm(axis))
            
    min_overlap = float('inf')
    best_axis = np.zeros(3)
    
    # Vector B -> A
    t = Cc - Cb
    
    for axis in axes:
        # Project OBB B
        # Radius = sum(abs(dot(axis, axis_i)) * extent_i)
        rb = sum(abs(np.dot(axis, Ab[:, i])) * Eb[i] for i in range(3))
        
        # Project Cylinder A
        # Cylinder projection = (Projection of Height) + (Projection of Radius)
        # Height Proj = abs(dot(axis, cyl_axis)) * hc
        # Radius Proj = radius * length(axis projected on plane perpendicular to cyl_axis)
        #             = radius * sin(angle between axis and cyl_axis)
        #             = radius * length(cross(axis, cyl_axis))
        #             = radius * sqrt(1 - dot(axis, cyl_axis)^2)
        
        dot_cyl = abs(np.dot(axis, cyl_axis))
        h_proj = dot_cyl * hc
        r_proj = rc * np.sqrt(max(0, 1.0 - dot_cyl**2))
        
        ra = h_proj + r_proj
        
        # Distance projection
        dist_proj = abs(np.dot(t, axis))
        
        overlap = (ra + rb) - dist_proj
        
        if overlap < 0:
            return None
            
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis
            
    # Ensure normal B -> A
    if np.dot(best_axis, t) < 0:
        best_axis = -best_axis
        
    return CollisionManifold(best_axis, min_overlap)


def closest_point_on_triangle(p, a, b, c):
    # Check if P in vertex region outside A
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a

    # Check if P in vertex region outside B
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b

    # Check if P in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab

    # Check if P in vertex region outside C
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c

    # Check if P in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac

    # Check if P in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

def sphere_vs_mesh_manifold(sph: Collider, mesh: Collider) -> Optional[CollisionManifold]:
    # 1. Transform sphere to Model Space of the Mesh
    # Mesh data: (vertices, faces, model_matrix)
    if mesh.mesh_data is None:
        return None
    vertices, faces, model_mat = mesh.mesh_data
    
    cs_world, rs_world = sph.get_world_sphere()
    
    # Invert Model Matrix
    # We assume uniform scale for simplicity in radius transform
    # M = T * R * S
    # InvM * P_world = P_local
    try:
        inv_model = np.linalg.inv(model_mat)
    except np.linalg.LinAlgError:
        return None
        
    cs_local_4 = inv_model @ np.array([cs_world[0], cs_world[1], cs_world[2], 1.0])
    cs_local = cs_local_4[:3]
    
    # Scale radius?
    # Scale factor is length of a basis vector in model mat (assuming uniform)
    scale_sq = np.dot(model_mat[:3, 0], model_mat[:3, 0])
    scale = np.sqrt(scale_sq)
    rs_local = rs_world / scale
    
    min_dist_sq = rs_local * rs_local
    closest_pt_local = None
    
    # 2. Iterate Faces
    # This loop is slow for Python. Ideally use AABB tree or BVH.
    # Check AABB of mesh first?
    # Mesh Local AABB is usually around origin or computed?
    # We skip for now.
    
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        pt = closest_point_on_triangle(cs_local, v0, v1, v2)
        
        diff = cs_local - pt
        dist_sq = np.dot(diff, diff)
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_pt_local = pt
            
    if closest_pt_local is None:
        return None
        
    # 3. Transform result back to World
    # Closest Point World
    cp_local_4 = np.array([closest_pt_local[0], closest_pt_local[1], closest_pt_local[2], 1.0])
    cp_world_4 = model_mat @ cp_local_4
    cp_world = cp_world_4[:3]
    
    # Distance in World Space
    diff_world = cs_world - cp_world
    dist_world = np.linalg.norm(diff_world)
    
    if dist_world > rs_world:
        return None # Should not happen if logic matches, but precision
        
    if dist_world < 1e-6:
        # Deep penetration / exact center. Push out via triangle normal?
        # We need the normal of the triangle we hit.
        # Recompute normal of closest_pt_local's triangle?
        # For now, simplistic push up?
        normal = np.array([0, 1, 0], dtype=np.float32) 
        depth = rs_world
    else:
        normal = diff_world / dist_world
        depth = rs_world - dist_world
        
    return CollisionManifold(normal, depth)

def cylinder_vs_mesh_manifold(cyl: Collider, mesh: Collider) -> Optional[CollisionManifold]:
    # Approximate Cylinder as a Sphere for Mesh collision (Simplest robust step)
    # A vertical capsule approximation would be better (check segment vs mesh)
    # But for "Player vs Floor/Stairs", a Sphere check at the feet (or center) is often acceptable
    # if the cylinder is roughly as wide as it is tall.
    # If tall cylinder, this might miss head collisions.
    # Let's try to do a Sphere check with a radius that covers the cylinder?
    # No, that's too big.
    # Let's use the cylinder's actual sphere approximation (center, radius=max(r, h))
    
    # Better: Treat it as Sphere vs Mesh but use Cylinder's world sphere.
    # This is what get_world_sphere() returns for Cylinder.
    return sphere_vs_mesh_manifold(cyl, mesh)

def aabb_overlap(a: Collider, b: Collider) -> bool:
    # Fast AABB broadphase (cheaper reject than sphere for boxes)
    amin, amax = a.get_world_aabb()
    bmin, bmax = b.get_world_aabb()
    return not (amax[0] < bmin[0] or amax[1] < bmin[1] or amax[2] < bmin[2] or
                amin[0] > bmax[0] or amin[1] > bmax[1] or amin[2] > bmax[2])

def get_collision_manifold(a: Collider, b: Collider) -> Optional[CollisionManifold]:
    # Broad phase: AABB then sphere (faster rejects)
    if not aabb_overlap(a, b):
        return None
    if not sphere_vs_sphere_manifold(a, b): 
        return None
        
    type_a = getattr(a, "type", ColliderType.CUBE)
    type_b = getattr(b, "type", ColliderType.CUBE)

    # MESH Handling
    if type_a == ColliderType.MESH and type_b == ColliderType.MESH:
        return None # Mesh vs Mesh too expensive/not supported
    
    if type_a == ColliderType.SPHERE and type_b == ColliderType.MESH:
        return sphere_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.SPHERE:
        m = sphere_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.MESH:
        return cylinder_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    if type_a == ColliderType.CUBE and type_b == ColliderType.MESH:
        # Fallback: Approximate Cube as Sphere
        return sphere_vs_mesh_manifold(a, b)
    if type_a == ColliderType.MESH and type_b == ColliderType.CUBE:
        m = sphere_vs_mesh_manifold(b, a)
        if m: m.normal = -m.normal
        return m

    # ... (Rest of existing logic)
    # 1. Sphere vs Sphere
    if type_a == ColliderType.SPHERE and type_b == ColliderType.SPHERE:
        return sphere_vs_sphere_manifold(a, b)

    # 2. Cube vs Cube
    if type_a == ColliderType.CUBE and type_b == ColliderType.CUBE:
        return obb_vs_obb_manifold(a, b)
        
    # 3. Sphere vs Cube
    if type_a == ColliderType.SPHERE and type_b == ColliderType.CUBE:
        return sphere_vs_obb_manifold(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.SPHERE:
        m = sphere_vs_obb_manifold(b, a)
        if m: m.normal = -m.normal
        return m
        
    # 4. Cylinder vs Cylinder
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CYLINDER:
        return cylinder_vs_cylinder_manifold(a, b)
        
    # 5. Cylinder vs Sphere
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.SPHERE:
        return cylinder_vs_sphere_manifold(a, b)
    if type_a == ColliderType.SPHERE and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_sphere_manifold(b, a) # b is cyl, a is sphere.
        if m: m.normal = -m.normal
        return m
        
    # 6. Cylinder vs Cube
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CUBE:
        return cylinder_vs_obb_manifold(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.CYLINDER:
        m = cylinder_vs_obb_manifold(b, a)
        if m: m.normal = -m.normal
        return m

    # Fallback
    return obb_vs_obb_manifold(a, b)


def objects_collide(a: Collider, b: Collider) -> bool:
    """Legacy wrapper returning boolean."""
    return get_collision_manifold(a, b) is not None


def collide_point_with_radius(point: np.ndarray, collider: Collider, radius: float = 1.0) -> bool:
    """
    Check collision treating the point as a sphere with a given radius.
    """
    point_proxy = Collider(ColliderType.SPHERE)
    point_proxy.sphere = (point, radius)
    return objects_collide(point_proxy, collider)


