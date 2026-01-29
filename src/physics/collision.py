import numpy as np

from src.physics.collider import Collider
from src.physics import ColliderType


def sphere_vs_sphere(a: Collider, b: Collider):
    ca, ra = a.get_world_sphere()
    cb, rb = b.get_world_sphere()
    diff = ca - cb
    return diff.dot(diff) <= (ra + rb) ** 2


def _obb_overlap(Ca, Aa, Ea, Cb, Ab, Eb):
    R = Aa.T @ Ab
    t = Ab.T @ (Cb - Ca)

    absR = np.abs(R) + 1e-6

    for i in range(3):
        ra = Ea[i]
        rb = Eb @ absR[i, :]
        if abs(t[i]) > ra + rb:
            return False

    for i in range(3):
        ra = Ea @ absR[:, i]
        rb = Eb[i]
        if abs(t @ R[:, i]) > ra + rb:
            return False

    for i in range(3):
        for j in range(3):
            ra = Ea[(i + 1) % 3] * absR[(i + 2) % 3, j] + Ea[(i + 2) % 3] * absR[(i + 1) % 3, j]
            rb = Eb[(j + 1) % 3] * absR[i, (j + 2) % 3] + Eb[(j + 2) % 3] * absR[i, (j + 1) % 3]
            if abs(t[(i + 2) % 3] * R[(i + 1) % 3, j] - t[(i + 1) % 3] * R[(i + 2) % 3, j]) > ra + rb:
                return False

    return True


def obb_vs_obb(a: Collider, b: Collider):
    Ca, Aa, Ea = a.get_world_obb()
    Cb, Ab, Eb = b.get_world_obb()
    return _obb_overlap(Ca, Aa, Ea, Cb, Ab, Eb)


def sphere_vs_obb(sphere_obj: Collider, obb_obj: Collider):
    cs, rs = sphere_obj.get_world_sphere()
    Cb, Ab, Eb = obb_obj.get_world_obb()

    d = cs - Cb
    local = Ab.T @ d
    closest_local = np.clip(local, -Eb, Eb)
    closest_world = Cb + Ab @ closest_local
    diff = cs - closest_world
    return diff.dot(diff) <= rs ** 2


def sphere_vs_cylinder(sphere_obj: Collider, cyl_obj: Collider):
    cs, rs = sphere_obj.get_world_sphere()
    Cc, rc, hc = cyl_obj.get_world_cylinder()

    # Closest point on infinite cylinder axis (aligned with Y)
    clamped_y = np.clip(cs[1], Cc[1] - hc, Cc[1] + hc)
    dx = cs[0] - Cc[0]
    dz = cs[2] - Cc[2]
    dist_xz_sq = dx * dx + dz * dz
    if dist_xz_sq > rc * rc:
        scale = rc / np.sqrt(dist_xz_sq)
        dx *= scale
        dz *= scale
    closest = np.array([Cc[0] + dx, clamped_y, Cc[2] + dz], dtype=np.float32)
    diff = cs - closest
    return diff.dot(diff) <= rs ** 2


def cylinder_vs_cylinder(a: Collider, b: Collider):
    Ca, ra, ha = a.get_world_cylinder()
    Cb, rb, hb = b.get_world_cylinder()

    # Vertical overlap
    if abs(Ca[1] - Cb[1]) > ha + hb:
        return False

    dx = Ca[0] - Cb[0]
    dz = Ca[2] - Cb[2]
    return dx * dx + dz * dz <= (ra + rb) ** 2


def cylinder_vs_obb(cyl_obj: Collider, obb_obj: Collider):
    # Approximate cylinder as an upright box for collision against OBB
    Cc, rc, hc = cyl_obj.get_world_cylinder()
    Ca = Cc
    Aa = np.eye(3, dtype=np.float32)
    Ea = np.array([rc, hc, rc], dtype=np.float32)

    Cb, Ab, Eb = obb_obj.get_world_obb()
    return _obb_overlap(Ca, Aa, Ea, Cb, Ab, Eb)


def objects_collide(a: Collider, b: Collider):
    # Broad phase using bounding spheres
    if not sphere_vs_sphere(a, b):
        return False

    type_a = getattr(a, "type", ColliderType.CUBE)
    type_b = getattr(b, "type", ColliderType.CUBE)

    if type_a == ColliderType.SPHERE and type_b == ColliderType.SPHERE:
        return True  # already passed sphere test

    if type_a == ColliderType.CUBE and type_b == ColliderType.CUBE:
        return obb_vs_obb(a, b)

    if type_a == ColliderType.SPHERE and type_b == ColliderType.CUBE:
        return sphere_vs_obb(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.SPHERE:
        return sphere_vs_obb(b, a)

    if type_a == ColliderType.SPHERE and type_b == ColliderType.CYLINDER:
        return sphere_vs_cylinder(a, b)
    if type_a == ColliderType.CYLINDER and type_b == ColliderType.SPHERE:
        return sphere_vs_cylinder(b, a)

    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CYLINDER:
        return cylinder_vs_cylinder(a, b)

    if type_a == ColliderType.CYLINDER and type_b == ColliderType.CUBE:
        return cylinder_vs_obb(a, b)
    if type_a == ColliderType.CUBE and type_b == ColliderType.CYLINDER:
        return cylinder_vs_obb(b, a)

    # Fallback to OBB test for any unknown combination
    return obb_vs_obb(a, b)


def collide_point_with_radius(point: np.ndarray, collider: Collider, radius: float = 1.0) -> bool:
    """
    Check collision treating the point as a sphere with a given radius.
    Useful for 'thick' raycasts or forgiving mouse picking.
    """
    # Create a temporary proxy collider
    point_proxy = Collider(ColliderType.SPHERE)
    # We only need to populate the sphere cache for Sphere-vs-X checks
    point_proxy.sphere = (point, radius)
    
    return objects_collide(point_proxy, collider)

