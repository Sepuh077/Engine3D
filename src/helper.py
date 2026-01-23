import numpy as np


def rotate_points_batch(points, center, angles):
    """
    Rotate a batch of 3D points around a center by angles (a1, a2, a3).

    points : np.ndarray of shape (N, 3)
    center : tuple or array (cx, cy, cz)
    angles : tuple of radians (a1, a2, a3) for rotation around X, Y, Z

    Returns: np.ndarray of shape (N, 3)
    """
    # Unpack angles
    is_numpy = isinstance(points, np.ndarray)

    a1, a2, a3 = angles
    cosx, sinx = np.cos(a1), np.sin(a1)
    cosy, siny = np.cos(a2), np.sin(a2)
    cosz, sinz = np.cos(a3), np.sin(a3)

    # Translate points to origin
    pts = points - np.array(center)

    # --- Rotate around X ---
    y = pts[:, 1] * cosx - pts[:, 2] * sinx
    z = pts[:, 1] * sinx + pts[:, 2] * cosx
    if is_numpy:
        pts[:, 1], pts[:, 2] = y, z
    else:
        pts = pts.at[:, 1].set(y)
        pts = pts.at[:, 2].set(z)

    # --- Rotate around Y ---
    x = pts[:, 0] * cosy + pts[:, 2] * siny
    z = -pts[:, 0] * siny + pts[:, 2] * cosy
    if is_numpy:
        pts[:, 0], pts[:, 2] = x, z
    else:
        pts = pts.at[:, 0].set(x)
        pts = pts.at[:, 2].set(z)

    # --- Rotate around Z ---
    x = pts[:, 0] * cosz - pts[:, 1] * sinz
    y = pts[:, 0] * sinz + pts[:, 1] * cosz
    if is_numpy:
        pts[:, 0], pts[:, 1] = x, y
    else:
        pts = pts.at[:, 0].set(x)
        pts = pts.at[:, 1].set(y)

    # Translate back
    pts += np.array(center)

    return pts


def rotate_point(point, center, angles):
    return rotate_points_batch(np.array([point]), center, angles)[0]
