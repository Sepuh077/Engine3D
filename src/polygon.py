import pygame
import numpy as np
import random

from .camera import Camera


class Polygon:
    # OPTIMIZATION: __slots__ for faster attribute access and less memory
    __slots__ = ['entity', 'indexes', 'color']
    
    def __init__(self, entity, indexes):
        self.entity = entity
        self.indexes = np.array(indexes, dtype=np.int32)  # OPTIMIZATION: numpy array for faster indexing
        self.color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

    def is_visible(self, points: np.ndarray, camera: Camera):
        x, y = points[:, 0], points[:, 1]
    
        inside = (x >= camera.left) & (x <= camera.right) & (y >= camera.bottom) & (y <= camera.top)
        return np.any(inside)

    def get_points(self, camera: Camera):
        points = camera.world_to_cam(self.entity.camera_points[self.indexes])
        return self.is_visible(points, camera), points

    @property
    def world_z(self):
        return self.entity.vertices[self.indexes][:, 2].max()

    def draw(self, camera: Camera):
        # if self.is_visible(self.entity.camera_points[self.indexes], camera):
        if np.all(self.entity.valid_points[self.indexes]):
            pygame.draw.polygon(camera, self.color, self.entity.camera_points[self.indexes])
