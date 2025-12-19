import math
import numpy as np
import pygame


class Camera(pygame.Surface):
    def __init__(self, size, position: tuple = (0, 0, 0), angle: float = 0, spread: float = math.pi / 6):
        super().__init__(size)
        self.position = np.array([*position])
        self.angle = angle # Does not have any effect yet
        self.spread = spread

    @property
    def left(self):
        return self.position[0]

    @property
    def right(self):
        return self.left + self.get_width()

    @property
    def bottom(self):
        return self.position[0]

    @property
    def top(self):
        return self.bottom + self.get_height()

    def world_to_cam(self, points):
        depth = points[:, 2] - self.position[2]
        near = 1e-6

        valid = depth > near

        diff = depth * math.tan(self.spread)
        h = self.get_height() + 2 * diff
        w = self.get_width() + 2 * diff

        screen = np.column_stack((
            (points[:, 0] - self.left + diff) / w * self.get_width(),
            (points[:, 1] - self.bottom + diff) / h * self.get_height()
        ))

        return screen, valid
