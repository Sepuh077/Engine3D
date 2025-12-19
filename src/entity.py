from typing import List
import numpy as np

from src.polygon import Polygon
from src.camera import Camera
from src.helper import rotate_points_batch


class Entity:
    def __init__(self, filename: str, position: tuple = (0, 0, 0), scale: float = 1):
        self._load_obj(filename)
        self.position = np.array(position)
        self.scale = scale
        self._angle = np.zeros(3)
        self._updated = True
        self._camera_points = None
        self._valid_points = None

    def _load_obj(self, filename):
        vertices = []
        self.polygons: List[Polygon] = []
        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append(
                        list(map(lambda x: float(x), line.split()[1:]))
                    )
                elif line.startswith("f "):
                    self.polygons.append(
                        Polygon(
                            self,
                            [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                        )
                    )

        self.vertices = np.array(vertices)
        self._find_position()

    def _find_position(self):
        self._position = self.vertices.mean(axis=0)
        self._scale = 1

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if self.scale == value:
            return
        self.vertices += self.position + (self.vertices - self.position) / self.scale * value - self.vertices
        self._scale = value
        self._updated = True

    @property
    def angle(self):
        return np.copy(self._angle)

    @angle.setter
    def angle(self, value: tuple):
        if np.all(self.angle == value):
            return
        diff_angle = np.array(value) - self.angle
        self.vertices += rotate_points_batch(
            self.vertices,
            self.position,
            diff_angle
        ) - self.vertices
        self._angle = np.array(value)
        self._updated = True

    @property
    def position(self):
        return np.copy(self._position)

    @position.setter
    def position(self, value):
        if np.all(self.position == value):
            return
        diff = value - self.position
        self.vertices += diff
        self._position = value
        self._updated = True

    @property
    def camera_points(self):
        return self._camera_points

    @property
    def valid_points(self):
        return self._valid_points

    def set_camera_points(self, camera: Camera):
        if not self._updated:
            return self._camera_points
        self._camera_points, self._valid_points = camera.world_to_cam(self.vertices)
        self._updated = False

    def draw(self, camera: Camera):
        self.set_camera_points(camera)
        self.polygons = sorted(self.polygons, key=lambda p: p.world_z)
        for polygon in self.polygons:
            polygon.draw(camera)
