import numpy as np

from src.engine3d.component import Component, Time


class Rigidbody(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False):
        super().__init__()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static

    def add_force(self, force):
        """Simple force application."""
        self.velocity += np.array(force, dtype=np.float32) / self.mass

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        delta_time = Time.delta_time
        if self.use_gravity:
            # Simple gravity: 9.81 m/s^2 downwards
            self.velocity[1] -= 9.81 * delta_time

        # if self.game_object and np.any(self.velocity):
        #     # Apply velocity to position
        #     self.game_object.transform.move(*(self.velocity * delta_time))
