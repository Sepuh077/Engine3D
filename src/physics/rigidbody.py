import numpy as np

from src.engine3d.component import Component, Time, InspectorField


class Rigidbody(Component):
    """Physics body for velocity, forces etc. Similar to Unity Rigidbody."""
    
    # Inspector fields
    use_gravity = InspectorField(bool, default=True, tooltip="Whether gravity affects this body")
    is_kinematic = InspectorField(bool, default=False, tooltip="If true, physics won't move this object")
    is_static = InspectorField(bool, default=False, tooltip="If true, this object never moves")
    mass = InspectorField(float, default=1.0, min_value=0.001, max_value=10000.0, step=0.1, decimals=2, tooltip="Mass of the rigidbody")
    drag = InspectorField(float, default=0.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Drag coefficient")

    def __init__(self, use_gravity: bool = True, is_kinematic: bool = False, is_static: bool = False, drag: float = 0.0):
        super().__init__()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.mass = 1.0
        self.is_static = is_static
        self.drag = drag

    def add_force(self, force):
        """Simple force application."""
        self.velocity += np.array(force, dtype=np.float32) / self.mass

    def update(self):
        if self.is_static or self.is_kinematic:
            return

        delta_time = Time.delta_time
            
        if self.drag > 0.0:
            # Apply drag to gradually decrease velocity
            drag_factor = max(0.0, 1.0 - self.drag * delta_time)
            self.velocity[0] *= drag_factor
            self.velocity[2] *= drag_factor
            if not self.use_gravity:
                self.velocity[1] *= drag_factor

        if self.use_gravity:
            # Simple gravity: 9.81 m/s^2 downwards
            self.velocity[1] -= 9.81 * delta_time

        if self.game_object and np.any(self.velocity):
            # Apply velocity to position
            self.game_object.transform.move(*(self.velocity * delta_time))
