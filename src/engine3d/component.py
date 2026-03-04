from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .gameobject import GameObject

class Component:
    """Base for attachable components like Transform, Object3D, Collider, Rigidbody."""

    def __init__(self):
        self.game_object: Optional['GameObject'] = None

    def on_attach(self):
        pass

    def update(self, delta_time: float):
        pass
