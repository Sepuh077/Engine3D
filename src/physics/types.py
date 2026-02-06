from enum import IntEnum

class ColliderType(IntEnum):
    SPHERE = 0
    CYLINDER = 1
    CUBE = 2
    MESH = 3

    @staticmethod
    def all():
        return [
            ColliderType.SPHERE,
            ColliderType.CYLINDER,
            ColliderType.CUBE,
            ColliderType.MESH
        ]


class CollisionRelation(IntEnum):
    """Collision relation between ObjectGroups.
    IGNORE: No collision detection.
    TRIGGER: Detect collisions but allow pass-through.
    SOLID: Detect collisions and prevent pass-through (block).
    """
    IGNORE = 0
    TRIGGER = 1
    SOLID = 2
