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
