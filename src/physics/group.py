from typing import List

from .types import CollisionRelation


class ObjectGroup:
    _registry = {}

    def __init__(self, name: str = "default"):
        if name in ObjectGroup._registry:
            raise ValueError(f"Group with name '{name}' already exists")
        self.name = name
        self.ignore_collisions: List['ObjectGroup'] = []
        self.detect_pass_through: List['ObjectGroup'] = []
        self.detect_block: List['ObjectGroup'] = []
        ObjectGroup._registry[name] = self

    def get_relation(self, other: 'ObjectGroup') -> CollisionRelation:
        if other in self.ignore_collisions:
            return CollisionRelation.IGNORE
        if other in self.detect_pass_through:
            return CollisionRelation.TRIGGER
        if other in self.detect_block:
            return CollisionRelation.SOLID
        return CollisionRelation.IGNORE

    def get_groups_for_relation(self, relation: CollisionRelation) -> List['ObjectGroup']:
        if relation == CollisionRelation.IGNORE:
            return self.ignore_collisions
        if relation == CollisionRelation.TRIGGER:
            return self.detect_pass_through
        if relation == CollisionRelation.SOLID:
            return self.detect_block
        return []

    def add_group(self, other: 'ObjectGroup', relation: CollisionRelation):
        if other is self:
            raise ValueError("Cannot add self to group")
        # Check no existing relation (any type) to avoid conflicts
        for rel in [CollisionRelation.IGNORE, CollisionRelation.TRIGGER, CollisionRelation.SOLID]:
            if other in self.get_groups_for_relation(rel) or self in other.get_groups_for_relation(rel):
                raise ValueError(f"Group '{other.name}' already related to '{self.name}'")
        # Auto-symmetric
        groups = self.get_groups_for_relation(relation)
        if other not in groups:
            groups.append(other)
        other_groups = other.get_groups_for_relation(relation)
        if self not in other_groups:
            other_groups.append(self)
