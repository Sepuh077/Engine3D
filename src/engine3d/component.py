from typing import Optional, TYPE_CHECKING, Generator, Type, TypeVar, List, Any, Generic, Tuple, Union
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .gameobject import GameObject


T = TypeVar('T', bound="Component")


class InspectorFieldType(Enum):
    """Types of inspector fields supported."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    COLOR = "color"
    VECTOR3 = "vector3"
    ENUM = "enum"
    LIST = "list"
    COMPONENT_REF = "component_ref"


@dataclass
class InspectorFieldInfo:
    """
    Metadata for an inspector field.
    
    Attributes:
        name: Display name in the inspector
        field_type: Type of the field (float, int, bool, etc.)
        default_value: Default value for the field
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step increment (for numeric types)
        decimals: Number of decimal places (for float types)
        enum_options: List of (value, label) tuples for enum types
        tooltip: Optional tooltip text
        list_item_type: The type of items in a list field (for LIST type)
        component_type: The component class type (for COMPONENT_REF type)
    """
    name: str
    field_type: InspectorFieldType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    decimals: Optional[int] = None
    enum_options: Optional[List[Tuple[Any, str]]] = None
    tooltip: Optional[str] = None
    list_item_type: Optional[Union[type, InspectorFieldType]] = None
    component_type: Optional[type] = None


class InspectorField:
    """
    Descriptor for defining inspector-visible fields on components.
    
    This allows fields to be used naturally in code while providing
    metadata for the inspector UI.
    
    Example:
        class MyComponent(Component):
            speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0)
            enabled = InspectorField(bool, default=True)
            color = InspectorField(color, default=(1.0, 1.0, 1.0))
            targets = InspectorField(list, default=[], list_item_type=float)
            player_transform = InspectorField(Component, default=None, component_type=Transform)
            
            def update(self):
                # Use speed as a regular float
                self.transform.position[0] += self.speed * Time.delta_time
    """
    
    def __init__(
        self,
        field_type: Union[type, InspectorFieldType],
        default: Any = None,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        decimals: Optional[int] = None,
        enum_options: Optional[List[Tuple[Any, str]]] = None,
        tooltip: Optional[str] = None,
        list_item_type: Optional[Union[type, InspectorFieldType]] = None,
        component_type: Optional[type] = None,
    ):
        """
        Initialize an inspector field.
        
        Args:
            field_type: The type of the field (float, int, bool, str, color, vector3, list, Component)
                       or an InspectorFieldType enum value.
            default: Default value for the field (if None, a sensible default is used)
            min_value: Minimum value for numeric types
            max_value: Maximum value for numeric types
            step: Step increment for numeric types
            decimals: Decimal places for float types
            enum_options: List of (value, label) tuples for enum types
            tooltip: Tooltip text for the inspector
            list_item_type: The type of items in a list (for list fields)
            component_type: The component class type (for component reference fields)
        """
        # Convert Python type to InspectorFieldType
        if isinstance(field_type, InspectorFieldType):
            self.field_type = field_type
        elif field_type == float:
            self.field_type = InspectorFieldType.FLOAT
        elif field_type == int:
            self.field_type = InspectorFieldType.INT
        elif field_type == bool:
            self.field_type = InspectorFieldType.BOOL
        elif field_type == str:
            self.field_type = InspectorFieldType.STRING
        elif isinstance(field_type, _ColorType):
            self.field_type = InspectorFieldType.COLOR
        elif isinstance(field_type, _Vector3Type):
            self.field_type = InspectorFieldType.VECTOR3
        elif field_type == list:
            self.field_type = InspectorFieldType.LIST
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            self.field_type = InspectorFieldType.ENUM
            # Auto-generate enum options if not provided
            if enum_options is None:
                enum_options = [(e.value, e.name) for e in field_type]
        elif isinstance(field_type, type) and issubclass(field_type, Component):
            self.field_type = InspectorFieldType.COMPONENT_REF
            # Auto-set component_type if not provided
            if component_type is None:
                component_type = field_type
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
        
        # Set sensible default values if not provided
        if default is None:
            default = self._get_type_default()
        
        self.default_value = default
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.decimals = decimals
        self.enum_options = enum_options
        self.tooltip = tooltip
        self.list_item_type = list_item_type
        self.component_type = component_type
        
        # Will be set by __set_name__
        self.name: Optional[str] = None
        self.private_name: Optional[str] = None
    
    def _get_type_default(self) -> Any:
        """Get the default value for this field type when none is specified."""
        if self.field_type == InspectorFieldType.FLOAT:
            return 0.0
        elif self.field_type == InspectorFieldType.INT:
            return 0
        elif self.field_type == InspectorFieldType.BOOL:
            return False
        elif self.field_type == InspectorFieldType.STRING:
            return ""
        elif self.field_type == InspectorFieldType.COLOR:
            return (1.0, 1.0, 1.0)  # White
        elif self.field_type == InspectorFieldType.VECTOR3:
            return (0.0, 0.0, 0.0)
        elif self.field_type == InspectorFieldType.ENUM:
            # Return first enum option value if available
            if self.enum_options:
                return self.enum_options[0][0]
            return None
        elif self.field_type == InspectorFieldType.LIST:
            return []  # Empty list by default
        elif self.field_type == InspectorFieldType.COMPONENT_REF:
            return None  # No component reference by default
        return None
    
    def __set_name__(self, owner: type, name: str):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        self.private_name = f"_inspector_{name}"
    
    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Get the field value, returning the default if not set."""
        if obj is None:
            return self
        
        value = getattr(obj, self.private_name, None)
        if value is None:
            return self.default_value
        return value
    
    def __set__(self, obj: Any, value: Any):
        """Set the field value."""
        setattr(obj, self.private_name, value)
    
    def get_info(self) -> InspectorFieldInfo:
        """Get the metadata for this inspector field."""
        return InspectorFieldInfo(
            name=self.name or "",
            field_type=self.field_type,
            default_value=self.default_value,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step,
            decimals=self.decimals,
            enum_options=self.enum_options,
            tooltip=self.tooltip,
            list_item_type=self.list_item_type,
            component_type=self.component_type,
        )


# Type markers for inspector fields (use these as field_type in InspectorField)
class _ColorType:
    """Marker type for color fields (RGB/RGBA tuple in 0-1 range)."""
    pass

class _Vector3Type:
    """Marker type for vector3 fields (3D position/direction tuple)."""
    pass

# Export these as singletons for use in InspectorField
color = _ColorType()
vector3 = _Vector3Type()


def inspector_field(
    field_type: Union[type, InspectorFieldType],
    default: Any = None,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    step: Optional[float] = None,
    decimals: Optional[int] = None,
    enum_options: Optional[List[Tuple[Any, str]]] = None,
    tooltip: Optional[str] = None,
    list_item_type: Optional[Union[type, InspectorFieldType]] = None,
    component_type: Optional[type] = None,
) -> InspectorField:
    """
    Convenience function to create an InspectorField.
    
    This is equivalent to using InspectorField directly but may be more
    readable in some contexts.
    """
    return InspectorField(
        field_type,
        default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        decimals=decimals,
        enum_options=enum_options,
        tooltip=tooltip,
        list_item_type=list_item_type,
        component_type=component_type,
    )


class Time:
    """
    Global time utility similar to Unity's Time class.
    """
    delta_time: float = 0.0
    scale: float = 1.0


class Component:
    """Base for attachable components like Transform, Object3D, Collider, Rigidbody."""

    def __init__(self):
        self.game_object: Optional['GameObject'] = None

    def on_attach(self):
        pass

    def update(self):
        pass

    @property
    def transform(self):
        if self.game_object:
            return self.game_object.transform
        return None

    def add_component(self, component: "Component") -> "Component":
        if not self.game_object:
            raise AttributeError("Current component must contain 'game_object' before adding a new component!")
        return self.game_object.add_component(component)

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        if not self.game_object:
            return None
        return self.game_object.get_component(component_type)

    def get_components(self, component_type: Type[T]) -> List[T]:
        if not self.game_object:
            return None
        return self.game_object.get_components(component_type)

    @classmethod
    def get_inspector_fields(cls) -> List[Tuple[str, InspectorFieldInfo]]:
        """
        Get all inspector fields defined on this component class.
        
        Returns a list of (attribute_name, InspectorFieldInfo) tuples for all
        InspectorField descriptors defined on this class and its parents.
        
        Example:
            for name, info in component.get_inspector_fields():
                print(f"{name}: {info.field_type} = {info.default_value}")
        """
        fields = []
        seen = set()
        
        # Walk through the MRO (Method Resolution Order) to get inherited fields
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if name in seen:
                    continue
                if isinstance(value, InspectorField):
                    fields.append((name, value.get_info()))
                    seen.add(name)
        
        return fields

    def get_inspector_field_value(self, name: str) -> Any:
        """
        Get the current value of an inspector field by name.
        
        Args:
            name: The name of the inspector field
            
        Returns:
            The current value of the field
        """
        # Get the descriptor from the class
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            return getattr(self, name)
        return None

    def set_inspector_field_value(self, name: str, value: Any) -> None:
        """
        Set the value of an inspector field by name.
        
        Args:
            name: The name of the inspector field
            value: The new value to set
        """
        # Get the descriptor from the class
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            setattr(self, name, value)


class WaitForSeconds:
    """
    Yield instruction to wait for a specified number of seconds.
    
    Example:
        yield WaitForSeconds(2.0)
    """
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.elapsed = 0.0

    def is_done(self, delta_time: float) -> bool:
        self.elapsed += delta_time
        return self.elapsed >= self.seconds


class WaitForFrames:
    """
    Yield instruction to wait for a number of frames.
    
    Used internally to support `yield None` as "wait one frame".
    """
    def __init__(self, frames: int = 1):
        self.frames = frames

    def step(self) -> bool:
        self.frames -= 1
        return self.frames <= 0


class WaitEndOfFrame:
    """
    Yield instruction to resume at the end of the current frame.

    Example:
        yield WaitEndOfFrame()
    """
    pass


class Script(Component):
    """
    Base class for user-defined scripts/components.
    
    Similar to Unity's MonoBehaviour, scripts can be attached to GameObjects
    and receive lifecycle callbacks:
    - awake(): Called once when the script is first created (before start)
    - start(): Called once when play mode begins
    - update(): Called every frame
    - on_collision_enter(other): Called when collision starts
    - on_collision_stay(other): Called every frame while colliding
    - on_collision_exit(other): Called when collision ends
    
    Example:
        class PlayerController(Script):
            def awake(self):
                # Initialize references here
                self.speed = 5.0
                
            def start(self):
                # Called when play begins
                self.start_coroutine(self.delayed_action())
                
            def update(self):
                # Move player logic here
                pass
                
            def on_collision_enter(self, other):
                print(f"Collided with {other.game_object.name}")

            def delayed_action(self):
                print("Waiting...")
                yield WaitForSeconds(2.0)
                print("Done waiting!")
    """
    
    def __init__(self):
        super().__init__()
        self._started = False
        self._awoken = False
    
    def awake(self):
        """
        Called once when the script is first created, before start().
        Use this for initialization that doesn't depend on other objects.
        Override to set up initial state.
        """
        pass
    
    def start(self):
        """
        Called once when play mode begins.
        Override to set up initial state that may depend on other objects.
        """
        pass
    
    def update(self):
        """
        Called every frame.
        Override to implement per-frame behavior.
        """
        pass
    
    def on_collision_enter(self, other):
        """
        Called when this object's collider starts touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass
    
    def on_collision_stay(self, other):
        """
        Called every frame while this object's collider is touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass
    
    def on_collision_exit(self, other):
        """
        Called when this object's collider stops touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass

    def start_coroutine(self, routine: Generator) -> Generator:
        """
        Starts a coroutine.
        
        Args:
            routine: A generator function (using yield)
            
        Returns:
            The routine generator
        """
        if self.game_object:
            return self.game_object.start_coroutine(routine)
        return routine
