from typing import Optional, TYPE_CHECKING, Generator, Type, TypeVar, List

if TYPE_CHECKING:
    from .gameobject import GameObject


T = TypeVar('T', bound="Component")


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
        if not self.game_object:
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
    - start(): Called once when the object is created/initialized
    - update(): Called every frame
    - on_collision_enter(other): Called when collision starts
    - on_collision_stay(other): Called every frame while colliding
    - on_collision_exit(other): Called when collision ends
    
    Example:
        class PlayerController(Script):
            def start(self):
                self.speed = 5.0
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
    
    def start(self):
        """
        Called once when the script is first initialized.
        Override to set up initial state.
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
