from typing import List, Optional, Type, TypeVar, Generator, Any, Dict
import numpy as np

from .component import Component, Script, WaitForSeconds, WaitEndOfFrame, WaitForFrames, Time
from .transform import Transform

T = TypeVar('T', bound=Component)

class GameObject:
    def __init__(self, name: str = "GameObject"):
        self.name = name
        self.tag = None
        self.components: List[Component] = []
        
        # Every GameObject has a Transform
        self.transform = Transform()
        self.add_component(self.transform)

        # Coroutines state
        self._active_coroutines: List[Dict[str, Any]] = []
        self._end_of_frame_coroutines: List[Dict[str, Any]] = []

    def add_component(self, component: Component) -> Component:
        component.game_object = self
        self.components.append(component)
        component.on_attach()
        return component

    def start_coroutine(self, routine: Generator) -> Generator:
        """Starts a coroutine on this GameObject."""
        # Initial step to get first yield value
        try:
            yield_val = next(routine)
            self._queue_coroutine(routine, yield_val)
        except StopIteration:
            pass
        return routine

    def _queue_coroutine(self, routine: Generator, wait_instruction: Any):
        """Queues a coroutine based on wait instruction."""
        wait = wait_instruction
        # Treat None as wait one frame
        if wait is None:
            wait = WaitForFrames(1)
        
        entry = {
            "generator": routine,
            "wait_instruction": wait
        }
        if isinstance(wait, WaitEndOfFrame):
            self._end_of_frame_coroutines.append(entry)
        else:
            self._active_coroutines.append(entry)

    def _step_coroutines(
        self,
        coroutines: List[Dict[str, Any]],
        delta_time: float,
        allow_end_of_frame: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Advance coroutines and return active + end-of-frame queues."""
        still_active = []
        end_of_frame_queue = []
        for coro in coroutines:
            gen = coro["generator"]
            wait = coro["wait_instruction"]

            is_ready = True
            if isinstance(wait, WaitForSeconds):
                if not wait.is_done(delta_time):
                    is_ready = False
            elif isinstance(wait, WaitForFrames):
                if not wait.step():
                    is_ready = False

            if is_ready:
                try:
                    new_wait = next(gen)
                except StopIteration:
                    continue

                if new_wait is None:
                    new_wait = WaitForFrames(1)

                entry = {
                    "generator": gen,
                    "wait_instruction": new_wait,
                }
                if isinstance(new_wait, WaitEndOfFrame):
                    if allow_end_of_frame:
                        end_of_frame_queue.append(entry)
                    else:
                        still_active.append(entry)
                else:
                    still_active.append(entry)
            else:
                still_active.append(coro)
        
        return still_active, end_of_frame_queue

    def _update_coroutines(self, delta_time: float):
        """Processes active coroutines during the main update phase."""
        self._active_coroutines, new_end_of_frame = self._step_coroutines(
            self._active_coroutines,
            delta_time,
            allow_end_of_frame=True,
        )
        if new_end_of_frame:
            self._end_of_frame_coroutines.extend(new_end_of_frame)

    def _update_end_of_frame_coroutines(self, delta_time: float):
        """Processes coroutines waiting for end of frame."""
        self._end_of_frame_coroutines, deferred_end_of_frame = self._step_coroutines(
            self._end_of_frame_coroutines,
            delta_time,
            allow_end_of_frame=False,
        )
        if deferred_end_of_frame:
            self._end_of_frame_coroutines.extend(deferred_end_of_frame)

    def update(self):
        # Update components
        for comp in self.components:
            comp.update()
        
        # Update coroutines (main phase)
        self._update_coroutines(Time.delta_time)

    def update_end_of_frame(self):
        """Called by Window3D to process end-of-frame coroutines."""
        self._update_end_of_frame_coroutines(Time.delta_time)

    def start_scripts(self):
        """Call start() on all Script components that haven't been started yet."""
        for comp in self.components:
            if isinstance(comp, Script) and not comp._started:
                comp.start()
                comp._started = True

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None

    def get_components(self, component_type: Type[T]) -> List[T]:
        return [comp for comp in self.components if isinstance(comp, component_type)]

    def __repr__(self):
        return f"GameObject(name='{self.name}', position={self.transform.position})"
