"""
Example game demonstrating the new ObjectGroup collision system.
Tests ignore, detect-pass-through (triggers), and solid (block) groups.
Also demonstrates OnCollisionEnter/Exit/Stay callbacks.
"""
import os
import sys
import math
import pygame

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from src.engine3d import Window3D, Keys, Color
from src.engine3d.object3d import create_cube, create_plane, Object3D
from src.physics import ColliderType, CollisionRelation, ObjectGroup


# Player uses custom OnCollision* (set on instance for demo)
def make_player_callbacks(player):
    def on_enter(other):
        if hasattr(other, 'color_on_trigger'):
            other.color = other.color_on_trigger
        print(f"Player entered collision with {other.name or 'obj'}")
    def on_exit(other):
        if hasattr(other, 'color_normal'):
            other.color = other.color_normal
        print(f"Player exited collision with {other.name or 'obj'}")
    def on_stay(other: Object3D):
        if other.group.name == "Walls":
            print(f"Player Stayed ---------------- with {other.name or 'obj'}")
    player.OnCollisionEnter = on_enter
    player.OnCollisionExit = on_exit
    player.OnCollisionStay = on_stay


class CollisionGroupsExample(Window3D):
    """Tests all ObjectGroup features."""
    
    def setup(self):
        # Create collision groups
        self.player_group = ObjectGroup("Player")
        self.wall_group = ObjectGroup("Walls")
        self.trigger_group = ObjectGroup("Triggers")
        self.ignore_group = ObjectGroup("Ignore")
        
        # Set rules (auto-symmetric via add_group for all types)
        # Player blocks with walls
        self.player_group.add_group(self.wall_group, CollisionRelation.SOLID)
        # Player triggers with trigger objects
        self.player_group.add_group(self.trigger_group, CollisionRelation.TRIGGER)
        # Player ignores ignore_group
        self.player_group.add_group(self.ignore_group, CollisionRelation.IGNORE)
        
        # Floor (solid with player)
        floor = self.add_object(create_plane(30, 30, color=Color.DARK_GRAY, collider_type=ColliderType.CUBE))
        floor.position = (0, -0.5, 0)
        floor.static = True
        floor.group = self.wall_group
        floor.name = "Floor"
        
        # Walls (solid)
        self.walls = []
        wall_positions = [(-10, 1, 0), (10, 1, 0), (0, 1, -10), (0, 1, 10)]
        for pos in wall_positions:
            wall = self.add_object(create_cube(2.0, color=Color.GRAY, collider_type=ColliderType.CUBE))
            wall.position = pos
            wall.static = True
            wall.group = self.wall_group
            wall.name = "Wall"
            self.walls.append(wall)
        
        # Trigger objects (pass through, change color on contact)
        self.triggers = []
        trigger_pos = [(-5, 1, 5), (5, 1, 5)]
        for i, pos in enumerate(trigger_pos):
            trig = self.add_object(create_cube(1.5, color=Color.YELLOW, collider_type=ColliderType.SPHERE))
            trig.position = pos
            trig.group = self.trigger_group
            trig.name = f"Trigger{i}"
            trig.color_normal = Color.YELLOW
            trig.color_on_trigger = Color.PURPLE
            self.triggers.append(trig)
        
        # Ignore objects (can overlap freely, no events)
        self.ignores = []
        ignore_pos = [(-5, 1, -5), (5, 1, -5)]
        for i, pos in enumerate(ignore_pos):
            ign = self.add_object(create_cube(1.5, color=Color.ORANGE, collider_type=ColliderType.CUBE))
            ign.position = pos
            ign.group = self.ignore_group
            ign.name = f"Ignore{i}"
            self.ignores.append(ign)
        
        # Player (cube collider + mesh)
        player_base = create_cube(1.0, color=Color.BLUE, collider_type=ColliderType.CUBE)
        self.player = self.add_object(player_base)
        self.player.scale = 1.0
        self.player.position = (0, 0.5, 0)
        self.player.group = self.player_group
        self.player.name = "Player"
        self.player.move_speed = 10.0
        # Attach callbacks
        make_player_callbacks(self.player)
        
        # Camera
        self.camera.position = (0, 15, 20)
        self.camera.look_at((0, 0, 0))
        
        # Light
        self.light.direction = (0.5, -0.8, -0.5)
        self.light.ambient = 0.4
        
        # UI state
        self.show_colliders = True
        self.collision_count = 0
    
    def on_update(self, delta_time):
        # Player movement with WASD + arrows for Y
        dx = dy = dz = 0.0
        speed = self.player.move_speed * delta_time
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dz -= speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dz += speed
        if keys[pygame.K_SPACE]:
            dy += speed
        if keys[pygame.K_LSHIFT]:
            dy -= speed
        
        if dx or dy or dz:
            self.move_object(self.player, (dx, dy, dz))
        
        # Count active collisions for display
        self.collision_count = len(self.player._current_collisions)
        
        # Update caption
        pos = self.player.position
        self.set_caption(
            f"Groups Demo - Player: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
            f"Collisions: {self.collision_count} | "
            f"FPS: {self.fps:.0f}"
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.close()
        elif key == Keys.SPACE:
            self.show_colliders = not self.show_colliders
    
    def on_draw(self):
        # Draw colliders if enabled
        if self.show_colliders:
            for obj in self.objects:
                if obj.group:
                    # Color by group type
                    if obj.group == self.player_group:
                        col = Color.BLUE
                    elif obj.group.name == "Walls":
                        col = Color.RED
                    elif obj.group.name == "Triggers":
                        col = Color.PURPLE
                    else:
                        col = Color.WHITE
                    obj.draw_collider(self, col)
        # Note: on_draw can add 2D UI if needed


if __name__ == "__main__":
    print("=== ObjectGroup Collision System Demo ===")
    print("Controls:")
    print("  WASD/Arrows - Move player")
    print("  SPACE - Toggle colliders")
    print("  ESC - Quit")
    print()
    print("Groups:")
    print("  Blue player (cube): collides with walls (red, blocks), triggers (purple, pass-thru), ignores orange")
    print("  Red walls: solid, block player")
    print("  Purple triggers: pass through, change color on contact, OnCollision* called")
    print("  Orange ignores: overlap freely, no detection/events")
    print()
    print("Watch console for Enter/Exit prints and color changes.")
    print()
    game = CollisionGroupsExample(900, 600, "Engine3D - ObjectGroup Collision Demo")
    game.run()
