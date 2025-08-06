#
# This file defines the task to pick and place an object using a Franka robot with path planning.
# By: Jackson Crandell
#

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
import random


MATERIALS = ["Iron", "Brass", "Aluminum"]

class PickPlacePathPlanningTask(BaseTask):
    def __init__(
        self,
        name: str,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        pick_position: Optional[np.ndarray] = None,
        pick_orientation: Optional[np.ndarray] = None,
        place_position: Optional[np.ndarray] = None,
        place_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:

        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_name = target_name
        self._target = None
        self._target_prim_path = target_prim_path
        self.pick_position = pick_position
        self.pick_orientation = pick_orientation
        self._place_position = place_position
        self._place_orientation = place_orientation
        self._target_visual_material = None
        self._obstacles = OrderedDict()
        
        # Multiple target management for sampling
        self._targets = {}  # Dictionary to store all target objects
        self._available_materials = MATERIALS.copy()  # Materials not yet sampled
        self._sampled_materials = []  # Materials that have been sampled
        self._current_target = None  # Currently selected target for manipulation
        self._sampling_complete = False
        self._material_index = 0
        
        if self.pick_orientation is None:
            self.pick_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
        if self._place_position is None:
            self._place_position = np.array([0.0, -0.45, 0.41]) / get_stage_units()
        return
    

    def create_targets(self):
        """Create all three material targets as rectangular prisms"""
        for i, material in enumerate(MATERIALS):
            target_name = f"target_{material}"
            target_prim_path = f"/World/{target_name}"
            # Position targets in a line on the turn table
            target_position = np.array([0.2 - i * 0.2, -0.6, 0.45])
            target_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
            
            colors = {
                "Aluminum": np.array([0.8, 0.8, 0.8]),  
                "Brass": np.array([0.8, 0.7, 0.3]),     
                "Iron": np.array([0.4, 0.4, 0.4])      
            }
            
            target = self.scene.add(
                DynamicCuboid(
                    name=target_name,
                    prim_path=target_prim_path,
                    position=target_position,
                    orientation=target_orientation,
                    color=colors[material],
                    size=1.0,
                    scale=np.array([0.03, 0.03, 0.1]) / get_stage_units(),
                )
            )
            self._targets[material] = target
            self._task_objects[target_name] = target

    def get_targets(self) -> dict:
        """Get all target objects"""
        return self._targets

    def select_next_target(self):
        if not self._available_materials:
            self._sampling_complete = True
            return None
        
        selected_material = self._available_materials[0]
        self._current_target = self._targets[selected_material]
        print(f"Selected target: {selected_material}")
        return self._current_target

    def mark_target_sampled(self):
        """Mark the current target as sampled and move it to sampled list"""
        if self._current_target is None:
            return
        
        # Find which material this target represents
        current_material = None
        for material, target in self._targets.items():
            if target == self._current_target:
                current_material = material
                break
        
        if current_material and current_material in self._available_materials:
            self._available_materials.remove(current_material)
            self._sampled_materials.append(current_material)
            print(f"Material {current_material} has been sampled")
            
            # Change target color to indicate it's been sampled
            visual_material = self._current_target.get_applied_visual_material()
            if visual_material and hasattr(visual_material, "set_color"):
                visual_material.set_color(np.array([0, 1.0, 0]))  # Green for sampled
        
        self._current_target = None

    def all_targets_sampled(self) -> bool:
        """Check if all targets have been sampled"""
        return len(self._available_materials) == 0

    def get_current_target(self):
        """Get the currently selected target"""
        return self._current_target

    def get_sampling_status(self) -> dict:
        """Get current sampling status"""
        return {
            "available_materials": self._available_materials.copy(),
            "sampled_materials": self._sampled_materials.copy(),
            "current_target": self._current_target.name if self._current_target else None,
            "sampling_complete": self.all_targets_sampled()
        }

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        if self._place_orientation is None:
            self._place_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

        # Create all material targets
        self.create_targets()

        # Select the first target for sampling
        self.select_next_target()

        self.turn_table = scene.add(FixedCuboid(prim_path="/World/turn_table",
                                            name="turn_table",
                                            position=np.array([0, -0.75, 0.2]),
                                            scale=np.array([0.8, 0.8, 0.4]),
                                            color=np.array([0.5, 0.5, 0.5])))

        self.weigh_table = scene.add(FixedCuboid(prim_path="/World/weigh_table",
                                            name="weigh_table",
                                            position=np.array([0.65, -0.01125, 0.2]),
                                            scale=np.array([0.3, 0.3, 0.4]),
                                            color=np.array([0.5, 0.5, 0.5])))        

        return


    def get_params(self) -> dict:
        params_representation = dict()
        if self._current_target is not None:
            params_representation["target_prim_path"] = {"value": self._current_target.prim_path, "modifiable": True}
            params_representation["target_name"] = {"value": self._current_target.name, "modifiable": True}
            position, orientation = self._current_target.get_local_pose()
            params_representation["target_position"] = {"value": position, "modifiable": True}
            params_representation["target_orientation"] = {"value": orientation, "modifiable": True}
            params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_task_objects(self) -> dict:
        return self._task_objects

    def get_observations(self) -> dict:
        joints_state = self._robot.get_joints_state()
        observations = {
            self._robot.name: {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            }
        }
        
        # Add current target observations if one is selected
        if self._current_target is not None:
            target_position, target_orientation = self._current_target.get_local_pose()
            observations[self._current_target.name] = {
                "position": np.array(target_position),
                "orientation": np.array(target_orientation)
            }
            
        return observations

    def target_reached(self) -> bool:
        # Check if current target is reached
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        
        if self._current_target is not None:
            target_position, _ = self._current_target.get_world_pose()
        else:
            return False
            
        if np.mean(np.abs(np.array(end_effector_position) - np.array(target_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        if self._target_visual_material is not None:
            if hasattr(self._target_visual_material, "set_color"):
                if self.target_reached():
                    self._target_visual_material.set_color(color=np.array([0, 1.0, 0]))
                else:
                    self._target_visual_material.set_color(color=np.array([1.0, 0, 0]))

        return

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacles.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacles[obstacle_to_delete]
        return

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return None, None
    
    def get_obstacles(self) -> List[SingleXFormPrim]:
        """Returns the list of obstacles added to the task."""
        return list(self._obstacles.values())
    
    def add_obstacles(self):
        # wall_prim_path = find_unique_string_name(initial_name="/World/WallObstacle", is_unique_fn=lambda x: not is_prim_path_valid(x))
        # wall_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not is_prim_path_valid(x))
        # self.wall = self.scene.add(DynamicCuboid(prim_path=wall_prim_path,
        #                                     name=wall_name,
        #                                     position=np.array([0.51, -0.29, 0.266]),
        #                                     orientation=euler_angles_to_quat(np.array([0, 0, np.pi / 4])),
        #                                     scale=np.array([0.04, 0.46, 0.56]),
        #                                     color=np.array([0.5, 0.5, 0.5])))
        # self._obstacles[self.wall.name] = self.wall
        self._obstacles[self.turn_table.name] = self.turn_table
        self._obstacles[self.weigh_table.name] = self.weigh_table
        


class FrankaPathPlanningTask(PickPlacePathPlanningTask):
    def __init__(
        self,
        name: str,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        pick_position: Optional[np.ndarray] = None,
        pick_orientation: Optional[np.ndarray] = None,
        place_position: Optional[np.ndarray] = None,
        place_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        franka_prim_path: Optional[str] = None,
        franka_robot_name: Optional[str] = None,
    ) -> None:
        PickPlacePathPlanningTask.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            pick_position=pick_position,
            pick_orientation=pick_orientation,
            place_position=place_position,
            place_orientation=place_orientation,
            offset=offset,
        )
        self._franka_prim_path = franka_prim_path
        self._franka_robot_name = franka_robot_name
        self._franka = None
        return

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        if self._franka_prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._franka_robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        self._franka = Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name)
        return self._franka

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return (1e15 * np.ones(9), 1e13 * np.ones(9))
