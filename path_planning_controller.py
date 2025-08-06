import os
from typing import Optional

import carb
import isaacsim.core.api.objects
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
import numpy as np
import omni.kit
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from isaacsim.robot_motion.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.path_planning_interface import PathPlanner
from isaacsim.core.utils.rotations import euler_angles_to_quat


class PathPlannerController(BaseController):
    def __init__(
        self,
        name: str,
        path_planner_visualizer: PathPlannerVisualizer,
        cspace_trajectory_generator: LulaCSpaceTrajectoryGenerator,
        physics_dt=1 / 60.0,
        rrt_interpolation_max_dist=0.01,
    ):
        BaseController.__init__(self, name)

        self._robot = path_planner_visualizer.get_robot_articulation()

        self._cspace_trajectory_generator = cspace_trajectory_generator
        self._path_planner = path_planner_visualizer.get_path_planner()
        self._path_planner_visualizer = path_planner_visualizer

        self._last_solution = None
        self._action_sequence = None

        self._physics_dt = physics_dt
        self._rrt_interpolation_max_dist = rrt_interpolation_max_dist

        self.home_position = np.array([0.5,0,0.5])
        self.home_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))  # Quaternion for no rotation

        self._event = 0
        self._wait_counter = 0
        self._wait_duration = 120  # Wait for 2 seconds at 60 FPS
        self._current_target = None
        self._operation_complete = False

        self.original_pick_position = None
        self.original_pick_orientation = None

        self.intermediate_position = np.array([0, -0.4, 0.6])  # Position to move to after picking up the material

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        # This example uses the LulaCSpaceTrajectoryGenerator to convert RRT waypoints to a cspace trajectory.
        # In general this is not theoretically guaranteed to work since the trajectory generator uses spline-based
        # interpolation and RRT only guarantees that the cspace position of the robot can be linearly interpolated between
        # waypoints.  For this example, we verified experimentally that a dense interpolation of cspace waypoints with a maximum
        # l2 norm of .01 between waypoints leads to a good enough approximation of the RRT path by the trajectory generator.

        interpolated_path = self._path_planner_visualizer.interpolate_path(rrt_plan, self._rrt_interpolation_max_dist)
        trajectory = self._cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)

        return art_trajectory.get_action_sequence()

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> None:
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()

        path_planner_visualizer = PathPlannerVisualizer(self._robot, self._path_planner)
        active_joints = path_planner_visualizer.get_active_joints_subset()
        if self._last_solution is None:
            start_pos = active_joints.get_joint_positions()
        else:
            start_pos = self._last_solution

        self._path_planner.set_max_iterations(10000)
        self._rrt_plan = self._path_planner.compute_path(start_pos, np.array([]))

        if self._rrt_plan is None or len(self._rrt_plan) <= 1:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))
            self._action_sequence = []
            return

        print(len(self._rrt_plan))

        self._action_sequence = self._convert_rrt_plan_to_trajectory(self._rrt_plan)
        self._last_solution = self._action_sequence[-1].joint_positions

    def set_current_target(self, target):
        self._current_target = target
        self._operation_complete = False

    def is_operation_complete(self) -> bool:
        """Check if the current pick-place-return operation is complete"""
        return self._operation_complete

    def reset_to_pick_state(self):
        self._event = 0
        self._action_sequence = None
        self._last_solution = None
        self._operation_complete = False
        self._wait_counter = 0

    def open_gripper(self) -> ArticulationAction:
        print("Opening the gripper")
        self._event += 1
        self._action_sequence = None
        return ArticulationAction(joint_positions=np.array([1.0, 1.0]), joint_indices=np.array([7, 8]))
    
    def close_gripper(self) -> ArticulationAction:
        print("Closing the gripper")
        self._event += 1
        self._action_sequence = None
        return ArticulationAction(joint_positions=np.array([0.0, 0.0]), joint_indices=np.array([7, 8]))
    
    def execute_motion_plan(self, position: np.ndarray, orientation: Optional[np.ndarray] = None, buffer = np.zeros(3)) -> ArticulationAction:
        if self._action_sequence is None:
            self._make_new_plan(position + buffer, orientation)

        if len(self._action_sequence) == 1:
            self._event += 1
            self._action_sequence = None
            #return ArticulationAction(joint_positions=np.array([1.0, 1.0]), joint_indices=np.array([7, 8]))
            return ArticulationAction()

        return self._action_sequence.pop(0)

    def forward(
        self, initial_pick_position: np.ndarray, initial_pick_orientation: Optional[np.ndarray],place_position: np.ndarray, place_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        
        if self._event == 0:
            # Save for later
            self.original_pick_position = initial_pick_position
            self.original_pick_orientation = initial_pick_orientation
            return self.open_gripper()

        elif self._event == 1:
            return self.execute_motion_plan(initial_pick_position, initial_pick_orientation, buffer=np.array([0, 0, 0.05]))

        elif self._event == 2:
            self._event += 1
            self._action_sequence = None
            return self.close_gripper()
        
        elif self._event == 3:
            # Move to intermediate position
            return self.execute_motion_plan(self.intermediate_position, initial_pick_orientation)

        elif self._event == 4:
            # Move up
            return self.execute_motion_plan(initial_pick_position, initial_pick_orientation, buffer=np.array([0, 0, 0.1]))

        elif self._event == 5:
            return self.execute_motion_plan(place_position, place_orientation)
    
        elif self._event == 6:
            # Move down
            return self.execute_motion_plan(place_position, place_orientation,buffer=np.array([0, 0, -0.07]))
        
        elif self._event == 7:
            return self.open_gripper()
        
        elif self._event == 8:
            # Return Home
            return self.execute_motion_plan(self.home_position, self.home_orientation)
        
        elif self._event == 9:
            # Wait at weigh station
            self._wait_counter += 1
            if self._wait_counter >= self._wait_duration:
                self._event += 1
                self._wait_counter = 0
                print("Finished waiting at weigh station")
            # Return current position to maintain position during wait
            return ArticulationAction()
        
        elif self._event == 10:
            # Move to final position 
            return self.execute_motion_plan(place_position, place_orientation)
    
        elif self._event == 11:
            # Move down
            return self.execute_motion_plan(place_position, place_orientation,buffer=np.array([0, 0, -0.07]))
        
        elif self._event == 12:
            return self.close_gripper()
        
        elif self._event == 13:
            # Move back up
            return self.execute_motion_plan(place_position, place_orientation,buffer=np.array([0,0,0.05]))
        
        elif self._event == 14:
            # Return material back to original pick position
            return self.execute_motion_plan(self.original_pick_position, self.original_pick_orientation,buffer=np.array([0,0,0.05]))

        elif self._event == 15:
            return self.open_gripper()

        elif self._event == 16:
            # Operation complete - signal that this cycle is done
            self._operation_complete = True
            self._event += 1
            print("Pick-place-return operation complete")
            return ArticulationAction()
        
        else:
            # Default case - return to idle
            return ArticulationAction()
        
    def add_obstacle(self, obstacle: isaacsim.core.api.objects, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: isaacsim.core.api.objects) -> None:
        self._path_planner.remove_obstacle(obstacle)

    def reset(self) -> None:
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        self._action_sequence = None
        self._last_solution = None

    def get_path_planner(self) -> PathPlanner:
        return self._path_planner


class FrankaRrtController(PathPlannerController):
    def __init__(
        self,
        name,
        robot_articulation: SingleArticulation,
    ):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.examples.interactive")
        examples_extension_path = ext_manager.get_extension_path(ext_id)

        # Load default RRT config files stored in the isaacsim.robot_motion.motion_generation extension
        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")

        # Replace the default robot description file with a copy that has inflated collision spheres
        rrt_config["robot_description_path"] = os.path.join(
            examples_extension_path,
            "isaacsim",
            "examples",
            "interactive",
            "path_planning",
            "path_planning_example_assets",
            "franka_conservative_spheres_robot_description.yaml",
        )
        rrt = RRT(**rrt_config)

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            rrt_config["robot_description_path"], rrt_config["urdf_path"]
        )

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        for i in range(len(rrt.get_active_joints())):
            assert cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i)
            assert cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i)

        visualizer = PathPlannerVisualizer(robot_articulation, rrt)

        PathPlannerController.__init__(self, name, visualizer, cspace_trajectory_generator)
