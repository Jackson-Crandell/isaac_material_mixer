from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.core.api.tasks import BaseTask
import numpy as np
from .pick_place_path_planning_task import FrankaPathPlanningTask
from .path_planning_controller import FrankaRrtController
    
PLACE_POSITION = np.array([0.62, 0.0, 0.55]) # Weigh table position

class MaterialMix(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._sampling_initialized = False

    def setup_scene(self):
        world = self.get_world()
        world.add_task(FrankaPathPlanningTask("Mix",place_position=PLACE_POSITION))
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        return

    def world_cleanup(self):
        self._controller = None
        return

    async def setup_post_load(self):
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()
        my_franka = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        my_franka.disable_gravity()
        self._controller = FrankaRrtController(name="franka_rrt_controller", robot_articulation=my_franka)
        self._articulation_controller = my_franka.get_articulation_controller()
        self._pass_world_state_to_controller()
        world = self.get_world()
        await world.play_async()
        if not world.physics_callback_exists("sim_step"):
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)
        return

    def _pass_world_state_to_controller(self):
        self._controller.reset()
        self._franka_task.add_obstacles()
        for wall in self._franka_task.get_obstacles():
            self._controller.add_obstacle(wall, static=True)

    def _on_follow_target_simulation_step(self, step_size):
        observations = self._world.get_observations()
        
        # Initialize sampling on first run
        if not self._sampling_initialized:
            self._initialize_sampling()
            self._sampling_initialized = True
            return
        
        # Check if current operation is complete
        if self._controller.is_operation_complete():
            # Mark current target as sampled
            self._franka_task.mark_target_sampled()
            
            # Check if all targets have been sampled
            if self._franka_task.all_targets_sampled():
                print("All materials have been sampled! Sampling complete.")
                return
            
            # Select next target for sampling
            next_target = self._franka_task.select_next_target()
            if next_target is not None:
                self._controller.set_current_target(next_target)
                self._controller.reset_to_pick_state()
                # Update task params for the new target
                self._update_task_params_for_current_target()
        
        # Get current target information
        current_target_name = None
        if self._franka_task.get_current_target() is not None:
            current_target_name = self._franka_task.get_current_target().name
        elif self._task_params["target_name"]["value"] in observations:
            current_target_name = self._task_params["target_name"]["value"]
        
        if current_target_name and current_target_name in observations:
            # Execute pick-place-return sequence
            actions = self._controller.forward(
                initial_pick_position=observations[current_target_name]["position"],
                initial_pick_orientation=observations[current_target_name]["orientation"],
                place_position=PLACE_POSITION,
                place_orientation=observations[current_target_name]["orientation"],
            )
            
            kps, kds = self._franka_task.get_custom_gains()
            self._articulation_controller.set_gains(kps, kds)
            self._articulation_controller.apply_action(actions)
        
        return

    def _initialize_sampling(self):
        print("Sample Material")
        first_target = self._franka_task.select_next_target()
        if first_target is not None:
            self._controller.set_current_target(first_target)
            self._update_task_params_for_current_target()
            print(f"Starting sampling with first target: {first_target.name}")

    def _update_task_params_for_current_target(self):
        current_target = self._franka_task.get_current_target()
        if current_target is not None:
            # Update the task params to point to the current target
            self._task_params["target_name"]["value"] = current_target.name
            position, orientation = current_target.get_local_pose()
            self._task_params["target_position"]["value"] = position
            self._task_params["target_orientation"]["value"] = orientation

