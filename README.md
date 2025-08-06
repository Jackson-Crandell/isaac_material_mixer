# Material Mixer Station Logic

Automated sampling system for materials (e.g., **Iron**, **Brass**, and **Aluminum**) using **RRT-based path planning**.  
The system randomly selects material, transports them to a weigh station, waits for measurement (2 seconds), then returns material to original positions (object turns green after it has been sampled).

*Note: It runs a bit slow as it is NOT using CUDA.*

---

## Short Demo
![Material Mixer Demo](short_mix.gif)

## [Full Demo Download](https://drive.google.com/file/d/1L-N2A3Ie_bkHQ52ZcOFm34ysKu_qWoc4/view?usp=sharing)

## `pick_place_path_planing_task.py`
- Sets up the pick/place task: objects, materials (targets)
- Handles sampling and task-finished logic

## `hello_world.py`
- Boilerplate Isaac Sim code to run a scene  
- Handles the target sampling and switching

## `path_planning_controller.py`
- Motion planning and execution using **RRT** for collision-free planning of the robotic arm  
- Contains sequential controller for complete pick-place cycle

---

## How to Run
To run on your own computer, put in `user_examples`  
[Isaac Sim Custom Interactive Examples](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/custom_interactive_examples.html)

---

## Sim2Real Development Pipeline

1. **Integrate ROS2**  
   - [How to Use MoveIt with Isaac Sim: A Step-by-Step Guide](https://www.youtube.com/watch?v=pGje2slp6-s&pp=0gcJCfwAo7VqN5tD)
2. **Setup MoveIt 2 ROS planner** and achieve the same performance (or better) as the built-in planner  
3. **Add basic logic for dropped objects** (integrate MoveIt Grasps)  
4. **Add sensors for object pose estimation** with planning  
   - RGBD Camera for weight table, material table, etc.  
5. **Substitute Cuboids for real-world items** (e.g., material bottles, turn table, etc.)  
6. **Add more logic for failures/error handling**  
7. **Optimize arm motion planning** (can be done concurrently)  
8. **Start scaling/domain randomization** to understand limitations  
   - Vary position  
   - Sensor noise  
   - Actuator speed/limits (drive above and below real-world)

---

## Data Replay

Given **rosbag data**, you could:  
- Setup this scene  
- Spawn items and replay the arm motion  
- Observe if failures, errors, or successes can be replicated

---

## CI/CD

Testing a **new end-to-end workflow** can be achieved with Isaac Sim:  
1. Isaac Sim provides sensor inputs  
2. Control code sends joint commands to the arm  
3. Output can be defined as success/failure (or any custom metric)

---

## Other Cases
- Workcell Optimization  
- Edge-case Simulation  
- Synthetic Data Generation  

---


## Why Isaac Sim Over Others

- **High Fidelity Physics + ROS2 Integration**  
- **Drake** might be the most accurate for Sim2Real, ROS integration is less straightforward  
- Isaac Sim is computationally heavy, but with a capable GPU, it’s very fast  
- **Gazebo** is great for quick simulator development  
  - But lacks photorealism and advanced PhysX 5 contact physics offered by Isaac Sim

---

