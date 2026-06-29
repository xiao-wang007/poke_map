# Quintic polynomial feedforward for finger robot

**Session ID:** ses_0ebb9e6c5ffeyLRZLrK4zhfQ01
**Created:** 6/29/2026, 5:46:06 PM
**Updated:** 6/29/2026, 10:03:14 PM

---

## User

now look at the new file quintic_poly.py. I am construting a quintic scalar polynomial for my last action, velocity magnitude, assuming the striking phase has travel distance L = 0.3 (m). Then with this polynomial, I can compute a feedforward force to my finger robot.

Have a look of all necessary files, generate a implementation plan

---

## Assistant (Plan · DeepSeek V4 Pro · 67.4s)

_Thinking:_

Let me understand the user's request. They want me to:

1. Look at the new file `quintic_poly.py`
2. Understand what's needed for a quintic scalar polynomial for their last action (velocity magnitude)
3. Consider a striking phase with travel distance L = 0.3m
4. Generate an implementation plan that uses this polynomial to compute feedforward force for their finger robot

Let me first explore the codebase to understand the structure, then read the relevant files.

**Tool: task**

**Input:**
```json
{
  "description": "Explore codebase structure",
  "prompt": "Explore the codebase at /home/xiao/0_codes/poke_map thoroughly. I need to understand:\n\n1. The overall project structure - list all files and directories\n2. Read quintic_poly.py in full\n3. Read any files related to \"finger robot\", \"feedforward\", \"trajectory\", \"polynomial\", \"striking\", \"poke\"\n4. Read any main scripts or entry points\n5. Look for any existing trajectory generation or control code\n6. Look for any configuration files or constants\n\nReturn the full content of quintic_poly.py and any related files, along with a summary of the project structure and purpose.",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_0ebb9d2cfffePeH3cVuYwtmTxv" state="completed">
<task_result>
Now I have a comprehensive understanding of the codebase. Let me provide the complete summary.

---

## Project Summary: `poke_map`

### Purpose
This project is a **reinforcement learning system for robotic pushing/poking tasks** using NVIDIA Isaac Sim. The agent learns to **push/poke objects (an L-shaped object and a cylinder) toward target positions** on a planar workspace using a **3-DOF prismatic finger robot** (a "finger robot" with X, Y, Z prismatic joints). The system uses a **spatial actor-critic architecture with a U-Net backbone and FiLM-conditioned Q-head** (inspired by the HACMan paper).

---

### Full Project Structure

```
/home/xiao/0_codes/poke_map/
├── README.md                           # Minimal (1 line: "# poke_map")
├── config.yaml                          # Central configuration (112 lines)
├── train_it.md                          # Training workflow instructions
├── .gitignore
├── .opencodeignore
│
├── env/                                 # Isaac Sim environment & execution
│   ├── README.md                        # Environment documentation
│   ├── config_loader.py                 # YAML config loader (11 lines)
│   ├── quintic_poly.py                  # Quintic polynomial trajectory (10 lines)
│   ├── make_finger_robot.py             # Builds 3-DOF prismatic finger robot (347 lines)
│   ├── object_sampler.py                # Stub (6 lines)
│   ├── scene_setup.py                   # Simple scene setup tutorial (102 lines)
│   ├── scene_setup_articulated_vectorized.py   # Vectorized scene with articulated finger robot (542 lines)
│   ├── scene_setup_standalone_vectorized.py    # Vectorized scene with standalone striker spheres (581 lines)
│   ├── poke_executor_articulated_vectorized.py # Poke executor for articulated finger (222 lines)
│   ├── poke_executor_standalone_vectorized.py  # Poke executor for standalone strikers (315 lines)
│   └── tutorials/
│       ├── actuator_setup.py
│       ├── add_panda.py
│       └── control_finger_robot.py      # Tutorial: controlling the finger robot (120 lines)
│
├── nn/                                  # Neural network training
│   ├── networks.py                      # SpatialActorCritic, U-Net, FiLM (336 lines)
│   ├── train.py                         # Main training script (1525 lines)
│   ├── overview.md                      # Architecture overview & plan (214 lines)
│   ├── on_rewards.md                    # Reward function analysis (14 lines)
│   ├── training_plan.txt                # Detailed DPG+FiLM training plan (223 lines)
│   ├── why_FiLM.md                      # FiLM spatial discrimination explanation (21 lines)
│   └── go_standalone_headless_training_later.md  # Future production workflow plan (45 lines)
│
├── vision/                              # Vision pipeline
│   ├── __init__.py
│   ├── camera.py                        # Full vision pipeline: camera, segmentation, contours (807 lines)
│   └── demo_script_editor.py            # Demo script for vision pipeline
│
├── concepts_for_writing/
│   └── affordance_learning.md           # Design discussion log
│
├── implementation_log/                  # Development logs (chronological)
│   ├── 0_add_finger_robot.md
│   ├── 1_lesson_learnt_from_adding_a_finger_robot.md
│   ├── 2_plan.md                        # RL training plan (DDPG+DQN hybrid)
│   ├── 3_overall_training_flow.md       # Training flow details & bug fixes
│   ├── 4_fixing_RL_issues1.md
│   ├── 5_fixing_RL_issues2.md
│   ├── 6_log_fixed_RL_issues_set2.md
│   ├── 7_log_auto-reset_for_performance_later.md
│   ├── 8_on_update_scheduling.md
│   ├── 9_how_does_greedy_pixel_work.md
│   ├── 10_the_practicals_in_Q_bootstrapping.md
│   ├── 11_curriculum_later_for_orientation.md
│   ├── 12_curriculum_compatible_implementation.md
│   ├── 13_manual_curriculum_is_better_for_debugging.md
│   ├── 14_decoupled_curriculum.md
│   └── 15_sim2real_with_TO.md
│
├── prior_obsolete/
│   └── poke_prior.py                    # Observed physics-based prior (241 lines)
│
└── vision_outputs/                      # Debug contour images (.npy and .png files)
    └── (many contour_*_env_*.npy and .png files)
```

---

### `quintic_poly.py` (Full Content)

```python
def quintic_poly_query(u):
    return 6 * u**5 - 15 * u**4 + 10 * u**3

def compute_T(v_mid, L):
    T = (L * 1.875) / v_mid
    return T


#TODO: with a predefined L = 0.3 m, and v_mid from param-head output (cap at 1.5 m/s)
#TODO: for the control cycle, num_steps = T * 60 (isaac sim default control cycle is 60Hz)
```

**Key files and their roles:**

| File | Lines | Purpose |
|------|-------|---------|
| `/home/xiao/0_codes/poke_map/env/quintic_poly.py` | 10 | Quintic polynomial trajectory: `quintic_poly_query(u)` computes smooth interpolation between start/end (6u^5-15u^4+10u^3), and `compute_T(v_mid, L)` calculates trajectory duration. Planned for feedforward trajectory generation (not yet integrated). |
| `/home/xiao/0_codes/poke_map/env/make_finger_robot.py` | 347 | Builds a 3-DOF prismatic finger robot (X/Y/Z prismatic joints with PD control via USD DriveAPI). The finger consists of a base fixed to World, three prismatic links, and a sphere fingertip with collision. |
| `/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py` | 542 | Creates vectorized environments with the articulated finger robot, L-shaped object, and cylinder. Cloned via GridCloner. Exposes globals for interactive use. |
| `/home/xiao/0_codes/poke_map/env/scene_setup_standalone_vectorized.py` | 581 | Alternative scene setup: replaces the articulated finger with standalone RigidPrim spheres ("strikers") that are moved by directly setting linear velocity (no joint control). |
| `/home/xiao/0_codes/poke_map/env/poke_executor_articulated_vectorized.py` | 222 | Commands the articulated finger robots: `command_fingers_to_targets()`, `sample_random_targets()`, velocity targets, and drive configuration. |
| `/home/xiao/0_codes/poke_map/env/poke_executor_standalone_vectorized.py` | 315 | Commands standalone strikers: `strike()` (single env) and `strike_all()` (batched) — teleport, descend, set velocity, step physics, retract. |
| `/home/xiao/0_codes/poke_map/nn/networks.py` | 336 | **SpatialActorCritic** with shared U-Net backbone, FiLM-conditioned Q-head, and per-pixel param-head. Implements `greedy_pixel()` with top-K re-evaluation. |
| `/home/xiao/0_codes/poke_map/nn/train.py` | 1525 | **Main training script**: ReplayBuffer, action selection (epsilon-greedy + heuristic prior), environment stepping, DPG+DQN training, checkpointing. |
| `/home/xiao/0_codes/poke_map/vision/camera.py` | 807 | Complete vision pipeline: overhead camera intrinsics, ground-truth segmentation via projection, contour extraction, goal contour rendering, input stacking. |
| `/home/xiao/0_codes/poke_map/config.yaml` | 112 | Central configuration: scene/physics parameters, object properties, finger robot parameters, striker settings, control parameters, training hyperparameters (gamma, lr, batch_size, etc.), yaw curriculum, checkpointing. |
| `/home/xiao/0_codes/poke_map/prior_obsolete/poke_prior.py` | 241 | Physics-based heuristic prior: `simulate_poke()` applies impulse and measures displacement, `compute_prior()` builds a per-pixel heatmap for exploration biasing. |
| `/home/xiao/0_codes/poke_map/train_it.md` | 61 | Step-by-step training workflow: run scene setup, configure Stage A (translation only) then Stage B (yaw curriculum), resume from checkpoints. |

---

### Key Architecture Details

1. **Action space**: hybrid discrete-continuous — `(pixel_ij, d_x, d_y, velocity)`. Pixel selection from contour pixels, direction as unit vector, velocity magnitude capped at 1.5 m/s.

2. **Network architecture** (SpatialActorCritic):
   - Input: `(B, 2, 128, 128)` — contour_current + contour_goal
   - Shared U-Net backbone → feature map `(B, 32, 128, 128)`
   - **Q-head**: Conv2d(32, 1) produces per-pixel Q-values, **conditioned on action params via FiLM** (Feature-wise Linear Modulation)
   - **Param-head**: Conv2d(32, 3) produces per-pixel `(d_x, d_y, v)` with tanh/normalize activation

3. **Training algorithm**: Modified DDPG + DQN hybrid
   - **Loss 1 (Q-loss)**: Huber TD error on Q(s, pixel, stored_action)
   - **Loss 2 (DPG)**: Deterministic policy gradient — maximize Q for the actor's output parameters
   - Target network with polyak averaging (tau=0.005)

4. **Two environment modes**:
   - **Articulated**: 3-DOF prismatic finger robot with PD joint control
   - **Standalone**: Direct velocity-controlled striker spheres (simpler, no joint control)

5. **Quintic polynomial** (`quintic_poly.py`): Planned for feedforward trajectory generation between finger waypoints. The `quintic_poly_query(u)` function implements the classic smooth step function (zero velocity/acceleration at boundaries). Not yet integrated into the main execution loop.

6. **Configuration** (`config.yaml`): All hyperparameters are centralized — gamma (0.95), lr (0.0003), batch_size (128), buffer_capacity (100k), epsilon decay schedule, noise decay, reward weights, velocity limits, workspace bounds, yaw curriculum parameters, etc.
</task_result>
</task>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 6.8s)

_Thinking:_

Now I have a thorough understanding of the codebase. Let me read the key files I need to understand more deeply:

1. `quintic_poly.py` - already read, it's small
2. The poke executor files - to understand how the finger robot is commanded
3. The config - to understand velocity limits and control parameters
4. The network files - to understand how velocity magnitude is output

Let me now think about what the user wants:

The user wants to:
1. Use `quintic_poly.py` to construct a quintic scalar polynomial for velocity magnitude
2. The striking phase has travel distance L = 0.3m
3. With this polynomial, compute a feedforward force to the finger robot

Key understanding:
- The quintic polynomial `6*u^5 - 15*u^4 + 10*u^3` is the classic smoothstep function. For u ∈ [0,1], it goes from 0 to 1 with zero first and second derivatives at endpoints.
- `compute_T(v_mid, L)` computes the trajectory duration T based on v_mid (velocity magnitude from the policy) and L (travel distance). The constant 1.875 comes from the ratio of average velocity to v_mid for the quintic polynomial.
- For a position trajectory: `s(t) = L * quintic_poly_query(t/T)` where s(0)=0, s(T)=L
- For velocity: `v(t) = s'(t) = L/T * quintic_poly_query'(t/T)`
- The feedforward force could be derived from the acceleration trajectory: `a(t) = s''(t)` and F = m*a

Let me read more specific files to understand the integration points.

Let me read the key files I need for the plan.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/quintic_poly.py"
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/quintic_poly.py</path>
<type>file</type>
<content>
1: def quintic_poly_query(u):
2:     return 6 * u**5 - 15 * u**4 + 10 * u**3
3: 
4: def compute_T(v_mid, L):
5:     T = (L * 1.875) / v_mid
6:     return T
7: 
8: 
9: #TODO: with a predefined L = 0.3 m, and v_mid from param-head output (cap at 1.5 m/s)
10: #TODO: for the control cycle, num_steps = T * 60 (isaac sim default control cycle is 60Hz)

(End of file - total 10 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/poke_executor_articulated_vectorized.py"
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/poke_executor_articulated_vectorized.py</path>
<type>file</type>
<content>
1: from pathlib import Path
2: import sys
3: import __main__
4: 
5: import numpy as np
6: 
7: CURRENT_DIR = Path(__file__).resolve().parent
8: PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")
9: 
10: for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
11:     if module_dir.exists() and str(module_dir) not in sys.path:
12:         sys.path.insert(0, str(module_dir))
13: 
14: from isaacsim.core.experimental.prims import Articulation, RigidPrim
15: import isaacsim.core.experimental.utils.app as app_utils
16: import isaacsim.core.experimental.utils.stage as stage_utils
17: from omni.kit.async_engine import run_coroutine
18: 
19: from config_loader import load_config
20: from make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives
21: 
22: 
23: CONFIG = load_config()
24: SCENE_CONFIG = CONFIG["scene"]
25: FINGER_CONFIG = CONFIG["finger"]
26: CONTROL_CONFIG = CONFIG["control"]
27: 
28: ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
29: FINGER_LOCAL_ROOT_PATH = FINGER_CONFIG["local_root_path"]
30: FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_ROOT_PATH}"
31: FINGER_TIP_LINK_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
32: SOURCE_FINGER_PATH = (
33:     f"{ENVS_ROOT_PATH}/{SCENE_CONFIG['source_env_name']}/{FINGER_LOCAL_ROOT_PATH}"
34: )
35: 
36: TARGET_SEED = CONTROL_CONFIG["target_seed"]
37: NUM_CONTROL_STEPS = CONTROL_CONFIG["num_control_steps"]
38: TARGET_LOW = np.array(LIMIT_LOWER, dtype=np.float32)
39: TARGET_HIGH = np.array(LIMIT_UPPER, dtype=np.float32)
40: 
41: # Velocity targets: sign controls direction, magnitude controls speed at arrival.
42: # None means the damping term drives velocity to zero (classic PD settle).
43: # Shape must be (num_fingers, 3) or broadcastable, units m/s.
44: ARRIVAL_VELOCITY = CONTROL_CONFIG["arrival_velocity"]
45: RANDOM_VELOCITY_SPEED_RANGE = CONTROL_CONFIG["random_velocity_speed_range"]
46: 
47: 
48: def as_numpy(values):
49:     if hasattr(values, "numpy"):
50:         values = values.numpy()
51:     return np.asarray(values, dtype=np.float32)
52: 
53: 
54: def get_vectorized_fingers():
55:     stage = stage_utils.get_current_stage(backend="usd")
56:     if not stage.GetPrimAtPath(SOURCE_FINGER_PATH):
57:         raise RuntimeError(
58:             f"No vectorized finger robot found at {SOURCE_FINGER_PATH}. "
59:             "Run env/scene_setup_vectorized.py first."
60:         )
61: 
62:     finger_articulations = Articulation(paths=FINGER_ROOT_PATTERN)
63:     finger_tip_links = RigidPrim(paths=FINGER_TIP_LINK_PATTERN)
64:     return finger_articulations, finger_tip_links
65: 
66: 
67: def sample_random_targets(num_fingers, seed=None, low=TARGET_LOW, high=TARGET_HIGH):
68:     rng = np.random.default_rng(seed)
69:     return rng.uniform(low, high, size=(num_fingers, 3)).astype(np.float32)
70: 
71: 
72: def sample_random_velocity_targets(
73:     num_fingers,
74:     speed_range=RANDOM_VELOCITY_SPEED_RANGE,
75:     seed=None,
76: ):
77:     """Sample random unit-direction velocities with random magnitudes.
78: 
79:     Each finger gets an independent random direction in 3-D joint space
80:     scaled to a random speed drawn from *speed_range* (m/s).
81:     The drive law then becomes  tau = kp*(q_target - q) + kd*(v_target - dq/dt),
82:     so the finger arrives at *q_target* while still moving at *v_target*.
83:     """
84:     rng = np.random.default_rng(seed)
85:     directions = rng.standard_normal(size=(num_fingers, 3)).astype(np.float32)
86:     norms = np.linalg.norm(directions, axis=1, keepdims=True)
87:     norms = np.where(norms < 1e-6, 1.0, norms)  # guard against zero vectors
88:     directions /= norms
89:     speeds = rng.uniform(speed_range[0], speed_range[1], size=(num_fingers, 1)).astype(np.float32)
90:     return directions * speeds
91: 
92: 
93: def ensure_simulation_started():
94:     app_utils.play()
95: 
96: 
97: def get_randomized_finger_properties():
98:     return globals().get(
99:         "randomized_finger_properties",
100:         getattr(__main__, "randomized_finger_properties", None),
101:     )
102: 
103: 
104: def configure_vectorized_drives(finger_articulations, finger_properties=None):
105:     if finger_properties is None:
106:         configure_drives(finger_articulations)
107:         print("Using default finger stiffness/damping.")
108:         return
109: 
110:     configure_drives(
111:         finger_articulations,
112:         stiffnesses=finger_properties["stiffnesses"],
113:         dampings=finger_properties["dampings"],
114:     )
115:     print("Using randomized finger stiffness/damping from scene setup.")
116: 
117: 
118: async def command_fingers_to_targets_async(
119:     finger_articulations,
120:     targets,
121:     num_steps=NUM_CONTROL_STEPS,
122:     arrival_velocity=None,
123: ):
124:     """Command all fingers to *targets* (shape: num_fingers x 3).
125: 
126:     arrival_velocity : array-like, shape (num_fingers, 3) or None
127:         When provided, each DOF's velocity target is set alongside the position
128:         target.  The PhysX drive then becomes:
129:             tau = kp*(q_target - q) + kd*(v_target - dq/dt)
130:         so the finger reaches *targets* while moving at *arrival_velocity*.
131:         Pass None (default) to keep the classic settle-to-stop behaviour
132:         (equivalent to v_target = 0).
133:     """
134:     dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
135:     finger_articulations.set_dof_position_targets(
136:         positions=targets.tolist(),
137:         dof_indices=dof_indices,
138:     )
139:     if arrival_velocity is not None:
140:         velocity_array = np.asarray(arrival_velocity, dtype=np.float32)
141:         finger_articulations.set_dof_velocity_targets(
142:             velocities=velocity_array.tolist(),
143:             dof_indices=dof_indices,
144:         )
145:     await app_utils.update_app_async(steps=num_steps)
146: 
147: 
148: def command_fingers_to_targets(
149:     finger_articulations,
150:     targets,
151:     num_steps=NUM_CONTROL_STEPS,
152:     arrival_velocity=None,
153: ):
154:     return run_coroutine(
155:         command_fingers_to_targets_async(
156:             finger_articulations, targets, num_steps, arrival_velocity
157:         )
158:     )
159: 
160: 
161: def get_tip_positions(finger_tip_links):
162:     positions, orientations = finger_tip_links.get_world_poses()
163:     return as_numpy(positions), as_numpy(orientations)
164: 
165: 
166: async def main_async(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS, arrival_velocity=ARRIVAL_VELOCITY):
167:     finger_articulations, finger_tip_links = get_vectorized_fingers()
168:     ensure_simulation_started()
169:     await app_utils.update_app_async(steps=1)
170: 
171:     finger_properties = get_randomized_finger_properties()
172:     configure_vectorized_drives(finger_articulations, finger_properties)
173: 
174:     targets = sample_random_targets(
175:         num_fingers=len(finger_articulations),
176:         seed=seed,
177:     )
178: 
179:     #! targets are passed in to move the robot; arrival_velocity lets the
180:     #! finger reach the target at a non-zero velocity (see docstring above).
181:     await command_fingers_to_targets_async(
182:         finger_articulations=finger_articulations,
183:         targets=targets,
184:         num_steps=num_steps,
185:         arrival_velocity=arrival_velocity,
186:     )
187: 
188:     tip_positions, tip_orientations = get_tip_positions(finger_tip_links)
189:     joint_positions = finger_articulations.get_dof_positions()
190: 
191:     print("✅ Vectorized finger control complete")
192:     print(f"Finger count : {len(finger_articulations)}")
193:     print(f"DOF names    : {finger_articulations.dof_names}")
194:     print(f"Targets      :\n{targets}")
195:     print(f"Joint pos    :\n{joint_positions}")
196:     print(f"Tip pos      :\n{tip_positions}")
197: 
198:     #! This is useful in Isaac Script Editor. Because after main_async() finishes,
199:     #! local variables like 'targets' would disappear. Using globals(), they can 
200:     #! be used interactively later.
201:     globals().update(
202:         {
203:             "finger_articulations": finger_articulations,
204:             "finger_tip_links": finger_tip_links,
205:             "finger_targets": targets,
206:             "finger_arrival_velocity": arrival_velocity,
207:             "finger_tip_positions": tip_positions,
208:             "finger_tip_orientations": tip_orientations,
209:             "finger_properties": finger_properties,
210:             "command_fingers_to_targets": command_fingers_to_targets,
211:             "sample_random_targets": sample_random_targets,
212:             "sample_random_velocity_targets": sample_random_velocity_targets,
213:         }
214:     )
215:     return finger_articulations, finger_tip_links, targets
216: 
217: 
218: def main(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS, arrival_velocity=ARRIVAL_VELOCITY):
219:     return run_coroutine(main_async(seed=seed, num_steps=num_steps, arrival_velocity=arrival_velocity))
220: 
221: 
222: finger_control_task = main()

(End of file - total 222 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/config.yaml"
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/config.yaml</path>
<type>file</type>
<content>
1: scene:
2:   object_height: 0.05
3:   l_arm_length: 0.08
4:   l_thickness: 0.03
5:   cylinder_radius: 0.05
6:   l_collision_approximation: convexHull
7:   cylinder_collision_approximation: convexHull
8:   num_envs: 64
9:   env_spacing: 1.8
10:   envs_root_path: /World/envs
11:   source_env_name: env_0
12:   l_object_local_position: [-0.10, -0.06, 0.0]
13:   cylinder_local_xy: [0.10, 0.02]
14:   randomize_objects_on_start: true
15:   randomize_finger_properties_on_start: false
16:   random_seed: 2
17:   settle_steps: 10
18: 
19: workspace:
20:   x_min: -0.4
21:   x_max: 0.4
22:   y_min: -0.4
23:   y_max: 0.4
24: 
25: objects:
26:   l_position_area_min: [-0.25, -0.25]
27:   l_position_area_max: [0.25, 0.25]
28:   cylinder_position_area_min: [-0.25, -0.25]
29:   cylinder_position_area_max: [0.25, 0.25]
30:   min_initial_clearance: 0.003
31:   pose_resample_attempts: 100
32:   l_yaw_range: [-3.141592653589793, 3.141592653589793]
33:   cylinder_yaw_range: [-3.141592653589793, 3.141592653589793]
34: 
35: finger:
36:   local_root_path: FingerRobot
37:   standalone_root_path: /World/FingerRobot
38:   sphere_radius: 0.0125
39:   box_half: 0.008
40:   limit_lower: [-0.4, -0.4, 0.025]
41:   limit_upper: [0.4, 0.4, 0.25]
42:   drive_stiffness: 1500.0
43:   drive_damping: 180.0
44:   drive_max_force: 500.0
45:   default_fingertip_mass: 0.3
46:   drive_stiffness_range: [1200.0, 1800.0]
47:   drive_damping_range: [140.0, 220.0]
48:   fingertip_mass_range: [0.2, 0.5]
49:   default_xyz: [0.0, 0.0, 0.025]
50: 
51: striker:
52:   local_path: Striker
53:   radius: 0.02
54:   default_z: 0.15
55:   mass_range: [0.02, 0.10]
56:   default_color: [1.0, 0.2, 0.2]
57: 
58: control:
59:   target_seed: null
60:   num_control_steps: 120
61:   arrival_velocity: None
62:   random_velocity_speed_range: [0.05, 0.20]
63: 
64: training:
65:   gamma: 0.95
66:   lr: 0.0003
67:   batch_size: 128
68:   buffer_capacity: 100000
69:   tau: 0.005
70:   train_after: 1024
71:   train_every: 1
72:   grad_updates_per_step: 4
73: 
74:   eps_start: 1.0
75:   eps_end: 0.05
76:   eps_decay_episodes: 250
77:   sigma_start: 0.3
78:   sigma_end: 0.05
79: 
80:   max_steps_per_episode: 30
81:   c_step: 0.01
82:   c_success: 10.0
83:   c_oob: -5.0
84:   success_threshold: 0.02
85:   stop_poke_threshold: 0.035
86: 
87:   velocity_max: 1.5
88:   velocity_integration_t: 0.5
89:   impact_steps: 80
90:   standoff_clearance: 0.01
91:   finger_track_tol: 0.003
92: 
93:   yaw_target_mode: preserve
94:   c_yaw: 0.1
95:   yaw_reward_enabled: true
96:   yaw_success_enabled: false
97:   yaw_threshold_deg: 10.0
98:   yaw_curriculum_start_ep: 0
99:   yaw_curriculum_end_ep: 500
100: 
101:   checkpoint_every: 100
102:   checkpoint_keep: 3
103: 
104:   show_plane_overlay: true
105:   plane_overlay_thickness: 0.003
106:   plane_overlay_opacity: 0.18
107:   show_target_overlay: true
108:   target_overlay_z_offset: 0.015
109:   target_overlay_thickness: 0.006
110:   target_overlay_opacity: 0.2
111:   show_action_overlay: true
112:   action_overlay_z_offset: 0.01

(End of file - total 112 lines)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 4.2s)

_Thinking:_

Now let me also read the training script to understand how actions are currently constructed (the velocity magnitude, how the action is applied, etc.) and the networks to understand the param-head output. Let me also read the finger robot construction and drive details.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/make_finger_robot.py",
  "offset": 1,
  "limit": 60
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/make_finger_robot.py</path>
<type>file</type>
<content>
1: """
2: Add an actuated spherical "finger robot" with 3 prismatic DOFs (XYZ) to the
3: current stage.  Designed to be run *after* ``scene_setup_vectorized.py`` from
4: the Isaac Sim script editor.
5: 
6: The finger robot is a lightweight cartesian manipulator:
7:   - Fixed base link
8:   - Three prismatic joints along X, Y, Z
9:   - A colored sphere at the tip as the "finger"
10: 
11: Control is via USD DriveAPI (PD at the solver level) so you can command
12: the tip to any (x, y, z) within the joint limits.
13: 
14: Usage (script editor)::
15: 
16:     # 1. Run scene_setup_vectorized.py first
17:     # 2. Run this file
18:     # 3. Interact via globals: finger_articulation etc.
19:     move_finger_to(finger_articulation, [0.1, -0.05, 0.08])
20: 
21: """
22: 
23: from __future__ import annotations
24: 
25: import gc
26: from pathlib import Path
27: import sys
28: 
29: import isaacsim.core.experimental.utils.stage as stage_utils
30: import numpy as np
31: import omni.kit.app
32: import omni.timeline
33: from isaacsim.core.experimental.objects import DistantLight, GroundPlane
34: from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
35: from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
36: 
37: CURRENT_DIR = Path(__file__).resolve().parent
38: PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")
39: 
40: for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
41:     if module_dir.exists() and str(module_dir) not in sys.path:
42:         sys.path.insert(0, str(module_dir))
43: 
44: from config_loader import load_config
45: 
46: CONFIG = load_config()
47: FINGER_CONFIG = CONFIG["finger"]
48: 
49: # ---------------------------------------------------------------------------
50: #  Constants
51: # ---------------------------------------------------------------------------
52: FINGER_ROOT_PATH = FINGER_CONFIG["standalone_root_path"]
53: BASE_PATH = f"{FINGER_ROOT_PATH}/base"
54: X_LINK_PATH = f"{FINGER_ROOT_PATH}/x_link"
55: Y_LINK_PATH = f"{FINGER_ROOT_PATH}/y_link"
56: Z_LINK_PATH = f"{FINGER_ROOT_PATH}/z_link"
57: 
58: X_JOINT_PATH = f"{FINGER_ROOT_PATH}/x_joint"
59: Y_JOINT_PATH = f"{FINGER_ROOT_PATH}/y_joint"
60: Z_JOINT_PATH = f"{FINGER_ROOT_PATH}/z_joint"

(Showing lines 1-60 of 347. Use offset=61 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/networks.py"
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/networks.py</path>
<type>file</type>
<content>
1: import torch
2: import torch.nn as nn
3: import torch.nn.functional as F
4: 
5: 
6: #* ================================================================
7: #*  Component networks
8: #* ================================================================
9: 
10: class DoubleConv(nn.Module):
11:     """(Conv → GN → ReLU) × 2.  Padding=1 keeps spatial dims unchanged."""
12: 
13:     def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
14:         super().__init__()
15:         mid_ch = mid_ch or out_ch
16:         self.conv = nn.Sequential(
17:             nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
18:             nn.GroupNorm(8, mid_ch),
19:             nn.ReLU(inplace=True),
20:             nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
21:             nn.GroupNorm(8, out_ch),
22:             nn.ReLU(inplace=True),
23:         )
24: 
25:     def forward(self, x):
26:         return self.conv(x)
27: 
28: 
29: class Down(nn.Module):
30:     """MaxPool → DoubleConv."""
31: 
32:     def __init__(self, in_ch: int, out_ch: int):
33:         super().__init__()
34:         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
35:         self.conv = DoubleConv(in_ch, out_ch)
36: 
37:     def forward(self, x):
38:         return self.conv(self.pool(x))
39: 
40: 
41: class Up(nn.Module):
42:     """Upsample → concat(skip) → DoubleConv."""
43: 
44:     def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
45:         super().__init__()
46:         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
47:         self.conv = DoubleConv(prev_ch + skip_ch, out_ch, (prev_ch + skip_ch) // 2)
48: 
49:     def forward(self, x_prev, x_skip):
50:         x_prev = self.up(x_prev)
51:         diff_y = x_skip.size(-2) - x_prev.size(-2)
52:         diff_x = x_skip.size(-1) - x_prev.size(-1)
53:         x_prev = F.pad(x_prev, [diff_x // 2, diff_x - diff_x // 2,
54:                                  diff_y // 2, diff_y - diff_y // 2])
55:         return self.conv(torch.cat([x_skip, x_prev], dim=1))
56: 
57: 
58: #* ================================================================
59: #*  U-Net (feature extractor only, no output head)
60: #* ================================================================
61: 
62: class UNet(nn.Module):
63:     """Lightweight U-Net backbone.
64: 
65:     Input  (B, C_in, H, W) — binary contour channels.
66:     Output (B, C, H, W)     — feature map at original resolution.
67:     """
68: 
69:     def __init__(self, in_channels: int = 2, base_channels: int = 32):
70:         super().__init__()
71:         c = base_channels
72:         self.inc   = DoubleConv(in_channels, c)
73:         self.down1 = Down(c, c * 2)
74:         self.down2 = Down(c * 2, c * 4)
75:         self.down3 = Down(c * 4, c * 8)
76:         self.up2   = Up(c * 8, c * 4, c * 4)
77:         self.up3   = Up(c * 4, c * 2, c * 2)
78:         self.up4   = Up(c * 2, c, c)
79: 
80:     def get_features(self, x: torch.Tensor) -> torch.Tensor:
81:         x1 = self.inc(x)         # (B, C, H, W)
82:         x2 = self.down1(x1)      # (B, 2C, H/2, W/2)
83:         x3 = self.down2(x2)      # (B, 4C, H/4, W/4)
84:         x4 = self.down3(x3)      # (B, 8C, H/8, W/8)
85:         x  = self.up2(x4, x3)    # (B, 4C, H/4, W/4)
86:         x  = self.up3(x, x2)     # (B, 2C, H/2, W/2)
87:         x  = self.up4(x, x1)     # (B, C, H, W)
88:         return x
89: 
90: 
91: #* ================================================================
92: #*  FiLM (Feature-wise Linear Modulation)
93: #* ================================================================
94: 
95: class FiLM(nn.Module):
96:     """Map action params (B, 3) → per-channel scale γ and shift β.
97: 
98:     γ, β have the same channel count as the feature map being modulated.
99:     """
100: 
101:     def __init__(self, param_dim: int = 3, feature_dim: int = 32, hidden_dim: int = 64):
102:         super().__init__()
103:         self.mlp = nn.Sequential(
104:             nn.Linear(param_dim, hidden_dim),
105:             nn.ReLU(inplace=True),
106:             nn.Linear(hidden_dim, feature_dim * 2),  # γ || β
107:         )
108: 
109:     def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
110:         """Returns (γ, β), each shape (B, feature_dim)."""
111:         out = self.mlp(params)                     # (B, feature_dim * 2)
112:         gamma = out[:, :out.shape[-1] // 2]         # (B, feature_dim)
113:         beta  = out[:, out.shape[-1] // 2:]          # (B, feature_dim)
114:         return gamma, beta
115: 
116: 
117: #* ================================================================
118: #*  SpatialActorCritic
119: #* ================================================================
120: 
121: class SpatialActorCritic(nn.Module):
122:     """Actor-Critic with shared U-Net backbone and FiLM-conditioned Q-head.
123: 
124:     Input  (B, 2, H, W) — contour_current + contour_goal
125:     Output q_map  (B, 1, H, W) — per-pixel Q-values (critic)
126:            params (B, 3, H, W) — (d̑_x, d̑_y, v) per pixel, activated
127: 
128:     Architecture
129:     ------------
130:       x → U-Net → f (B, C, H, W) ─┬─ param_head → params_raw → (tanh,sigmoid) → params
131:                                   │
132:                                   │  params pooled or indexed → (B,3)
133:                                   │       │
134:                                   │       ▼
135:                                   │  FiLM(γ,β) ──────────┐
136:                                   │                       │
137:                                   └── f ──────────────────┤
138:                                                           ▼
139:                                                     f_mod = γ·f + β
140:                                                           │
141:                                                           ▼
142:                                                      q_head → Q-map
143: 
144:     Training:
145:       Q-loss    updates q_head + FiLM + shared U-Net
146:       DPG loss  updates param_head + FiLM    (q_head and U-Net frozen)
147:     """
148: 
149:     def __init__(
150:         self,
151:         in_channels: int = 2,
152:         base_channels: int = 32,
153:         velocity_max: float = 0.4,
154:     ):
155:         super().__init__()
156:         c = base_channels
157: 
158:         #! shared backbone for both critic and actor
159:         self.unet = UNet(in_channels=in_channels, base_channels=c)
160:         
161:         #! critic 
162:         self.q_head = nn.Conv2d(c, 1, 1)              # per-pixel Q
163:         self.film = FiLM(param_dim=3, feature_dim=c, hidden_dim=64)
164: 
165:         #! actor
166:         self.param_head = nn.Conv2d(c, 3, 1)          # raw (d̑_x, d̑_y, v)
167:         self.velocity_max = velocity_max
168: 
169:     #* -- helpers -------------------------------------------------------
170: 
171:     #! bound the action params to valid ranges
172:     def _activate_params(self, raw: torch.Tensor) -> torch.Tensor:
173:         """Apply tanh/sigmoid to raw param output; normalise direction."""
174:         d_xy  = torch.tanh(raw[:, :2])                    # (B, 2, H, W) in [-1, 1]
175:         d_xy  = F.normalize(d_xy, dim=1, eps=1e-6)       # unit length per pixel
176:         v = torch.sigmoid(raw[:, 2:]) * self.velocity_max  # (B, 1, H, W) m/s
177:         return torch.cat([d_xy, v], dim=1)              # (B, 3, H, W)
178: 
179:     def params_at_pixel(
180:         self, params: torch.Tensor, pixel_ij: torch.Tensor
181:     ) -> torch.Tensor:
182:         """Extract (d̑, Δd) at specific pixels.  pixel_ij: (B, 2)."""
183:         B = pixel_ij.shape[0]
184:         batch_idx = torch.arange(B, device=params.device)
185:         return params[batch_idx, :, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B, 3)
186: 
187:     def q_vals_at_pixel(
188:         self, q_map: torch.Tensor, pixel_ij: torch.Tensor
189:     ) -> torch.Tensor:
190:         """Extract Q-values at specific pixels."""
191:         B = pixel_ij.shape[0]
192:         batch_idx = torch.arange(B, device=q_map.device)
193:         return q_map[batch_idx, 0, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B,)
194: 
195:     #* -- forward -------------------------------------------------------
196: 
197:     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
198:         """Full forward pass; FiLM uses contour-masked spatial mean of params.
199:         Used for TD target pixel selection in train_step (cheap ranking before
200:         per-pixel re-evaluation) and get_values_at_pixel convenience."""
201:         f = self.unet.get_features(x)              # (B, C, H, W)
202:         params_raw = self.param_head(f)            # (B, 3, H, W)
203:         params = self._activate_params(params_raw) # (B, 3, H, W)
204: 
205:         # contour-masked mean — only object pixels contribute to FiLM
206:         contour_mask = (x[:, 0:1] > 0.0).float()   # (B, 1, H, W)
207:         counts = contour_mask.sum(dim=[-2, -1]).clamp(min=1)  # (B, 1)
208:         params_global = (params * contour_mask).sum(dim=[-2, -1]) / counts  # (B, 3)
209:         gamma, beta = self.film(params_global)
210:         f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
211:         q_map = self.q_head(f_mod)                 # (B, 1, H, W)
212: 
213:         return q_map, params
214: 
215:     def q_map_with_params(
216:         self,
217:         x: torch.Tensor,
218:         params_at_pixel: torch.Tensor,  # (B, 3)
219:     ) -> torch.Tensor:
220:         """Q-map conditioned on explicit per-sample action params.
221: 
222:         Used during training to compute Q(s, pixel*, a) where a comes
223:         from the replay buffer (Q-loss) or from the current actor (DPG).
224:         The shared encoder is recomputed because x may differ in each call.
225: 
226:         Returns: q_map (B, 1, H, W)
227:         """
228:         f = self.unet.get_features(x)          # (B, C, H, W)
229:         gamma, beta = self.film(params_at_pixel)
230:         f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
231:         return self.q_head(f_mod)
232: 
233:     def features_and_params(self, x: torch.Tensor) -> tuple:
234:         """Return U-Net features + activated params (used in DPG step)."""
235:         f = self.unet.get_features(x)
236:         params_raw = self.param_head(f)
237:         params = self._activate_params(params_raw)
238:         return f, params, params_raw
239: 
240:     def get_values_at_pixel(
241:         self,
242:         x: torch.Tensor,
243:         pixel_ij: torch.Tensor,     # (B, 2) int
244:     ) -> tuple[torch.Tensor, torch.Tensor]:
245:         """Convenience: return (Q_val, params) at chosen pixels."""
246:         q_map, params = self.forward(x)
247:         B = pixel_ij.shape[0]
248:         batch_idx = torch.arange(B, device=x.device)
249: 
250:         q_vals = q_map[batch_idx, 0, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B,)
251:         p_vals = self.params_at_pixel(params, pixel_ij)                # (B, 3)
252:         return q_vals, p_vals
253: 
254:     def greedy_pixel(
255:         self,
256:         f: torch.Tensor,                 # (B, C, H, W) — pre-computed U-Net features
257:         params: torch.Tensor,            # (B, 3, H, W) — per-pixel action params
258:         contour_mask: torch.Tensor,      # (B, H, W) bool
259:         top_k: int = 5,
260:     ) -> tuple[torch.Tensor, torch.Tensor]:
261:         """Per-pixel FiLM-conditioned greedy pixel selection.
262: 
263:         Ranks pixels by cheap mean-conditioned Q, then re-evaluates the
264:         top-K with each pixel's own action params via per-pixel FiLM.
265:         One U-Net pass is assumed (caller provides f).
266: 
267:         Note: this is exact only when top_k ≥ all contour pixels.
268:         With typical top_k=5 it is an approximation — the globally best
269:         per-pixel conditioned pixel is found if it ranks in the top-K
270:         mean-conditioned candidates, which is empirically nearly always true.
271: 
272:         Returns:
273:           pixels: (B, 2) long — (row, col) of best pixel per batch item
274:           q_vals: (B,)  float — per-pixel conditioned Q at winning pixel
275:         """
276:         B, C, H, W = f.shape
277:         batch_idx = torch.arange(B, device=f.device)
278: 
279:         # cheap mean-conditioned Q for candidate ranking
280:         mask_f = contour_mask.unsqueeze(1).float()
281:         counts = mask_f.sum(dim=[-2, -1]).clamp(min=1)
282:         params_mean = (params * mask_f).sum(dim=[-2, -1]) / counts  # (B, 3)
283:         gamma_m, beta_m = self.film(params_mean)
284:         f_mean = gamma_m[:, :, None, None] * f + beta_m[:, :, None, None]
285:         q_mean = self.q_head(f_mean).squeeze(1)                    # (B, H, W)
286:         q_mean[~contour_mask] = -float("inf")
287: 
288:         # top-K candidates per batch item
289:         q_flat = q_mean.view(B, -1)                               # (B, H*W)
290:         candidates = min(top_k, contour_mask.sum(dim=(-2, -1)).max().item())
291:         _, topk_idx = torch.topk(q_flat, candidates, dim=1)       # (B, K)
292:         topk_row = topk_idx // W                                   # (B, K)
293:         topk_col = topk_idx % W                                    # (B, K)
294: 
295:         best_q = torch.full((B,), -float("inf"), device=f.device)
296:         best_row = torch.zeros(B, dtype=torch.long, device=f.device)
297:         best_col = torch.zeros(B, dtype=torch.long, device=f.device)
298: 
299:         for k in range(candidates):
300:             row = topk_row[:, k]                                    # (B,)
301:             col = topk_col[:, k]                                    # (B,)
302:             pixel_params = params[batch_idx, :, row, col]           # (B, 3)
303:             gamma, beta = self.film(pixel_params)                   # (B, C)
304:             f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
305:             q_cand = self.q_head(f_mod)                             # (B, 1, H, W)
306:             q_val = q_cand[batch_idx, 0, row, col]                 # (B,)
307:             valid = contour_mask[batch_idx, row, col]
308:             q_val = q_val.masked_fill(~valid, -float("inf"))
309: 
310:             better = q_val > best_q
311:             best_q[better] = q_val[better]
312:             best_row[better] = row[better]
313:             best_col[better] = col[better]
314: 
315:         pixels = torch.stack([best_row, best_col], dim=1)          # (B, 2)
316:         return pixels, best_q
317: 
318: 
319: #! FiLM (Feature-wise Linear Modulation) is a conditioning technique from paper:
320: #! Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
321: #?
322: #? Instead of concatenating extra input channels, we modulate existing features with
323: #? a learned affine transform per channel. Here is the flow: 
324: #?
325: #?    f ------------(B, 32, H, W)    ← U-net output features
326: #?    params -------(B, 3)           ← (d̂_x, d̂_y, Δd)
327: #?    γ  =  MLP_γ(params) ----- (B, 32)           ← learned scale per channel
328: #?    β  =  MLP_β(params) ----- (B, 32)           ← learned shift per channel
329: #?    f_mod = γ * f + β   ----- (B, 32, H, W)     ← same shape, modulated
330: #?    Q_map = Conv2d(32, 1)(f_mod)                ← Q-head sees conditioned features
331: #?
332: #? γ and β are produced by a tiny MLP (3 → 32 → 32 neurons, ~2K parameters). 
333: #? They scale and shift each of the 32 feature channels based on the action params.
334: 
335: 
336: 

(End of file - total 336 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "offset": 1,
  "limit": 80
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
1: """
2: Training script for Spatial Actor-Critic with FiLM-conditioned Q-head.
3: 
4: Runs inside the Isaac Sim Script Editor.  Before executing:
5:   1. Run ``scene_setup_articulated_vectorized.py``
6:   2. Run this file (may take a few seconds to import)
7:   3. Call ``train()`` or ``main()``
8: 
9: Architecture
10: ------------
11:   - SpatialActorCritic with shared U-Net backbone
12:   - FiLM-conditioned Q-head: action params modulate features before Q
13:   - Two losses: TD (Huber) for Q-head, DPG for param-head
14:   - Replay buffer, ε-greedy pixel + Gaussian param noise
15:   - Target network with soft (polyak) updates
16: """
17: 
18: from __future__ import annotations
19: 
20: import __main__
21: import gc
22: from collections import deque
23: from pathlib import Path
24: import sys
25: import time
26: 
27: import numpy as np
28: import torch
29: import torch.nn.functional as F
30: 
31: #* -- project paths --------------------------------------------------------
32: def find_project_root() -> Path:
33:     candidates = [
34:         Path(__file__).resolve().parents[1],
35:         Path.cwd(),
36:         Path("/home/xiao/0_codes/poke_map"),
37:     ]
38:     for candidate in candidates:
39:         if ((candidate / "config.yaml").exists() and
40:             (candidate / "nn").exists() and
41:             (candidate / "vision").exists() and
42:             (candidate / "env").exists()):
43:             return candidate
44:     raise RuntimeError("Could not find poke_map project root.")
45: 
46: 
47: PROJECT_ROOT = find_project_root()
48: for p in (PROJECT_ROOT, PROJECT_ROOT / "nn", PROJECT_ROOT / "env",
49:           PROJECT_ROOT / "vision"):
50:     if p.exists() and str(p) not in sys.path:
51:         sys.path.insert(0, str(p))
52: 
53: #* -- Isaac Sim ------------------------------------------------------------
54: try:
55:     from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
56:     import isaacsim.core.experimental.utils.app as app_utils
57:     import isaacsim.core.experimental.utils.stage as stage_utils
58:     from omni.kit.async_engine import run_coroutine
59:     from pxr import Gf, Sdf, UsdGeom, UsdShade
60:     _HAS_ISAAC = True
61: except ModuleNotFoundError:
62:     Gf = None
63:     Sdf = None
64:     UsdGeom = None
65:     UsdShade = None
66:     stage_utils = None
67:     _HAS_ISAAC = False
68: 
69: #* -- project modules ------------------------------------------------------
70: from config_loader import load_config
71: from vision.camera import (
72:     RESOLUTION,
73:     get_camera_intrinsics,
74:     get_object_poses_vectorized,
75:     build_vision_observation,
76:     get_object_masks_2d,
77:     mask_to_contour,
78:     pixel_to_world,
79: )
80: from nn.networks import SpatialActorCritic

(Showing lines 1-80 of 1525. Use offset=81 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 4.3s)

_Thinking:_

Let me now read more of the training script to understand the critical sections: how actions are applied to the environment, the `poke_action` function, how velocity magnitude is used, and the physics stepping.

**Tool: grep**

**Input:**
```json
{
  "pattern": "velocity|poke_action|finger_target|v_mid|velocity_mag|impact_steps|num_control|arrival_velocity|num_steps",
  "include": "train.py",
  "path": "/home/xiao/0_codes/poke_map"
}
```

**Output:**
```
Found 25 matches
/home/xiao/0_codes/poke_map/nn/train.py:
  Line 121: VELOCITY_MAX: float = TRAIN_CONFIG["velocity_max"]

  Line 122: VELOCITY_INTEGRATION_T: float = TRAIN_CONFIG["velocity_integration_t"]

  Line 123: IMPACT_STEPS: int = TRAIN_CONFIG["impact_steps"]

  Line 179:         self.velocity = torch.zeros(capacity, 1)

  Line 192:         velocity: float,

  Line 204:         self.velocity[idx] = torch.tensor([velocity])

  Line 220:             "velocity":  self.velocity[indices].to(self.device),

  Line 584: def heuristic_poke_actions(

  Line 689:     delta_d  : (B,)  float — poke velocity in [0, VELOCITY_MAX]

  Line 697:     velocity_arr = np.zeros(B, dtype=np.float32)

  Line 708:                 velocity_arr[b] = 0.0           # no-op poke (empty contour)

  Line 736:             velocity_arr[b] = np.clip(dd_noisy, 0.0, actor_critic.velocity_max)

  Line 738:     return pixel_ij, d_xy_arr, velocity_arr

  Line 914:     #* ── velocity → displacement for PD geometry ────────────────────

  Line 918:     # integration: velocity (m/s) × time (s) = distance (m)

  Line 940:         fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)

  Line 984:     #* pushing force kp*Δd.  No velocity target avoids oscillation at contact.

  Line 991:     fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)

  Line 1032:     v_st    = batch["velocity"]      # (B, 1)  stored velocity (m/s)

  Line 1134:         self.actor_critic = SpatialActorCritic(velocity_max=VELOCITY_MAX).to(DEVICE)

  Line 1135:         self.target_net   = SpatialActorCritic(velocity_max=VELOCITY_MAX).to(DEVICE)

  Line 1219:         self.fingers.set_dof_velocity_targets(velocities=init_vel.tolist(), dof_indices=dof_idx)

  Line 1279:             heuristic_actions = heuristic_poke_actions(

  Line 1403:             "buffer_velocity": self.buffer.velocity[:self.buffer.size],

  Line 1443:         self.buffer.velocity[:s].copy_(ckpt["buffer_velocity"][:s])

```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "offset": 80,
  "limit": 100
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
80: from nn.networks import SpatialActorCritic
81: from env.make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives
82: from env.scene_setup_articulated_vectorized import randomize_object_poses
83: 
84: CONFIG = load_config()
85: SCENE_CONFIG = CONFIG["scene"]
86: FINGER_CONFIG = CONFIG["finger"]
87: WORKSPACE_CONFIG = CONFIG["workspace"]
88: 
89: OBJECT_HEIGHT = SCENE_CONFIG["object_height"]
90: L_ARM_LENGTH = SCENE_CONFIG["l_arm_length"]
91: L_THICKNESS = SCENE_CONFIG["l_thickness"]
92: CYLINDER_RADIUS = SCENE_CONFIG["cylinder_radius"]
93: ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
94: FINGER_LOCAL_PATH = FINGER_CONFIG["local_root_path"]
95: FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_PATH}"
96: FINGER_TIP_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
97: NUM_ENVS = SCENE_CONFIG["num_envs"]
98: TRAIN_CONFIG = CONFIG["training"]
99: 
100: 
101: #* ================================================================
102: #*  Hyper-parameters  (sourced from config.yaml → training)
103: #* ================================================================
104: 
105: GAMMA: float = TRAIN_CONFIG["gamma"]
106: LR: float = TRAIN_CONFIG["lr"]
107: BATCH_SIZE: int = TRAIN_CONFIG["batch_size"]
108: BUFFER_CAPACITY: int = TRAIN_CONFIG["buffer_capacity"]
109: EPS_START: float = TRAIN_CONFIG["eps_start"]
110: EPS_END: float = TRAIN_CONFIG["eps_end"]
111: EPS_DECAY: int = TRAIN_CONFIG["eps_decay_episodes"]
112: SIGMA_START: float = TRAIN_CONFIG["sigma_start"]
113: SIGMA_END: float = TRAIN_CONFIG["sigma_end"]
114: TAU: float = TRAIN_CONFIG["tau"]
115: MAX_STEPS: int = TRAIN_CONFIG["max_steps_per_episode"]
116: C_STEP: float = TRAIN_CONFIG["c_step"]
117: C_SUCCESS: float = TRAIN_CONFIG["c_success"]
118: C_OOB: float = TRAIN_CONFIG["c_oob"]
119: SUCCESS_THRESHOLD: float = TRAIN_CONFIG["success_threshold"]
120: STOP_POKE_THRESHOLD: float = TRAIN_CONFIG["stop_poke_threshold"]
121: VELOCITY_MAX: float = TRAIN_CONFIG["velocity_max"]
122: VELOCITY_INTEGRATION_T: float = TRAIN_CONFIG["velocity_integration_t"]
123: IMPACT_STEPS: int = TRAIN_CONFIG["impact_steps"]
124: FINGERTIP_RADIUS: float = FINGER_CONFIG["sphere_radius"]
125: STANDOFF_CLEARANCE: float = TRAIN_CONFIG["standoff_clearance"]
126: OVERTRAVEL: float = 0.05          # DEPRECATED — delta_d now controls penetration symmetrically
127: POKE_Z: float = OBJECT_HEIGHT * 0.5
128: SAFE_Z: float = 0.2
129: FINGER_TRACK_TOL: float = TRAIN_CONFIG["finger_track_tol"]
130: 
131: # yaw training controls
132: YAW_TARGET_MODE: str = TRAIN_CONFIG["yaw_target_mode"]
133: C_YAW: float = TRAIN_CONFIG["c_yaw"]
134: YAW_REWARD_ENABLED: bool = TRAIN_CONFIG["yaw_reward_enabled"]
135: YAW_SUCCESS_ENABLED: bool = TRAIN_CONFIG["yaw_success_enabled"]
136: YAW_THRESHOLD: float = np.deg2rad(TRAIN_CONFIG["yaw_threshold_deg"])
137: YAW_CURRICULUM_START: int = TRAIN_CONFIG["yaw_curriculum_start_ep"]
138: YAW_CURRICULUM_END: int = TRAIN_CONFIG["yaw_curriculum_end_ep"]
139: YAW_FULL_RANGE: float = np.pi
140: 
141: TRAIN_AFTER: int = TRAIN_CONFIG["train_after"]
142: TRAIN_EVERY: int = TRAIN_CONFIG["train_every"]
143: GRAD_UPDATES_PER_STEP: int = TRAIN_CONFIG["grad_updates_per_step"]
144: DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
145: CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
146: CHECKPOINT_EVERY: int = TRAIN_CONFIG["checkpoint_every"]
147: CHECKPOINT_KEEP: int = TRAIN_CONFIG["checkpoint_keep"]
148: SHOW_PLANE_OVERLAY: bool = TRAIN_CONFIG["show_plane_overlay"]
149: PLANE_OVERLAY_THICKNESS: float = TRAIN_CONFIG["plane_overlay_thickness"]
150: PLANE_OVERLAY_OPACITY: float = TRAIN_CONFIG["plane_overlay_opacity"]
151: SHOW_TARGET_OVERLAY: bool = TRAIN_CONFIG["show_target_overlay"]
152: TARGET_OVERLAY_Z_OFFSET: float = TRAIN_CONFIG["target_overlay_z_offset"]
153: TARGET_OVERLAY_THICKNESS: float = TRAIN_CONFIG["target_overlay_thickness"]
154: TARGET_OVERLAY_OPACITY: float = TRAIN_CONFIG["target_overlay_opacity"]
155: SHOW_ACTION_OVERLAY: bool = TRAIN_CONFIG["show_action_overlay"]
156: ACTION_OVERLAY_Z_OFFSET: float = TRAIN_CONFIG["action_overlay_z_offset"]
157: 
158: 
159: #* ================================================================
160: #*  Replay buffer
161: #* ================================================================
162: 
163: class ReplayBuffer:
164:     """Ring buffer storing per-transition data (one env, one step).
165: 
166:     Each element is a flat dict so we can mix across envs and episodes.
167:     x / x_next are stored as uint8 (binary contours) to save memory.
168:     Mask is reconstructed on-the-fly from x in sample() / train_step().
169:     """
170: 
171:     def __init__(self, capacity: int = BUFFER_CAPACITY, device: str = DEVICE):
172:         self.capacity = capacity
173:         self.device = device
174: 
175:         H, W = RESOLUTION
176:         self.x        = torch.zeros(capacity, 2, H, W, dtype=torch.uint8)
177:         self.pixel    = torch.zeros(capacity, 2, dtype=torch.long)
178:         self.d_xy     = torch.zeros(capacity, 2)
179:         self.velocity = torch.zeros(capacity, 1)

(Showing lines 80-179 of 1525. Use offset=180 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 3.3s)

_Thinking:_

Let me now read the critical sections in train.py where the poke action is actually executed - the `execute_poke` or equivalent function, and how velocity is converted to movement.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "offset": 580,
  "limit": 170
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
580:         return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
581:     return (v / norm).astype(np.float32)
582: 
583: 
584: def heuristic_poke_actions(
585:     contour_masks: torch.Tensor,
586:     poses_before: dict,
587:     targets_pos: dict[str, np.ndarray],
588:     env_root_pos: np.ndarray,
589:     K: np.ndarray,
590: ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
591:     """Physically plausible exploration pokes for epsilon actions.
592: 
593:     Pick the object that is farther from its target, choose a contour pixel on
594:     the backside relative to target displacement, and push toward the target.
595:     """
596:     B = contour_masks.shape[0]
597:     pixel_ij = np.zeros((B, 2), dtype=np.int32)
598:     d_xy = np.zeros((B, 2), dtype=np.float32)
599:     delta_d = np.full(B, 0.05, dtype=np.float32)
600: 
601:     masks_cpu = contour_masks.detach().cpu()
602:     env_offsets = env_root_pos[:, :2]
603: 
604:     for b in range(B):
605:         valid_pixels_t = torch.nonzero(masks_cpu[b], as_tuple=False)
606:         if valid_pixels_t.numel() == 0:
607:             continue
608: 
609:         object_infos = []
610:         for obj_name, target in targets_pos.items():
611:             current_xy = poses_before[obj_name][0][b, :2] - env_offsets[b]
612:             target_xy = target[b, :2]
613:             to_target = target_xy - current_xy
614:             dist = float(np.linalg.norm(to_target))
615:             object_infos.append((dist, obj_name, current_xy, target_xy, to_target))
616: 
617:         movable_infos = [info for info in object_infos if info[0] >= STOP_POKE_THRESHOLD]
618:         if not movable_infos:
619:             movable_infos = object_infos
620:         _, obj_name, current_xy, _, to_target = max(movable_infos, key=lambda item: item[0])
621:         goal_dir = _normalise_xy(to_target)
622: 
623:         local_position = poses_before[obj_name][0][b:b + 1].copy()
624:         local_position[:, :2] -= env_offsets[b]
625:         object_mask = get_object_masks_2d(
626:             object_name=obj_name,
627:             object_positions=local_position,
628:             object_orientations=poses_before[obj_name][1][b:b + 1],
629:             K=K,
630:             resolution=RESOLUTION,
631:         )[0]
632:         object_contour, _ = mask_to_contour(object_mask)
633:         valid_pixels = np.argwhere(object_contour).astype(np.int32)
634:         if valid_pixels.shape[0] == 0:
635:             valid_pixels = valid_pixels_t.numpy().astype(np.int32)
636:         contact_xys = _pixels_to_local_xy(valid_pixels, K)
637: 
638:         if obj_name == "Cylinder":
639:             desired_contact_xy = current_xy - goal_dir * CYLINDER_RADIUS
640:             nearest = int(np.argmin(np.linalg.norm(contact_xys - desired_contact_xy[None, :], axis=1)))
641:             pixel_ij[b] = valid_pixels[nearest]
642:             noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.05
643:             d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
644:             delta_d[b] = np.float32(0.045 + abs(np.random.randn()) * 0.015)
645:             continue
646: 
647:         inward_scores = (current_xy[None, :] - contact_xys) @ goal_dir
648:         candidate_mask = inward_scores > 0.0
649:         candidate_pixels = valid_pixels[candidate_mask]
650:         candidate_xys = contact_xys[candidate_mask]
651: 
652:         if candidate_pixels.shape[0] == 0:
653:             candidate_pixels = valid_pixels
654:             candidate_xys = contact_xys
655: 
656:         backside_scores = (candidate_xys - current_xy[None, :]) @ (-goal_dir)
657:         k = min(25, candidate_pixels.shape[0])
658:         top_idx = np.argpartition(backside_scores, -k)[-k:]
659:         chosen = int(np.random.choice(top_idx))
660: 
661:         pixel_ij[b] = candidate_pixels[chosen]
662:         noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.10
663:         d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
664:         delta_d[b] = np.float32(0.045 + abs(np.random.randn()) * 0.015)
665: 
666:     return pixel_ij, d_xy, delta_d
667: 
668: 
669: def select_action(
670:     actor_critic: SpatialActorCritic,
671:     x: torch.Tensor,                       # (B, 2, H, W)
672:     contour_masks: torch.Tensor,           # (B, H, W) bool
673:     epsilon: float,
674:     noise_std: float,
675:     heuristic_actions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
676:     top_k: int = 5,
677: ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
678:     """Select actions for a batch of observations.
679: 
680:     Greedy envs use per-pixel FiLM-conditioned pixel selection (top-K
681:     re-evaluation).  ε-greedy envs use the provided heuristic action prior
682:     when available, so early exploration produces meaningful pushes instead
683:     of untrained param-head thrashing.
684: 
685:     Returns
686:     -------
687:     pixel_ij : (B, 2) int   — (row, col) per env
688:     d_xy     : (B, 2) float — normalised unit vector
689:     delta_d  : (B,)  float — poke velocity in [0, VELOCITY_MAX]
690:     """
691:     B = x.shape[0]
692:     device = x.device
693:     actor_critic.eval()
694: 
695:     pixel_ij = np.zeros((B, 2), dtype=np.int32)
696:     d_xy_arr = np.zeros((B, 2), dtype=np.float32)
697:     velocity_arr = np.zeros(B, dtype=np.float32)
698: 
699:     with torch.no_grad():
700:         # one U-Net pass — features + per-pixel params
701:         f, params, _ = actor_critic.features_and_params(x)
702: 
703:         # batched greedy pixel selection (per-pixel FiLM conditioned)
704:         greedy_pix, _ = actor_critic.greedy_pixel(f, params, contour_masks, top_k=top_k)
705: 
706:         for b in range(B):
707:             if contour_masks[b].sum() == 0:
708:                 velocity_arr[b] = 0.0           # no-op poke (empty contour)
709:                 continue
710: 
711:             if np.random.random() < epsilon:
712:                 if heuristic_actions is not None:
713:                     h_pixel, h_dxy, h_delta = heuristic_actions
714:                     pixel_ij[b] = h_pixel[b]
715:                     d_noisy = h_dxy[b]
716:                     dd_noisy = h_delta[b]
717:                 else:
718:                     valid = torch.nonzero(contour_masks[b], as_tuple=True)
719:                     r = np.random.randint(len(valid[0]))
720:                     pixel_ij[b, 0] = valid[0][r].item()
721:                     pixel_ij[b, 1] = valid[1][r].item()
722:                     d_noisy = np.random.randn(2).astype(np.float32)
723:                     dd_noisy = 0.05 + abs(np.float32(np.random.randn())) * 0.02
724:             else:
725:                 # greedy: pre-computed per-pixel-conditioned best pixel
726:                 pixel_ij[b, 0] = greedy_pix[b, 0].item()
727:                 pixel_ij[b, 1] = greedy_pix[b, 1].item()
728:                 # read params at selected pixel
729:                 p = params[b, :, pixel_ij[b, 0], pixel_ij[b, 1]].cpu().numpy()  # (3,)
730:                 d_noisy = p[:2] + np.random.randn(2).astype(np.float32) * noise_std
731:                 dd_noisy = p[2] + abs(np.float32(np.random.randn())) * noise_std
732: 
733:             # clamp & normalise direction
734:             norm = np.linalg.norm(d_noisy) + 1e-8
735:             d_xy_arr[b] = d_noisy / norm
736:             velocity_arr[b] = np.clip(dd_noisy, 0.0, actor_critic.velocity_max)
737: 
738:     return pixel_ij, d_xy_arr, velocity_arr
739: 
740: 
741: 
742: #* ================================================================
743: #*  Reward & termination
744: #* ================================================================
745: 
746: def _yaw_error(quats: np.ndarray, target_quats: np.ndarray) -> np.ndarray:
747:     """Wrapped angle difference between quaternion yaws (radians, [0, π])."""
748:     yaw = 2.0 * np.arctan2(quats[:, 3], quats[:, 0])
749:     t_yaw = 2.0 * np.arctan2(target_quats[:, 3], target_quats[:, 0])

(Showing lines 580-749 of 1525. Use offset=750 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "offset": 850,
  "limit": 200
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
850:         inside &= (local_xy[:, 1] >= wy_min) & (local_xy[:, 1] <= wy_max)
851:     return inside
852: 
853: 
854: def near_translation_targets(
855:     poses: dict,
856:     targets: dict,
857:     env_root_pos: np.ndarray,
858:     threshold: float = STOP_POKE_THRESHOLD,
859: ) -> np.ndarray:
860:     """Return envs where all object centers are close enough to targets."""
861:     num_envs = list(poses.values())[0][0].shape[0]
862:     near = np.ones(num_envs, dtype=bool)
863:     offset = env_root_pos[:, :2]
864:     for obj_name, (positions, _) in poses.items():
865:         local_xy = positions[:, :2].copy() - offset
866:         dist = np.linalg.norm(local_xy - targets[obj_name][:, :2], axis=1)
867:         near &= dist < threshold
868:     return near
869: 
870: 
871: #* ================================================================
872: #*  Environment step — articulated finger (batched, all envs)
873: #* ================================================================
874: 
875: async def env_step_async(
876:     fingers: Articulation,
877:     pixel_ij: np.ndarray,        # (B, 2)
878:     d_xy: np.ndarray,            # (B, 2)
879:     delta_d: np.ndarray,         # (B,)
880:     K: np.ndarray,
881:     active: np.ndarray | None = None,  # (B,) bool — which envs to control
882: ) -> tuple[dict, np.ndarray]:
883:     """Execute strikes with the prismatic XYZ finger robot.
884: 
885:     Phases:
886:       0. move to side-poke height
887:       1. move to standoff_xy at side-poke height
888:       2. settle briefly at standoff
889:       3. strike — push Δd forward past the contour (world_xy is midpoint)
890:       4. retract back to standoff_xy at side-poke height
891:       5. settle objects
892: 
893:     Inactive envs (where active=False) are frozen at their current position
894:     throughout all phases — no strike, no movement.
895:     """
896:     B = len(fingers)
897:     dof_indices = fingers.get_dof_indices(fingers.dof_names)
898: 
899:     #* ── convert pixel → world XY ───────────────────────────────────
900:     world_xy = np.zeros((B, 2), dtype=np.float32)
901:     for b in range(B):
902:         w = pixel_to_world(tuple(pixel_ij[b]), K)
903:         world_xy[b] = w[:2]
904: 
905:     #* ── normalise direction ────────────────────────────────────────
906:     dir_norm = np.linalg.norm(d_xy, axis=1, keepdims=True)
907:     dir_norm = np.where(dir_norm < 1e-8, 1.0, dir_norm)
908:     dirs = d_xy / dir_norm
909: 
910:     zero_v = np.zeros((B, 3), dtype=np.float32)
911:     xy_low  = np.array(LIMIT_LOWER[:2], dtype=np.float32)
912:     xy_high = np.array(LIMIT_UPPER[:2], dtype=np.float32)
913: 
914:     #* ── velocity → displacement for PD geometry ────────────────────
915:     #* world_xy is a contour point. The controlled point is the sphere centre,
916:     #* so keep at least one fingertip radius plus clearance outside the object.
917:     min_standoff = FINGERTIP_RADIUS + STANDOFF_CLEARANCE
918:     # integration: velocity (m/s) × time (s) = distance (m)
919:     delta_d_m = delta_d * VELOCITY_INTEGRATION_T
920:     max_displacement = VELOCITY_MAX * VELOCITY_INTEGRATION_T
921:     delta_d_clipped = np.clip(delta_d_m, min_standoff, max_displacement)
922:     standoff_xy = world_xy - dirs * delta_d_clipped[:, None]  # (B, 2) run-up (half Δd behind)
923:     strike_xy = world_xy + dirs * delta_d_clipped[:, None]     # (B, 2) symmetric penetration forward
924: 
925:     #* ── snap initial positions (inactive envs hold these throughout) ──
926:     try:
927:         q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)
928:     except AssertionError:
929:         app_utils.play()
930:         await _step_physics(2)
931:         q_cur = as_numpy(fingers.get_dof_positions()).copy()
932: 
933:     def _hold_inactive(targets: np.ndarray):
934:         if active is not None:
935:             targets[~active] = q_cur[~active]
936: 
937:     async def _drive_to(targets: np.ndarray, max_steps: int,
938:                         tol: float = FINGER_TRACK_TOL):
939:         fingers.set_dof_position_targets(positions=targets.tolist(), dof_indices=dof_indices)
940:         fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
941:         if active is not None and not np.any(active):
942:             await _step_physics(1)
943:             return
944:         for step_idx in range(max_steps):
945:             await _step_physics(1)
946:             try:
947:                 q_now = as_numpy(fingers.get_dof_positions())
948:             except AssertionError:
949:                 app_utils.play()
950:                 await _step_physics(2)
951:                 try:
952:                     q_now = as_numpy(fingers.get_dof_positions())
953:                 except AssertionError:
954:                     await _step_physics(max(0, max_steps - step_idx - 1))
955:                     break
956:             err = np.linalg.norm(q_now - targets, axis=1)
957:             if active is not None:
958:                 err = err[active]
959:             if err.size > 0 and float(np.max(err)) <= tol:
960:                 break
961: 
962:     #* ── Phase 0: move to side-poke height above current XY ──────────
963:     q0 = q_cur.copy()
964:     q0[:, 2] = SAFE_Z
965:     _hold_inactive(q0)
966:     await _drive_to(q0, max_steps=60)
967: 
968:     #* ── Phase 1: move to standoff at side-poke height ──────────────
969:     q1 = np.zeros((B, 3), dtype=np.float32)
970:     q1[:, :2] = np.clip(standoff_xy, xy_low, xy_high)
971:     q1[:, 2] = SAFE_Z
972:     _hold_inactive(q1)
973:     await _drive_to(q1, max_steps=100)
974: 
975:     #* ── Phase 2: hold standoff and drop down before impact ───────────────────────
976:     q2 = q1.copy()
977:     q2[:, 2] = POKE_Z
978:     _hold_inactive(q2)
979:     await _drive_to(q2, max_steps=30)
980: 
981:     #* ── Phase 3: strike — symmetric push past contour ────────────
982:     #* world_xy is the midpoint; the finger pushes Δd forward past it.
983:     #* Position-only PD (v_target=0): OVERTRAVEL via Δd creates a steady
984:     #* pushing force kp*Δd.  No velocity target avoids oscillation at contact.
985:     q3 = np.zeros((B, 3), dtype=np.float32)
986:     q3[:, :2] = np.clip(strike_xy, xy_low, xy_high)
987:     q3[:, 2] = POKE_Z
988:     _hold_inactive(q3)
989: 
990:     fingers.set_dof_position_targets(positions=q3.tolist(), dof_indices=dof_indices)
991:     fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
992:     await _step_physics(IMPACT_STEPS)
993: 
994:     #* ── Phase 4: retract back to standoff at side-poke height ───────
995:     q4 = q2.copy()
996:     q4[:, 2] = SAFE_Z
997:     _hold_inactive(q4)
998:     await _drive_to(q4, max_steps=80)
999: 
1000:     #* ── Phase 5: settle briefly, then clear residual object velocities ─
1001:     SETTLE_STEPS = 30
1002:     l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
1003:     cyl_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
1004: 
1005:     await _step_physics(SETTLE_STEPS)
1006:     zero_vel = np.zeros((B, 3), dtype=np.float32)
1007:     l_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_vel)
1008:     cyl_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_vel)
1009:     await _step_physics(1)
1010: 
1011:     #* ── query new object poses ─────────────────────────────────────
1012:     poses, env_root_pos = get_object_poses_vectorized()
1013:     return poses, env_root_pos
1014: 
1015: 
1016: #* ================================================================
1017: #*  Training step
1018: #* ================================================================
1019: 
1020: def train_step(
1021:     actor_critic: SpatialActorCritic,
1022:     target_net: SpatialActorCritic,
1023:     optimizer_q: torch.optim.Optimizer,
1024:     optimizer_mu: torch.optim.Optimizer,
1025:     batch: dict[str, torch.Tensor],
1026:     gamma: float = GAMMA,
1027: ):
1028:     """One DQN + DPG training step on a minibatch."""
1029:     x       = batch["x"]
1030:     pixel   = batch["pixel"]       # (B, 2)
1031:     d_xy_st = batch["d_xy"]        # (B, 2)  stored (with noise)
1032:     v_st    = batch["velocity"]      # (B, 1)  stored velocity (m/s)
1033:     r       = batch["r"]           # (B, 1)
1034:     x_next  = batch["x_next"]
1035:     done    = batch["done"]        # (B, 1)
1036:     mask_next = batch["mask_next"]  # (B, H, W) — pre-computed in sample()
1037: 
1038:     B = x.shape[0]
1039:     batch_idx = torch.arange(B, device=x.device)
1040:     actor_critic.train()
1041:     target_net.eval()
1042: 
1043:     #* -- stored action params (with exploration noise, as executed) -----
1044:     a_stored = torch.cat([d_xy_st, v_st], dim=1)  # (B, 3)
1045: 
1046:     #* =================================================================
1047:     #*  LOSS 1 — Q-Loss (Huber TD)
1048:     #* =================================================================
1049: 

(Showing lines 850-1049 of 1525. Use offset=1050 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 85.2s)

_Thinking:_

Now I have a very thorough understanding of the codebase. Let me synthesize what I know and create an implementation plan.

## Current State Understanding

### What the user has:
1. **`quintic_poly.py`** — Contains:
   - `quintic_poly_query(u)` — classic smoothstep function (6u⁵ - 15u⁴ + 10u³), maps u∈[0,1] to [0,1] with zero velocity/acceleration at boundaries
   - `compute_T(v_mid, L)` — computes trajectory duration T = L*1.875/v_mid
   - TODOs: use L=0.3m, v_mid from param-head output (cap at 1.5 m/s), control cycle steps = T*60 at 60Hz

2. **Network architecture** — `SpatialActorCritic` outputs per-pixel params (dx, dy, v) where v is velocity magnitude ∈ [0, VELOCITY_MAX=1.5]

3. **Current control flow** in `env_step_async()`:
   - Pixel → world XY
   - Delta_d (velocity) → displacement via `delta_d_m = delta_d * VELOCITY_INTEGRATION_T` (0.5s integration time)
   - Then PD position control: set_dof_position_targets to the displacement-augmented position
   - Position-only PD (v_target=0) with IMPACT_STEPS=80 for the strike phase
   - `kp*Δd` creates the pushing force

### What the user wants:
1. **Quintic polynomial for velocity magnitude** during the striking phase
   - The polynomial defines a smooth velocity profile: zero velocity at start, peak in middle, zero velocity at end
   - L = 0.3m travel distance → the actual displacement trajectory s(t) = L * quintic_poly_query(t/T)
   - T computed from v_mid (from param-head) and L
   
2. **Feedforward force computation** — use the polynomial's derivatives to compute feedforward force:
   - Position: s(t) = L * f(u) where u = t/T, f(u) = 6u⁵ - 15u⁴ + 10u³
   - Velocity: v(t) = s'(t) = (L/T) * f'(u) = (L/T) * (30u⁴ - 60u³ + 30u²)
   - Acceleration: a(t) = s''(t) = (L/T²) * f''(u) = (L/T²) * (120u³ - 180u² + 60u)
   - Feedforward force: F_ff = m * a(t)

3. **Integration into the control loop** — replace the current static PD position control with a time-varying trajectory that follows the quintic polynomial

Let me now think about the derivatives of the quintic polynomial:
- f(u) = 6u⁵ - 15u⁴ + 10u³
- f'(u) = 30u⁴ - 60u³ + 30u² = 30u²(u² - 2u + 1) = 30u²(1-u)²
- f''(u) = 120u³ - 180u² + 60u = 60u(2u² - 3u + 1) = 60u(2u-1)(u-1)

Key properties:
- f(0) = 0, f(1) = 1
- f'(0) = 0, f'(1) = 0
- f''(0) = 0, f''(1) = 0
- f'(0.5) = 30*(0.25)*(0.25) = 30*0.0625 = 1.875 ← max velocity at midpoint

This explains the constant 1.875 in `compute_T`: it's the maximum value of f'(u). So v_mid (the param-head velocity output) would equal the peak velocity of the polynomial = L*1.875/T. Rearranging: T = L*1.875/v_mid.

Now the key question: **how to use this for feedforward force?**

The current approach in `env_step_async()`:
- Pre-compute standoff_xy and strike_xy based on `delta_d * VELOCITY_INTEGRATION_T` displacement
- Then step physics for `IMPACT_STEPS` with a fixed PD position target (strike_xy)
- This gives `force = kp * (strike_xy - q) + kd * (0 - dq/dt)` — a purely passive PD approach

With the quintic polynomial, we could do time-varying position+velocity targets:
- At each timestep k ∈ [0, num_steps], compute u = k/num_steps
- Position target: s(k) = L * f(u) — but this needs to be in the right coordinate frame
- Velocity target: v(k) = (L/T) * f'(u)
- Feedforward force: F_ff = m_a * a(k) = m_a * (L/T²) * f''(u) — this is the inertial feedforward

Actually, let me re-read the current approach more carefully. The finger robot uses PD drive:
- `tau = kp*(q_target - q) + kd*(v_target - dq/dt)`

Currently, the strike phase sets:
- q_target = strike_xy (a fixed position)
- v_target = 0

This means the PD controller generates force proportionally to the position error. When the finger hits the object, the error builds up (finger can't reach through the object), creating a pushing force ~kp*(strike_xy - actual_x).

With feedforward, we want to:
1. Add a velocity profile so the finger follows a smooth quintic trajectory
2. Add feedforward force = m*a(t) to compensate for inertia

But wait — the key insight is that the current system is **position-controlled**, not force-controlled. The PD drive computes force internally, but we command position and velocity targets.

Looking more carefully at the architecture:

The `set_dof_velocity_targets` sets v_target. In the current code, it's always set to 0 (classic PD settle). If we set a non-zero v_target, the drive law becomes:
```
tau = kp*(q_target - q) + kd*(v_target - dq/dt)
```

This means we can inject a desired velocity trajectory that the PD controller tries to track.

For feedforward force, there are a few approaches:
1. **Velocity feedforward**: Set v_target = desired_velocity(t) — PD controller will apply force to track this velocity
2. **Position + velocity trajectory**: Set both q_target(t) and v_target(t) along the quintic trajectory — standard trajectory tracking
3. **Torque/force feedforward**: If the DriveAPI allows direct force input, add F_ff = m*a(t) to the drive output

Let me look at the finger robot drive configuration more carefully to understand what APIs are available.

Actually, looking at the `configure_drives` function and the USD DriveAPI, the available parameters are:
- stiffness (kp)
- damping (kd)  
- max_force
- target_position (q_target)
- target_velocity (v_target)

There's no direct force feedforward in the DriveAPI. But we can compute it ourselves:
- If we know the desired acceleration a(t), we can solve for the position target that would produce this force
- Or we can set v_target and rely on kd to provide the damping-force component

The cleanest approach for integrating quintic polynomial control in this architecture:

**Approach A: Time-varying trajectory tracking**
- Compute the full trajectory: time series of positions and velocities
- At each control step, set both position target and velocity target
- The PD controller tracks the trajectory

**Approach B: Feedforward force computation (external to DriveAPI)**
- Compute the feedforward force F_ff = m * a(t)
- Convert to equivalent position offset: Δq = F_ff / kp
- Add to the position target

Let me think about which approach makes more sense...

For the striking phase specifically:
- The finger moves in a straight line along the poke direction
- We want the velocity to follow a quintic profile: start from 0, peak, then end at 0
- The position target at the end should be strike_xy (past the object)

Let me trace through the math more carefully:

Starting position = standoff_xy (or more precisely, the position after Phase 2)
Ending position = strike_xy 
Travel distance L = 0.3m (given by user)

But wait — the current code computes `standoff_xy` and `strike_xy` symmetrically around world_xy based on delta_d. So the travel distance during the strike phase is:
- From standoff_xy to strike_xy = 2 * delta_d_clipped

The user says L = 0.3m, which seems like a fixed strike travel distance. So the end position would be `start_pos + dir * 0.3`.

But actually, rethinking: the user says "the striking phase has travel distance L = 0.3 (m)". This is a fixed parameter for the polynomial. The quintic polynomial then determines how the position evolves smoothly over this 0.3m distance.

Now, what is the "feedforward force"? In robotics, feedforward force is typically:
- The force needed to achieve desired acceleration, added to the PD controller output
- F_total = kp*e + kd*ė + F_ff
- where F_ff = m*a_desired(t)

In the DriveAPI context, we can achieve this by setting position and velocity targets at each timestep. But we can also approximate the feedforward by:
1. Setting v_target = desired velocity — this uses the damping term to apply velocity tracking force
2. The position target can be set to the desired position — this uses stiffness for position tracking

The combination of position + velocity targets in DriveAPI effectively gives us:
```
tau = kp*(q_des - q) + kd*(v_des - ė)
```
which is exactly what we want for tracking a trajectory (qd, vd).

**So the implementation plan should:**

1. **Expand `quintic_poly.py`** with:
   - Derivative functions: `quintic_poly_derivative(u)` → f'(u), `quintic_poly_second_derivative(u)` → f''(u)
   - `compute_trajectory(v_mid, L, dt=1/60)` → returns arrays of position_scale, velocity_scale, acceleration_scale for each timestep
   - Or just a function that computes s, v, a given u

2. **Modify the strike phase in `env_step_async()`** to:
   - Compute T from v_mid (from param-head) and L=0.3
   - Compute num_steps = T * 60
   - For each timestep, compute position target and velocity target using the quintic polynomial
   - Set both target_position and target_velocity for the drive API at each timestep

3. **Feedforward force**: Since we're using trajectory tracking (position + velocity targets), the PD controller already provides implicit feedforward. The velocity target gives the kd*(v_des - v) term which is essentially feedforward. But if we want explicit feedforward force added, we'd need to find a way to inject it. 

Actually, let me reconsider. Maybe the user is thinking about it differently. Let me re-read the TODO in quintic_poly.py:

```python
#TODO: with a predefined L = 0.3 m, and v_mid from param-head output (cap at 1.5 m/s)
#TODO: for the control cycle, num_steps = T * 60 (isaac sim default control cycle is 60Hz)
```

So the user wants:
1. L = 0.3m fixed
2. v_mid comes from param-head (the velocity output of the network, currently v ∈ [0, 1.5])
3. T is computed from L and v_mid
4. num_steps = T * 60

Then they mention "feedforward force to my finger robot". With the quintic polynomial and the trajectory computed, the feedforward force can be computed as:
- F_ff = m_fingertip * a(t)

where m_fingertip is the mass of the fingertip (default 0.3 kg from config, or randomized from [0.2, 0.5]).

In the DriveAPI, we can't directly set force, but we can:
- Set position targets to track the quintic trajectory
- Set velocity targets to track the quintic velocity profile

The effective feedforward force from velocity tracking is kd * v_des, and from position tracking is kp * q_des. So the trajectory tracking approach implicitly provides the feedforward.

But if the user specifically wants to compute and add an explicit feedforward force, we could:
- Compute acceleration a(t) from polynomial
- Compute F_ff = m * a(t)
- Convert to equivalent velocity feedforward: since tau = kd*(v_target - v), if we want to add F_ff on top of the PD output, we could set v_target such that kd*v_target contributes F_ff: v_target_ff = F_ff / kd
- Or convert to position: q_target_ff = F_ff / kp (+ actual desired position)

Hmm, but this gets confusing. Let me think about what's cleanest.

Actually, I think the simplest and most standard approach is:

**Use trajectory tracking with the quintic polynomial.** At each control timestep k:
1. Compute u = (k+1) / num_steps
2. Compute desired position: q_des = q_start + dir * L * f(u) 
3. Compute desired velocity: v_des = dir * (L/T) * f'(u)
4. Set both as drive targets

This gives smooth acceleration from rest, smooth deceleration to rest, and the PD controller provides the necessary forces. This is effectively feedforward because the velocity target term `kd*(v_des - v)` provides a feedforward damping force.

Now, if the user also wants to add explicit inertial feedforward, we'd compute:
- a_des = dir * (L/T²) * f''(u)
- F_ff = m_fingertip * a_des

But in DriveAPI without direct force control, the cleanest way to add F_ff is to augment the position target:
- q_target_with_ff = q_des + F_ff / kp
- This way, the PD term kp*(q_target_with_ff - q) includes both the position tracking and the feedforward force

Or more directly, since tau = kp*(q_target - q) + kd*(v_target - v), and at each step we want tau = F_ff + kp*(q_des - q) + kd*(v_des - v):
- q_target = q_des + F_ff/kp
- v_target = v_des

This gives us tau = kp*(q_des + F_ff/kp - q) + kd*(v_des - v) = kp*(q_des - q) + kd*(v_des - v) + F_ff

That's the clean approach! Set position target = desired_position + F_ff/kp and velocity target = desired_velocity.

Wait, but actually in a well-tuned PD controller, if the trajectory is smooth (which the quintic is), the tracking error will be small, and the PD controller will naturally provide the required forces without explicit feedforward. The feedforward would mainly help during high-acceleration phases.

Let me think about what the user actually needs vs. wants. They said "I can compute a feedforward force to my finger robot." This suggests they want to compute the feedforward force explicitly and add it to the control.

OK, let me now draft the plan.

Let me also check the finger mass and drive parameters more carefully.

From config:
- `drive_stiffness: 1500.0` — kp
- `drive_damping: 180.0` — kd  
- `drive_max_force: 500.0`
- `default_fingertip_mass: 0.3` (kg)
- `fingertip_mass_range: [0.2, 0.5]`

Fingertip mass is 0.3 kg default, could be randomized 0.2-0.5. For feedforward, we need the effective mass at the fingertip.

Now, the finger robot has 3 prismatic joints. The mass to move for each axis depends on the robot's kinematics. But as a first approximation, the fingertip mass (0.3 kg) plus some reflected inertia of the links could be used.

Actually, looking at `make_finger_robot.py` more carefully, the finger has:
- base (fixed to World)
- x_link → x_joint (prismatic along X)
- y_link → y_joint (prismatic along Y)  
- z_link → z_joint (prismatic along Z)

The z_link includes the fingertip sphere. The masses are set by `UsdPhysics.MassAPI`. Let me check what mass values are set...

From the scene setup, I see `default_fingertip_mass: 0.3` and `fingertip_mass_range: [0.2, 0.5]`. For feedforward, we'd need to know the mass of each moving link.

Actually, for simplicity and robustness, the feedforward can use the fingertip mass as the total effective mass, or the user can specify it. This is a reasonable approximation since the prismatic links are lightweight boxes.

OK, let me now formulate the implementation plan.

## Implementation Plan

### 1. Expand `quintic_poly.py`

Add derivative functions and trajectory generation:

```python
def quintic_poly_query(u):
    """Position profile: s(u) / L where s(0)=0, s(T)=L"""
    return 6 * u**5 - 15 * u**4 + 10 * u**3

def quintic_poly_derivative(u):
    """Velocity profile: v(u) * T / L = f'(u)"""
    return 30 * u**4 - 60 * u**3 + 30 * u**2

def quintic_poly_second_derivative(u):
    """Acceleration profile: a(u) * T² / L = f''(u)"""
    return 120 * u**3 - 180 * u**2 + 60 * u

def compute_T(v_mid, L):
    """Compute trajectory duration from peak velocity and travel distance."""
    T = (L * 1.875) / v_mid
    return T

def compute_trajectory(v_mid, L, dt=1/60):
    """Generate full trajectory time series.
    
    Returns:
        t: time array
        s: position array (scale 0→1, multiply by L*dir + q_start to get actual)
        v: velocity array (m/s, multiply by dir to get actual)
        a: acceleration array (m/s², multiply by dir to get actual)
    """
    T = compute_T(v_mid, L)
    num_steps = max(2, int(T / dt))
    t = np.linspace(0, T, num_steps)
    u = t / T
    s = L * quintic_poly_query(u)
    v = (L / T) * quintic_poly_derivative(u)
    a = (L / T**2) * quintic_poly_second_derivative(u)
    return t, s, v, a, T

def compute_feedforward_position_offset(a, mass, kp):
    """Convert feedforward force to equivalent position offset for DriveAPI."""
    F_ff = mass * a
    return F_ff / kp
```

### 2. Modify `env_step_async()` — Phase 3 (strike phase)

Replace the static position-based strike with a time-varying trajectory:

**Before (current):**
- Pre-compute strike_xy
- Set fixed position target
- Step physics for IMPACT_STEPS with v_target=0

**After (proposed):**
- Import trajectory computation from quintic_poly.py
- Compute T from v_mid (velocity magnitude from action) and L=0.3
- For each control timestep k ∈ [0, num_steps]:
  - Compute u = k / num_steps
  - Compute position, velocity, acceleration from polynomial
  - Apply direction vector
  - Compute feedforward position offset: Δq = F_ff/kp = m*a(t)/kp
  - Set position target = s(t)*dir + q_start + Δq
  - Set velocity target = v(t)*dir
  - Step physics 1 frame

### 3. Handle mass parameter

Need to retrieve fingertip mass (possibly randomized per env). Could:
- Read from config default (0.3 kg)
- Or read from the randomized finger properties (if used)
- Or add a config parameter for feedforward mass

### 4. Handle edge cases

- v_mid = 0 → no strike (already handled by active flag)
- v_mid very small → very long trajectory → cap num_steps
- v_mid at max → very short trajectory → minimum of a few steps
- Need to ensure the starting and ending positions align with the existing standoff/strike geometry

Wait, actually I need to reconsider the integration with the existing code. Currently:
- Phase 1: Move to standoff_xy at SAFE_Z
- Phase 2: Drop to POKE_Z at standoff position
- Phase 3: Strike — push from standoff_xy to strike_xy (both computed from delta_d)
  - travel distance = 2*delta_d_clipped
- Phase 4: Retract

With the quintic polynomial and L=0.3m, the travel distance is fixed at 0.3m (not 2*delta_d_clipped). So the strike phase would:
- Start at: the position after Phase 2 (standoff_xy at POKE_Z)
- End at: start + dir * 0.3m
- This might overshoot or undershoot the object depending on the standing distance

Actually, maybe the user wants L=0.3m to be the total travel distance, and the standoff distance is determined separately. Or maybe L=0.3m replaces the current delta_d-based computation for the strike phase only.

Let me re-read the user's request: "quintic scalar polynomial for my last action, velocity magnitude, assuming the striking phase has travel distance L = 0.3 (m)."

So L = 0.3m is the travel distance for the striking phase. The velocity magnitude from the param-head determines how fast this 0.3m is traversed (faster v_mid = shorter T = more aggressive).

But the current system uses delta_d to determine both the travel distance (2*delta_d_clipped) AND the velocity influence. With the quintic polynomial approach, we'd:
- Fix L = 0.3m (total travel distance during strike)
- Use v_mid from param-head to control speed/duration
- Derive q_start = standoff position (after Phase 2)
- The end position = q_start + dir * 0.3m

This changes the semantics somewhat — previously delta_d controlled both velocity and displacement. Now L is fixed and v_mid controls only the speed.

Hmm, but wait — the current architecture already computes delta_d from the velocity output. Let me trace through:
1. Action selects pixel, direction, velocity_mag
2. `env_step_async` receives delta_d (the velocity magnitude)
3. Computes: `delta_d_m = delta_d * VELOCITY_INTEGRATION_T` (velocity×time = distance)
4. `standoff_xy = world_xy - dirs * delta_d_m` (half the travel behind contour)
5. `strike_xy = world_xy + dirs * delta_d_m` (half the travel past contour)
6. Total travel = 2 * delta_d_m

So the total travel distance depends on the velocity. With the new approach:
- Total travel = L = 0.3m (fixed)
- standoff_xy stays the same (just before the object)
- strike_xy = standoff_xy + dir * 0.3m
- OR: we recompute geometry to keep symmetry around the contour point

Actually, I think the simplest integration is:
- Keep the standoff determination as-is (pixel-based)
- In Phase 3, instead of setting a fixed target at strike_xy and stepping for IMPACT_STEPS, use the quintic trajectory from standoff position to a target that's L=0.3m away
- During trajectory execution, at each step, set both position_target and velocity_target

This is cleaner because it doesn't change the geometric planning (standoff is still based on object location), just the execution of the strike motion.

Let me also think about: should the trajectory start from the current position after Phase 2, or from standoff_xy? It should start from wherever the finger actually is after settling at Phase 2 (which should be close to standoff_xy).

Let me draft the full plan now.

Actually, one more consideration: the feedforward force. Looking at the DriveAPI more carefully:

The drive equation is:
```
τ = kp * (q_target - q) + kd * (v_target - ė)
```

If we track a trajectory (q_des, v_des) with:
- q_target = q_des + F_ff/kp  
- v_target = v_des

Then:
```
τ = kp*(q_des + F_ff/kp - q) + kd*(v_des - ė)
  = kp*(q_des - q) + kd*(v_des - ė) + F_ff
  = PD feedback + inertial feedforward
```

This is standard computed-torque feedforward. Good.

But wait, F_ff = m*a_des, and if we also set q_target = q_des + m*a_des/kp, this is adding the feedforward through the position channel. An alternative is:
- q_target = q_des
- v_target = v_des + m*a_des/kd

But the position-based approach is more standard because kp is typically larger/more reliable than kd for force injection.

Either way works. Let me go with the position-offset approach since it's cleaner.

OK one more thing — the mass. The finger is a 3-DOF prismatic robot. The mass to accelerate for each axis might differ. Specifically:
- X and Y axes move horizontal masses
- Z axis moves vertical mass (against gravity too)

For the simplest approach, use the fingertip mass (0.3 kg default) as the effective mass for all axes since the links are lightweight.

Now, should I read more of `make_finger_robot.py` to understand the mass distribution? Let me check that.

Actually I think I have enough information. Let me formulate the concise plan.

Let me also consider: the user mentioned "feedforward force to my finger robot". In the context of this finger robot, the "feedforward" likely means:
1. Computing the desired acceleration trajectory from the quintic polynomial
2. Computing the corresponding inertial force: F = m * a(t)
3. Adding this to the control command

But in the DriveAPI, we can only set position and velocity targets. So the feedforward is implemented by augmenting the position target as described above.

Alternatively, maybe the user wants to bypass the PD controller entirely and directly command force/torque? But looking at the code, only position/velocity targets are used. The DriveAPI in PhysX might not support direct force control without changing the drive type.

Actually, let me look at the USD DriveAPI to see if there's a force drive option...

The code uses `PhysxSchema.PhysxDriveAPI` with `type="force"` — wait, no. Let me check.

In `make_finger_robot.py`, the drive is configured via `configure_drives` which sets stiffness, damping, and max_force. The drive type is the default USD Physics drive which is a PD position controller. There's no direct force input.

So the plan stays: implement trajectory tracking with position+velocity targets + position-based feedforward offset.

Let me now write up the plan concisely.

Now let me read the finger robot drive configuration and the impact phase more carefully:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/make_finger_robot.py",
  "offset": 200,
  "limit": 150
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/make_finger_robot.py</path>
<type>file</type>
<content>
200:     # axis (plain string auto-converts to TfToken: "X", "Y", or "Z")
201:     joint.CreateAxisAttr(axis_token)
202: 
203:     # limits
204:     joint.CreateLowerLimitAttr().Set(lower)
205:     joint.CreateUpperLimitAttr().Set(upper)
206: 
207: 
208: # ---------------------------------------------------------------------------
209: #  Finger articulation construction
210: # ---------------------------------------------------------------------------
211: def build_finger_articulation(
212:     root_path=FINGER_ROOT_PATH,
213:     root_position=None,
214:     drive_stiffness=DRIVE_STIFFNESS,
215:     drive_damping=DRIVE_DAMPING,
216:     fingertip_mass=DEFAULT_FINGERTIP_MASS,
217: ):
218:     """Build the 3-DOF prismatic finger robot hierarchy from scratch.
219: 
220:     Fixed-base articulation pattern (PhysX-compatible):
221: 
222:         /World
223:           └── FingerRobot                         ← Xform + ArticulationRootAPI
224:                 ├── base              (RigidBody + CollisionAPI)
225:                 │     anchored to /World via FixedJoint
226:                 ├── x_link            (RigidBody + CollisionAPI)
227:                 ├── y_link            (RigidBody + CollisionAPI)
228:                 ├── z_link            (RigidBody + CollisionAPI)
229:                 ├── base_fixed_joint  FixedJoint:      /World → base
230:                 ├── x_joint           PrismaticJoint:  base   → x_link  (X)
231:                 ├── y_joint           PrismaticJoint:  x_link → y_link  (Y)
232:                 └── z_joint           PrismaticJoint:  y_link → z_link  (Z)
233: 
234:         Chain:  /World ──[fixed]──▶ base ──[X]──▶ x_link ──[Y]──▶ y_link ──[Z]──▶ z_link
235:     """
236:     stage = stage_utils.get_current_stage(backend="usd")
237:     paths = get_finger_paths(root_path)
238:     if root_position is None:
239:         root_position = [0.0, 0.0, 0.0]
240: 
241:     # --- Ensure /World exists ---
242:     _ensure_xform(stage, "/World")
243: 
244:     # --- FingerRobot container with ArticulationRootAPI ---
245:     _ensure_xform(stage, paths["root"])
246:     root_prim = stage.GetPrimAtPath(paths["root"])
247:     UsdPhysics.ArticulationRootAPI.Apply(root_prim)
248:     physx_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
249:     physx_api.CreateSleepThresholdAttr(0.0)
250: 
251:     # --- Links (RigidBodyAPI + explicit MassAPI; collision on geometry children) ---
252:     _ensure_xform(stage, paths["base"])
253:     _apply_rigid_body(stage, paths["base"], mass_kg=0.1)
254:     _add_box_visual(stage, f"{paths['base']}/base_visual", (0.5, 0.5, 0.5))
255: 
256:     _ensure_xform(stage, paths["x_link"])
257:     _apply_rigid_body(stage, paths["x_link"], mass_kg=0.05)
258:     _add_box_visual(stage, f"{paths['x_link']}/x_visual", (0.9, 0.5, 0.5))
259: 
260:     _ensure_xform(stage, paths["y_link"])
261:     _apply_rigid_body(stage, paths["y_link"], mass_kg=0.05)
262:     _add_box_visual(stage, f"{paths['y_link']}/y_visual", (0.5, 0.9, 0.5))
263: 
264:     _ensure_xform(stage, paths["z_link"])
265:     _apply_rigid_body(stage, paths["z_link"], mass_kg=fingertip_mass)
266:     _add_sphere(
267:         stage, f"{paths['z_link']}/sphere", (1.0, 0.2, 0.2)
268:     )
269: 
270:     # --- Fixed joint: anchor base to /World (world is implicit body0) ---
271:     fixed_joint = UsdPhysics.FixedJoint.Define(
272:         stage, paths["base_fixed_joint"]
273:     )
274:     PhysxSchema.JointStateAPI.Apply(fixed_joint.GetPrim(), "linear")
275:     fixed_joint.CreateBody1Rel().SetTargets([paths["base"]])
276: 
277:     #! the conventions here is:
278:     #! a joint has two sides, body0 and body1 here; so CreateLocalPos0Attr()
279:     #! and CreateLocalPos1Attr() refers to the two bodies respectively.
280:     fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
281:     fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
282:     fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0)) #* [1, 0, 0, 0] identify rotation
283:     fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
284: 
285:     # --- Prismatic joints ---
286:     _create_prismatic_joint(
287:         stage, paths["x_joint"], paths["base"], paths["x_link"],
288:         axis_token="X",
289:         lower=LIMIT_LOWER[0], upper=LIMIT_UPPER[0],
290:         drive_stiffness=drive_stiffness,
291:         drive_damping=drive_damping,
292:     )
293:     _create_prismatic_joint(
294:         stage, paths["y_joint"], paths["x_link"], paths["y_link"],
295:         axis_token="Y",
296:         lower=LIMIT_LOWER[1], upper=LIMIT_UPPER[1],
297:         drive_stiffness=drive_stiffness,
298:         drive_damping=drive_damping,
299:     )
300:     _create_prismatic_joint(
301:         stage, paths["z_joint"], paths["y_link"], paths["z_link"],
302:         axis_token="Z",
303:         lower=LIMIT_LOWER[2], upper=LIMIT_UPPER[2],
304:         drive_stiffness=drive_stiffness,
305:         drive_damping=drive_damping,
306:     )
307: 
308:     # --- Root Xform for world positioning ---
309:     root_xform = XformPrim(
310:         paths=paths["root"],
311:         positions=root_position,
312:         reset_xform_op_properties=True,
313:     )
314:     return root_xform
315: 
316: 
317: #* ---------------------------------------------------------------------------
318: #*  Articulation setup
319: #* ---------------------------------------------------------------------------
320: def setup_finger_articulation(root_path=FINGER_ROOT_PATH) -> Articulation:
321:     """Wrap the finger robot prim with an Articulation for joint control."""
322:     articulation = Articulation(root_path)
323:     print(f"Finger robot – {articulation.num_dofs} DOFs: {articulation.dof_names}")
324:     return articulation
325: 
326: 
327: #! The USD actuation setup should be authored before simulation starts (like the 
328: #! URDF, or .world in gazebo). But here, it writes into the runtime PhysX articulation
329: #! /tensor handle. This overwrites what's pre-defined in the USD; but does not 
330: #! necessarily rewrite the authored USD attributes on disk/stage; it writes into the 
331: #! live PhysX/articulation runtime data.
332: #! Then if the statge is stopped/reloaded/recreated, PhysX will go back to whatever is
333: #! authored is USD unless set_dof_gains() is called again.
334: def configure_drives(
335:     articulation: Articulation,
336:     stiffnesses=DRIVE_STIFFNESS,
337:     dampings=DRIVE_DAMPING,
338: ):
339:     """Push stiffness/damping into the physics tensor after sim starts.
340: 
341:     USD DriveAPI attrs are only read at simulation start-up.  After play(),
342:     the tensor must be updated explicitly — otherwise gains are zero.
343:     """
344:     articulation.set_dof_gains(
345:         stiffnesses=stiffnesses,
346:         dampings=dampings,
347:     )

(End of file - total 347 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "offset": 910,
  "limit": 60
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
910:     zero_v = np.zeros((B, 3), dtype=np.float32)
911:     xy_low  = np.array(LIMIT_LOWER[:2], dtype=np.float32)
912:     xy_high = np.array(LIMIT_UPPER[:2], dtype=np.float32)
913: 
914:     #* ── velocity → displacement for PD geometry ────────────────────
915:     #* world_xy is a contour point. The controlled point is the sphere centre,
916:     #* so keep at least one fingertip radius plus clearance outside the object.
917:     min_standoff = FINGERTIP_RADIUS + STANDOFF_CLEARANCE
918:     # integration: velocity (m/s) × time (s) = distance (m)
919:     delta_d_m = delta_d * VELOCITY_INTEGRATION_T
920:     max_displacement = VELOCITY_MAX * VELOCITY_INTEGRATION_T
921:     delta_d_clipped = np.clip(delta_d_m, min_standoff, max_displacement)
922:     standoff_xy = world_xy - dirs * delta_d_clipped[:, None]  # (B, 2) run-up (half Δd behind)
923:     strike_xy = world_xy + dirs * delta_d_clipped[:, None]     # (B, 2) symmetric penetration forward
924: 
925:     #* ── snap initial positions (inactive envs hold these throughout) ──
926:     try:
927:         q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)
928:     except AssertionError:
929:         app_utils.play()
930:         await _step_physics(2)
931:         q_cur = as_numpy(fingers.get_dof_positions()).copy()
932: 
933:     def _hold_inactive(targets: np.ndarray):
934:         if active is not None:
935:             targets[~active] = q_cur[~active]
936: 
937:     async def _drive_to(targets: np.ndarray, max_steps: int,
938:                         tol: float = FINGER_TRACK_TOL):
939:         fingers.set_dof_position_targets(positions=targets.tolist(), dof_indices=dof_indices)
940:         fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
941:         if active is not None and not np.any(active):
942:             await _step_physics(1)
943:             return
944:         for step_idx in range(max_steps):
945:             await _step_physics(1)
946:             try:
947:                 q_now = as_numpy(fingers.get_dof_positions())
948:             except AssertionError:
949:                 app_utils.play()
950:                 await _step_physics(2)
951:                 try:
952:                     q_now = as_numpy(fingers.get_dof_positions())
953:                 except AssertionError:
954:                     await _step_physics(max(0, max_steps - step_idx - 1))
955:                     break
956:             err = np.linalg.norm(q_now - targets, axis=1)
957:             if active is not None:
958:                 err = err[active]
959:             if err.size > 0 and float(np.max(err)) <= tol:
960:                 break
961: 
962:     #* ── Phase 0: move to side-poke height above current XY ──────────
963:     q0 = q_cur.copy()
964:     q0[:, 2] = SAFE_Z
965:     _hold_inactive(q0)
966:     await _drive_to(q0, max_steps=60)
967: 
968:     #* ── Phase 1: move to standoff at side-poke height ──────────────
969:     q1 = np.zeros((B, 3), dtype=np.float32)

(Showing lines 910-969 of 1525. Use offset=970 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 46.2s)

_Thinking:_

Now I have a very complete picture. Let me also see the `train_step` function to understand how velocity is used in the buffer and how the DPG gradient relates to velocity.

Actually, I have enough to write the plan. Let me also check one thing — the `_step_physics` helper function in train.py, and check what happens in Phase 3 of env_step_async more carefully.

I think I have enough now. Let me write the implementation plan.

Key decisions to make:
1. **How to compute T**: T = L * 1.875 / v_mid, where v_mid is the velocity from param-head (capped at 1.5)
2. **How many steps**: num_steps = T * 60 (60Hz control)
3. **Starting position**: Current finger position after Phase 2 (should be standoff_xy at POKE_Z)
4. **Ending position**: start + dir * 0.3m
5. **Velocity target**: (L/T) * f'(u) along the direction
6. **Feedforward**: Compute a(t) = (L/T²) * f''(u), then F_ff = m * a(t), add as q_target offset: q_target = q_des + F_ff/kp
7. **Mass for feedforward**: Use fingertip mass (0.3 kg default, potentially randomized)

Let me also consider whether we should modify the existing `standoff_xy`/`strike_xy` geometry or if the quintic trajectory replaces it. The quintic trajectory from the current position to current_position + dir*0.3m replaces the strike_xy target. But we still need the standoff position (Phase 1-2) to be before the object.

Actually, I realize there's a subtlety: the current code computes standoff_xy as world_xy - dirs * delta_d_clipped. This places the standoff a certain distance behind the contour. With L=0.3m fixed, if we start from this standoff and travel 0.3m, we'd push 0.3m past the standoff. The object may be somewhere in between. This is fine — the finger will contact the object during the trajectory at some point and stop there (PD error will build up, applying force).

But wait, with the quintic trajectory, the position target keeps moving even after contact. The PD error (= des - actual) grows very large when the finger hits the object, creating large forces. This is exactly the same behavior as the current static position target approach, except with a smooth velocity profile.

Actually, let me reconsider. The current approach is:
- Set position target well past the object
- Let PD controller push
- After IMPACT_STEPS, retract

This works because there's no velocity target → the damping term kd*(0 - v) resists motion, and the stiffness term kp*(target - q) pushes. When hitting the object, q is stuck, so error = target - q grows → force = kp*error grows.

With the quintic approach:
- Position target moves along the trajectory
- Velocity target is set to match the desired velocity
- When hitting the object, q is stuck, but q_des keeps moving → error = q_des - q grows → force = kp*error grows

This is equivalent but smoother. The key difference is:
- Current: abrupt position change → large initial error → large initial force spike
- Quintic: smooth position change → gradual error buildup → smoother force profile

This is actually the whole point! The quintic polynomial gives a smooth acceleration profile, avoiding the initial jerk/force spike.

Now for the feedforward: even with position+velocity targets, the PD controller may lag behind during acceleration. Feedforward adds a pre-computed force to compensate:
- F_ff = m * a_des(t)
- Since τ = kp*(q_target - q) + kd*(v_target - v), we want τ ≈ F_ff + PD_feedback
- Set q_target = q_des + F_ff/kp so that kp*(q_des + F_ff/kp - q) = kp*(q_des - q) + F_ff
- Set v_target = v_des

This gives us the computed-torque-like feedforward within the DriveAPI constraints.

OK, one more consideration: the existing `standoff_xy` computation uses `delta_d_clipped = delta_d * VELOCITY_INTEGRATION_T`. With the new approach, should we still use delta_d to determine standoff, or should we use L? I think we should keep using delta_d for standoff planning (determines how far behind the object the finger starts), and use L=0.3m for the strike trajectory length. These are independent decisions.

But wait, if L=0.3m and standoff is e.g. 0.02m behind the contour, then the finger travels L=0.3m total, of which ~0.02m is free space and ~0.28m tries to penetrate the object. This is fine — the object stops the finger and the PD error generates the pushing force.

The user might want L=0.3m to be the free-space travel distance only, from standoff to the contour plane. But I think the simpler interpretation is total travel distance from standoff position forward = 0.3m.

Alright, let me finalize the plan. I'll also need to think about:

1. **Edge case: v_mid = 0 or very small** → T = infinity. Need to handle this.
2. **Edge case: v_mid at maximum** → T = 0.3 * 1.875 / 1.5 = 0.375s → num_steps = 0.375 * 60 = 22.5 → 23 steps. This is much shorter than IMPACT_STEPS=80. So the phase is shorter but more intense.
3. **The existing `delta_d` parameter**: Currently, `delta_d` is used both in standoff planning AND the implicit pushing force (via the fixed position target). With the quintic approach, we still need `delta_d` for standoff planning, but the strike force is determined by the trajectory. Should we decouple these?
4. **What about the direction vector?**: The direction d_xy for the strike is already computed in env_step_async. We use this same direction for the trajectory.

Let me trace through a complete example:
- v_mid = 0.4 m/s (from param-head)
- L = 0.3 m
- T = 0.3 * 1.875 / 0.4 = 1.40625 s
- num_steps = ceil(1.40625 * 60) = 85 steps ≈ 1.42s
- At u=0.5 (midpoint), v_max = (L/T) * f'(0.5) = (0.3/1.40625) * 1.875 = 0.4 m/s ✓
- At u=0.5, a = 0 (f''(0.5) = 0)
- At u=0.2, a = (0.3/1.40625²) * 60*0.2*(2*0.04 - 3*0.2 + 1) = ... let me not compute exactly but it's reasonable

Now, let me also consider: the quintic smoothstep's f'(u) = 30u²(1-u)² has maximum 1.875 at u=0.5. So v_mid = 0.4 gives v_max = 0.4 m/s, which is exactly what the user's network outputs. Good.

Now the implementation plan:

## Plan

### File 1: `quintic_poly.py` — Expand

Add:
- `quintic_poly_derivative(u)`: f'(u) = 30u⁴ - 60u³ + 30u²
- `quintic_poly_second_derivative(u)`: f''(u) = 120u³ - 180u² + 60u
- Hard-code L = 0.3 (or make it configurable)
- `generate_strike_trajectory(v_mid, dir_xy, q_start_xy, dt=1/60, mass=0.3, kp=1500.0)`:
  - Computes T = L*1.875/v_mid
  - Computes num_steps = max(2, int(T/dt))
  - Returns per-step: position targets, velocity targets (for setting DriveAPI)

### File 2: `train.py` — Modify env_step_async Phase 3

Replace the static position-target strike (lines ~981-992) with:
- Import trajectory generation from quintic_poly.py
- Compute T from v_mid (delta_d) and L=0.3
- Sample current position as q_start
- Generate trajectory from q_start → q_start + dir * 0.3
- For each timestep, set position + velocity targets
- Step physics 1 frame at a time

### Questions for the user:
1. Should L=0.3m be the total travel distance from the current position (including free space), or from the contour point?
2. Should the standoff geometry (Phase 1-2) change, or remain based on delta_d as before?
3. For feedforward force, should I use the default fingertip mass (0.3 kg) or read the potentially randomized mass per env?
4. Should I compute feedforward as a position offset (q_target += F_ff/kp) or as a velocity augment (v_target += F_ff/kd)?

Actually, let me not over-engineer the questions. The user said "generate an implementation plan". Let me just present the plan clearly and they can give feedback.

Now I have a thorough understanding of the codebase. Here is my analysis and implementation plan:

---

## Analysis

**Current Phase 3 (strike)** — `nn/train.py:981-992`:
- Computes `standoff_xy` and `strike_xy` geometrically from `delta_d * VELOCITY_INTEGRATION_T`
- Sets a **fixed** position target (`strike_xy`) with `v_target=0`
- Steps physics for `IMPACT_STEPS=80` — force comes from kp*(error) building up when the finger hits the object
- This is an **abrupt position jump** → large initial force spike (no velocity profile)

**What the quintic polynomial gives you:**
- S(x) = L · (6x⁵ - 15x⁴ + 10x³) — smooth position from 0→L with zero initial/final velocity and acceleration
- V(x) = (L/T) · (30x⁴ - 60x³ + 30x²) — velocity profile, peak = 1.875·L/T = v_mid at x=0.5
- A(x) = (L/T²) · (120x³ - 180x² + 60x) — acceleration, zero at endpoints, antisymmetric about x=0.5
- T = L·1.875 / v_mid — trajectory duration; for v_mid=0.4 m/s, T ≈ 1.4s (~85 steps at 60Hz)

---

## Implementation Plan

### Step 1 — Expand `env/quintic_poly.py`

Add the polynomial derivatives and a trajectory generator:

| Function | Purpose |
|----------|---------|
| `quintic_poly_derivative(u)` | f'(u) for velocity scaling |
| `quintic_poly_second_derivative(u)` | f''(u) for acceleration/feedforward |
| `generate_strike_trajectory(v_mid, L=0.3, dt=1/60)` | Returns arrays: position_scale (0→1), velocity (m/s), acceleration (m/s²), num_steps, T |

Key math:
- `num_steps = max(2, ceil(T/dt))` where `dt = 1/60`
- At each timestep k: `u = (k+1) / num_steps`, then evaluate s, v, a

### Step 2 — Modify Phase 3 in `nn/train.py:env_step_async`

Replace the static strike (lines ~975-992) with a per-timestep trajectory loop:

```
# Capture current position after Phase 2 as q_start
q_start = get_dof_positions()           # (B, 3)

# Compute trajectory
T, num_steps, s_scale, v_profile, a_profile = generate_strike_trajectory(delta_d[b])

# Per-timestep loop (replaces await _step_physics(IMPACT_STEPS)):
for k in range(num_steps):
    # Desired position + velocity + feedforward
    q_des[:, :2] = q_start[:, :2] + dirs * 0.3 * s_scale[k]    # XY travel
    q_des[:, 2] = q_start[:, 2]                                  # Z stays at POKE_Z
    
    # Velocity target (DriveAPI: kd*(v_target - v) term)
    v_des[:, :2] = dirs * v_profile[k]     # along direction
    v_des[:, 2] = 0.0
    
    # Feedforward: F_ff = m*a, inject via position offset (q_target += F_ff/kp)
    F_ff_xy = dirs * fingertip_mass * a_profile[k]
    q_des[:, :2] += F_ff_xy / kp
    
    fingers.set_dof_position_targets(q_des)
    fingers.set_dof_velocity_targets(v_des)
    await _step_physics(1)
```

After the loop, proceed to Phase 4 (retract) as before.

### Step 3 — Feedforward force design

Since DriveAPI only accepts position/velocity targets (no direct force input), feedforward is injected by offsetting the position target:

```
τ = kp*(q_target - q) + kd*(v_target - v)

Set  q_target = q_des + F_ff/kp   (where F_ff = m · a_des)
Set  v_target = v_des

⇒ τ = kp*(q_des - q) + kd*(v_des - v) + F_ff   ← PD feedback + inertial feedforward
```

Mass for F_ff: use the fingertip mass from config (default 0.3 kg, potentially randomized per env).

### Step 4 — Edge cases

- **v_mid = 0**: skip the trajectory, treat as no-op (already handled by `active` flag)
- **v_mid very small**: num_steps could exceed ~200; cap at some maximum (e.g., 120 steps ≈ 2s)
- **v_mid at max (1.5 m/s)**: num_steps ≈ 23 — shorter but more intense strike. This replaces `IMPACT_STEPS=80` with a variable-length phase.

---

## Open questions for you

1. **Standoff geometry**: The existing `standoff_xy` computation (from delta_d/contour) determines where Phase 1-2 places the finger before striking. Should I keep this unchanged, or should L=0.3m also influence the standoff distance? (E.g., standoff at 0.1m behind contour so that 0.1m of the 0.3m travel is free-space?)

2. **Mass source for feedforward**: Use `config["finger"]["default_fingertip_mass"]` (0.3 kg), or query the live PhysX mass from the articulation? Note that masses can be randomized per env.

3. **Phase 4/5 unchanged?** Phase 4 (retract) and Phase 5 (settle + zero object velocities) remain as-is, correct?

4. **Z-axis**: The quintic profile only applies to XY (the poke direction). Z stays at POKE_Z during the strike. Does this match your intention, or should Z also follow the profile (e.g., lift during the strike)?

Let me know your answers and I'll implement.

---

## User

for open question 1: keep L = 0.3 but make the contact point the mid point which is the expression in quintic_poly.py assumes. For open question 2: use the mass from each env from randomization. Open question 3: yes. 4. yes, keep them unchanged 

---

## Assistant (Plan · DeepSeek V4 Pro · 20.1s)

_Thinking:_

The user has answered my open questions. Let me now refine the implementation plan based on their answers:

1. **L = 0.3m with contact point as midpoint**: This means the quintic position profile s(u) goes from -0.15m to +0.15m relative to the contact/contour point, with s(0.5) = 0 at the contour. So the finger starts 0.15m behind the contour, passes through it at midpoint, and ends 0.15m past it.

   Actually, re-reading the quintic_poly.py more carefully: `quintic_poly_query(u)` goes from 0 to 1. With L=0.3, the position is `L * f(u)` going from 0 to 0.3.

   But the user says "make the contact point the mid point which is the expression in quintic_poly.py assumes." Looking at `compute_T(v_mid, L)`: the constant 1.875 comes from f'(0.5) = 1.875. This means at u=0.5 (midpoint of normalized time), the velocity is maximum (v_mid). At this point, f(0.5) = 6*(0.5)^5 - 15*(0.5)^4 + 10*(0.5)^3 = 6*(1/32) - 15*(1/16) + 10*(1/8) = 6/32 - 15/16 + 10/8 = 3/16 - 15/16 + 20/16 = 8/16 = 0.5.

   So at the time midpoint (u=0.5), the position is at L*0.5 = 0.15m, which is half the total travel. If the contour point is at the halfway mark, the finger travels from -0.15m (behind contour) to +0.15m (past contour), passing through the contour at the time midpoint.

   This is perfect! The existing `standoff_xy` computation puts the finger at `world_xy - dir * delta_d` (behind), and `strike_xy = world_xy + dir * delta_d` (forward). With L=0.3 and midpoint at contour, we have:
   - Start position: contour - dir * 0.15
   - End position: contour + dir * 0.15
   - The trajectory passes through the contour at u=0.5

   This replaces the current standoff/strike computation for Phase 3. The standoff is now fixed at 0.15m behind the contour (instead of delta_d-based).

   Wait, but Phase 1-2 currently move to standoff_xy. If the quintic trajectory starts behind the contour at -0.15m, and Phase 2 already places the finger at standoff_xy... I need to make sure these align.

   With the quintic approach:
   - Phase 1: Move to (contour_xy - dir * 0.15) at SAFE_Z
   - Phase 2: Drop to POKE_Z
   - Phase 3: Execute quintic trajectory from (contour_xy - dir * 0.15) at POKE_Z to (contour_xy + dir * 0.15) at POKE_Z

   This means Phase 1 standoff should be contour_xy - dir * 0.15, NOT based on delta_d. The trajectory handles the rest.

   Actually, wait. The quintic polynomial defines the position as s(u) = L * f(u) going from 0 to L. To make the contour the midpoint:
   - s(0) = 0 → start position
   - s(0.5) = L/2 = 0.15 → contour point
   - s(1) = L = 0.3 → end position

   So: start_pos = contour_xy - dir * (L/2) = contour_xy - dir * 0.15
       end_pos = contour_xy + dir * (L/2) = contour_xy + dir * 0.15

   The standoff_xy for Phase 1 should be: contour_xy - dir * 0.15
   The strike end should be: contour_xy + dir * 0.15

   This replaces the existing `standoff_xy` = `world_xy - dirs * delta_d_clipped` and `strike_xy` = `world_xy + dirs * delta_d_clipped`.

   So the standoff geometry changes to use the fixed L/2 = 0.15m instead of delta_d-based distance.

2. **Mass from each env randomization**: Need to read the randomized finger properties (which include masses). Looking at the code, `randomized_finger_properties` is stored as a global variable with fields like `stiffnesses`, `dampings`, and probably `masses` or `fingertip_masses`. I need to check how the randomization stores mass info.

   Actually, I need to look more carefully at where the fingertip mass randomization happens and how it's stored. Let me check the scene setup files.

3. **Phase 4/5 unchanged**: retract and settle remain as-is.

4. **Z stays fixed**: Only XY follows the quintic profile.

Let me now refine the implementation plan with these specifics.

Now, about the mass for feedforward: I need to understand how the randomized finger properties expose mass. Let me check the scene setup files to see the structure.

Also, I need to reconsider the plan more carefully:

### Revised Plan Details

#### Phase 1-2 (standoff geometry)
Instead of computing standoff_xy from delta_d:
- `standoff_xy = world_xy - dirs * 0.15` (half of L behind contour)
- Phase 1: Move to standoff_xy at SAFE_Z
- Phase 2: Drop to POKE_Z

#### Phase 3 (quintic trajectory strike)
- Capture q_start = current position after Phase 2
- Compute T = L * 1.875 / v_mid = 0.3 * 1.875 / v_mid = 0.5625 / v_mid
- num_steps = max(2, int(T * 60))
- For each step k:
  - u = (k+1) / num_steps
  - s = L * f(u) = 0.3 * (6u⁵ - 15u⁴ + 10u³)  →  the position along the direction, starting from 0 at q_start
  - v = (L/T) * f'(u) → velocity along direction
  - a = (L/T²) * f''(u) → acceleration along direction
  
  - q_des[:, :2] = q_start[:, :2] + dirs * s  (where s goes from 0 to 0.3)
  - q_des[:, 2] = q_start[:, 2] (keeps POKE_Z)
  
  - v_des[:, :2] = dirs * v
  - v_des[:, 2] = 0.0
  
  - F_ff_xy = dirs * mass * a  (per-env mass)
  - q_des[:, :2] += F_ff_xy / kp  (feedforward offset)
  
  - Set targets, step physics

But wait — there's a subtlety. The starting position `q_start` is captured after Phase 2, which placed the finger at `standoff_xy` (contour - dir*0.15) at POKE_Z. If the trajectory runs from this position to `standoff_xy + dir*0.3`, the end position would be `contour + dir*0.15`. This aligns correctly with the midpoint being the contour.

But actually, I'm using s from 0 to 0.3 relative to q_start. Since q_start = contour - dir*0.15, the trajectory endpoint = contour - dir*0.15 + dir*0.3 = contour + dir*0.15. And at s=0.15 (u=0.5, midpoint), the position = contour, which is the contact point. ✓

#### Mass for feedforward
I need to check how the randomized finger properties expose mass. Let me look at the scene setup file.

Actually, I should also consider: since the user wants per-env mass for feedforward, I need to get the mass from live PhysX properties or from the stored randomization data. Let me check how finger properties are randomized.

Let me also think about edge cases more carefully:
- v_mid=0: skip Phase 3 entirely
- v_mid very small (<0.01): T > 56s, num_steps > 3375. Cap num_steps at some max (e.g., 200-300).
- v_mid at max (1.5): T = 0.5625/1.5 = 0.375s, num_steps = 23

And the `active` mask handling: inactive envs should be frozen throughout.

Let me now check the scene setup for mass randomization info, then present the refined plan.

Good, let me refine the plan based on your answers. First, let me check how mass randomization is stored so I can access per-env masses for feedforward:

**Tool: grep**

**Input:**
```json
{
  "pattern": "finger_properties|fingertip_mass|masses|randomized_finger",
  "path": "/home/xiao/0_codes/poke_map/env",
  "include": "*.py"
}
```

**Output:**
```
Found 46 matches
/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py:
  Line 67:     "randomize_finger_properties_on_start",

  Line 109:         "randomized_finger_properties",

  Line 153:     l_object = RigidPrim(paths=path, masses=1.0)

  Line 169:     cylinder_object = RigidPrim(paths=path, masses=1.0)

  Line 339: def randomize_finger_properties(finger_articulations, finger_tip_links, seed=None):

  Line 354:     fingertip_masses = rng.uniform(

  Line 360:     finger_tip_links.set_masses(fingertip_masses)

  Line 365:         "fingertip_masses": fingertip_masses,

  Line 384:     finger_properties=None,

  Line 389:         if finger_properties is None:

  Line 394:                 stiffnesses=finger_properties["stiffnesses"],

  Line 395:                 dampings=finger_properties["dampings"],

  Line 415:     finger_properties=None,

  Line 418:         settle_scene_async(steps, finger_articulations, finger_target, finger_properties)

  Line 466:     randomized_finger_properties = None

  Line 475:         randomized_finger_properties = randomize_finger_properties(

  Line 486:         finger_properties=randomized_finger_properties,

  Line 524:             "randomized_finger_properties": randomized_finger_properties,


/home/xiao/0_codes/poke_map/env/poke_executor_articulated_vectorized.py:
  Line 97: def get_randomized_finger_properties():

  Line 99:         "randomized_finger_properties",

  Line 100:         getattr(__main__, "randomized_finger_properties", None),

  Line 104: def configure_vectorized_drives(finger_articulations, finger_properties=None):

  Line 105:     if finger_properties is None:

  Line 112:         stiffnesses=finger_properties["stiffnesses"],

  Line 113:         dampings=finger_properties["dampings"],

  Line 171:     finger_properties = get_randomized_finger_properties()

  Line 172:     configure_vectorized_drives(finger_articulations, finger_properties)

  Line 209:             "finger_properties": finger_properties,


/home/xiao/0_codes/poke_map/env/scene_setup_standalone_vectorized.py:
  Line 85:     "randomize_striker_masses_on_start",

  Line 139:         "randomized_striker_masses",

  Line 235:     l_object = RigidPrim(paths=path, masses=1.0)

  Line 253:     cylinder_object = RigidPrim(paths=path, masses=1.0)

  Line 399: def randomize_striker_masses(

  Line 402:     """Randomise striker masses for domain randomization."""

  Line 405:     masses = rng.uniform(

  Line 408:     strikers.set_masses(masses)

  Line 409:     return masses

  Line 522:     #* ── Randomize initial poses & striker masses ────────────────────

  Line 524:     randomized_striker_masses = None

  Line 533:         randomized_striker_masses = randomize_striker_masses(

  Line 571:             "randomized_striker_masses": randomized_striker_masses,


/home/xiao/0_codes/poke_map/env/make_finger_robot.py:
  Line 74: DEFAULT_FINGERTIP_MASS = FINGER_CONFIG["default_fingertip_mass"]  # kg, authored on z_link rigid body

  Line 79: FINGERTIP_MASS_RANGE = FINGER_CONFIG["fingertip_mass_range"]      # kg

  Line 105:         "fingertip_mass": float(rng.uniform(FINGERTIP_MASS_RANGE[0], FINGERTIP_MASS_RANGE[1])),

  Line 216:     fingertip_mass=DEFAULT_FINGERTIP_MASS,

  Line 265:     _apply_rigid_body(stage, paths["z_link"], mass_kg=fingertip_mass)

```

---

## Assistant (Plan · DeepSeek V4 Pro · 3.4s)

_Thinking:_

Now I need to see the full `randomize_finger_properties` function and the `settle_scene_async` function to understand how finger properties are structured and stored:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py",
  "offset": 96,
  "limit": 30
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py</path>
<type>file</type>
<content>
96:         "cylinder_shape",
97:         "cylinder_geometry",
98:         "cylinder_object",
99:         "cylinder_objects",
100:         "source_env",
101:         "env_roots",
102:         "env_paths",
103:         "cloner",
104:         "randomized_poses",
105:         "source_finger_root_xform",
106:         "finger_articulations",
107:         "finger_tip_links",
108:         "finger_settle_task",
109:         "randomized_finger_properties",
110:     ):
111:         globals().pop(handle_name, None)
112:     gc.collect()
113: 
114: 
115: def get_l_object_part_paths(path):
116:     return [
117:         f"{path}/VerticalLeg",
118:         f"{path}/HorizontalLeg",
119:     ]
120: 
121: 
122: def create_l_shaped_object(path, origin, material):
123:     stage = stage_utils.get_current_stage(backend="usd")
124:     l_part_paths = get_l_object_part_paths(path)
125: 

(Showing lines 96-125 of 542. Use offset=126 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py",
  "offset": 339,
  "limit": 40
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py</path>
<type>file</type>
<content>
339: def randomize_finger_properties(finger_articulations, finger_tip_links, seed=None):
340:     rng = np.random.default_rng(seed)
341:     num_fingers = len(finger_articulations)
342:     num_dofs = finger_articulations.num_dofs
343: 
344:     stiffnesses = rng.uniform(
345:         DRIVE_STIFFNESS_RANGE[0],
346:         DRIVE_STIFFNESS_RANGE[1],
347:         size=(num_fingers, num_dofs),
348:     ).astype(np.float32)
349:     dampings = rng.uniform(
350:         DRIVE_DAMPING_RANGE[0],
351:         DRIVE_DAMPING_RANGE[1],
352:         size=(num_fingers, num_dofs),
353:     ).astype(np.float32)
354:     fingertip_masses = rng.uniform(
355:         FINGERTIP_MASS_RANGE[0],
356:         FINGERTIP_MASS_RANGE[1],
357:         size=(num_fingers, 1),
358:     ).astype(np.float32)
359: 
360:     finger_tip_links.set_masses(fingertip_masses)
361: 
362:     return {
363:         "stiffnesses": stiffnesses,
364:         "dampings": dampings,
365:         "fingertip_masses": fingertip_masses,
366:     }
367: 
368: 
369: #! need to step the app in async mode as the script editor is already in this mode
370: #! so app_utils.update_app(steps=DELAY_TO_SETTLE) gives erros
371: def move_fingers_to(finger_articulations, xyz, num_steps=120):
372:     dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
373:     finger_articulations.set_dof_position_targets(
374:         positions=list(xyz),
375:         dof_indices=dof_indices,
376:     )
377:     return run_coroutine(settle_scene_async(num_steps))
378: 

(Showing lines 339-378 of 542. Use offset=379 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py",
  "offset": 375,
  "limit": 150
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py</path>
<type>file</type>
<content>
375:         dof_indices=dof_indices,
376:     )
377:     return run_coroutine(settle_scene_async(num_steps))
378: 
379: 
380: async def settle_scene_async(
381:     steps,
382:     finger_articulations=None,
383:     finger_target=None,
384:     finger_properties=None,
385: ):
386:     app_utils.play()
387:     await app_utils.update_app_async(steps=1)
388:     if finger_articulations is not None:
389:         if finger_properties is None:
390:             configure_drives(finger_articulations)
391:         else:
392:             configure_drives(
393:                 finger_articulations,
394:                 stiffnesses=finger_properties["stiffnesses"],
395:                 dampings=finger_properties["dampings"],
396:             )
397:         if finger_target is not None:
398:             dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
399:             finger_articulations.set_dof_position_targets(
400:                 positions=list(finger_target),
401:                 dof_indices=dof_indices,
402:             )
403:     if steps > 1:
404:         await app_utils.update_app_async(steps=steps - 1)
405:     print(f"Settled scene for {steps} app update steps.")
406: 
407: 
408: #! run_coroutine is needed as async function cannot be called directly from the
409: #! main(). run_coroutine() tells isaac to run the async task safely through Kit's
410: #! event loop
411: def schedule_settle_scene(
412:     steps,
413:     finger_articulations=None,
414:     finger_target=None,
415:     finger_properties=None,
416: ):
417:     return run_coroutine(
418:         settle_scene_async(steps, finger_articulations, finger_target, finger_properties)
419:     )
420: 
421: 
422: def main():
423:     clear_previous_handles()
424:     stage_utils.create_new_stage()
425:     stage_utils.set_stage_units(meters_per_unit=1.0)
426: 
427:     ground_plane = GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
428: 
429:     distant_light = DistantLight("/World/DistantLight")
430:     distant_light.set_intensities(500)
431: 
432:     l_material = PreviewSurfaceMaterial("/VisualMaterials/l_object_blue")
433:     l_material.set_input_values("diffuseColor", [0.1, 0.35, 1.0])
434: 
435:     cylinder_material = PreviewSurfaceMaterial("/VisualMaterials/cylinder_orange")
436:     cylinder_material.set_input_values("diffuseColor", [1.0, 0.45, 0.05])
437: 
438: 
439:     #! create a source env with L-shaped object, cylinder, and finger robot
440:     (
441:         source_env,
442:         l_parts,
443:         l_geometry,
444:         l_transform,
445:         l_object,
446:         cylinder_shape,
447:         cylinder_geometry,
448:         cylinder_object,
449:         source_finger_root_xform,
450:     ) = create_source_env(SOURCE_ENV_PATH, l_material, cylinder_material)
451: 
452:     #! replicate the source env here
453:     cloner, env_paths = clone_envs(
454:         source_env_path=SOURCE_ENV_PATH,
455:         num_envs=NUM_ENVS,
456:         spacing=ENV_SPACING,
457:     )
458: 
459:     env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
460:     l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
461:     cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
462:     finger_articulations = Articulation(paths=FINGER_ROOT_PATTERN)
463:     finger_tip_links = RigidPrim(paths=FINGER_TIP_LINK_PATTERN)
464: 
465:     randomized_poses = None
466:     randomized_finger_properties = None
467:     if RANDOMIZE_OBJECTS_ON_START:
468:         randomized_poses = randomize_object_poses(
469:             env_roots=env_roots,
470:             l_objects=l_objects,
471:             cylinder_objects=cylinder_objects,
472:             seed=RANDOM_SEED,
473:         )
474:     if RANDOMIZE_FINGER_PROPERTIES_ON_START:
475:         randomized_finger_properties = randomize_finger_properties(
476:             finger_articulations=finger_articulations,
477:             finger_tip_links=finger_tip_links,
478:             seed=RANDOM_SEED + 1,
479:         )
480:     
481:     #! asynchronously forward the physics to resolve overlap after randomization
482:     settle_task = schedule_settle_scene(
483:         DELAY_TO_SETTLE,
484:         finger_articulations=finger_articulations,
485:         finger_target=DEFAULT_XYZ,
486:         finger_properties=randomized_finger_properties,
487:     )
488: 
489:     ViewportManager.set_camera_view(
490:         "/OmniverseKit_Persp",
491:         eye=[2.8, 2.8, 2.0],
492:         target=[0.0, 0.0, 0.45],
493:     )
494: 
495: 
496:     #! This turns local variables from main() into global variables, so after 
497:     #! main() finishes, these objects are still accessible. This lets us interact
498:     #! with these objects after running the scripts
499:     globals().update(
500:         {
501:             "ground_plane": ground_plane,
502:             "distant_light": distant_light,
503:             "l_material": l_material,
504:             "cylinder_material": cylinder_material,
505:             "l_parts": l_parts,
506:             "l_geometry": l_geometry,
507:             "l_transform": l_transform,
508:             "l_object": l_object,
509:             "l_objects": l_objects,
510:             "cylinder_shape": cylinder_shape,
511:             "cylinder_geometry": cylinder_geometry,
512:             "cylinder_object": cylinder_object,
513:             "cylinder_objects": cylinder_objects,
514:             "source_env": source_env,
515:             "env_roots": env_roots,
516:             "env_paths": env_paths,
517:             "cloner": cloner,
518:             "randomized_poses": randomized_poses,
519:             "settle_task": settle_task,
520:             "source_finger_root_xform": source_finger_root_xform,
521:             "finger_articulations": finger_articulations,
522:             "finger_tip_links": finger_tip_links,
523:             "move_fingers_to": move_fingers_to,
524:             "randomized_finger_properties": randomized_finger_properties,

(Showing lines 375-524 of 542. Use offset=525 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 4.6s)

_Thinking:_

Good, now I understand the mass structure. The `randomized_finger_properties` dict contains:
- `"stiffnesses"`: (num_fingers, num_dofs) = (64, 3)
- `"dampings"`: (num_fingers, num_dofs) = (64, 3)  
- `"fingertip_masses"`: (num_fingers, 1) = (64, 1)

The `fingertip_masses` is stored as `(num_fingers, 1)` and can be used directly for per-env feedforward.

In `train.py`, there's already a `get_randomized_finger_properties()` function imported that accesses this global. I can use the same pattern.

Now, let me also check the train.py to see how `env_step_async` currently accesses finger properties and kp/kd:

Actually, `env_step_async` doesn't currently access finger properties. It's used only in `train_step` via `configure_vectorized_drives`. But in `train.py`'s main training loop, the fingers and properties are set up. Let me check how the trainer accesses the properties.

Let me look at the Trainer class or main training loop in train.py to see how it accesses the finger robot properties and how env_step_async is called.

**Tool: grep**

**Input:**
```json
{
  "include": "train.py",
  "pattern": "env_step|finger_properties|get_randomized_finger|fingers|stiffness|damping|DRIVE_STIFFNESS",
  "path": "/home/xiao/0_codes/poke_map"
}
```

**Output:**
```
Found 26 matches
/home/xiao/0_codes/poke_map/nn/train.py:
  Line 255:     fingers = Articulation(paths=FINGER_ROOT_PATTERN)

  Line 258:     return fingers, tips, env_roots

  Line 875: async def env_step_async(

  Line 876:     fingers: Articulation,

  Line 896:     B = len(fingers)

  Line 897:     dof_indices = fingers.get_dof_indices(fingers.dof_names)

  Line 927:         q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)

  Line 931:         q_cur = as_numpy(fingers.get_dof_positions()).copy()

  Line 939:         fingers.set_dof_position_targets(positions=targets.tolist(), dof_indices=dof_indices)

  Line 940:         fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)

  Line 947:                 q_now = as_numpy(fingers.get_dof_positions())

  Line 952:                     q_now = as_numpy(fingers.get_dof_positions())

  Line 990:     fingers.set_dof_position_targets(positions=q3.tolist(), dof_indices=dof_indices)

  Line 991:     fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)

  Line 1162:         finger_props = globals().get("randomized_finger_properties",

  Line 1163:                         getattr(__main__, "randomized_finger_properties", None))

  Line 1165:             configure_drives(self.fingers,

  Line 1166:                              stiffnesses=finger_props["stiffnesses"],

  Line 1167:                              dampings=finger_props["dampings"])

  Line 1169:             configure_drives(self.fingers)

  Line 1176:         self.fingers, self.tips, self.env_roots = get_finger_handles()

  Line 1215:         dof_idx = self.fingers.get_dof_indices(self.fingers.dof_names)

  Line 1218:         self.fingers.set_dof_position_targets(positions=init_pos.tolist(), dof_indices=dof_idx)

  Line 1219:         self.fingers.set_dof_velocity_targets(velocities=init_vel.tolist(), dof_indices=dof_idx)

  Line 1294:             poses_after, _ = await env_step_async(

  Line 1295:                 self.fingers, pixel_ij, d_xy, delta_d, self.K,

```

---

## Assistant (Plan · DeepSeek V4 Pro · 2.4s)

_Thinking:_

Let me check lines 1160-1170 to see how finger_props are accessed in the Trainer class:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "limit": 80,
  "offset": 1155
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
1155:         self.ep_lengths: deque[int]   = deque(maxlen=100)
1156: 
1157:     async def setup(self):
1158:         """Async init — starts sim, gets handles, configures drives."""
1159:         await ensure_sim_running_async()
1160:         self.refresh_scene_handles()
1161: 
1162:         finger_props = globals().get("randomized_finger_properties",
1163:                         getattr(__main__, "randomized_finger_properties", None))
1164:         if finger_props is not None:
1165:             configure_drives(self.fingers,
1166:                              stiffnesses=finger_props["stiffnesses"],
1167:                              dampings=finger_props["dampings"])
1168:         else:
1169:             configure_drives(self.fingers)
1170: 
1171:         self.K = get_camera_intrinsics()
1172:         update_plane_overlays(self.K)
1173: 
1174:     def refresh_scene_handles(self):
1175:         """Recreate Isaac wrappers after stage/timeline reinitialization."""
1176:         self.fingers, self.tips, self.env_roots = get_finger_handles()
1177: 
1178:     def _decay_schedule(self, episode: int):
1179:         frac = min(1.0, episode / EPS_DECAY)
1180:         self.epsilon  = EPS_START + (EPS_END - EPS_START) * frac
1181:         self.noise_std = SIGMA_START + (SIGMA_END - SIGMA_START) * frac
1182: 
1183: 
1184:     #! Reset 
1185:     async def _reset_episode(self, episode: int) -> tuple[torch.Tensor, list, torch.Tensor, dict]:
1186:         """Randomise objects + targets, return first observation."""
1187:         self.refresh_scene_handles()
1188: 
1189:         #* RigidPrim handles (use same patterns as scene setup)
1190:         l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
1191:         cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
1192: 
1193:         try:
1194:             randomize_object_poses(
1195:                 self.env_roots, l_objects, cylinder_objects,
1196:                 seed=None,
1197:             )
1198:         except AssertionError:
1199:             await ensure_sim_running_async()
1200:             self.refresh_scene_handles()
1201:             l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
1202:             cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
1203:             randomize_object_poses(
1204:                 self.env_roots, l_objects, cylinder_objects,
1205:                 seed=None,
1206:             )
1207:         # clear residual velocities from previous episode
1208:         zero_vel = np.zeros((NUM_ENVS, 3), dtype=np.float32)
1209:         zero_ang = np.zeros((NUM_ENVS, 3), dtype=np.float32)
1210:         l_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_ang)
1211:         cylinder_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_ang)
1212:         await _step_physics(SCENE_CONFIG["settle_steps"])
1213: 
1214:         #* ── reposition finger at object level (centre of each env) ──
1215:         dof_idx = self.fingers.get_dof_indices(self.fingers.dof_names)
1216:         init_pos = np.tile([0.0, 0.0, POKE_Z], (NUM_ENVS, 1)).astype(np.float32)
1217:         init_vel = np.zeros((NUM_ENVS, 3), dtype=np.float32)
1218:         self.fingers.set_dof_position_targets(positions=init_pos.tolist(), dof_indices=dof_idx)
1219:         self.fingers.set_dof_velocity_targets(velocities=init_vel.tolist(), dof_indices=dof_idx)
1220:         await _step_physics(30)  # let PD converge to centre
1221: 
1222:         # random position targets
1223:         targets_pos = sample_target_poses(self.rng, NUM_ENVS)
1224: 
1225:         # query current object poses for yaw curriculum
1226:         poses_before, env_root_pos = get_object_poses_vectorized()
1227: 
1228:         # target orientations — Cylinder is identity, LObject uses curriculum
1229:         targets_ori = {"Cylinder": np.zeros((NUM_ENVS, 4), dtype=np.float32)}
1230:         targets_ori["Cylinder"][:, 0] = 1.0
1231: 
1232:         l_quats = poses_before["LObject"][1]
1233:         l_yaws = 2.0 * np.arctan2(l_quats[:, 3], l_quats[:, 0])
1234: 

(Showing lines 1155-1234 of 1525. Use offset=1235 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 2.1s)

_Thinking:_

Let me also check line 1294-1300 to see how `env_step_async` is called and what parameters are passed:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/nn/train.py",
  "limit": 50,
  "offset": 1260
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/nn/train.py</path>
<type>file</type>
<content>
1260:         ep_len = 0
1261:         active_counts = []
1262:         ep_active_steps = 0
1263:         final_has_contour = np.ones(NUM_ENVS, dtype=bool)
1264: 
1265:         for step in range(MAX_STEPS):
1266:             has_contour = contour_masks.any(dim=(-2, -1)).cpu().numpy()
1267:             final_has_contour = has_contour
1268:             near_target = near_translation_targets(
1269:                 poses_before, self._targets_pos, self._env_root_pos
1270:             )
1271:             oob_before = ~objects_in_workspace(poses_before, self._env_root_pos)
1272:             self._done_once |= near_target
1273:             self._done_once |= oob_before
1274:             was_active = ~self._done_once & has_contour  # done OR invisible OR oob → frozen
1275:             active_counts.append(int(was_active.sum()))
1276:             ep_active_steps += int(was_active.sum())
1277: 
1278:             #* — select actions ——————————————————————————————
1279:             heuristic_actions = heuristic_poke_actions(
1280:                 contour_masks,
1281:                 poses_before,
1282:                 self._targets_pos,
1283:                 self._env_root_pos,
1284:                 self.K,
1285:             )
1286:             pixel_ij, d_xy, delta_d = select_action(
1287:                 self.actor_critic, x, contour_masks,
1288:                 self.epsilon, self.noise_std,
1289:                 heuristic_actions=heuristic_actions,
1290:             )
1291:             update_action_overlay(pixel_ij, d_xy, delta_d, self.K, active=was_active)
1292: 
1293:             #* — execute (only active envs move; inactive held frozen) —
1294:             poses_after, _ = await env_step_async(
1295:                 self.fingers, pixel_ij, d_xy, delta_d, self.K,
1296:                 active=was_active,
1297:             )
1298:             rewards, dones, self._done_once = compute_rewards(
1299:                 poses_before, poses_after,
1300:                 self._targets_pos, self._env_root_pos, self._done_once,
1301:                 target_oris=self._targets_ori,
1302:                 yaw_reward_enabled=YAW_REWARD_ENABLED,
1303:                 yaw_success_enabled=YAW_SUCCESS_ENABLED)
1304:             self._done_once |= near_translation_targets(
1305:                 poses_after, self._targets_pos, self._env_root_pos
1306:             )
1307:             self._done_once |= ~objects_in_workspace(poses_after, self._env_root_pos)
1308:             rewards[~was_active] = 0.0
1309: 

(Showing lines 1260-1309 of 1525. Use offset=1310 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 40.3s)

_Thinking:_

Now I have a complete picture. Let me finalize the plan. Let me also check how `env_step_async` currently exposes or doesn't expose kp/kd values:

The current `env_step_async` doesn't use kp/kd directly — it just sets position and velocity targets. For feedforward, I need kp (drive_stiffness) to convert F_ff to a position offset. I also need the fingertip mass.

Key observations:
1. `finger_props` is accessed in the Trainer.setup() and stored as a local variable `finger_props`. It's not passed to `env_step_async`.
2. The mass is in `finger_props["fingertip_masses"]` shape (num_fingers, 1)
3. kp is in `finger_props["stiffnesses"]` shape (num_fingers, 3)

I have two options for getting mass and kp into env_step_async:
- Option A: Pass them as new parameters
- Option B: Access globals inside env_step_async (like `get_randomized_finger_properties()` already does)

Option A is cleaner. I'll add `finger_properties` as an optional parameter to `env_step_async`.

But wait, there's a subtlety with kp — the stiffness for each DOF could be different (though in practice they're randomized per-DOF, not per-axis-type). For feedforward, since we're moving the fingertip in XY, we need the combined effective stiffness. But the drives are independent per DOF, so:
- For X motion: use stiffnesses[:, 0] (X DOF)
- For Y motion: use stiffnesses[:, 1] (Y DOF)
- The feedforward force in each direction uses the corresponding DOF's stiffness

Actually, that's the correct approach — per DOF feedforward. The position offset for each axis is F_ff_i / kp_i.

But hmm, this gets complicated. Let me simplify: use a single scalar kp (the config default 1500.0) for all DOFs. The randomization changes stiffness between 1200-1800, so the approximation is close enough.

Or, use the per-DOF stiffness for the precise feedforward:
- q_des[:, 0] += F_ff_x * mass / kp[:, 0]
- q_des[:, 1] += F_ff_y * mass / kp[:, 1]

where kp[:, 0] is the X joint stiffness, kp[:, 1] is the Y joint stiffness.

Let me go with the simpler scalar kp first (using config default). The user can refine later.

OK, now let me also think about the standoff/fingertip radius constraint. Currently:
```python
min_standoff = FINGERTIP_RADIUS + STANDOFF_CLEARANCE
```

With L=0.3, the start position is 0.15m behind the contour. This is larger than min_standoff (0.0125 + 0.01 = 0.0225), so it's fine. But I should still enforce minimum bounds.

Now let me think about the full trajectory math one more time:

For env b with velocity magnitude v_mid[b]:
- T = L * 1.875 / v_mid[b] = 0.5625 / v_mid[b]
- num_steps = int(T * 60) but bounded

Wait, but different envs might have different v_mid, so different T and different num_steps. I need to handle this:
- The trajectory loop should run for max(num_steps) across all envs
- For envs whose trajectory has already finished, hold at their final position

Or simpler: use a single global num_steps based on the mean v_mid or min v_mid. But that loses per-env precision.

Actually, the cleanest approach: compute the trajectory per-env but run a fixed number of steps equal to the max needed. Envs that finish earlier stay at their endpoint. This is essentially what the current code does with `_hold_inactive`.

Let me think about this differently. The trajectory per step gives us position_scale and velocity values. For each env, at step k:
- If k < num_steps[b]: use trajectory values
- If k >= num_steps[b]: hold at endpoint (strike_xy)

Let me compute num_steps per env and then take the max as the loop length.

Actually, to keep it simpler and avoid jitter, I could use a common num_steps based on the median v_mid. But the user might want per-env precision.

Let me go with per-env trajectory and max-steps loop. This is the most correct approach.

But wait, active envs with delta_d=0 have T=infinity... Actually, the `active` mask already handles this — envs with no contour (and thus delta_d=0 from `select_action`) are marked inactive. So all active envs have nonzero v_mid.

Actually, looking at `select_action` more carefully: when contour is empty, `velocity_arr[b] = 0.0`. But then in `env_step_async`, `delta_d` could be 0 for some active envs if contour exists but velocity is clipped to 0? No, `velocity_arr` is clipped to [0, velocity_max], and a velocity of 0 means the finger doesn't move. But if an env is marked active (has contour), it would have v_mid from the policy which is >0 (sigmoid * 1.5).

Edge case: v_mid could be very close to 0 (e.g., 0.001) → T = 562.5s → way too many steps. I should cap T at some maximum (e.g., 2s → 120 steps).

Let me settle the plan now.

Also, I realize the `delta_d` parameter coming into `env_step_async` is actually the velocity magnitude. Currently it's used to compute displacement via `delta_d * VELOCITY_INTEGRATION_T`. With the quintic approach, this same `delta_d` is used as `v_mid` for the polynomial.

But the direction of delta_d has already been applied in standoff_xy/strike_xy computation. With the quintic approach, I'm changing this — the standoff is fixed at L/2 = 0.15 from contour, and the trajectory runs from -0.15 to +0.15.

The key question is: do I still compute standoff_xy using the OLD method (delta_d-based) for Phase 1-2 positioning, or use the NEW method (L/2 = 0.15m from contour)?

The user said "keep L = 0.3 but make the contact point the mid point which is the expression in quintic_poly.py assumes."

"make the contact point the mid point" means the contour point = the midpoint of the trajectory. So:
- Start position = contour_point - dir * 0.15  (standoff behind object)
- End position = contour_point + dir * 0.15    (past the object)
- At u=0.5 (midpoint of time), the finger is at the contour point

This means the standoff for Phase 1 should be `contour_point - dir * 0.15` (at SAFE_Z), not the old delta_d-based computation. Phase 3 trajectory runs from this point to `contour_point + dir * 0.15` (at POKE_Z).

So the standoff_xy computation changes! Old:
```python
standoff_xy = world_xy - dirs * delta_d_clipped[:, None]
strike_xy = world_xy + dirs * delta_d_clipped[:, None]
```

New:
```python
standoff_xy = world_xy - dirs * (L/2)  # 0.15m behind
strike_xy = world_xy + dirs * (L/2)    # 0.15m ahead
```

But I need to ensure standoff_xy is within joint limits, and the fingertip doesn't start inside the object. The constraint: standoff_xy should put the fingertip at least `FINGERTIP_RADIUS + STANDOFF_CLEARANCE` from the contour. Since L/2 = 0.15m >> 0.0225m, this is fine.

OK, I also need to clip standoff_xy to joint limits. The current code does `np.clip(standoff_xy, xy_low, xy_high)`. I'll keep this.

Alright, let me now write up the final refined plan.

One more thing: the `env_step_async` function signature. Currently:
```python
async def env_step_async(
    fingers: Articulation,
    pixel_ij: np.ndarray,
    d_xy: np.ndarray,
    delta_d: np.ndarray,
    K: np.ndarray,
    active: np.ndarray | None = None,
)
```

I need to add:
- `finger_properties: dict | None = None` for mass and stiffness access

Or better: just pass `fingertip_masses: np.ndarray | None = None` and use a default kp scalar. Let me keep it simple.

Actually, let me re-read the user's question 2 answer: "use the mass from each env from randomization." This means I need to access `randomized_finger_properties["fingertip_masses"]`. The cleanest way is to either:
1. Pass it as a parameter to `env_step_async`
2. Access globals inside `env_step_async`

I'll go with option 1 — add an optional `fingertip_masses` parameter. The trainer already accesses finger_props, so it can pass the masses.

For kp, I'll use `FINGER_CONFIG["drive_stiffness"]` = 1500.0 as default, which is fine for feedforward (the exact kp only matters for converting force to position offset, and small errors here just mean slightly suboptimal feedforward, which the PD feedback compensates).

Wait, but the user might have randomized kp per env... In that case, using a single scalar introduces error. However:
1. The kp range is 1200-1800, which is ±20% around the default 1500
2. The feedforward is an augmentation — PD feedback handles the rest
3. For a first implementation, scalar kp is acceptable

If needed, I can also read per-DOF stiffness from finger_properties later.

OK, final plan:

### Revised Implementation Plan

#### Step 1: Expand `env/quintic_poly.py`

Add:
- `L = 0.3` (constant)
- `CONTROL_FREQUENCY = 60` (constant)
- `MAX_TRAJECTORY_DURATION = 2.0` (max T in seconds, caps num_steps ≤ 120)
- `MIN_TRAJECTORY_STEPS = 3`
- `quintic_poly_derivative(u)` → f'(u)
- `quintic_poly_second_derivative(u)` → f''(u)
- `generate_strike_trajectory(v_mid, dt=1/60)` → returns `(T, num_steps, s_scale, v_profile, a_profile)` where:
  - `T = clamp(L * 1.875 / v_mid, dt * MIN_TRAJECTORY_STEPS, MAX_TRAJECTORY_DURATION)`
  - `s_scale` is unitless (0→1), positions = q_start + dir * L * s_scale
  - `v_profile` in m/s (scalar, multiply by direction)
  - `a_profile` in m/s² (scalar, multiply by direction)

#### Step 2: Modify Phase 3 in `env_step_async`

**Change standoff computation** (lines 914-923):
```
OLD: standoff_xy = world_xy - dirs * delta_d_clipped
     strike_xy = world_xy + dirs * delta_d_clipped
NEW: standoff_xy = world_xy - dirs * (L/2)      # L = 0.3
     strike_xy = world_xy + dirs * (L/2)
```

**Replace Phase 3 (lines 981-992)** — the static strike block:
```
# Capture current position
q_start = as_numpy(fingers.get_dof_positions())

# Compute trajectories per active env
for each active env b:
    T, n_steps, s_scale, v_prof, a_prof = generate_strike_trajectory(delta_d[b])
    store in arrays
    
global_n_steps = max(n_steps across active envs)

for step k in range(global_n_steps):
    # Compute position + velocity + feedforward for all envs
    q_des = q_start.copy()  # default: hold current
    v_des = np.zeros((B, 3))
    
    for active env b where k < n_steps[b]:
        s = s_scale[b][k]     # 0→1
        v = v_prof[b][k]      # m/s
        a = a_prof[b][k]      # m/s²
        
        # Position along direction
        q_des[b, :2] = q_start[b, :2] + dirs[b] * L * s
        q_des[b, 2] = q_start[b, 2]  # Z stays at POKE_Z
        
        # Velocity target
        v_des[b, :2] = dirs[b] * v
        
        # Feedforward: F_ff = m*a, converted to position offset
        if fingertip_masses is not None:
            F_ff = fingertip_masses[b] * a
            q_des[b, :2] += dirs[b] * F_ff / kp
    
    # Set targets & step
    fingers.set_dof_position_targets(q_des)
    fingers.set_dof_velocity_targets(v_des)
    await _step_physics(1)
```

#### Step 3: Update `env_step_async` signature

Add optional parameter: `fingertip_masses: np.ndarray | None = None`  (num_fingers,) or (num_fingers, 1)

#### Step 4: Update call site in Trainer

In the training loop (~line 1294), pass `fingertip_masses` from `finger_properties["fingertip_masses"]`:

```python
finger_props = globals().get("randomized_finger_properties", ...)
finger_masses = finger_props["fingertip_masses"].ravel() if finger_props else None

poses_after, _ = await env_step_async(
    self.fingers, pixel_ij, d_xy, delta_d, self.K,
    active=was_active,
    fingertip_masses=finger_masses,
)
```

#### Step 5: Constants

- `L = 0.3` — total strike travel distance (m)
- `kp = 1500.0` — default drive stiffness for feedforward position-offset conversion
- `fingertip_mass_fallback = 0.3` — fallback if no randomization

---

Let me now present this plan concisely to the user.

Here's the refined plan:

---

## Implementation Plan

### File 1 — `env/quintic_poly.py`: Add derivatives + trajectory generator

Add:
- `L = 0.3` (m) — total strike travel; contact point at midpoint
- `quintic_poly_derivative(u)` → `30u⁴ - 60u³ + 30u²`
- `quintic_poly_second_derivative(u)` → `120u³ - 180u² + 60u`
- `generate_strike_trajectory(v_mid, dt=1/60)` → returns `(T, num_steps, s_scale, v_profile, a_profile)`
  - `T = clamp(L * 1.875 / v_mid, 3*dt, 2.0)` — caps between 3 steps and 2s
  - `s_scale` ∈ [0, 1] — multiply by `L` and direction to get position
  - `v_profile` in m/s, `a_profile` in m/s²

### File 2 — `env/quintic_poly.py`: Change standoff geometry

Replace the `delta_d`-based standoff with fixed L/2:

```
# Old:  standoff_xy = world_xy - dirs * delta_d_clipped
#        strike_xy   = world_xy + dirs * delta_d_clipped
# New:
standoff_xy = world_xy - dirs * L/2   # 0.15m behind contour
strike_xy   = world_xy + dirs * L/2   # 0.15m past contour (contact at midpoint)
```

### File 3 — `nn/train.py:env_step_async`: Replace static Phase 3 strike

**Signature change**: add `fingertip_masses: np.ndarray | None = None`

**Replace lines ~981-992** (current `set_dof_position_targets(q3)` + `_step_physics(IMPACT_STEPS)` block) with:

1. Capture `q_start` after Phase 2 (finger is at standoff_xy at POKE_Z)
2. Per active env: call `generate_strike_trajectory(delta_d[b])` to get per-env T, steps, profiles
3. Loop `k` from 0 to `max(num_steps)`:
   - `u = (k+1) / n_steps[b]`
   - `q_des[b,:2] = q_start[b,:2] + dirs[b] * L * s_scale[b,k]`
   - `v_des[b,:2] = dirs[b] * v_profile[b,k]`
   - Feedforward: `F_ff = mass[b] * a_profile[b,k]`, then `q_des[b,:2] += dirs[b] * F_ff / kp`
   - Z stays at `POKE_Z` throughout
   - Inactive envs and envs past their n_steps hold at current position
4. At each step: `set_dof_position_targets` + `set_dof_velocity_targets` + `_step_physics(1)`

### File 4 — `nn/train.py`: Update call site (~line 1294)

Pass `fingertip_masses` from `randomized_finger_properties["fingertip_masses"]`:

```python
finger_props = globals().get("randomized_finger_properties", ...)
finger_masses = finger_props["fingertip_masses"].ravel() if finger_props else None
```

With fallback mass = 0.3 kg and fallback kp = 1500.0 when randomization is off.

---

### Summary of changes

| What | Where | Change |
|------|-------|--------|
| Polynomial toolkit | `quintic_poly.py` | +3 functions, 1 constant (40→~70 lines) |
| Standoff geometry | `train.py` line ~920 | delta_d-based → fixed L/2 = 0.15m |
| Strike execution | `train.py` lines ~981-992 | static position jump → per-timestep trajectory loop |
| Call signature | `train.py` line ~875 | +`fingertip_masses` param |
| Call site | `train.py` line ~1294 | pass masses from randomization |

Phase 0/1/2/4/5 — **unchanged**. Z-axis — **unchanged** (stays at POKE_Z).

Does this plan look correct? Ready for me to implement?

---

## User

Also, check thoroughly for articulation in isaac sim; whether it allows feedforward? 

---

## Assistant (Plan · DeepSeek V4 Pro · 6.2s)

_Thinking:_

The user wants me to check thoroughly whether Isaac Sim's articulation (DriveAPI) supports feedforward directly. Let me search for relevant documentation about Isaac Sim's articulation drive API, particularly:

1. Whether `set_dof_velocity_targets` is the only way to influence the drive
2. Whether there's a `set_dof_efforts` or `set_dof_force_targets` for direct force/torque feedforward
3. What the PhysX articulation drive model actually supports

Let me use the Context7 tool to look up Isaac Sim documentation about the articulation API, and also check the existing code for any effort/torque-related methods.

Let me research Isaac Sim's articulation DriveAPI thoroughly to see what feedforward options exist.

**Tool: context7_resolve-library-id**

**Input:**
```json
{
  "query": "Isaac Sim articulation DriveAPI set dof efforts force feedforward control",
  "libraryName": "Isaac Sim"
}
```

**Output:**
```
Available Libraries:

- Title: NVIDIA Isaac Sim
- Context7-compatible library ID: /websites/isaacsim_omniverse_nvidia_5_1_0
- Description: NVIDIA Isaac Sim is a reference application for developing, simulating, and testing AI-driven robots in physically-based virtual environments built on NVIDIA Omniverse.
- Code Snippets: 12150
- Source Reputation: High
- Benchmark Score: 84.7
----------
- Title: NVIDIA Isaac Sim
- Context7-compatible library ID: /websites/isaacsim_omniverse_nvidia
- Description: NVIDIA Isaac Sim is a reference application built on NVIDIA Omniverse that enables developers to develop, simulate, and test AI-driven robots in physically-based virtual environments.
- Code Snippets: 23089
- Source Reputation: High
- Benchmark Score: 71.3
----------
- Title: Isaac Sim
- Context7-compatible library ID: /isaac-sim/isaacsim
- Description: Isaac Sim is a simulation platform built on NVIDIA Omniverse for developing, testing, training, and deploying AI-powered robots with high-fidelity physics engines and synthetic data generation.
- Code Snippets: 471
- Source Reputation: High
- Benchmark Score: 55.92
- Versions: v5.1.0
----------
- Title: Isaac Lab
- Context7-compatible library ID: /websites/isaac-sim_github_io_isaaclab_main
- Description: Isaac Lab is a unified and modular framework for robot learning, built on NVIDIA Isaac Sim, designed to simplify workflows in robotics research like reinforcement learning and motion planning with fast, efficient, and photo-realistic simulation.
- Code Snippets: 10717
- Source Reputation: High
- Benchmark Score: 83.22
----------
- Title: Isaac Sim App Template
- Context7-compatible library ID: /isaac-sim/isaacsim-app-template
- Description: Isaac Sim App Template provides pre-built and customizable NVIDIA Isaac Sim applications with templates for creating extensions and applications based on the Omniverse Kit SDK.
- Code Snippets: 238
- Source Reputation: High
- Benchmark Score: 97
```

**Tool: grep**

**Input:**
```json
{
  "pattern": "set_dof_effort|set_dof_force|set_dof_torque|effort_target|force_target|feedforward|set_dof_drive",
  "path": "/home/xiao/0_codes/poke_map"
}
```

**Output:**
```
Found 5 matches
/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py:
  Line 232:     # <start-feedforward-effort-snippet>

  Line 236:     feedforward_efforts = [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  Line 237:     actuated.set_dof_feedforward_effort_targets(

  Line 238:         feedforward_efforts,

  Line 241:     # <end-feedforward-effort-snippet>

```

---

## Assistant (Plan · DeepSeek V4 Pro · 4.6s)

_Thinking:_

Very interesting! There's `set_dof_feedforward_effort_targets` in the tutorials! This is exactly what I need. Let me look at that tutorial file in detail, and also query the Isaac Sim documentation for more details.

Excellent find! There's `set_dof_feedforward_effort_targets` in the tutorials. Let me read that tutorial and also query the docs for details.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py",
  "limit": 60,
  "offset": 220
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py</path>
<type>file</type>
<content>
220:     # Values match the Franka robot descriptor's default_q (home pose).
221:     target_positions = [0.8, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]
222:     arm_dof_indices = articulation.get_dof_indices(ARM_JOINTS)
223:     articulation.set_dof_position_targets(
224:         positions=target_positions,
225:         dof_indices=arm_dof_indices,
226:     )
227: 
228:     for _ in range(num_steps):  # ~4 seconds at 60 Hz
229:         simulation_app.update()
230:     # <end-drive-to-target-snippet>
231: 
232:     # <start-feedforward-effort-snippet>
233:     # Feedforward effort is added on top of the controller output every tick.
234:     # With kp = kd = 0 it becomes the entire output — a pure open-loop torque drive.
235:     #! this only take effect for joints with explicit actuators
236:     feedforward_efforts = [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
237:     actuated.set_dof_feedforward_effort_targets(
238:         feedforward_efforts,
239:         dof_indices=arm_dof_indices,
240:     )
241:     # <end-feedforward-effort-snippet>
242: 
243:     for _ in range(num_steps):
244:         simulation_app.update()
245: 
246: 
247: # ============================================================================
248: # Entry point
249: # ============================================================================
250: def main() -> None:
251:     SimulationManager.set_physics_dt(1.0 / 60.0)
252:     simulation_app.run_coroutine(setup_stage_with_franka())
253:     if args.non_ideal:
254:         print("Using non-ideal actuators (PD + per-joint effort clamp + 2-step delay).")
255:         construct = construct_articulation_actuators_non_ideal
256:     else:
257:         construct = construct_articulation_actuators
258:     try:
259:         # <start-context-manager-snippet>
260:         # Recommended teardown pattern: construct ``ArticulationActuators`` inside
261:         # the ``with`` statement so the wrapper's lifetime is bounded by the block.
262:         # ``__exit__`` calls ``actuated.close()``, which deregisters every
263:         # ``SimulationManager`` lifecycle callback owned by the instance and is
264:         # guaranteed to run even if the body raises.
265:         with construct() as actuated:
266:             drive_to_target(actuated)
267:         # <end-context-manager-snippet>
268:     finally:
269:         omni.timeline.get_timeline_interface().stop()
270:     print("Newton actuators Python example complete.")
271: 
272: 
273: if __name__ == "__main__":
274:     main()
275:     simulation_app.close()

(End of file - total 275 lines)
</content>
```

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "Articulation set_dof_feedforward_effort_targets drive API feedforward effort control prismatic joint"
}
```

**Output:**
```
### Set Velocity Targets for Joints

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/robots_simulation.html

Sets drive stiffness to 0 for revolute and prismatic joints, then targets joint velocities for an articulation. Ensure the articulation is acquired after stiffness is set.

```python
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()
for prim in stage.TraverseAll():
    prim_type = prim.GetTypeName()
    if prim_type in ["PhysicsRevoluteJoint" , "PhysicsPrismaticJoint"]:
        if prim_type == "PhysicsRevoluteJoint":
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        else:
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if drive:
            drive.GetStiffnessAttr().Set(0)
from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
dc = _dynamic_control.acquire_dynamic_control_interface()
#Note: getting the articulation has to happen after changing the drive stiffness
articulation = dc.get_articulation("/Franka")
dc.wake_up_articulation(articulation)
joint_vels = [-np.random.rand(9)*10]
dc.set_articulation_dof_velocity_targets(articulation, joint_vels)
```

--------------------------------

### Set Joint Efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.manipulators/docs/index.html

Applies specified efforts to the articulation's joints. This method is suitable for effort control, provided no joint drives are active or their stiffness/damping is zero.

```python
>>> # set all the robot joint efforts to 0.0
>>> prim.set_joint_efforts(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
>>>
>>> # set only the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
>>> prim.set_joint_efforts(np.array([10, 10]), joint_indices=np.array([7, 8]))
```

--------------------------------

### set_joint_efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html

Sets the articulation joint efforts. This method can be used for effort control. For this purpose, there must be no joint drive or the stiffness and damping must be set to zero.

```APIDOC
## set_joint_efforts(_efforts : ndarray_, _joint_indices : List | ndarray | None = None_)

### Description
Sets the articulation joint efforts. This method can be used for effort control. For this purpose, there must be no joint drive or the stiffness and damping must be set to zero.

### Method
POST (implied)

### Endpoint
N/A (SDK method)

### Parameters
#### Request Body
- **efforts** (np.ndarray) - Required - Articulation joint efforts.
- **joint_indices** (Optional[Union[list, np.ndarray]]) - Optional - Indices to specify which joints to manipulate. Defaults to None (all joints).

### Example
```python
# set all the robot joint efforts to 0.0
```
```

--------------------------------

### Set Torque Control for Articulation

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/robots_simulation.html

Applies joint efforts (torques) to an articulation. This is useful for direct torque control of robot joints.

```python
from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
dc = _dynamic_control.acquire_dynamic_control_interface()
articulation = dc.get_articulation("/Franka")
dc.wake_up_articulation(articulation)
joint_efforts = [-np.random.rand(9) * 1000]
dc.set_articulation_dof_efforts(articulation, joint_efforts)
```

--------------------------------

### set_joint_efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html

Sets the effort values for specified joints of the robot. This method allows direct control over the forces applied to the joints.

```APIDOC
## set_joint_efforts

### Description
Sets the articulation joint efforts.
Warning
This method will immediately set the affected joints to the indicated value. Use the `apply_action` method to control robot joints.

### Parameters
* **efforts** (_np.ndarray_) – articulation joint efforts
* **joint_indices** (_Optional_ _[__Union_ _[__list_ _,__np.ndarray_ _]__]__,__optional_) – indices to specify which joints to manipulate. Defaults to None (all joints)

### Hint
This method belongs to the methods used to set the articulation kinematic state:
`set_linear_velocity`, `set_angular_velocity`, `set_joint_positions`, `set_joint_velocities`, `set_joint_efforts`

### Example
```python
>>> # set all the robot joints efforts to 0.0
>>> prim.set_joint_efforts(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
>>>
>>> # set only the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
>>> prim.set_joint_efforts(np.array([10, 10]), joint_indices=np.array([7, 8]))
```
```
```

---

## Assistant (Plan · DeepSeek V4 Pro · 14.9s)

_Thinking:_

This is very important! I found that Isaac Sim has `set_dof_feedforward_effort_targets` which is exactly what we need for feedforward control. Let me understand the difference between the two APIs:

1. **`Articulation`** (from `isaacsim.core.experimental.prims`) — This is the "new" core API used in the current code. Let me check what methods it has.

2. **`ArticulationActuators`** (from the actuator_setup tutorial) — Uses `set_dof_feedforward_effort_targets`. The tutorial says: "Feedforward effort is added on top of the controller output every tick. With kp = kd = 0 it becomes the entire output — a pure open-loop torque drive. This only takes effect for joints with explicit actuators."

But wait, the current codebase uses `Articulation` from `isaacsim.core.experimental.prims`, not `ArticulationActuators`. Let me check if `Articulation` also has `set_dof_feedforward_effort_targets` or if we need to use a different API.

Also, looking at the tutorial more carefully:

```python
# In actuator_setup.py
with construct() as actuated:   # actuated is an ArticulationActuators instance
    actuated.set_dof_feedforward_effort_targets(
        feedforward_efforts,
        dof_indices=arm_dof_indices,
    )
```

The `ArticulationActuators` is a wrapper around an articulation that provides additional actuator-specific methods. But the current code uses the bare `Articulation`.

Let me check if `Articulation` from `isaacsim.core.experimental.prims` also supports `set_dof_feedforward_effort_targets`. I need to search the Isaac Sim docs or check the available methods.

Actually, I should also check if the finger robot has "explicit actuators" — the tutorial says feedforward only works for joints with explicit actuators. The finger robot uses DriveAPI (which is the PhysX drive mechanism), not explicit actuators. So `set_dof_feedforward_effort_targets` might NOT work with the current finger robot setup.

Let me check:

1. What methods does `Articulation` from `isaacsim.core.experimental.prims` have?
2. Does the finger robot's DriveAPI support feedforward?
3. Is there a difference between DriveAPI-based control and actuator-based control?

Looking at the Isaac Sim docs:

From the tutorial: "Feedforward effort is added on top of the controller output every tick." and "this only takes effect for joints with explicit actuators."

The key phrase is "explicit actuators" — this means the joint must have an `UsdPhysics.DriveAPI` applied (which the finger robot does) but the feedforward might require the "new" actuator API (`ArticulationActuators`) rather than the simple `Articulation` wrapper.

Actually, let me re-read the important part: from the PhysX drive equation:
```
τ = kp*(q_target - q) + kd*(v_target - dq/dt) + ff_target
```

Where `ff_target` is the feedforward effort. This is the standard PD+FF control law in PhysX!

The question is: does the Python API expose this? 

Looking at the current code:
- `fingers.set_dof_position_targets(...)` — sets q_target
- `fingers.set_dof_velocity_targets(...)` — sets v_target

But is there a `fingers.set_dof_feedforward_effort_targets(...)` or similar?

From the tutorial, `ArticulationActuators` has it. But does `Articulation` from `isaacsim.core.experimental.prims` also have it?

Let me search for this in the Isaac Sim documentation more specifically.

Actually, the key distinction is:
- `Articulation` from `isaacsim.core.experimental.prims` — wraps the raw PhysX articulation
- `ArticulationActuators` — a higher-level wrapper that adds actuator-specific features including feedforward

Let me check what the current codebase actually imports and what methods are available on `Articulation`.

For now, the important design decision is:

**Option A: Use `Articulation.set_dof_feedforward_effort_targets` if available**
- Direct feedforward force injection
- Clean separation: PD for feedback, FF for inertial compensation

**Option B: Use position-offset feedforward through DriveAPI**
- `q_target += F_ff / kp` 
- Implicit feedforward through the position channel
- Works with current code

**Option C: Use `ArticulationActuators` wrapper**
- Adds `set_dof_feedforward_effort_targets` 
- Requires wrapping the existing articulation
- May need additional setup

Let me search for what methods the `Articulation` class actually has, and whether feedforward effort is available on it.

Also, I realize there's a subtle but important thing: the finger robot uses **prismatic** joints with **linear** drives (PD position control). The feedforward effort for a prismatic joint is a force (N), while for a revolute joint it's a torque (N·m). The API should handle this correctly.

Let me look at more documentation about the `Articulation` class from `isaacsim.core.experimental.prims`.

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "Articulation class set_dof_position_targets set_dof_velocity_targets set_dof_feedforward_effort_targets methods API prismatic joint drive"
}
```

**Output:**
```
### Set Velocity Targets for Joints

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/robots_simulation.html

Sets drive stiffness to 0 for revolute and prismatic joints, then targets joint velocities for an articulation. Ensure the articulation is acquired after stiffness is set.

```python
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()
for prim in stage.TraverseAll():
    prim_type = prim.GetTypeName()
    if prim_type in ["PhysicsRevoluteJoint" , "PhysicsPrismaticJoint"]:
        if prim_type == "PhysicsRevoluteJoint":
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        else:
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if drive:
            drive.GetStiffnessAttr().Set(0)
from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
dc = _dynamic_control.acquire_dynamic_control_interface()
#Note: getting the articulation has to happen after changing the drive stiffness
articulation = dc.get_articulation("/Franka")
dc.wake_up_articulation(articulation)
joint_vels = [-np.random.rand(9)*10]
dc.set_articulation_dof_velocity_targets(articulation, joint_vels)
```

--------------------------------

### set_articulation_dof_position_targets

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/deprecated/omni.isaac.dynamic_control/docs/index.html

Sets the target positions for an actor's degrees of freedom. This is used to command joints to move to specific positions.

```APIDOC
## set_articulation_dof_position_targets

### Description
Sets an actor’s degree-of-freedom position targets.

### Method
POST (assumed, as it modifies state)

### Endpoint
`/dynamic_control/articulation/{arg0}/dof_position_targets` (assumed structure)

### Parameters
#### Path Parameters
- **arg0** (int) - Required - The identifier of the actor.

#### Request Body
- **targets** (numpy.ndarray[numpy.float32]) - Required - An array of target positions for the degrees of freedom.
```

--------------------------------

### set_articulation_dof_velocity_targets

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/deprecated/omni.isaac.dynamic_control/docs/index.html

Sets the target velocities for an actor's degrees of freedom. This is used to command joints to move at specific speeds.

```APIDOC
## set_articulation_dof_velocity_targets

### Description
Sets an actor’s degree-of-freedom velocity targets.

### Method
POST (assumed, as it modifies state)

### Endpoint
`/dynamic_control/articulation/{arg0}/dof_velocity_targets` (assumed structure)

### Parameters
#### Path Parameters
- **arg0** (int) - Required - The identifier of the actor.

#### Request Body
- **targets** (numpy.ndarray[numpy.float32]) - Required - An array of target velocities for the degrees of freedom.
```

--------------------------------

### Set Single DOF Velocity Target

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/robots_simulation.html

Sets drive stiffness to 0 for a specific joint and then targets a velocity for that single degree of freedom. Ensure the articulation is acquired after stiffness is set.

```python
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()
panda_joint2_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/Franka/panda_link1/panda_joint2"), "angular")
panda_joint2_drive.GetStiffnessAttr().Set(0)
from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
dc = _dynamic_control.acquire_dynamic_control_interface()
#Note: getting the articulation has to happen after changing the drive stiffness
articulation = dc.get_articulation("/Franka")
dc.wake_up_articulation(articulation)
dof_ptr = dc.find_articulation_dof(articulation, "panda_joint2")
dc.set_dof_velocity_target(dof_ptr, 0.2)
```

### Prims API > Wrappers > set_dof_velocity_targets

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

The `set_dof_velocity_targets()` method sets the desired target velocity for DOFs, not instantaneous velocities. It may take several simulation steps to reach the target. High stiffness moves the DOF faster, while high damping softens and slows movement. For velocity control, set zero stiffness and non-zero damping.
```

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "ArticulationActuators set_dof_feedforward_effort_targets explicit actuator drive feedforward PD controller"
}
```

**Output:**
```
### Switch DOF Control Mode

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Switches the control mode (position, velocity, or effort) for specified degrees of freedom (DOFs) of prims. This affects the PD controller's gains. Use 'dof_indices' to target specific DOFs.

```python
>>> # switch to 'velocity' control mode for all prims' arm DOFs (except for the fingers)
>>> prims.switch_dof_control_mode("velocity", dof_indices=np.arange(7))
>>>
>>> # switch to 'effort' control mode for all prims' fingers (last 2 DOFs)
>>> prims.switch_dof_control_mode("effort", dof_indices=[7, 8])
```

--------------------------------

### Set DOF Gains (Stiffness and Damping) for Prims

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Set the stiffness and damping gains for the implicit Proportional-Derivative (PD) controller of prim DOFs. This allows fine-tuning the response of joints and actuators.

```python
>>> # set the DOF gains for all prims
>>> stiffnesses = np.array([100000, 100000, 100000, 100000, 80000, 80000, 80000, 50000, 50000])
>>> dampings = np.array([8000, 8000, 8000, 8000, 5000, 5000, 5000, 2000, 2000])
>>> prims.set_dof_gains(stiffnesses, dampings)
```

--------------------------------

### set_dof_gains

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Sets the implicit Proportional-Derivative (PD) controller's gains, including stiffnesses and dampings, for the degrees of freedom (DOFs) of the prims.

```APIDOC
## set_dof_gains

### Description
Sets the implicit Proportional-Derivative (PD) controller’s gains (stiffnesses and dampings) of the degrees of freedom (DOFs) of the prims.

### Method
`set_dof_gains`

### Parameters
* **stiffnesses** (float | list | np.ndarray | wp.array | None) - Stiffnesses (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted.
* **dampings** (float | list | np.ndarray | wp.array | None) - Dampings (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted.
* **indices** (int | list | np.ndarray | wp.array | None) - Indices of prims to process (shape `(N,)`). If not defined, all wrapped prims are processed.
* **dof_indices** (int | list | np.ndarray | wp.array | None) - Indices of DOFs to process (shape `(D,)`). If not defined, all DOFs are processed.
* **update_default_gains** (bool) - Whether to update the default gains with the given values.

### Raises
* **AssertionError** - If neither stiffnesses nor dampings are specified.
* **AssertionError** - Wrapped prims are not valid.

### Example
```python
>>> # set the DOF gains for all prims
>>> stiffnesses = np.array([100000, 100000, 100000, 100000, 80000, 80000, 80000, 50000, 50000])
>>> dampings = np.array([8000, 8000, 8000, 8000, 5000, 5000, 5000, 2000, 2000])
>>> prims.set_dof_gains(stiffnesses, dampings)
```
```

--------------------------------

### Setting Robot Drive Properties with UsdPhysics.DriveAPI

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html

Demonstrates how to get and set drive properties like velocity, damping, and stiffness for robot joints using the UsdPhysics.DriveAPI in Python. This is useful for controlling robot locomotion and actuators.

```python
1# get handle to the Drive API for both wheels
2left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/left_wheel"), "angular")
3right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/right_wheel"), "angular")
4
5# Set the velocity drive target in degrees/second
6left_wheel_drive.GetTargetVelocityAttr().Set(150)
7right_wheel_drive.GetTargetVelocityAttr().Set(150)
8
9# Set the drive damping, which controls the strength of the velocity drive
10left_wheel_drive.GetDampingAttr().Set(15000)
11right_wheel_drive.GetDampingAttr().Set(15000)
12
13# Set the drive stiffness, which controls the strength of the position drive
14# In this case because we want to do velocity control this should be set to zero
15left_wheel_drive.GetStiffnessAttr().Set(0)
16right_wheel_drive.GetStiffnessAttr().Set(0)
```

### Gain Tuner Extension > Overview > Understanding Joint Drives

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html

The formula for Joint Drives is τ = stiffness * (q - q_target) + damping * (q̇ - q̇_target), where q and q̇ are joint position and velocity. When q̇_target is zero, it functions as a conventional PD controller. The joint max force clamps τ, and the drive type determines if the effort is applied as torque/force or converted to acceleration.
```

---

## Assistant (Plan · DeepSeek V4 Pro · 11.9s)

_Thinking:_

Very important findings! Let me now check for `set_dof_feedforward_effort_targets` on the `Articulation` class specifically. I found that there's a `switch_dof_control_mode` method that allows switching to "effort" control mode. But what I really need is feedforward effort.

Let me search for `feedforward_effort` in the Articulation API more specifically.

Also, the tutorial `actuator_setup.py` uses `ArticulationActuators` (not `Articulation`) for `set_dof_feedforward_effort_targets`. The `ArticulationActuators` is a different wrapper. Let me look more carefully at what's available.

Actually, let me check: does `Articulation` (from `isaacsim.core.experimental.prims`) have `set_dof_feedforward_effort_targets`?

From the docs I found:
- `Articulation.set_dof_gains(stiffnesses, dampings)` ✓ (used in current code)
- `Articulation.set_dof_position_targets(positions, dof_indices)` ✓ (used in current code)
- `Articulation.set_dof_velocity_targets(velocities, dof_indices)` ✓ (used in current code)
- `Articulation.switch_dof_control_mode(mode, dof_indices)` — can switch to "position", "velocity", or "effort"

But I didn't find `set_dof_feedforward_effort_targets` on `Articulation` directly. The tutorial uses `ArticulationActuators` for this.

Let me check: maybe `Articulation` also has this method but it's not documented in the snippets I found. Or maybe it's only on `ArticulationActuators`.

Key difference:
- `Articulation` — low-level PhysX articulation wrapper, uses implicit PD controller
- `ArticulationActuators` — higher-level wrapper that adds actuator-specific features including:
  - `set_dof_feedforward_effort_targets` — adds feedforward on top of PD
  - Requires joints to have "explicit actuators" configured

The finger robot in the current code uses `Articulation` with DriveAPI (implicit PD). The drive is set up via USD attributes (stiffness, damping, maxForce on the DriveAPI). This uses the built-in PhysX drive.

For feedforward, we have two approaches:

**Approach 1: Use `ArticulationActuators` wrapper**
- Wrap the existing articulation with `ArticulationActuators`
- Use `set_dof_feedforward_effort_targets` to directly inject feedforward force
- The DriveAPI drive law becomes: τ = kp*(q_target - q) + kd*(v_target - dq/dt) + ff_target

**Approach 2: Stay with `Articulation` and use position-offset**
- Same as my original plan: inject feedforward via position target offset
- `q_target = q_des + F_ff/kp`
- Works with current API

Let me check if `Articulation` itself has `set_dof_feedforward_effort_targets`. Let me search the code or docs more specifically.

Actually, I noticed the key difference: the `ArticulationActuators` from the tutorial is specifically for joints that have "explicit actuators." The finger robot's joints use DriveAPI (built-in PD), which may or may not support feedforward.

In PhysX, the joint drive has a `targetForce` or `forceLimit` attribute. But the standard DriveAPI has stiffness, damping, maxForce, and target_position/velocity. There's no explicit feedforward in the DriveAPI schema.

However, PhysX internally supports feedforward if the joint has explicit actuators configured (via `PhysxSchema.PhysxDriveActuatorAPI` or similar). The `ArticulationActuators` wrapper configures this.

Let me search for what "explicit actuators" means in Isaac Sim and whether the finger robot's joints can be upgraded.

Actually, let me look at the tutorial `actuator_setup.py` more carefully to understand how `ArticulationActuators` is constructed:

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "Articulation set_dof_feedforward_effort_targets method feedforward effort target force prismatic joint API"
}
```

**Output:**
```
### set_joint_efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html

Sets the articulation joint efforts. This method can be used for effort control. For this purpose, there must be no joint drive or the stiffness and damping must be set to zero.

```APIDOC
## set_joint_efforts(_efforts : ndarray_, _joint_indices : List | ndarray | None = None_)

### Description
Sets the articulation joint efforts. This method can be used for effort control. For this purpose, there must be no joint drive or the stiffness and damping must be set to zero.

### Method
POST (implied)

### Endpoint
N/A (SDK method)

### Parameters
#### Request Body
- **efforts** (np.ndarray) - Required - Articulation joint efforts.
- **joint_indices** (Optional[Union[list, np.ndarray]]) - Optional - Indices to specify which joints to manipulate. Defaults to None (all joints).

### Example
```python
# set all the robot joint efforts to 0.0
```
```

--------------------------------

### Set Joint Efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.manipulators/docs/index.html

Applies specified efforts to the articulation's joints. This method is suitable for effort control, provided no joint drives are active or their stiffness/damping is zero.

```python
>>> # set all the robot joint efforts to 0.0
>>> prim.set_joint_efforts(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
>>>
>>> # set only the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
>>> prim.set_joint_efforts(np.array([10, 10]), joint_indices=np.array([7, 8]))
```

--------------------------------

### set_joint_efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html

Sets the effort values for specified joints of the robot. This method allows direct control over the forces applied to the joints.

```APIDOC
## set_joint_efforts

### Description
Sets the articulation joint efforts.
Warning
This method will immediately set the affected joints to the indicated value. Use the `apply_action` method to control robot joints.

### Parameters
* **efforts** (_np.ndarray_) – articulation joint efforts
* **joint_indices** (_Optional_ _[__Union_ _[__list_ _,__np.ndarray_ _]__]__,__optional_) – indices to specify which joints to manipulate. Defaults to None (all joints)

### Hint
This method belongs to the methods used to set the articulation kinematic state:
`set_linear_velocity`, `set_angular_velocity`, `set_joint_positions`, `set_joint_velocities`, `set_joint_efforts`

### Example
```python
>>> # set all the robot joints efforts to 0.0
>>> prim.set_joint_efforts(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
>>>
>>> # set only the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
>>> prim.set_joint_efforts(np.array([10, 10]), joint_indices=np.array([7, 8]))
```
```

--------------------------------

### setArticulationDofEfforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structomni_1_1isaac_1_1dynamic__control_1_1_dynamic_control.html

Sets the efforts for all degrees of freedom in an articulation. This function applies specific effort (force/torque) values to the articulation's joints.

```APIDOC
## setArticulationDofEfforts

### Description
Sets the efforts of all degrees of freedom in an articulation.

### Parameters
#### Path Parameters
- **artHandle** (DcHandle) - in - Handle to the articulation
- **efforts** (const float *) - in - Array of efforts to set

### Return
True if successful, false otherwise
```

--------------------------------

### set_articulation_dof_efforts

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/deprecated/omni.isaac.dynamic_control/docs/index.html

Sets the efforts for an actor's degrees of freedom. This allows for direct control over the forces applied to specific joints or actuators.

```APIDOC
## set_articulation_dof_efforts

### Description
Sets efforts on an actor’s degrees-of-freedom.

### Method
POST (assumed, as it modifies state)

### Endpoint
`/dynamic_control/articulation/{arg0}/dof_efforts` (assumed structure)

### Parameters
#### Path Parameters
- **arg0** (int) - Required - The identifier of the actor.

#### Request Body
- **efforts** (numpy.ndarray[numpy.float32]) - Required - An array of efforts to apply to the degrees of freedom.
```
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py",
  "limit": 100,
  "offset": 1
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py</path>
<type>file</type>
<content>
1: 
2: #! USD joint drive: 
3: #!      the drive is part of the physics model. The solver uses the joint target, 
4: #!      stiffness, damping, and related parameters to compute constraint forces 
5: #!      that help the joint move toward the target.
6: #! Newton ideal actuator: 
7: #!      your control code computes a torque/effort command from the target error, 
8: #!      then the solver just applies that torque as an external input and integrates 
9: #!      the dynamics.
10: #! So the distinction is:
11: #!      USD drive: “the solver itself is helping enforce the target through the joint model”
12: #!      Newton ideal actuator: “your controller outputs torque, and the solver responds to that torque”
13: """
14: ================================================================================
15: This file contains code snippets that are displayed in the Newton Actuators
16: "Adding Actuators from Python" tutorial.  Keep the
17: ``<start-...-snippet>`` / ``<end-...-snippet>`` markers in sync with
18: ``docs/isaacsim/newton_actuators_tutorials/newton_actuators_python.rst``.
19: ================================================================================
20: 
21: Runs end-to-end as a standalone script:
22: 
23:     ./python.sh standalone_examples/api/isaacsim.core.experimental.actuators/newton_actuators_python_example.py
24: 
25: Pass ``--non-ideal`` to swap the default ideal PD actuators for the
26: non-ideal variant (PD + per-joint effort clamp + 2-step input delay):
27: 
28:     ./python.sh standalone_examples/api/isaacsim.core.experimental.actuators/newton_actuators_python_example.py --non-ideal
29: """
30: 
31: from __future__ import annotations
32: 
33: # ============================================================================
34: # 1. Parse arguments and launch Simulation App
35: # ============================================================================
36: import argparse
37: 
38: _parser = argparse.ArgumentParser(description="Newton actuators Python tutorial example.")
39: _parser.add_argument(
40:     "--non-ideal",
41:     action="store_true",
42:     help=(
43:         "Use non-ideal actuators (PD + per-joint effort clamp + 2-step input delay) "
44:         "instead of the default ideal PD actuators."
45:     ),
46: )
47: args, _ = _parser.parse_known_args()
48: 
49: from isaacsim import SimulationApp
50: 
51: simulation_app = SimulationApp({"headless": False})
52: 
53: import isaacsim.core.experimental.utils.stage as stage_utils
54: import numpy as np
55: import omni.kit.app
56: import omni.timeline
57: import warp as wp
58: from isaacsim.core.experimental.utils.stage import add_reference_to_stage
59: from isaacsim.core.simulation_manager import SimulationManager
60: from isaacsim.storage.native import get_assets_root_path_async
61: 
62: FRANKA_USD_REL_PATH = "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
63: FRANKA_PRIM_PATH = "/World/Franka"
64: ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
65: 
66: 
67: async def setup_stage_with_franka() -> None:
68:     """Create a fresh stage with a PhysicsScene and reference in a Franka Panda.
69: 
70:     Async so the kit app stays responsive while the Franka USD is fetched
71:     (especially from Nucleus).  Drive it via ``simulation_app.run_coroutine``.
72:     """
73:     await stage_utils.create_new_stage_async(template="default stage")
74:     stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
75:     assets_root_path = await get_assets_root_path_async()
76:     usd_path = f"{assets_root_path}/{FRANKA_USD_REL_PATH}"
77:     add_reference_to_stage(usd_path, FRANKA_PRIM_PATH)
78:     await omni.kit.app.get_app().next_update_async()
79: 
80: 
81: # ============================================================================
82: # 2. Building a stock PD actuator config
83: # ============================================================================
84: def build_pd_actuator_config(n_robots: int, kp: float, kd: float) -> ActuatorConfig:
85:     """Build an ActuatorConfig with a Newton ``ControllerPD``."""
86:     # <start-build-pd-config-snippet>
87:     import warp as wp
88:     from isaacsim.core.experimental.actuators import ActuatorConfig
89:     from newton.actuators import ControllerPD
90: 
91:     # Per-robot Warp arrays.  Even a single-robot scene must size them with
92:     # n_robots = len(Articulation(paths)).
93:     pd_config = ActuatorConfig(
94:         controller=ControllerPD(
95:             kp=wp.array([kp] * n_robots, dtype=wp.float32),
96:             kd=wp.array([kd] * n_robots, dtype=wp.float32),
97:         )
98:     )
99:     # <end-build-pd-config-snippet>
100:     return pd_config

(Showing lines 1-100 of 275. Use offset=101 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 14.2s)

_Thinking:_

Now I understand the distinction very clearly:

1. **USD Joint Drive** (what the finger robot currently uses):
   - Built into the PhysX solver
   - τ = kp*(q_target - q) + kd*(v_target - dq/dt)
   - Clamped by maxForce
   - Controlled via `Articulation.set_dof_position_targets()` and `set_dof_velocity_targets()`

2. **Newton Ideal Actuator** (`ArticulationActuators`):
   - External controller that computes torque/effort command
   - Solver applies it as external input
   - Supports `set_dof_feedforward_effort_targets()` — adds feedforward on top of PD output
   - More features: PD config, effort clamping, input delay

The finger robot uses approach 1 (USD Joint Drive). To use `set_dof_feedforward_effort_targets`, we'd need to switch to approach 2 (`ArticulationActuators`), which is a significant refactor.

But wait — approach 1 (USD Drive) SHOULD also support feedforward through the DriveAPI! In PhysX, the joint drive has:
- `targetPosition`
- `targetVelocity`  
- `stiffness`
- `damping`
- `maxForce`

The drive equation is: `τ = stiffness * (targetPosition - q) + damping * (targetVelocity - dq/dt)`

There's no explicit feedforward in the USD DriveAPI schema.

However, from the `switch_dof_control_mode` documentation I found earlier, `Articulation` supports switching to "effort" control mode. This might allow direct effort setting. But it would require kp=kd=0, meaning no PD feedback.

So the conclusion is:

**For the USD DriveAPI approach (current code):**
- No explicit feedforward effort target
- Feedforward must be implemented via position/velocity offset
- `q_target = q_des + F_ff/kp` is the correct way to inject feedforward force

**For the `ArticulationActuators` approach:**
- Has `set_dof_feedforward_effort_targets` for direct feedforward
- Requires switching from DriveAPI to Newton actuators
- This is a major refactor (build PD config, wrap articulation, etc.)

Given the user's question about whether articulation allows feedforward, the answer is:
- The `Articulation` class from `isaacsim.core.experimental.prims` (DriveAPI) does NOT have a `set_dof_feedforward_effort_targets` method
- The `ArticulationActuators` class (Newton actuators) DOES have `set_dof_feedforward_effort_targets`
- BUT switching to `ArticulationActuators` requires significant refactoring

So the position-offset feedforward approach (`q_target += F_ff/kp`) remains the correct approach for the current DriveAPI-based control.

Let me verify one more thing: does the `Articulation` class from `isaacsim.core.experimental.prims` actually have `set_dof_feedforward_effort_targets`? Let me check what version of Isaac Sim is being used and if there's any method I missed.

Actually, let me look at the `make_finger_robot.py` to see how the drive is configured. Maybe there's a way to add feedforward to the DriveAPI setup.

Looking at the code in `make_finger_robot.py`, the drive is configured via:
```python
drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
drive.CreateStiffnessAttr(0.0)
drive.CreateDampingAttr(drive_damping)
drive.CreateMaxForceAttr(DRIVE_MAX_FORCE)
```

The `DriveAPI` schema includes: stiffness, damping, maxForce, targetPosition, targetVelocity. There's no feedforward attribute. So the USD DriveAPI does not support feedforward.

But there might be a `PhysxSchema` extension. Let me check if `PhysxSchema.PhysxDriveAPI` adds feedforward support. In the code, `PhysxSchema.PhysxDriveAPI` is imported but I need to check what it actually provides.

Let me search for what attributes `PhysxSchema.PhysxDriveAPI` provides.

Actually, looking at make_finger_robot.py more carefully, I see at line 35:
```python
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
```

And `PhysxSchema` is used for `PhysxArticulationAPI` and `PhysxSchema.JointStateAPI`, not for drive configuration. The drive is configured with `UsdPhysics.DriveAPI` (standard USD Physics).

So the conclusion is clear: the current DriveAPI setup does NOT support explicit feedforward effort. We must use the position-offset approach.

But let me present both options to the user and let them decide:
1. Position-offset feedforward (no API change, works with current code)
2. Switch to `ArticulationActuators` for native feedforward (major refactor)

Now let me also look at the full actuator_setup.py tutorial to understand what's involved in switching to Newton actuators, just in case the user wants to explore that option.

Let me also check the `Articulation` class methods directly from the docs:

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "isaacsim.core.experimental.prims Articulation class all methods set_dof API list drive feedforward"
}
```

**Output:**
```
### Articulation Wrapper Class

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

The Articulation class serves as a high-level wrapper for manipulating USD prims that have the Root Articulation API applied. It allows for efficient management of articulation properties and other attributes across single or multiple prims, supporting regular expressions for path matching.

```APIDOC
## class Articulation

### Description
High level wrapper for manipulating prims (that have the Root Articulation API applied) and their attributes. This class is a wrapper over one or more USD prims in the stage to provide high-level functionality for manipulating articulation properties, and other attributes. The prims are specified using paths that can include regular expressions for matching multiple prims.

### Parameters
* **paths** (str | list[str]) - Single path or list of paths to USD prims. Can include regular expressions for matching multiple prims.
* **resolve_paths** (bool) - Whether to resolve the given paths (true) or use them as is (false). Defaults to True.
* **positions** (list | np.ndarray | wp.array | None) - Positions in the world frame (shape `(N, 3)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **translations** (list | np.ndarray | wp.array | None) - Translations in the local frame (shape `(N, 3)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **orientations** (list | np.ndarray | wp.array | None) - Orientations in the world frame (shape `(N, 4)`, quaternion `wxyz`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **scales** (list | np.ndarray | wp.array | None) - Scales to be applied to the prims (shape `(N, 3)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **reset_xform_op_properties** (bool) - Whether to reset the transformation operation attributes of the prims to a standard set. Defaults to False.
* **enable_residual_reports** (bool) - Whether to enable residual reporting for the articulations. Defaults to False.

### Raises
* **ValueError** - If no prims are found matching the specified path(s).
* **AssertionError** - If both positions and translations are specified.

### Example
```python
>>> import numpy as np
>>> import omni.timeline
>>> from isaacsim.core.experimental.prims import Articulation
>>>
>>> # given a USD stage with the prims: /World/prim_0, /World/prim_1, and /World/prim_2
>>> # where each prim is a reference to the Isaac Sim's Franka Panda USD asset
>>> # - create wrapper over single prim
>>> prim = Articulation("/World/prim_0")  
>>> # - create wrapper over multiple prims using regex
>>> prims = Articulation(
...     "/World/prim_.*",
...     positions=[[x, 0, 0] for x in range(3)],
...     reset_xform_op_properties=True,
...     enable_residual_reports=True,
... )  
>>>
>>> # play the simulation so that the Physics tensor entity becomes valid
>>> omni.timeline.get_timeline_interface().play()
```
```

--------------------------------

### Set DOF Position Targets

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Sets the desired target position for the DOFs of prims. This method influences movement over simulation steps. Supports tensor and USD backends. Stiffness and damping can be tuned for control.

```python
>>> # set random DOF position targets for all prims
>>> prims.set_dof_position_targets(np.random.uniform(low=-0.25, high=0.25, size=(3, 9)))
>>>
>>> # open all the Franka Panda fingers (finger DOFs to 0.04)
>>> prims.set_dof_position_targets([0.04], dof_indices=[7, 8])
```

--------------------------------

### set_dof_limits

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Sets the limits for the degrees of freedom (DOFs) of the prims. This method allows configuration of lower and upper bounds for DOF movements.

```APIDOC
## set_dof_limits

### Description
Sets the limits of the degrees of freedom (DOFs) of the prims.

### Parameters
* **lower** (float | list | np.ndarray | wp.array | None) - Lower limits (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **upper** (float | list | np.ndarray | wp.array | None) - Upper limits (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **indices** (int | list | np.ndarray | wp.array | None) - Indices of prims to process (shape `(N,)`). If not defined, all wrapped prims are processed.
* **dof_indices** (int | list | np.ndarray | wp.array | None) - Indices of DOFs to process (shape `(D,)`). If not defined, all DOFs are processed.

### Raises
* **AssertionError** – If neither lower nor upper limits are specified.
* **AssertionError** – Wrapped prims are not valid.

### Example
```python
>>> # set the DOF lower limits for all prims to -0.25
>>> prims.set_dof_limits(lower=[-0.25])
```
```

--------------------------------

### set_dof_armatures

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Sets the armatures for the degrees of freedom (DOFs) of the prims. Supports tensor and USD backends. Armatures define the resistance to acceleration.

```APIDOC
## set_dof_armatures

### Description
Sets the armatures of the degrees of freedom (DOFs) of the prims.

### Method
`set_dof_armatures`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **armatures** (float | list | np.ndarray | wp.array) - Required - Armatures (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **indices** (int | list | np.ndarray | wp.array | None) - Optional - Indices of prims to process (shape `(N,)`). If not defined, all wrapped prims are processed.
* **dof_indices** (int | list | np.ndarray | wp.array | None) - Optional - Indices of DOFs to process (shape `(D,)`). If not defined, all DOFs are processed.

### Request Example
```python
# set the DOF armatures for all prims
prims.set_dof_armatures([0.5])

# set the armatures for the first and last prims' finger DOFs
prims.set_dof_armatures([1.5], indices=[0, 2], dof_indices=[7, 8])
```

### Response
#### Success Response (200)
None

#### Response Example
None

### Raises
* **AssertionError** – Wrapped prims are not valid.
```

--------------------------------

### set_dof_positions

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.prims/docs/index.html

Sets the current positions of the degrees of freedom (DOFs) of the prims. This method directly manipulates the instantaneous DOF positions.

```APIDOC
## set_dof_positions

### Description
Set the positions of the degrees of freedom (DOFs) of the prims.
Warning

### Parameters
* **positions** (float | list | np.ndarray | wp.array) - Positions (shape `(N, D)`). If the input shape is smaller than expected, data will be broadcasted (following NumPy broadcast rules).
* **indices** (int | list | np.ndarray | wp.array | None) - Indices of prims to process (shape `(N,)`). If not defined, all wrapped prims are processed.
* **dof_indices** (int | list | np.ndarray | wp.array | None) - Indices of DOFs to process (shape `(D,)`). If not defined, all DOFs are processed.
```
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py",
  "limit": 100,
  "offset": 100
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py</path>
<type>file</type>
<content>
100:     return pd_config
101: 
102: 
103: # ============================================================================
104: # 3. Adding clamping and delay to an actuator config
105: # ============================================================================
106: def build_pd_with_clamping_and_delay(n_robots: int, kp: float, kd: float, max_effort: float) -> ActuatorConfig:
107:     """Build an ActuatorConfig with PD control plus a max-effort clamp and an input delay.
108: 
109:     ``kp``, ``kd``, and ``max_effort`` are single-joint scalars; each is fanned
110:     out across ``n_robots`` instances.
111:     """
112:     # <start-build-pd-with-clamping-snippet>
113:     import warp as wp
114:     from isaacsim.core.experimental.actuators import ActuatorConfig
115:     from newton.actuators import ClampingMaxEffort, ControllerPD, Delay
116: 
117:     config = ActuatorConfig(
118:         controller=ControllerPD(
119:             kp=wp.array([kp] * n_robots, dtype=wp.float32),
120:             kd=wp.array([kd] * n_robots, dtype=wp.float32),
121:         ),
122:         clamping=[
123:             ClampingMaxEffort(max_effort=wp.array([max_effort] * n_robots, dtype=wp.float32)),
124:         ],
125:         delay=Delay(
126:             delay_steps=wp.array([2] * n_robots, dtype=wp.int32),
127:             max_delay=2,
128:         ),
129:     )
130:     # <end-build-pd-with-clamping-snippet>
131:     return config
132: 
133: 
134: # ============================================================================
135: # 4. Constructing ArticulationActuators with the configs
136: # ============================================================================
137: # Per-joint PD gains tuned for a single Franka.  Joints 1-4 are the heavy arm
138: # joints; joints 5-7 are the wrist.
139: KP_GAINS = [32.0, 32.0, 32.0, 16.0, 32.0, 32.0, 32.0]
140: KD_GAINS = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
141: MAX_EFFORTS = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
142: 
143: 
144: def construct_articulation_actuators() -> ArticulationActuators:
145:     """Attach a per-DOF ideal PD actuator to each of the seven Franka arm joints."""
146:     # <start-construct-from-actuators-snippet>
147:     from isaacsim.core.experimental.actuators import ArticulationActuators
148:     from isaacsim.core.experimental.prims import Articulation
149: 
150:     # FRANKA_PRIM_PATH = "/World/Franka"
151:     # ARM_JOINTS       = ["panda_joint1", "panda_joint2", ..., "panda_joint7"]
152:     # Single Franka instance, so n_robots = 1.
153:     n_robots = len(Articulation(FRANKA_PRIM_PATH))
154: 
155:     actuators = [
156:         (build_pd_actuator_config(n_robots, kp, kd), name) for name, kp, kd in zip(ARM_JOINTS, KP_GAINS, KD_GAINS)
157:     ]
158: 
159:     actuated = ArticulationActuators.from_actuators(
160:         FRANKA_PRIM_PATH,
161:         actuators=actuators,
162:     )
163: 
164:     # A small armature on every DOF improves numerical stability when joints
165:     # are driven externally (the Newton-actuator effort can excite high-frequency
166:     # modes that the implicit USD drive would normally damp).  Real motors and
167:     # gearboxes also carry rotor inertia, so a non-zero armature is closer to
168:     # physical reality regardless of stability concerns.
169:     actuated.articulation.set_dof_armatures(0.1)
170:     # <end-construct-from-actuators-snippet>
171:     return actuated
172: 
173: #? where non-idealness comes from:
174: #! A 2-step input delay means the actuator does not apply the command from 
175: #! the current simulation tick immediately. It holds it for 2 physics steps
176: #! first, so at 60 Hz that is about 33 ms of latency. That models the kind
177: #! of lag you get from communication, computation, filtering, or actuator
178: #! response time. 
179: 
180: def construct_articulation_actuators_non_ideal() -> ArticulationActuators:
181:     """Construct `ArticulationActuators` with per-joint clamping and a 2-step input delay."""
182:     from isaacsim.core.experimental.actuators import ArticulationActuators
183:     from isaacsim.core.experimental.prims import Articulation
184: 
185:     n_robots = len(Articulation(FRANKA_PRIM_PATH))
186: 
187:     actuators = [
188:         (build_pd_with_clamping_and_delay(n_robots, kp, kd, max_effort), name)
189:         for name, kp, kd, max_effort in zip(ARM_JOINTS, KP_GAINS, KD_GAINS, MAX_EFFORTS)
190:     ]
191: 
192:     actuated = ArticulationActuators.from_actuators(
193:         FRANKA_PRIM_PATH,
194:         actuators=actuators,
195:     )
196:     actuated.articulation.set_dof_armatures(0.1)
197:     return actuated
198: 
199: 

(Showing lines 100-199 of 275. Use offset=200 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 5.5s)

_Thinking:_

Now let me also check if the `Articulation` class has `set_dof_feedforward_effort_targets` by checking the full API docs. I'll also look at the actuator_setup tutorial to understand what other methods `ArticulationActuators` offers.

Let me also check the USD DriveAPI documentation more specifically - specifically about whether the drive can have a feedforward component in the PhysX solver.

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/websites/isaacsim_omniverse_nvidia_5_1_0",
  "query": "UsdPhysics.DriveAPI feedforward targetForce forceLimit attributes complete schema prismatic joint drive force"
}
```

**Output:**
```
### Setting Robot Drive Properties with UsdPhysics.DriveAPI

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html

Demonstrates how to get and set drive properties like velocity, damping, and stiffness for robot joints using the UsdPhysics.DriveAPI in Python. This is useful for controlling robot locomotion and actuators.

```python
1# get handle to the Drive API for both wheels
2left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/left_wheel"), "angular")
3right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/right_wheel"), "angular")
4
5# Set the velocity drive target in degrees/second
6left_wheel_drive.GetTargetVelocityAttr().Set(150)
7right_wheel_drive.GetTargetVelocityAttr().Set(150)
8
9# Set the drive damping, which controls the strength of the velocity drive
10left_wheel_drive.GetDampingAttr().Set(15000)
11right_wheel_drive.GetDampingAttr().Set(15000)
12
13# Set the drive stiffness, which controls the strength of the position drive
14# In this case because we want to do velocity control this should be set to zero
15left_wheel_drive.GetStiffnessAttr().Set(0)
16right_wheel_drive.GetStiffnessAttr().Set(0)
```

--------------------------------

### Set Velocity Targets for Joints

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/robots_simulation.html

Sets drive stiffness to 0 for revolute and prismatic joints, then targets joint velocities for an articulation. Ensure the articulation is acquired after stiffness is set.

```python
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()
for prim in stage.TraverseAll():
    prim_type = prim.GetTypeName()
    if prim_type in ["PhysicsRevoluteJoint" , "PhysicsPrismaticJoint"]:
        if prim_type == "PhysicsRevoluteJoint":
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        else:
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if drive:
            drive.GetStiffnessAttr().Set(0)
from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
dc = _dynamic_control.acquire_dynamic_control_interface()
#Note: getting the articulation has to happen after changing the drive stiffness
articulation = dc.get_articulation("/Franka")
dc.wake_up_articulation(articulation)
joint_vels = [-np.random.rand(9)*10]
dc.set_articulation_dof_velocity_targets(articulation, joint_vels)
```

### Joints and Drives > Drive Type

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html

Drive types can be set to 'Acceleration' or 'Force'. Acceleration drives are invariant to changes in robot mass, while force drives apply effort directly to the joint. Target types can be 'None', 'Position', or 'Velocity'. Position targets use radians for revolute joints or distance units for prismatic joints, while velocity targets use units per second. The target type cannot be changed for joints configured as 'Mimic'.

--------------------------------

### UrdfJointDriveType

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/enum__urdf_types_8h_1a4ff04ab1973699d8113e61b47a1aad90.html

FORCE: Represents a joint drive controlled by force.

--------------------------------

### DcAttractorProperties

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structomni_1_1isaac_1_1dynamic__control_1_1_dc_attractor_properties.html

The `forceLimit` property sets the maximum force that can be applied by the drive mechanism.
```

**Tool: read**

**Input:**
```json
{
  "offset": 200,
  "filePath": "/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py",
  "limit": 80
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/tutorials/actuator_setup.py</path>
<type>file</type>
<content>
200: # ============================================================================
201: # 5. Driving the robot to a position target
202: # ============================================================================
203: #? how it works:
204: #! By default, ArticulationActuators registers a callback that runs immediately
205: #! BEFORE every physics step. Once the ArticulationActuators wrapper is constructed,
206: #! the actuators are live.
207: 
208: def drive_to_target(actuated, num_steps: int = 240) -> None:
209:     #! actuated here is an ArticulationActuators object (a wrapper) with user-defined
210:     #! settings
211:     """Set position targets and step the simulation to watch the robot converge."""
212:     # <start-drive-to-target-snippet>
213:     articulation = actuated.articulation
214: 
215:     timeline = omni.timeline.get_timeline_interface()
216:     timeline.play()
217:     simulation_app.update()
218: 
219:     # Send a per-DOF position target.  The PD controller reads it each tick.
220:     # Values match the Franka robot descriptor's default_q (home pose).
221:     target_positions = [0.8, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]
222:     arm_dof_indices = articulation.get_dof_indices(ARM_JOINTS)
223:     articulation.set_dof_position_targets(
224:         positions=target_positions,
225:         dof_indices=arm_dof_indices,
226:     )
227: 
228:     for _ in range(num_steps):  # ~4 seconds at 60 Hz
229:         simulation_app.update()
230:     # <end-drive-to-target-snippet>
231: 
232:     # <start-feedforward-effort-snippet>
233:     # Feedforward effort is added on top of the controller output every tick.
234:     # With kp = kd = 0 it becomes the entire output — a pure open-loop torque drive.
235:     #! this only take effect for joints with explicit actuators
236:     feedforward_efforts = [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
237:     actuated.set_dof_feedforward_effort_targets(
238:         feedforward_efforts,
239:         dof_indices=arm_dof_indices,
240:     )
241:     # <end-feedforward-effort-snippet>
242: 
243:     for _ in range(num_steps):
244:         simulation_app.update()
245: 
246: 
247: # ============================================================================
248: # Entry point
249: # ============================================================================
250: def main() -> None:
251:     SimulationManager.set_physics_dt(1.0 / 60.0)
252:     simulation_app.run_coroutine(setup_stage_with_franka())
253:     if args.non_ideal:
254:         print("Using non-ideal actuators (PD + per-joint effort clamp + 2-step delay).")
255:         construct = construct_articulation_actuators_non_ideal
256:     else:
257:         construct = construct_articulation_actuators
258:     try:
259:         # <start-context-manager-snippet>
260:         # Recommended teardown pattern: construct ``ArticulationActuators`` inside
261:         # the ``with`` statement so the wrapper's lifetime is bounded by the block.
262:         # ``__exit__`` calls ``actuated.close()``, which deregisters every
263:         # ``SimulationManager`` lifecycle callback owned by the instance and is
264:         # guaranteed to run even if the body raises.
265:         with construct() as actuated:
266:             drive_to_target(actuated)
267:         # <end-context-manager-snippet>
268:     finally:
269:         omni.timeline.get_timeline_interface().stop()
270:     print("Newton actuators Python example complete.")
271: 
272: 
273: if __name__ == "__main__":
274:     main()
275:     simulation_app.close()

(End of file - total 275 lines)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 2.7s)

_Thinking:_

Now I have a very clear picture. Let me also check one more thing — the `UsdPhysics.DriveAPI` vs `PhysxSchema.PhysxDriveAPI` in the `make_finger_robot.py` to see which drive API is actually used:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/xiao/0_codes/poke_map/env/make_finger_robot.py",
  "limit": 60,
  "offset": 142
}
```

**Output:**
```
<path>/home/xiao/0_codes/poke_map/env/make_finger_robot.py</path>
<type>file</type>
<content>
142:     cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
143:     prim = stage.GetPrimAtPath(path)
144:     attr = prim.CreateAttribute(
145:         "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
146:     )
147:     attr.Set([Gf.Vec3f(*color_rgb)])
148: 
149: 
150: def _add_sphere(stage, path, color_rgb):
151:     """Sphere to mark the finger tip, with collision enabled."""
152:     sphere = UsdGeom.Sphere.Define(stage, path)
153:     sphere.CreateRadiusAttr(SPHERE_RADIUS)
154:     prim = stage.GetPrimAtPath(path)
155:     UsdPhysics.CollisionAPI.Apply(prim) #* raw USD API to turn on collision
156:     PhysxSchema.PhysxCollisionAPI.Apply(prim) #* ditto
157:     attr = prim.CreateAttribute(
158:         "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
159:     )
160:     attr.Set([Gf.Vec3f(*color_rgb)])
161: 
162: 
163: def _create_prismatic_joint(stage, joint_path, 
164:                             parent_path, child_path, axis_token, 
165:                             lower, upper,
166:                             drive_stiffness=DRIVE_STIFFNESS,
167:                             drive_damping=DRIVE_DAMPING):
168:     """Create a PhysicsPrismaticJoint coupling *parent_path* → *child_path*.
169: 
170:     Parameters
171:     ----------
172:     axis_token : str
173:         One of ``"X"``, ``"Y"``, ``"Z"`` — the slide direction.
174:     """
175:     joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
176: 
177:     # Tell PhysX this is a linear (prismatic) joint
178:     PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")
179: 
180:     # USD DriveAPI — physics solver applies PD directly at constraint level
181:     drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
182:     drive.CreateTypeAttr("force")
183:     drive.CreateStiffnessAttr(drive_stiffness)
184:     drive.CreateDampingAttr(drive_damping)
185:     drive.CreateMaxForceAttr(DRIVE_MAX_FORCE)
186:     drive.CreateTargetPositionAttr(0.0)  # must exist before Set() can be called
187: 
188:     # body targets
189:     joint.CreateBody0Rel().SetTargets([parent_path])
190:     joint.CreateBody1Rel().SetTargets([child_path])
191: 
192:     # local anchor positions — both at their own origin (links are colocated)
193:     joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
194:     joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
195: 
196:     # local rotations — identity
197:     joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
198:     joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
199: 
200:     # axis (plain string auto-converts to TfToken: "X", "Y", or "Z")
201:     joint.CreateAxisAttr(axis_token)

(Showing lines 142-201 of 347. Use offset=202 to continue.)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 12.5s)

_Thinking:_

Now I have the complete picture. Let me summarize my findings:

## Key Findings

### Drive API Used
The finger robot uses `UsdPhysics.DriveAPI` with `type="force"` on each prismatic joint. The attributes are:
- `stiffness` (kp) — position gain
- `damping` (kd) — velocity gain
- `maxForce` — force clamp
- `targetPosition` — position setpoint

There is **no feedforward attribute** in `UsdPhysics.DriveAPI`. The drive equation is purely:
```
τ = stiffness * (targetPosition - q) + damping * (targetVelocity - dq/dt)
```
where `targetVelocity` defaults to 0.

### Feedforward in ArticulationActuators (Newton Actuators)
`ArticulationActuators.set_dof_feedforward_effort_targets()` exists, but it requires:
1. Switching from USD DriveAPI to Newton actuators
2. Wrapping the articulation with `ArticulationActuators.from_actuators()`
3. Configuring explicit PD controllers per joint
4. This is a **major refactor** (and the tutorial says: "this only takes effect for joints with explicit actuators")

### Position-Offset Feedforward (Recommended)
Since USD DriveAPI has no feedforward, the correct approach is to inject feedforward through the position target:
```
τ = kp*(q_target - q) + kd*(v_target - dq/dt)

Set: q_target = q_des + F_ff/kp   (where F_ff = m * a_des)
     v_target = v_des

⇒ τ = kp*(q_des - q) + kd*(v_des - dq/dt) + F_ff
```

This is mathematically exact — the feedforward force F_ff is propagated through the stiffness channel. The only limitation is that it's clamped by `maxForce`, but the feedforward is bounded by the physics anyway.

### Alternative: switch to ArticulationActuators
This would give native `set_dof_feedforward_effort_targets()` but requires:
- Creating `ActuatorConfig` objects with `ControllerPD` per joint
- Wrapping with `ArticulationActuators.from_actuators()`
- Managing the actuator lifecycle (context manager)
- Loses the simple DriveAPI setup

This is unjustified complexity for our use case.

So my recommendation is clear: **position-offset feedforward is the correct approach** for the current DriveAPI-based architecture.

Now let me finalize the plan and present it to the user.

Now I have a complete answer. Here's the summary of my research:

---

## Feedforward in Isaac Sim — Verdict

**The finger robot uses `UsdPhysics.DriveAPI` (not Newton actuators).** The drive schema at `make_finger_robot.py:181-185` only has `stiffness`, `damping`, `maxForce`, and `targetPosition` — **no feedforward attribute exists**.

The drive law is strictly:
```
τ = kp*(q_target − q) + kd*(v_target − dq/dt)
```

**`ArticulationActuators.set_dof_feedforward_effort_targets()`** does exist (see `tutorials/actuator_setup.py:237`), but it requires switching from DriveAPI to the Newton actuator framework (`ArticulationActuators.from_actuators`, per-joint `ControllerPD` configs, lifecycle management). That's a major refactor and adds complexity for no real gain here.

**Recommended approach: Position-offset feedforward.** Since DriveAPI's output is `kp*(q_target − q)`, we can inject feedforward force by offsetting the position target:

```
Set:  q_target = q_des + F_ff / kp     (F_ff = m · a_des)
      v_target = v_des

⇒ τ = kp*(q_des − q) + kd*(v_des − dq/dt) + F_ff
    └───────── PD feedback ─────────┘   └── inertial FF ──┘
```

This is mathematically exact — no API change needed, no refactor. The only caveat is that `maxForce=500` clamps the total output, but that's fine (it already limits `kp*error` today for the same reason).

---

## Revised Final Plan

| # | File | Change |
|---|------|--------|
| **1** | `env/quintic_poly.py` | Add `quintic_poly_derivative(u)`, `quintic_poly_second_derivative(u)`, `L=0.3`, `generate_strike_trajectory(v_mid)` → returns `(T, num_steps, s/v/a profiles)` |
| **2** | `nn/train.py` line ~920 | Standoff geometry: `delta_d`-based → fixed `L/2 = 0.15m` behind/ahead of contour |
| **3** | `nn/train.py` lines ~981-992 | Replace static `set_dof_position_targets(q3)` + `_step_physics(IMPACT_STEPS)` with per-timestep quintic trajectory loop (each step sets `q_target`, `v_target` with `q_target += F_ff/kp` position-offset feedforward) |
| **4** | `nn/train.py` line ~875 | Add optional `fingertip_masses` parameter to `env_step_async` |
| **5** | `nn/train.py` line ~1294 | Pass `randomized_finger_properties["fingertip_masses"]` at call site; fallback `mass=0.3`, `kp=1500.0` |

Phases 0/1/2/4/5 — **unchanged**. Z-axis stays at `POKE_Z`. No API change — pure DriveAPI position-offset feedforward.

Ready to implement?

---

## User

what is the benefit of using driveX api versus Newton api? 

---

## Assistant (Plan · DeepSeek V4 Pro · 12.1s)

_Thinking:_

The user is asking about the tradeoffs between the USD DriveAPI (currently used) and the Newton actuators API (ArticulationActuators) for their finger robot. Let me analyze this based on what I've learned from the codebase and documentation.

Key differences:

**USD DriveAPI (current approach):**
- Drive is part of the physics solver — PhysX computes constraint forces internally
- The solver helps enforce the target through the joint model
- Simple API: `set_dof_position_targets()`, `set_dof_velocity_targets()`, `set_dof_gains()`
- Implicit PD controller built into solver
- No explicit feedforward support
- Force output is: τ = kp*(q_target - q) + kd*(v_target - dq/dt)
- Clamped by maxForce
- Very lightweight, no external callback overhead
- Works directly on the PhysX articulation tensor

**Newton Actuators (ArticulationActuators):**
- External controller that computes torque/effort command each tick
- Your control code outputs torque, solver applies it as external input
- Requires `ArticulationActuators.from_actuators()` wrapper setup
- Per-joint PD configuration with `ControllerPD(kp, kd)`
- Has explicit `set_dof_feedforward_effort_targets()` for direct feedforward
- Additional features: per-joint effort clamping, input delay modeling, custom controllers
- Registers a callback that runs BEFORE every physics step
- Lifecycle managed via context manager (`with` statement)
- Slightly more overhead (Warp kernel dispatch per step)

The user is asking for the benefits of each. Let me think about what matters for their use case:

For the poke_map finger robot:
- Simple 3-DOF prismatic cartesian robot
- Already working with DriveAPI
- Needs smooth trajectory tracking with feedforward
- Batch control of 64 parallel environments
- RL training loop — performance sensitive

The DriveAPI position-offset approach is simpler and sufficient. Newton actuators would add complexity without meaningful benefit for this application.

But let me present both sides fairly.

Here's the comparison:

## DriveAPI vs Newton Actuators

| | **USD DriveAPI** (current) | **Newton Actuators** |
|---|---|---|
| **How it works** | PhysX solver computes constraint forces internally to enforce joint targets | Your code computes torque, solver applies it as external input |
| **Control law** | Implicit in solver: `τ = kp·Δq + kd·Δv` | Explicit per-tick callback via `ControllerPD(kp, kd)` |
| **Feedforward** | Not supported — must inject via position-offset `q_target += F_ff/kp` | Native: `set_dof_feedforward_effort_targets(τ_ff)` |
| **API surface** | 3 methods: `set_dof_position_targets`, `set_dof_velocity_targets`, `set_dof_gains` | Full `ActuatorConfig` system: PD gains per joint, `ClampingMaxEffort`, `Delay`, custom controllers |
| **Overhead** | Near-zero — runs inside PhysX constraint solve | Per-step Warp kernel dispatch + callback |
| **Setup** | Already built-in. Drive authored in USD at construction time | Requires `ArticulationActuators.from_actuators()` wrapper + lifecycle management (`with` statement) |
| **Numeric stability** | PhysX dampens internally | May need explicit armature (`set_dof_armatures(0.1)`) to suppress high-freq modes |

## Verdict for this project

**Stick with DriveAPI + position-offset feedforward.** The Newton API's only benefit here would be native `set_dof_feedforward_effort_targets()`, but we get the **same math** with `q_target += m·a/kp` — just routed through the stiffness channel instead of a dedicated feedforward register. You gain nothing functionally while adding wrapper boilerplate, lifecycle management, and per-step callback overhead on 64 parallel envs.

Newton actuators shine when you need per-joint effort clamps, simulated input delay, or custom non-PD controllers. None of that applies to a 3-DOF prismatic poke robot.

---

