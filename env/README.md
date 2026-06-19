### `scene_setup_standalone_vectorized.py`

Replaces the articulated finger robot with standalone `RigidPrim` spheres. Each env gets a sphere at `{ENVS_ROOT_PATH}/env_*/Striker` with collision, mass, and a red visual.

**Key differences from the articulated version:**

| Articulated | Standalone |
|---|---|
| `build_finger_articulation()` → 3-DOF prismatic chain | `create_striker_sphere()` → one rigid-body sphere |
| `Articulation` prim | `RigidPrim` only |
| Joint PD control (`set_dof_position_targets`) | Direct velocity set (`set_linear_velocities`) |
| `randomize_finger_properties()` (stiffness, damping, mass) | `randomize_striker_masses()` (mass only) |
| `move_fingers_to()` with joint targets | `reset_strikers_to_safe()` + `set_striker_velocities()` |


### `poke_executor_standalone_vectorized.py`

Commands the strikers with two modes:
- **`strike()`** — single env: teleport above target → descend → set velocity → physics → retract
- **`strike_all()`** — vectorized batch: all envs strike simultaneously with per-env targets/directions/speeds