"""
Striker executor for the standalone-sphere vectorised scene.

After running ``scene_setup_standalone_vectorized.py`` from the Isaac Sim
Script Editor, import / run this module to command the striker spheres.

Control model
-------------
The striker is a pure RigidPrim — no joints, no articulation.  Movement is
achieved by directly setting its world position (to *teleport*) and its
linear velocity (to *strike*).

A typical strike sequence::

    executor.strike(
        env_idx=0,
        target_xy=(0.15, -0.05),
        direction=(0.8, -0.6),
        speed=2.0,
        steps=30,
    )

This:
  1. teleports the striker above ``target_xy`` at a safe height,
  2. descends to contact height,
  3. sets the striker velocity along ``direction``,
  4. steps physics so the striker collides with any object in its path,
  5. retracts the striker back to the safe height.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")

for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from isaacsim.core.experimental.prims import RigidPrim, XformPrim
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from omni.kit.async_engine import run_coroutine

from config_loader import load_config

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
STRIKER_CONFIG = CONFIG["striker"]

ENVS_ROOT_PATH: str = SCENE_CONFIG["envs_root_path"]
STRIKER_PATTERN: str = f"{ENVS_ROOT_PATH}/env_.*/{STRIKER_CONFIG['local_path']}"

STRIKER_RADIUS: float = STRIKER_CONFIG["radius"]
STRIKER_DEFAULT_Z: float = STRIKER_CONFIG["default_z"]
OBJECT_HEIGHT: float = SCENE_CONFIG["object_height"]


def as_numpy(values):
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def get_vectorized_strikers() -> tuple[RigidPrim, XformPrim]:
    """Return batch handles for strikers and env roots."""
    stage = stage_utils.get_current_stage(backend="usd")
    if not stage.GetPrimAtPath(f"{ENVS_ROOT_PATH}/env_0"):
        raise RuntimeError(
            "No vectorized envs found. Run scene_setup_standalone_vectorized.py first."
        )
    strikers = RigidPrim(paths=STRIKER_PATTERN)
    env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
    return strikers, env_roots


def ensure_simulation_started():
    app_utils.play()


async def _step_physics(steps: int) -> None:
    await app_utils.update_app_async(steps=steps)


def step_physics(steps: int = 1) -> None:
    run_coroutine(_step_physics(steps))


#* ========================================================================
#*  Single-env strike
#* ========================================================================

async def strike_async(
    strikers: RigidPrim,
    env_idx: int,
    target_xy: tuple[float, float],
    direction: tuple[float, float],
    speed: float = 2.0,
    sim_steps: int = 30,
    safe_z: float = STRIKER_DEFAULT_Z,
    contact_z: float | None = None,
) -> dict:
    """Strike in one environment.  Other envs are unaffected.

    Parameters
    ----------
    strikers : batch RigidPrim over all envs.
    env_idx : which environment to act in.
    target_xy : (x, y) world position to strike at (table plane).
    direction : (dx, dy) unit vector of the strike.
    speed : velocity magnitude (m/s) along *direction*.
    sim_steps : number of PhysX steps to run after setting velocity.
    safe_z : retract / teleport height (m).
    contact_z : height of the contact surface.  Defaults to object top.

    Returns
    -------
    info : dict with initial and final striker positions.
    """
    if contact_z is None:
        contact_z = OBJECT_HEIGHT

    num_envs = len(strikers)
    zero_vel = np.zeros((num_envs, 3), dtype=np.float32)

    #* -- 1. Teleport striker above target --------------------------------
    positions, _ = strikers.get_world_poses()
    positions = as_numpy(positions).copy()
    positions[env_idx] = [target_xy[0], target_xy[1], safe_z]
    strikers.set_world_poses(positions=positions.tolist())
    strikers.set_velocities(
        linear_velocities=zero_vel.tolist(),
        angular_velocities=zero_vel.tolist(),
    )
    await _step_physics(1)

    #* -- 2. Descend to contact height ------------------------------------
    positions[env_idx, 2] = contact_z
    strikers.set_world_poses(positions=positions.tolist())
    await _step_physics(2)

    #* record pre-strike position
    init_pos = positions[env_idx].copy()

    #* -- 3. Set velocity for the active env ------------------------------
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-8:
        dir_norm = 1.0
    vx = direction[0] / dir_norm * speed
    vy = direction[1] / dir_norm * speed

    velocities = np.zeros((num_envs, 3), dtype=np.float32)
    velocities[env_idx] = [vx, vy, 0.0]
    strikers.set_velocities(
        linear_velocities=velocities.tolist(),
        angular_velocities=zero_vel.tolist(),
    )

    #* -- 4. Step physics (collision happens here) ------------------------
    await _step_physics(sim_steps)

    #* -- 5. Read final position ------------------------------------------
    final_positions, _ = strikers.get_world_poses()
    final_positions = as_numpy(final_positions)
    final_pos = final_positions[env_idx].copy()

    #* -- 6. Retract to safe height ---------------------------------------
    final_positions[env_idx, 2] = safe_z
    strikers.set_world_poses(positions=final_positions.tolist())
    strikers.set_velocities(
        linear_velocities=zero_vel.tolist(),
        angular_velocities=zero_vel.tolist(),
    )
    await _step_physics(2)

    return {
        "env_idx": env_idx,
        "init_pos": init_pos,
        "final_pos": final_pos,
    }


def strike(
    strikers: RigidPrim,
    env_idx: int,
    target_xy: tuple[float, float],
    direction: tuple[float, float],
    speed: float = 2.0,
    sim_steps: int = 30,
) -> dict:
    return run_coroutine(
        strike_async(strikers, env_idx, target_xy, direction, speed, sim_steps)
    )


#* ========================================================================
#*  Batch strike — all envs at once
#* ========================================================================

async def strike_all_async(
    strikers: RigidPrim,
    target_xys: np.ndarray,        # (N, 2)
    directions: np.ndarray,        # (N, 2)
    speeds: np.ndarray | float = 2.0,
    sim_steps: int = 30,
    safe_z: float = STRIKER_DEFAULT_Z,
    contact_z: float | None = None,
) -> list[dict]:
    """Strike in all environments simultaneously.

    Each env receives its own target, direction, and speed.
    """
    if contact_z is None:
        contact_z = OBJECT_HEIGHT

    num_envs = len(strikers)
    if isinstance(speeds, (int, float)):
        speeds = np.full(num_envs, speeds, dtype=np.float32)
    speeds = np.asarray(speeds, dtype=np.float32)

    #* -- 1. Teleport all strikers above targets --------------------------
    positions = np.zeros((num_envs, 3), dtype=np.float32)
    positions[:, 0] = target_xys[:, 0]
    positions[:, 1] = target_xys[:, 1]
    positions[:, 2] = safe_z
    strikers.set_world_poses(positions=positions.tolist())
    strikers.set_velocities(
        linear_velocities=np.zeros_like(positions).tolist(),
        angular_velocities=np.zeros_like(positions).tolist(),
    )
    await _step_physics(1)

    #* -- 2. Descend to contact height ------------------------------------
    positions[:, 2] = contact_z
    strikers.set_world_poses(positions=positions.tolist())
    await _step_physics(2)

    init_positions = positions.copy()

    #* -- 3. Set velocities -----------------------------------------------
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    dirs_unit = directions / norms

    velocities = np.zeros((num_envs, 3), dtype=np.float32)
    velocities[:, 0] = dirs_unit[:, 0] * speeds
    velocities[:, 1] = dirs_unit[:, 1] * speeds
    strikers.set_velocities(linear_velocities=velocities.tolist())

    #* -- 4. Step physics -------------------------------------------------
    await _step_physics(sim_steps)

    #* -- 5. Read final positions, retract --------------------------------
    final_positions, _ = strikers.get_world_poses()
    final_positions = as_numpy(final_positions)

    final_positions[:, 2] = safe_z
    strikers.set_world_poses(positions=final_positions.tolist())
    strikers.set_velocities(linear_velocities=np.zeros_like(positions).tolist())
    await _step_physics(2)

    infos = []
    for i in range(num_envs):
        infos.append({
            "env_idx": i,
            "init_pos": init_positions[i],
            "final_pos": final_positions[i],
        })
    return infos


def strike_all(
    strikers: RigidPrim,
    target_xys: np.ndarray,
    directions: np.ndarray,
    speeds: np.ndarray | float = 2.0,
    sim_steps: int = 30,
) -> list[dict]:
    return run_coroutine(
        strike_all_async(strikers, target_xys, directions, speeds, sim_steps)
    )


#* ========================================================================
#*  Interactive entry point
#* ========================================================================

async def main_async():
    strikers, env_roots = get_vectorized_strikers()
    ensure_simulation_started()
    await _step_physics(1)

    print(f"Strikers ready: {len(strikers)} env(s).")
    globals().update(
        {
            "strikers": strikers,
            "env_roots": env_roots,
            "strike": strike,
            "strike_all": strike_all,
            "step_physics": step_physics,
        }
    )
    return strikers, env_roots


def main():
    return run_coroutine(main_async())


standalone_striker_task = main()
