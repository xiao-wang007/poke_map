from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")

for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from isaacsim.core.experimental.prims import Articulation, RigidPrim
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from omni.kit.async_engine import run_coroutine

from make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives


ENVS_ROOT_PATH = "/World/envs"
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/FingerRobot"
FINGER_TIP_LINK_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
SOURCE_FINGER_PATH = f"{ENVS_ROOT_PATH}/env_0/FingerRobot"

TARGET_SEED = None
NUM_CONTROL_STEPS = 120
TARGET_LOW = np.array(LIMIT_LOWER, dtype=np.float32)
TARGET_HIGH = np.array(LIMIT_UPPER, dtype=np.float32)


def as_numpy(values):
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def get_vectorized_fingers():
    stage = stage_utils.get_current_stage(backend="usd")
    if not stage.GetPrimAtPath(SOURCE_FINGER_PATH):
        raise RuntimeError(
            f"No vectorized finger robot found at {SOURCE_FINGER_PATH}. "
            "Run env/scene_setup_vectorized.py first."
        )

    finger_articulations = Articulation(paths=FINGER_ROOT_PATTERN)
    finger_tip_links = RigidPrim(paths=FINGER_TIP_LINK_PATTERN)
    return finger_articulations, finger_tip_links


def sample_random_targets(num_fingers, seed=None, low=TARGET_LOW, high=TARGET_HIGH):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(num_fingers, 3)).astype(np.float32)


def ensure_simulation_started():
    app_utils.play()


async def command_fingers_to_targets_async(finger_articulations, targets, num_steps=NUM_CONTROL_STEPS):
    dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
    finger_articulations.set_dof_position_targets(
        positions=targets.tolist(),
        dof_indices=dof_indices,
    )
    await app_utils.update_app_async(steps=num_steps)


def command_fingers_to_targets(finger_articulations, targets, num_steps=NUM_CONTROL_STEPS):
    return run_coroutine(command_fingers_to_targets_async(finger_articulations, targets, num_steps))


def get_tip_positions(finger_tip_links):
    positions, orientations = finger_tip_links.get_world_poses()
    return as_numpy(positions), as_numpy(orientations)


async def main_async(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS):
    finger_articulations, finger_tip_links = get_vectorized_fingers()
    ensure_simulation_started()
    await app_utils.update_app_async(steps=1)

    configure_drives(finger_articulations)

    targets = sample_random_targets(
        num_fingers=len(finger_articulations),
        seed=seed,
    )
    await command_fingers_to_targets_async(
        finger_articulations=finger_articulations,
        targets=targets,
        num_steps=num_steps,
    )

    tip_positions, tip_orientations = get_tip_positions(finger_tip_links)
    joint_positions = finger_articulations.get_dof_positions()

    print("✅ Vectorized finger control complete")
    print(f"Finger count : {len(finger_articulations)}")
    print(f"DOF names    : {finger_articulations.dof_names}")
    print(f"Targets      :\n{targets}")
    print(f"Joint pos    :\n{joint_positions}")
    print(f"Tip pos      :\n{tip_positions}")

    globals().update(
        {
            "finger_articulations": finger_articulations,
            "finger_tip_links": finger_tip_links,
            "finger_targets": targets,
            "finger_tip_positions": tip_positions,
            "finger_tip_orientations": tip_orientations,
            "command_fingers_to_targets": command_fingers_to_targets,
            "sample_random_targets": sample_random_targets,
        }
    )
    return finger_articulations, finger_tip_links, targets


def main(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS):
    return run_coroutine(main_async(seed=seed, num_steps=num_steps))


finger_control_task = main()
