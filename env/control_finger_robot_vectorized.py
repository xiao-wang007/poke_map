from pathlib import Path
import sys
import __main__

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

from config_loader import load_config
from make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives


CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]
CONTROL_CONFIG = CONFIG["control"]

ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
FINGER_LOCAL_ROOT_PATH = FINGER_CONFIG["local_root_path"]
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_ROOT_PATH}"
FINGER_TIP_LINK_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
SOURCE_FINGER_PATH = (
    f"{ENVS_ROOT_PATH}/{SCENE_CONFIG['source_env_name']}/{FINGER_LOCAL_ROOT_PATH}"
)

TARGET_SEED = CONTROL_CONFIG["target_seed"]
NUM_CONTROL_STEPS = CONTROL_CONFIG["num_control_steps"]
TARGET_LOW = np.array(LIMIT_LOWER, dtype=np.float32)
TARGET_HIGH = np.array(LIMIT_UPPER, dtype=np.float32)

# Velocity targets: sign controls direction, magnitude controls speed at arrival.
# None means the damping term drives velocity to zero (classic PD settle).
# Shape must be (num_fingers, 3) or broadcastable, units m/s.
ARRIVAL_VELOCITY = CONTROL_CONFIG["arrival_velocity"]
RANDOM_VELOCITY_SPEED_RANGE = CONTROL_CONFIG["random_velocity_speed_range"]


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


def sample_random_velocity_targets(
    num_fingers,
    speed_range=RANDOM_VELOCITY_SPEED_RANGE,
    seed=None,
):
    """Sample random unit-direction velocities with random magnitudes.

    Each finger gets an independent random direction in 3-D joint space
    scaled to a random speed drawn from *speed_range* (m/s).
    The drive law then becomes  tau = kp*(q_target - q) + kd*(v_target - dq/dt),
    so the finger arrives at *q_target* while still moving at *v_target*.
    """
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal(size=(num_fingers, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)  # guard against zero vectors
    directions /= norms
    speeds = rng.uniform(speed_range[0], speed_range[1], size=(num_fingers, 1)).astype(np.float32)
    return directions * speeds


def ensure_simulation_started():
    app_utils.play()


def get_randomized_finger_properties():
    return globals().get(
        "randomized_finger_properties",
        getattr(__main__, "randomized_finger_properties", None),
    )


def configure_vectorized_drives(finger_articulations, finger_properties=None):
    if finger_properties is None:
        configure_drives(finger_articulations)
        print("Using default finger stiffness/damping.")
        return

    configure_drives(
        finger_articulations,
        stiffnesses=finger_properties["stiffnesses"],
        dampings=finger_properties["dampings"],
    )
    print("Using randomized finger stiffness/damping from scene setup.")


async def command_fingers_to_targets_async(
    finger_articulations,
    targets,
    num_steps=NUM_CONTROL_STEPS,
    arrival_velocity=None,
):
    """Command all fingers to *targets* (shape: num_fingers x 3).

    arrival_velocity : array-like, shape (num_fingers, 3) or None
        When provided, each DOF's velocity target is set alongside the position
        target.  The PhysX drive then becomes:
            tau = kp*(q_target - q) + kd*(v_target - dq/dt)
        so the finger reaches *targets* while moving at *arrival_velocity*.
        Pass None (default) to keep the classic settle-to-stop behaviour
        (equivalent to v_target = 0).
    """
    dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
    finger_articulations.set_dof_position_targets(
        positions=targets.tolist(),
        dof_indices=dof_indices,
    )
    if arrival_velocity is not None:
        velocity_array = np.asarray(arrival_velocity, dtype=np.float32)
        finger_articulations.set_dof_velocity_targets(
            velocities=velocity_array.tolist(),
            dof_indices=dof_indices,
        )
    await app_utils.update_app_async(steps=num_steps)


def command_fingers_to_targets(
    finger_articulations,
    targets,
    num_steps=NUM_CONTROL_STEPS,
    arrival_velocity=None,
):
    return run_coroutine(
        command_fingers_to_targets_async(
            finger_articulations, targets, num_steps, arrival_velocity
        )
    )


def get_tip_positions(finger_tip_links):
    positions, orientations = finger_tip_links.get_world_poses()
    return as_numpy(positions), as_numpy(orientations)


async def main_async(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS, arrival_velocity=ARRIVAL_VELOCITY):
    finger_articulations, finger_tip_links = get_vectorized_fingers()
    ensure_simulation_started()
    await app_utils.update_app_async(steps=1)

    finger_properties = get_randomized_finger_properties()
    configure_vectorized_drives(finger_articulations, finger_properties)

    targets = sample_random_targets(
        num_fingers=len(finger_articulations),
        seed=seed,
    )

    #! targets are passed in to move the robot; arrival_velocity lets the
    #! finger reach the target at a non-zero velocity (see docstring above).
    await command_fingers_to_targets_async(
        finger_articulations=finger_articulations,
        targets=targets,
        num_steps=num_steps,
        arrival_velocity=arrival_velocity,
    )

    tip_positions, tip_orientations = get_tip_positions(finger_tip_links)
    joint_positions = finger_articulations.get_dof_positions()

    print("✅ Vectorized finger control complete")
    print(f"Finger count : {len(finger_articulations)}")
    print(f"DOF names    : {finger_articulations.dof_names}")
    print(f"Targets      :\n{targets}")
    print(f"Joint pos    :\n{joint_positions}")
    print(f"Tip pos      :\n{tip_positions}")

    #! This is useful in Isaac Script Editor. Because after main_async() finishes,
    #! local variables like 'targets' would disappear. Using globals(), they can 
    #! be used interactively later.
    globals().update(
        {
            "finger_articulations": finger_articulations,
            "finger_tip_links": finger_tip_links,
            "finger_targets": targets,
            "finger_arrival_velocity": arrival_velocity,
            "finger_tip_positions": tip_positions,
            "finger_tip_orientations": tip_orientations,
            "finger_properties": finger_properties,
            "command_fingers_to_targets": command_fingers_to_targets,
            "sample_random_targets": sample_random_targets,
            "sample_random_velocity_targets": sample_random_velocity_targets,
        }
    )
    return finger_articulations, finger_tip_links, targets


def main(seed=TARGET_SEED, num_steps=NUM_CONTROL_STEPS, arrival_velocity=ARRIVAL_VELOCITY):
    return run_coroutine(main_async(seed=seed, num_steps=num_steps, arrival_velocity=arrival_velocity))


finger_control_task = main()
