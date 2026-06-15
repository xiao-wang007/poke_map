from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")

for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from make_finger_robot import *

VECTOR_FINGER_ROOT_PATTERN = "/World/envs/env_.*/FingerRobot"
VECTOR_FINGER_TIP_PATTERN = "/World/envs/env_.*/FingerRobot/z_link"
SOURCE_VECTOR_FINGER_PATH = "/World/envs/env_0/FingerRobot"

#* ---------------------------------------------------------------------------
#*  Movement helpers
#* ---------------------------------------------------------------------------
def move_finger_to(articulation: Articulation, xyz, num_steps=120):
    """Command the finger tip to absolute (x, y, z) joint positions.

    Parameters
    ----------
    articulation : Articulation
    xyz : (3,) array-like
        Target joint positions [x, y, z] in meters.
    num_steps : int
        Simulation steps to advance after setting the target.
    """
    # For articulations, the physics tensor API is authoritative during
    # simulation — USD attribute changes are only read at sim start.
    # set_dof_position_targets writes directly to the PhysX articulation
    # drive cache, using the stiffness/damping configured via USD DriveAPI.
    dof_indices = articulation.get_dof_indices(articulation.dof_names)
    articulation.set_dof_position_targets(
        positions=list(xyz),
        dof_indices=dof_indices,
    )
    for _ in range(num_steps):
        omni.kit.app.get_app().update()


def get_tip_pose():
    """Return world (position, orientation) of the finger tip."""
    stage = stage_utils.get_current_stage(backend="usd")
    tip_path = VECTOR_FINGER_TIP_PATTERN if stage.GetPrimAtPath(SOURCE_VECTOR_FINGER_PATH) else Z_LINK_PATH
    tip = RigidPrim(tip_path)
    return tip.get_world_poses()


def set_finger_world_position(xform: XformPrim, position):
    """Move the entire finger robot root to a new world position."""
    xform.set_world_poses(positions=position)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    clear_previous_handles()

    # --- Ensure basic scene elements exist (idempotent for re-runs) ---
    stage = stage_utils.get_current_stage(backend="usd")
    if not stage.GetPrimAtPath("/World/GroundPlane"):
        GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    if not stage.GetPrimAtPath("/World/DistantLight"):
        DistantLight("/World/DistantLight").set_intensities(500)

    if stage.GetPrimAtPath(SOURCE_VECTOR_FINGER_PATH):
        finger_root_xform = XformPrim(paths="/World/envs/env_.*")
        finger_articulation = Articulation(paths=VECTOR_FINGER_ROOT_PATTERN)
        print(f"Using vectorized finger robots: {finger_articulation.num_dofs} DOFs")
    else:
        # --- Build the standalone finger robot ---
        finger_root_xform = build_finger_articulation()

        # --- Wrap with Articulation ---
        finger_articulation = setup_finger_articulation()

    # --- Start simulation ---
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
    omni.kit.app.get_app().update()

    # --- Push gains into physics tensor (USD attrs only read at sim start) ---
    configure_drives(finger_articulation)
    omni.kit.app.get_app().update()

    # --- Go to default pose ---
    move_finger_to(finger_articulation, DEFAULT_XYZ, num_steps=120)

    # --- Report ---
    pos, _ = get_tip_pose()
    print(f"✅ Finger robot ready at {FINGER_ROOT_PATH}")
    print(f"   DOFs        : {finger_articulation.dof_names}")
    print(f"   Joint pos   : {finger_articulation.get_dof_positions()}")
    print(f"   Tip world   : {pos}")

    # --- Expose to globals for interactive use ---
    globals().update(
        {
            "finger_articulation": finger_articulation,
            "finger_root_xform": finger_root_xform,
        }
    )
    return finger_articulation, finger_root_xform


_finger_articulation, _finger_root_xform = main()

# --- Try moving the finger ---
TARGET = [0.10, 0.05, 0.12]
move_finger_to(_finger_articulation, TARGET, num_steps=120)
pos, _ = get_tip_pose()
joints = _finger_articulation.get_dof_positions()
print(f"Target      : {TARGET}")
print(f"Joint pos   : {joints}")
print(f"Tip world   : {pos}")
