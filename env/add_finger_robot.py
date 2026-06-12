"""
Add an actuated spherical "finger robot" with 3 prismatic DOFs (XYZ) to the
current stage.  Designed to be run *after* ``scene_setup_vectorized.py`` from
the Isaac Sim script editor.

The finger robot is a lightweight cartesian manipulator:
  - Fixed base link
  - Three prismatic joints along X, Y, Z
  - A colored sphere at the tip as the "finger"

Control is via Newton ideal PD actuators (``ArticulationActuators``) so you
can command the tip to any (x, y, z) within the joint limits.

Usage (script editor)::

    # 1. Run scene_setup_vectorized.py first
    # 2. Run this file
    # 3. Interact via globals: finger_actuators, finger_articulation, etc.
    move_finger_to(finger_actuators, [0.1, -0.05, 0.08])

"""

from __future__ import annotations

import gc

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.timeline
import warp as wp
from isaacsim.core.experimental.actuators import ActuatorConfig, ArticulationActuators
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from newton.actuators import ControllerPD
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
FINGER_ROOT_PATH = "/World/FingerRobot"
BASE_PATH = f"{FINGER_ROOT_PATH}/base"
X_LINK_PATH = f"{FINGER_ROOT_PATH}/x_link"
Y_LINK_PATH = f"{FINGER_ROOT_PATH}/y_link"
Z_LINK_PATH = f"{FINGER_ROOT_PATH}/z_link"

X_JOINT_PATH = f"{FINGER_ROOT_PATH}/x_joint"
Y_JOINT_PATH = f"{FINGER_ROOT_PATH}/y_joint"
Z_JOINT_PATH = f"{FINGER_ROOT_PATH}/z_joint"

SPHERE_RADIUS = 0.025  # finger-tip sphere radius (m)
BOX_HALF = 0.008       # intermediate link visual half-size (m)

# Prismatic joint limits in meters (span the Franka EE workspace)
LIMIT_LOWER = [-0.25, -0.25, -0.05]
LIMIT_UPPER = [0.25, 0.25, 0.25]

# Per-axis PD gains (tuned for ~50 g links)
KP_DEFAULT = 150.0
KD_DEFAULT = 15.0

# Default starting pose
DEFAULT_XYZ = [0.0, 0.0, 0.05]


# ---------------------------------------------------------------------------
#  Cleanup helper
# ---------------------------------------------------------------------------
def clear_previous_handles():
    """Remove previous global handles so re-running the script is safe."""
    for name in (
        "finger_articulation",
        "finger_actuators",
        "finger_tip_sphere",
        "finger_root_xform",
    ):
        globals().pop(name, None)
    gc.collect()


# ---------------------------------------------------------------------------
#  Low-level USD helpers
# ---------------------------------------------------------------------------
def _ensure_xform(stage, path):
    if not stage.GetPrimAtPath(path):
        UsdGeom.Xform.Define(stage, path)


def _apply_rigid_body(stage, path, kinematic=False):
    prim = stage.GetPrimAtPath(path)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)


def _add_box_visual(stage, path, color_rgb):
    """Small box to mark a link's location."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(BOX_HALF * 2.0)
    prim = stage.GetPrimAtPath(path)
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _add_sphere_visual(stage, path, color_rgb):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(SPHERE_RADIUS)
    prim = stage.GetPrimAtPath(path)
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _create_prismatic_joint(stage, joint_path, parent_path, child_path, axis, lower, upper):
    """Create a PhysicsPrismaticJoint coupling *parent_path* → *child_path*."""
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)

    # body targets
    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])

    # local anchor positions — both at their own origin (links are colocated)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    # local rotations — identity
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # axis & limits
    joint.CreateAxisAttr().Set(Gf.Vec3f(*axis))
    joint.CreateLowerLimitAttr().Set(lower)
    joint.CreateUpperLimitAttr().Set(upper)


# ---------------------------------------------------------------------------
#  Finger articulation construction
# ---------------------------------------------------------------------------
def build_finger_articulation():
    """Build the 3-DOF prismatic finger robot hierarchy from scratch.

    USD hierarchy::

        /World/FingerRobot             ← Xform + ArticulationRootAPI
          ├── base                     ← Xform + RigidBodyAPI (kinematic/fixed)
          ├── x_link                   ← Xform + RigidBodyAPI
          │   └── x_visual             ← Cube
          ├── y_link                   ← Xform + RigidBodyAPI
          │   └── y_visual             ← Cube
          ├── z_link                   ← Xform + RigidBodyAPI
          │   └── sphere_visual        ← Sphere (finger tip)
          ├── x_joint                  ← PrismaticJoint base→x_link  (axis +X)
          ├── y_joint                  ← PrismaticJoint x_link→y_link (axis +Y)
          └── z_joint                  ← PrismaticJoint y_link→z_link (axis +Z)
    """
    stage = stage_utils.get_current_stage(backend="usd")

    # --- Ensure /World exists ---
    _ensure_xform(stage, "/World")

    # --- Articulation root ---
    _ensure_xform(stage, FINGER_ROOT_PATH)
    root_prim = stage.GetPrimAtPath(FINGER_ROOT_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # --- Links ---
    _ensure_xform(stage, BASE_PATH)
    _apply_rigid_body(stage, BASE_PATH, kinematic=True)

    _ensure_xform(stage, X_LINK_PATH)
    _apply_rigid_body(stage, X_LINK_PATH, kinematic=False)
    _add_box_visual(stage, f"{X_LINK_PATH}/x_visual", (0.9, 0.5, 0.5))

    _ensure_xform(stage, Y_LINK_PATH)
    _apply_rigid_body(stage, Y_LINK_PATH, kinematic=False)
    _add_box_visual(stage, f"{Y_LINK_PATH}/y_visual", (0.5, 0.9, 0.5))

    _ensure_xform(stage, Z_LINK_PATH)
    _apply_rigid_body(stage, Z_LINK_PATH, kinematic=False)
    _add_sphere_visual(
        stage, f"{Z_LINK_PATH}/sphere_visual", (1.0, 0.2, 0.2)
    )

    # --- Prismatic joints ---
    _create_prismatic_joint(
        stage, X_JOINT_PATH, BASE_PATH, X_LINK_PATH,
        axis=(1.0, 0.0, 0.0),
        lower=LIMIT_LOWER[0], upper=LIMIT_UPPER[0],
    )
    _create_prismatic_joint(
        stage, Y_JOINT_PATH, X_LINK_PATH, Y_LINK_PATH,
        axis=(0.0, 1.0, 0.0),
        lower=LIMIT_LOWER[1], upper=LIMIT_UPPER[1],
    )
    _create_prismatic_joint(
        stage, Z_JOINT_PATH, Y_LINK_PATH, Z_LINK_PATH,
        axis=(0.0, 0.0, 1.0),
        lower=LIMIT_LOWER[2], upper=LIMIT_UPPER[2],
    )

    # --- Root Xform for world positioning ---
    root_xform = XformPrim(
        paths=FINGER_ROOT_PATH,
        positions=[0.0, 0.0, 0.0],
        reset_xform_op_properties=True,
    )
    return root_xform


# ---------------------------------------------------------------------------
#  Actuator setup
# ---------------------------------------------------------------------------
def _make_pd_config(n_robots: int, kp: float, kd: float) -> ActuatorConfig:
    """Single-joint ideal PD actuator config (Warp-backed)."""
    return ActuatorConfig(
        controller=ControllerPD(
            kp=wp.array([kp] * n_robots, dtype=wp.float32),
            kd=wp.array([kd] * n_robots, dtype=wp.float32),
        )
    )


def setup_finger_actuators():
    """Wrap the finger robot with Newton PD actuators on all joints.

    Returns
    -------
    tuple[Articulation, ArticulationActuators]
        articulation : the Articulation wrapper
        actuated : the actuated wrapper with PD controllers
    """
    articulation = Articulation(FINGER_ROOT_PATH)
    n_robots = len(articulation)

    # Use the articulation's actual DOF names to build actuators
    dof_names = articulation.dof_names
    print(f"Finger robot – {articulation.num_dofs} DOFs: {dof_names}")

    # Build one actuator per DOF
    actuators = [
        (_make_pd_config(n_robots, KP_DEFAULT, KD_DEFAULT), name)
        for name in dof_names
    ]

    actuated = ArticulationActuators.from_actuators(
        FINGER_ROOT_PATH,
        actuators=actuators,
    )
    actuated.articulation.set_dof_armatures(0.005)
    return articulation, actuated


# ---------------------------------------------------------------------------
#  Movement helpers
# ---------------------------------------------------------------------------
def move_finger_to(actuated, xyz, num_steps=60):
    """Command the finger tip to absolute (x, y, z) joint positions.

    Parameters
    ----------
    actuated : ArticulationActuators
    xyz : (3,) array-like
        Target joint positions [x, y, z] in meters.
    num_steps : int
        Simulation steps to advance after setting the target.
    """
    dof_indices = actuated.articulation.dof_indices  # all DOFs
    actuated.set_joint_position_targets(
        positions=list(xyz),
        dof_indices=dof_indices,
    )
    for _ in range(num_steps):
        omni.kit.app.get_app().update()


def get_tip_pose():
    """Return world (position, orientation) of the finger tip."""
    tip = RigidPrim(Z_LINK_PATH)
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

    # --- Build the finger robot ---
    finger_root_xform = build_finger_articulation()

    # --- Wrap with articulation + actuators ---
    finger_articulation, finger_actuators = setup_finger_actuators()

    # --- Start simulation ---
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
    omni.kit.app.get_app().update()

    # --- Go to default pose ---
    dof_indices = finger_articulation.dof_indices  # all DOFs
    finger_actuators.set_joint_position_targets(
        positions=DEFAULT_XYZ,
        dof_indices=dof_indices,
    )
    for _ in range(60):
        omni.kit.app.get_app().update()

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
            "finger_actuators": finger_actuators,
            "finger_root_xform": finger_root_xform,
        }
    )


main()
