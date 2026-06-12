"""
Add an actuated spherical "finger robot" with 3 prismatic DOFs (XYZ) to the
current stage.  Designed to be run *after* ``scene_setup_vectorized.py`` from
the Isaac Sim script editor.

The finger robot is a lightweight cartesian manipulator:
  - Fixed base link
  - Three prismatic joints along X, Y, Z
  - A colored sphere at the tip as the "finger"

Control is via USD DriveAPI (PD at the solver level) so you can command
the tip to any (x, y, z) within the joint limits.

Usage (script editor)::

    # 1. Run scene_setup_vectorized.py first
    # 2. Run this file
    # 3. Interact via globals: finger_articulation etc.
    move_finger_to(finger_articulation, [0.1, -0.05, 0.08])

"""

from __future__ import annotations

import gc

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.timeline
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

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

# USD DriveAPI gains (applied by the physics solver directly)
# stiffness = position gain (N/m), damping = velocity gain (N·s/m)
DRIVE_STIFFNESS = 1000.0  # N/m
DRIVE_DAMPING   = 100.0   # N·s/m
DRIVE_MAX_FORCE = 500.0   # N

# Default starting pose
DEFAULT_XYZ = [0.0, 0.0, 0.05]


# ---------------------------------------------------------------------------
#  Cleanup helper
# ---------------------------------------------------------------------------
def clear_previous_handles():
    """Remove previous global handles so re-running the script is safe."""
    for name in (
        "finger_articulation",
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


def _apply_rigid_body(stage, path, mass_kg=0.05):
    """Apply RigidBodyAPI + MassAPI to a prim so it participates in physics."""
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)


def _add_box_visual(stage, path, color_rgb):
    """Small box to mark a link's location (visual only — no collision)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(BOX_HALF * 2.0)
    prim = stage.GetPrimAtPath(path)
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _add_sphere_visual(stage, path, color_rgb):
    """Sphere to mark the finger tip (visual only — no collision)."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(SPHERE_RADIUS)
    prim = stage.GetPrimAtPath(path)
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _create_prismatic_joint(stage, joint_path, parent_path, child_path, axis_token, lower, upper):
    """Create a PhysicsPrismaticJoint coupling *parent_path* → *child_path*.

    Parameters
    ----------
    axis_token : str
        One of ``"X"``, ``"Y"``, ``"Z"`` — the slide direction.
    """
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)

    # Tell PhysX this is a linear (prismatic) joint
    PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")

    # USD DriveAPI — physics solver applies PD directly at constraint level
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(DRIVE_STIFFNESS)
    drive.CreateDampingAttr(DRIVE_DAMPING)
    drive.CreateMaxForceAttr(DRIVE_MAX_FORCE)
    drive.CreateTargetPositionAttr(0.0)  # must exist before Set() can be called

    # body targets
    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])

    # local anchor positions — both at their own origin (links are colocated)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    # local rotations — identity
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # axis (plain string auto-converts to TfToken: "X", "Y", or "Z")
    joint.CreateAxisAttr(axis_token)

    # limits
    joint.CreateLowerLimitAttr().Set(lower)
    joint.CreateUpperLimitAttr().Set(upper)


# ---------------------------------------------------------------------------
#  Finger articulation construction
# ---------------------------------------------------------------------------
def build_finger_articulation():
    """Build the 3-DOF prismatic finger robot hierarchy from scratch.

    Fixed-base articulation pattern (PhysX-compatible):

        /World
          └── FingerRobot                         ← Xform + ArticulationRootAPI
                ├── base              (RigidBody + CollisionAPI)
                │     anchored to /World via FixedJoint
                ├── x_link            (RigidBody + CollisionAPI)
                ├── y_link            (RigidBody + CollisionAPI)
                ├── z_link            (RigidBody + CollisionAPI)
                ├── base_fixed_joint  FixedJoint:      /World → base
                ├── x_joint           PrismaticJoint:  base   → x_link  (X)
                ├── y_joint           PrismaticJoint:  x_link → y_link  (Y)
                └── z_joint           PrismaticJoint:  y_link → z_link  (Z)

        Chain:  /World ──[fixed]──▶ base ──[X]──▶ x_link ──[Y]──▶ y_link ──[Z]──▶ z_link
    """
    stage = stage_utils.get_current_stage(backend="usd")

    # --- Ensure /World exists ---
    _ensure_xform(stage, "/World")

    # --- FingerRobot container with ArticulationRootAPI ---
    _ensure_xform(stage, FINGER_ROOT_PATH)
    root_prim = stage.GetPrimAtPath(FINGER_ROOT_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    physx_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physx_api.CreateSleepThresholdAttr(0.0)

    # --- Links (RigidBodyAPI + explicit MassAPI; collision on geometry children) ---
    _ensure_xform(stage, BASE_PATH)
    _apply_rigid_body(stage, BASE_PATH, mass_kg=0.1)
    _add_box_visual(stage, f"{BASE_PATH}/base_visual", (0.5, 0.5, 0.5))

    _ensure_xform(stage, X_LINK_PATH)
    _apply_rigid_body(stage, X_LINK_PATH, mass_kg=0.05)
    _add_box_visual(stage, f"{X_LINK_PATH}/x_visual", (0.9, 0.5, 0.5))

    _ensure_xform(stage, Y_LINK_PATH)
    _apply_rigid_body(stage, Y_LINK_PATH, mass_kg=0.05)
    _add_box_visual(stage, f"{Y_LINK_PATH}/y_visual", (0.5, 0.9, 0.5))

    _ensure_xform(stage, Z_LINK_PATH)
    _apply_rigid_body(stage, Z_LINK_PATH, mass_kg=0.05)
    _add_sphere_visual(
        stage, f"{Z_LINK_PATH}/sphere_visual", (1.0, 0.2, 0.2)
    )

    # --- Fixed joint: anchor base to /World (world is implicit body0) ---
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage, f"{FINGER_ROOT_PATH}/base_fixed_joint"
    )
    PhysxSchema.JointStateAPI.Apply(fixed_joint.GetPrim(), "linear")
    fixed_joint.CreateBody1Rel().SetTargets([BASE_PATH])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # --- Prismatic joints ---
    _create_prismatic_joint(
        stage, X_JOINT_PATH, BASE_PATH, X_LINK_PATH,
        axis_token="X",
        lower=LIMIT_LOWER[0], upper=LIMIT_UPPER[0],
    )
    _create_prismatic_joint(
        stage, Y_JOINT_PATH, X_LINK_PATH, Y_LINK_PATH,
        axis_token="Y",
        lower=LIMIT_LOWER[1], upper=LIMIT_UPPER[1],
    )
    _create_prismatic_joint(
        stage, Z_JOINT_PATH, Y_LINK_PATH, Z_LINK_PATH,
        axis_token="Z",
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
#  Articulation setup
# ---------------------------------------------------------------------------
def setup_finger_articulation() -> Articulation:
    """Wrap the finger robot prim with an Articulation for joint control."""
    articulation = Articulation(FINGER_ROOT_PATH)
    print(f"Finger robot – {articulation.num_dofs} DOFs: {articulation.dof_names}")
    return articulation


def configure_drives(articulation: Articulation):
    """Push stiffness/damping into the physics tensor after sim starts.

    USD DriveAPI attrs are only read at simulation start-up.  After play(),
    the tensor must be updated explicitly — otherwise gains are zero.
    """
    articulation.set_dof_gains(
        stiffnesses=DRIVE_STIFFNESS,
        dampings=DRIVE_DAMPING,
    )


# ---------------------------------------------------------------------------
#  Movement helpers
# ---------------------------------------------------------------------------
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
