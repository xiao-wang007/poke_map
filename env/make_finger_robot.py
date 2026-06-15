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
from pathlib import Path
import sys

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.timeline
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")

for module_dir in (CURRENT_DIR, PROJECT_ENV_DIR):
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from config_loader import load_config

CONFIG = load_config()
FINGER_CONFIG = CONFIG["finger"]

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
FINGER_ROOT_PATH = FINGER_CONFIG["standalone_root_path"]
BASE_PATH = f"{FINGER_ROOT_PATH}/base"
X_LINK_PATH = f"{FINGER_ROOT_PATH}/x_link"
Y_LINK_PATH = f"{FINGER_ROOT_PATH}/y_link"
Z_LINK_PATH = f"{FINGER_ROOT_PATH}/z_link"

X_JOINT_PATH = f"{FINGER_ROOT_PATH}/x_joint"
Y_JOINT_PATH = f"{FINGER_ROOT_PATH}/y_joint"
Z_JOINT_PATH = f"{FINGER_ROOT_PATH}/z_joint"

SPHERE_RADIUS = FINGER_CONFIG["sphere_radius"]  # finger-tip sphere radius (m)
BOX_HALF = FINGER_CONFIG["box_half"]       # intermediate link visual half-size (m)

# Prismatic joint limits in meters (span the Franka EE workspace)
LIMIT_LOWER = FINGER_CONFIG["limit_lower"]
LIMIT_UPPER = FINGER_CONFIG["limit_upper"]

# USD DriveAPI gains (applied by the physics solver directly)
# stiffness = position gain (N/m), damping = velocity gain (N·s/m)
DRIVE_STIFFNESS = FINGER_CONFIG["drive_stiffness"]  # N/m
DRIVE_DAMPING   = FINGER_CONFIG["drive_damping"]   # N·s/m
DRIVE_MAX_FORCE = FINGER_CONFIG["drive_max_force"]   # N
DEFAULT_FINGERTIP_MASS = FINGER_CONFIG["default_fingertip_mass"]  # kg, authored on z_link rigid body

# Optional randomization ranges used by vectorized scene setup.
DRIVE_STIFFNESS_RANGE = FINGER_CONFIG["drive_stiffness_range"]  # N/m
DRIVE_DAMPING_RANGE = FINGER_CONFIG["drive_damping_range"]      # N·s/m
FINGERTIP_MASS_RANGE = FINGER_CONFIG["fingertip_mass_range"]      # kg

# Default starting pose
DEFAULT_XYZ = FINGER_CONFIG["default_xyz"] # (m) default z position 20cm 


def get_finger_paths(root_path=FINGER_ROOT_PATH):
    return {
        "root": root_path,
        "base": f"{root_path}/base",
        "x_link": f"{root_path}/x_link",
        "y_link": f"{root_path}/y_link",
        "z_link": f"{root_path}/z_link",
        "x_joint": f"{root_path}/x_joint",
        "y_joint": f"{root_path}/y_joint",
        "z_joint": f"{root_path}/z_joint",
        "base_fixed_joint": f"{root_path}/base_fixed_joint",
    }


def sample_finger_parameters(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return {
        "drive_stiffness": float(rng.uniform(DRIVE_STIFFNESS_RANGE[0], DRIVE_STIFFNESS_RANGE[1])),
        "drive_damping": float(rng.uniform(DRIVE_DAMPING_RANGE[0], DRIVE_DAMPING_RANGE[1])),
        "fingertip_mass": float(rng.uniform(FINGERTIP_MASS_RANGE[0], FINGERTIP_MASS_RANGE[1])),
    }


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
    """Hidden marker for intermediate links."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(BOX_HALF * 2.0)
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    prim = stage.GetPrimAtPath(path)
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _add_sphere(stage, path, color_rgb):
    """Sphere to mark the finger tip, with collision enabled."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(SPHERE_RADIUS)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI.Apply(prim) #* raw USD API to turn on collision
    PhysxSchema.PhysxCollisionAPI.Apply(prim) #* ditto
    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    attr.Set([Gf.Vec3f(*color_rgb)])


def _create_prismatic_joint(stage, joint_path, 
                            parent_path, child_path, axis_token, 
                            lower, upper,
                            drive_stiffness=DRIVE_STIFFNESS,
                            drive_damping=DRIVE_DAMPING):
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
    drive.CreateStiffnessAttr(drive_stiffness)
    drive.CreateDampingAttr(drive_damping)
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
def build_finger_articulation(
    root_path=FINGER_ROOT_PATH,
    root_position=None,
    drive_stiffness=DRIVE_STIFFNESS,
    drive_damping=DRIVE_DAMPING,
    fingertip_mass=DEFAULT_FINGERTIP_MASS,
):
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
    paths = get_finger_paths(root_path)
    if root_position is None:
        root_position = [0.0, 0.0, 0.0]

    # --- Ensure /World exists ---
    _ensure_xform(stage, "/World")

    # --- FingerRobot container with ArticulationRootAPI ---
    _ensure_xform(stage, paths["root"])
    root_prim = stage.GetPrimAtPath(paths["root"])
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    physx_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physx_api.CreateSleepThresholdAttr(0.0)

    # --- Links (RigidBodyAPI + explicit MassAPI; collision on geometry children) ---
    _ensure_xform(stage, paths["base"])
    _apply_rigid_body(stage, paths["base"], mass_kg=0.1)
    _add_box_visual(stage, f"{paths['base']}/base_visual", (0.5, 0.5, 0.5))

    _ensure_xform(stage, paths["x_link"])
    _apply_rigid_body(stage, paths["x_link"], mass_kg=0.05)
    _add_box_visual(stage, f"{paths['x_link']}/x_visual", (0.9, 0.5, 0.5))

    _ensure_xform(stage, paths["y_link"])
    _apply_rigid_body(stage, paths["y_link"], mass_kg=0.05)
    _add_box_visual(stage, f"{paths['y_link']}/y_visual", (0.5, 0.9, 0.5))

    _ensure_xform(stage, paths["z_link"])
    _apply_rigid_body(stage, paths["z_link"], mass_kg=fingertip_mass)
    _add_sphere(
        stage, f"{paths['z_link']}/sphere", (1.0, 0.2, 0.2)
    )

    # --- Fixed joint: anchor base to /World (world is implicit body0) ---
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage, paths["base_fixed_joint"]
    )
    PhysxSchema.JointStateAPI.Apply(fixed_joint.GetPrim(), "linear")
    fixed_joint.CreateBody1Rel().SetTargets([paths["base"]])

    #! the conventions here is:
    #! a joint has two sides, body0 and body1 here; so CreateLocalPos0Attr()
    #! and CreateLocalPos1Attr() refers to the two bodies respectively.
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0)) #* [1, 0, 0, 0] identify rotation
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # --- Prismatic joints ---
    _create_prismatic_joint(
        stage, paths["x_joint"], paths["base"], paths["x_link"],
        axis_token="X",
        lower=LIMIT_LOWER[0], upper=LIMIT_UPPER[0],
        drive_stiffness=drive_stiffness,
        drive_damping=drive_damping,
    )
    _create_prismatic_joint(
        stage, paths["y_joint"], paths["x_link"], paths["y_link"],
        axis_token="Y",
        lower=LIMIT_LOWER[1], upper=LIMIT_UPPER[1],
        drive_stiffness=drive_stiffness,
        drive_damping=drive_damping,
    )
    _create_prismatic_joint(
        stage, paths["z_joint"], paths["y_link"], paths["z_link"],
        axis_token="Z",
        lower=LIMIT_LOWER[2], upper=LIMIT_UPPER[2],
        drive_stiffness=drive_stiffness,
        drive_damping=drive_damping,
    )

    # --- Root Xform for world positioning ---
    root_xform = XformPrim(
        paths=paths["root"],
        positions=root_position,
        reset_xform_op_properties=True,
    )
    return root_xform


#* ---------------------------------------------------------------------------
#*  Articulation setup
#* ---------------------------------------------------------------------------
def setup_finger_articulation(root_path=FINGER_ROOT_PATH) -> Articulation:
    """Wrap the finger robot prim with an Articulation for joint control."""
    articulation = Articulation(root_path)
    print(f"Finger robot – {articulation.num_dofs} DOFs: {articulation.dof_names}")
    return articulation


#! The USD actuation setup should be authored before simulation starts (like the 
#! URDF, or .world in gazebo). But here, it writes into the runtime PhysX articulation
#! /tensor handle. This overwrites what's pre-defined in the USD; but does not 
#! necessarily rewrite the authored USD attributes on disk/stage; it writes into the 
#! live PhysX/articulation runtime data.
#! Then if the statge is stopped/reloaded/recreated, PhysX will go back to whatever is
#! authored is USD unless set_dof_gains() is called again.
def configure_drives(
    articulation: Articulation,
    stiffnesses=DRIVE_STIFFNESS,
    dampings=DRIVE_DAMPING,
):
    """Push stiffness/damping into the physics tensor after sim starts.

    USD DriveAPI attrs are only read at simulation start-up.  After play(),
    the tensor must be updated explicitly — otherwise gains are zero.
    """
    articulation.set_dof_gains(
        stiffnesses=stiffnesses,
        dampings=dampings,
    )
