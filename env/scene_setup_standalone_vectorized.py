"""
Vectorised scene setup with standalone striker spheres (no articulated finger robot).

Each environment contains:
  - One L-shaped object (rigid body)
  - One cylinder object (rigid body)
  - One standalone RigidPrim sphere (the "striker")

The striker has no joints — it is moved by directly setting its linear velocity.
This eliminates joint-space control from Level 1, isolating the task-level RL
problem (where to strike, in which direction, at what speed).

After running this script from the Isaac Sim Script Editor, use
``poke_executor_standalone_vectorized.py`` to command the strikers.

Workflow convention
-------------------
one USD stage
  └── many envs
      ├── env_0/Striker   ←  standalone sphere 0
      ├── env_0/LObject
      ├── env_0/Cylinder
      ├── env_1/Striker   ←  standalone sphere 1
      ├── ...
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np

#* Isaac Sim imports
from isaacsim.core.cloner import GridCloner
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube, Cylinder, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.rendering_manager import ViewportManager
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

from omni.kit.async_engine import run_coroutine

PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")
if PROJECT_ENV_DIR.exists() and str(PROJECT_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ENV_DIR))

from config_loader import load_config

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
OBJECT_CONFIG = CONFIG["objects"]
STRIKER_CONFIG = CONFIG["striker"]

#* ---------------------------------------------------------------------------
#*  Scene constants
#* ---------------------------------------------------------------------------
OBJECT_HEIGHT: float = SCENE_CONFIG["object_height"]
L_ARM_LENGTH: float = SCENE_CONFIG["l_arm_length"]
L_THICKNESS: float = SCENE_CONFIG["l_thickness"]
CYLINDER_RADIUS: float = SCENE_CONFIG["cylinder_radius"]
L_COLLISION_APPROXIMATION: str = SCENE_CONFIG["l_collision_approximation"]
CYLINDER_COLLISION_APPROXIMATION: str = SCENE_CONFIG["cylinder_collision_approximation"]

NUM_ENVS: int = SCENE_CONFIG["num_envs"]
ENV_SPACING: float = SCENE_CONFIG["env_spacing"]
ENVS_ROOT_PATH: str = SCENE_CONFIG["envs_root_path"]
SOURCE_ENV_PATH: str = f"{ENVS_ROOT_PATH}/{SCENE_CONFIG['source_env_name']}"

L_OBJECT_LOCAL_POSITION: list = SCENE_CONFIG["l_object_local_position"]
CYLINDER_LOCAL_POSITION: list = [
    SCENE_CONFIG["cylinder_local_xy"][0],
    SCENE_CONFIG["cylinder_local_xy"][1],
    OBJECT_HEIGHT * 0.5,
]

RANDOMIZE_ON_START: bool = SCENE_CONFIG["randomize_on_start"]
RANDOM_SEED: int = SCENE_CONFIG["random_seed"]

L_POSITION_AREA_MIN: list = OBJECT_CONFIG["l_position_area_min"]
L_POSITION_AREA_MAX: list = OBJECT_CONFIG["l_position_area_max"]
CYLINDER_POSITION_AREA_MIN: list = OBJECT_CONFIG["cylinder_position_area_min"]
CYLINDER_POSITION_AREA_MAX: list = OBJECT_CONFIG["cylinder_position_area_max"]

L_YAW_RANGE: list = OBJECT_CONFIG["l_yaw_range"]
CYLINDER_YAW_RANGE: list = OBJECT_CONFIG["cylinder_yaw_range"]

DELAY_TO_SETTLE: int = SCENE_CONFIG["settle_steps"]

#* ---------------------------------------------------------------------------
#*  Striker constants
#* ---------------------------------------------------------------------------
STRIKER_LOCAL_PATH: str = STRIKER_CONFIG["local_path"]
STRIKER_RADIUS: float = STRIKER_CONFIG["radius"]
STRIKER_DEFAULT_Z: float = STRIKER_CONFIG["default_z"]
STRIKER_MASS_RANGE: list = STRIKER_CONFIG["mass_range"]
STRIKER_DEFAULT_COLOR: list = STRIKER_CONFIG["default_color"]

STRIKER_PATTERN: str = f"{ENVS_ROOT_PATH}/env_.*/{STRIKER_LOCAL_PATH}"


#* ========================================================================
#*  Cleanup
#* ========================================================================

def clear_previous_handles() -> None:
    for handle_name in (
        "ground_plane",
        "distant_light",
        "l_material",
        "cylinder_material",
        "striker_material",
        "l_parts",
        "l_geometry",
        "l_transform",
        "l_object",
        "l_objects",
        "cylinder_shape",
        "cylinder_geometry",
        "cylinder_object",
        "cylinder_objects",
        "source_striker",
        "strikers",
        "source_env",
        "env_roots",
        "env_paths",
        "cloner",
        "randomized_poses",
        "randomized_striker_masses",
        "settle_task",
    ):
        globals().pop(handle_name, None)
    gc.collect()


#* ========================================================================
#*  Striker sphere creation
#* ========================================================================

def create_striker_sphere(
    path: str,
    position: tuple[float, float, float],
    radius: float = STRIKER_RADIUS,
    mass: float = 0.05,
    color: tuple[float, float, float] = (1.0, 0.2, 0.2),
) -> RigidPrim:
    """Create a standalone rigid-body sphere with collision at *path*.

    The sphere is NOT part of any articulation.  It is moved by directly
    setting ``set_velocities`` and ``set_world_poses`` — no joint
    control is needed.  This is the lightweight "proxy striker" for the
    prior and for early RL experiments.
    """
    stage = stage_utils.get_current_stage(backend="usd")

    #* -- Xform container ------------------------------------------------
    UsdGeom.Xform.Define(stage, path)

    #* -- Sphere geometry (visual + collision) ---------------------------
    geo_path = f"{path}/geo"
    sphere = UsdGeom.Sphere.Define(stage, geo_path)
    sphere.CreateRadiusAttr(radius)

    prim = stage.GetPrimAtPath(geo_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    display_color = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    )
    display_color.Set([Gf.Vec3f(*color)])

    #* -- Rigid body + mass on the container prim ------------------------
    body_prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.CreateMassAttr(mass)

    #* -- Wrap as RigidPrim for velocity / pose control ------------------
    striker_rb = RigidPrim(
        paths=path,
        positions=[list(position)],
        reset_xform_op_properties=True,
    )

    return striker_rb


#* ========================================================================
#*  Object creation (same as articulated setup)
#* ========================================================================

def get_l_object_part_paths(path: str) -> list[str]:
    return [f"{path}/VerticalLeg", f"{path}/HorizontalLeg"]


def create_l_shaped_object(
    path: str,
    origin: list,
    material,
):
    stage = stage_utils.get_current_stage(backend="usd")
    l_part_paths = get_l_object_part_paths(path)
    UsdGeom.Xform.Define(stage, path)

    l_parts = Cube(
        paths=l_part_paths,
        translations=[
            [L_THICKNESS * 0.5, L_ARM_LENGTH * 0.5, OBJECT_HEIGHT * 0.5],
            [L_ARM_LENGTH * 0.5, L_THICKNESS * 0.5, OBJECT_HEIGHT * 0.5],
        ],
        sizes=1.0,
        scales=[
            [L_THICKNESS, L_ARM_LENGTH, OBJECT_HEIGHT],
            [L_ARM_LENGTH, L_THICKNESS, OBJECT_HEIGHT],
        ],
    )
    l_parts.apply_visual_materials(material)
    l_geometry = GeomPrim(paths=l_part_paths, apply_collision_apis=True)
    l_geometry.set_collision_approximations(
        [L_COLLISION_APPROXIMATION] * len(l_part_paths)
    )

    l_transform = XformPrim(paths=path, positions=origin, reset_xform_op_properties=True)
    l_object = RigidPrim(paths=path, masses=1.0)
    return l_parts, l_geometry, l_transform, l_object


def create_cylinder(
    path: str,
    position: list,
    radius: float,
    height: float,
    material,
):
    cylinder_shape = Cylinder(
        paths=path, positions=position, radii=radius, heights=height, axes="Z"
    )
    cylinder_shape.apply_visual_materials(material)

    cylinder_geometry = GeomPrim(paths=path, apply_collision_apis=True)
    cylinder_geometry.set_collision_approximations(CYLINDER_COLLISION_APPROXIMATION)
    cylinder_object = RigidPrim(paths=path, masses=1.0)
    return cylinder_shape, cylinder_geometry, cylinder_object


#* ========================================================================
#*  Source env + cloning
#* ========================================================================

def create_source_env(
    path: str,
    l_material,
    cylinder_material,
    striker_color: tuple = (1.0, 0.2, 0.2),
):
    """Create a single source environment with objects + striker sphere."""
    stage = stage_utils.get_current_stage(backend="usd")
    UsdGeom.Xform.Define(stage, ENVS_ROOT_PATH)
    UsdGeom.Xform.Define(stage, path)

    source_env = XformPrim(paths=path, positions=[0.0, 0.0, 0.0], reset_xform_op_properties=True)

    #* -- Objects --------------------------------------------------------
    l_parts, l_geometry, l_transform, l_object = create_l_shaped_object(
        path=f"{path}/LObject",
        origin=L_OBJECT_LOCAL_POSITION,
        material=l_material,
    )
    cylinder_shape, cylinder_geometry, cylinder_object = create_cylinder(
        path=f"{path}/Cylinder",
        position=CYLINDER_LOCAL_POSITION,
        radius=CYLINDER_RADIUS,
        height=OBJECT_HEIGHT,
        material=cylinder_material,
    )

    #* -- Standalone striker sphere --------------------------------------
    striker_sphere = create_striker_sphere(
        path=f"{path}/{STRIKER_LOCAL_PATH}",
        position=(0.0, 0.0, STRIKER_DEFAULT_Z),
        radius=STRIKER_RADIUS,
        mass=0.05,
        color=striker_color,
    )

    return (
        source_env,
        l_parts,
        l_geometry,
        l_transform,
        l_object,
        cylinder_shape,
        cylinder_geometry,
        cylinder_object,
        striker_sphere,
    )


def clone_envs(source_env_path: str, num_envs: int, spacing: float):
    cloner = GridCloner(spacing=spacing)
    env_paths = cloner.generate_paths(f"{ENVS_ROOT_PATH}/env", num_envs)
    cloner.clone(
        source_prim_path=source_env_path,
        prim_paths=env_paths,
        replicate_physics=True,
        base_env_path=ENVS_ROOT_PATH,
    )
    return cloner, env_paths


#* ========================================================================
#*  Helpers
#* ========================================================================

def as_numpy(values):
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def sample_planar_positions(
    rng: np.random.Generator, count: int, area_min: list, area_max: list, z_height: float
) -> np.ndarray:
    xy_positions = rng.uniform(area_min, area_max, size=(count, 2))
    z_positions = np.full((count, 1), z_height)
    return np.concatenate([xy_positions, z_positions], axis=1)


def sample_yaw_orientations(
    rng: np.random.Generator, count: int, yaw_range: list
) -> tuple[np.ndarray, np.ndarray]:
    yaws = rng.uniform(yaw_range[0], yaw_range[1], size=count)
    orientations = np.zeros((count, 4), dtype=np.float32)
    orientations[:, 0] = np.cos(yaws * 0.5)
    orientations[:, 3] = np.sin(yaws * 0.5)
    return orientations, yaws


#* ========================================================================
#*  Randomization
#* ========================================================================

def randomize_object_poses(
    env_roots: XformPrim,
    l_objects: RigidPrim,
    cylinder_objects: RigidPrim,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    env_positions, _ = env_roots.get_world_poses()
    env_positions = as_numpy(env_positions)
    num_envs = len(env_positions)

    l_local_positions = sample_planar_positions(
        rng, num_envs, L_POSITION_AREA_MIN, L_POSITION_AREA_MAX, z_height=0.0
    )
    cylinder_local_positions = sample_planar_positions(
        rng, num_envs,
        CYLINDER_POSITION_AREA_MIN, CYLINDER_POSITION_AREA_MAX,
        z_height=OBJECT_HEIGHT * 0.5,
    )

    l_orientations, l_yaws = sample_yaw_orientations(rng, num_envs, L_YAW_RANGE)
    cylinder_orientations, cylinder_yaws = sample_yaw_orientations(
        rng, num_envs, CYLINDER_YAW_RANGE
    )

    l_world_positions = env_positions + l_local_positions
    cylinder_world_positions = env_positions + cylinder_local_positions

    l_objects.set_world_poses(
        positions=l_world_positions.tolist(),
        orientations=l_orientations.tolist(),
    )
    cylinder_objects.set_world_poses(
        positions=cylinder_world_positions.tolist(),
        orientations=cylinder_orientations.tolist(),
    )

    return {
        "l_local_positions": l_local_positions,
        "l_yaws": l_yaws,
        "cylinder_local_positions": cylinder_local_positions,
        "cylinder_yaws": cylinder_yaws,
    }


def randomize_striker_masses(
    strikers: RigidPrim, seed: int | None = None
) -> np.ndarray:
    """Randomise striker masses for domain randomization."""
    rng = np.random.default_rng(seed)
    num_strikers = len(strikers)
    masses = rng.uniform(
        STRIKER_MASS_RANGE[0], STRIKER_MASS_RANGE[1], size=num_strikers
    ).astype(np.float32)
    strikers.set_masses(masses)
    return masses


#* ========================================================================
#*  Striker control (position + velocity, no joints)
#* ========================================================================

def reset_strikers_to_safe(
    strikers: RigidPrim,
    env_roots: XformPrim | None = None,
) -> None:
    """Move all strikers to a safe default position (above table, at env origin)."""
    num_strikers = len(strikers)

    if env_roots is not None:
        env_positions, _ = env_roots.get_world_poses()
        env_positions = as_numpy(env_positions)
        positions = env_positions + np.array([0.0, 0.0, STRIKER_DEFAULT_Z])
    else:
        positions = np.tile(
            np.array([0.0, 0.0, STRIKER_DEFAULT_Z], dtype=np.float32),
            (num_strikers, 1),
        )

    strikers.set_world_poses(positions=positions.tolist())
    #* zero out residual velocity
    zero_vel = np.zeros((num_strikers, 3), dtype=np.float32)
    strikers.set_velocities(
        linear_velocities=zero_vel.tolist(),
        angular_velocities=zero_vel.tolist(),
    )


def set_striker_velocities(
    strikers: RigidPrim,
    velocities: np.ndarray,       # (N, 3) or (N_strikers, 3)
) -> None:
    """Set linear velocities for ALL strikers at once.

    To strike in a specific env, set non-zero velocity only for that index.
    Other envs retain zero velocity.
    """
    strikers.set_velocities(linear_velocities=velocities.tolist())


def set_striker_positions(
    strikers: RigidPrim,
    positions: np.ndarray,        # (N, 3)
) -> None:
    strikers.set_world_poses(positions=positions.tolist())


#* ========================================================================
#*  Physics stepping
#* ========================================================================

async def settle_scene_async(steps: int) -> None:
    app_utils.play()
    if steps > 0:
        await app_utils.update_app_async(steps=steps)
    print(f"Settled scene for {steps} app update steps.")


def schedule_settle_scene(steps: int):
    return run_coroutine(settle_scene_async(steps))


#* ========================================================================
#*  Main
#* ========================================================================

def main():
    clear_previous_handles()
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    #* ── Lighting & materials ────────────────────────────────────────
    ground_plane = GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

    distant_light = DistantLight("/World/DistantLight")
    distant_light.set_intensities(500)

    l_material = PreviewSurfaceMaterial("/VisualMaterials/l_object_blue")
    l_material.set_input_values("diffuseColor", [0.1, 0.35, 1.0])

    cylinder_material = PreviewSurfaceMaterial("/VisualMaterials/cylinder_orange")
    cylinder_material.set_input_values("diffuseColor", [1.0, 0.45, 0.05])

    #* ── Source env (single, cloned below) ───────────────────────────
    (
        source_env,
        l_parts,
        l_geometry,
        l_transform,
        l_object,
        cylinder_shape,
        cylinder_geometry,
        cylinder_object,
        source_striker,
    ) = create_source_env(
        SOURCE_ENV_PATH, l_material, cylinder_material,
        striker_color=tuple(STRIKER_DEFAULT_COLOR),
    )

    #* ── Clone into vectorized envs ──────────────────────────────────
    cloner, env_paths = clone_envs(SOURCE_ENV_PATH, NUM_ENVS, ENV_SPACING)

    #* ── Global prim handles (batch over all envs) ───────────────────
    env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
    l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
    cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
    strikers = RigidPrim(paths=STRIKER_PATTERN)

    #* ── Randomize initial poses & striker masses ────────────────────
    randomized_poses = None
    randomized_striker_masses = None
    if RANDOMIZE_ON_START:
        randomized_poses = randomize_object_poses(
            env_roots=env_roots,
            l_objects=l_objects,
            cylinder_objects=cylinder_objects,
            seed=RANDOM_SEED,
        )
        randomized_striker_masses = randomize_striker_masses(
            strikers=strikers, seed=RANDOM_SEED + 1
        )

    #* ── Move strikers to safe pose, settle physics ──────────────────
    reset_strikers_to_safe(strikers, env_roots)
    settle_task = schedule_settle_scene(DELAY_TO_SETTLE)

    #* ── View ────────────────────────────────────────────────────────
    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=[2.8, 2.8, 2.0],
        target=[0.0, 0.0, 0.45],
    )

    #* ── Expose handles to the global scope (interactive use) ────────
    globals().update(
        {
            "ground_plane": ground_plane,
            "distant_light": distant_light,
            "l_material": l_material,
            "cylinder_material": cylinder_material,
            "l_parts": l_parts,
            "l_geometry": l_geometry,
            "l_transform": l_transform,
            "l_object": l_object,
            "l_objects": l_objects,
            "cylinder_shape": cylinder_shape,
            "cylinder_geometry": cylinder_geometry,
            "cylinder_object": cylinder_object,
            "cylinder_objects": cylinder_objects,
            "source_striker": source_striker,
            "strikers": strikers,
            "source_env": source_env,
            "env_roots": env_roots,
            "env_paths": env_paths,
            "cloner": cloner,
            "randomized_poses": randomized_poses,
            "randomized_striker_masses": randomized_striker_masses,
            "settle_task": settle_task,
            "reset_strikers_to_safe": reset_strikers_to_safe,
            "set_striker_velocities": set_striker_velocities,
            "set_striker_positions": set_striker_positions,
            "schedule_settle_scene": schedule_settle_scene,
        }
    )


main()
