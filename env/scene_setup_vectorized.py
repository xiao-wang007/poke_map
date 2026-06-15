import gc
from pathlib import Path
import sys

from isaacsim.core.cloner import GridCloner
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube, Cylinder, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
import numpy as np
from pxr import UsdGeom

#! wait to settle sim by resolving object overlapping from initial randomization
import isaacsim.core.experimental.utils.app as app_utils

#! for running async task safely 
from omni.kit.async_engine import run_coroutine

PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")
if PROJECT_ENV_DIR.exists() and str(PROJECT_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ENV_DIR))

from make_finger_robot import (
    DEFAULT_XYZ,
    DRIVE_DAMPING_RANGE,
    DRIVE_STIFFNESS_RANGE,
    FINGERTIP_MASS_RANGE,
    build_finger_articulation,
    configure_drives,
)


OBJECT_HEIGHT = 0.05
L_ARM_LENGTH = 0.08
L_THICKNESS = 0.03
CYLINDER_RADIUS = 0.05
L_COLLISION_APPROXIMATION = "convexHull"
CYLINDER_COLLISION_APPROXIMATION = "convexHull"

NUM_ENVS = 12
ENV_SPACING = 1.2
ENVS_ROOT_PATH = "/World/envs"
SOURCE_ENV_PATH = f"{ENVS_ROOT_PATH}/env_0"
FINGER_LOCAL_ROOT_PATH = "FingerRobot"
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_ROOT_PATH}"
FINGER_TIP_LINK_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"

L_OBJECT_LOCAL_POSITION = [-0.10, -0.06, 0.0]
CYLINDER_LOCAL_POSITION = [0.10, 0.02, OBJECT_HEIGHT * 0.5]

RANDOMIZE_ON_START = True
RANDOM_SEED = 2

FRANKA_EE_RANGE_MIN_X = -0.25
FRANKA_EE_RANGE_MAX_X = 0.25
FRANKA_EE_RANGE_MIN_Y = -0.25
FRANKA_EE_RANGE_MAX_Y = 0.25

# L_POSITION_AREA_MAX = [-0.03, 0.20]
# CYLINDER_POSITION_AREA_MIN = [0.03, -0.20]
# CYLINDER_POSITION_AREA_MAX = [0.20, 0.20]
L_POSITION_AREA_MIN = [FRANKA_EE_RANGE_MIN_X, FRANKA_EE_RANGE_MIN_Y]
L_POSITION_AREA_MAX = [FRANKA_EE_RANGE_MAX_X, FRANKA_EE_RANGE_MAX_Y]
CYLINDER_POSITION_AREA_MIN = [FRANKA_EE_RANGE_MIN_X, FRANKA_EE_RANGE_MIN_Y]
CYLINDER_POSITION_AREA_MAX = [FRANKA_EE_RANGE_MAX_X, FRANKA_EE_RANGE_MAX_Y]

L_YAW_RANGE = [-np.pi, np.pi]
CYLINDER_YAW_RANGE = [-np.pi, np.pi]

DELAY_TO_SETTLE = 10 # 30 physics steps


def clear_previous_handles():
    for handle_name in (
        "ground_plane",
        "distant_light",
        "l_material",
        "cylinder_material",
        "l_parts",
        "l_geometry",
        "l_transform",
        "l_object",
        "l_objects",
        "cylinder_shape",
        "cylinder_geometry",
        "cylinder_object",
        "cylinder_objects",
        "source_env",
        "env_roots",
        "env_paths",
        "cloner",
        "randomized_poses",
        "source_finger_root_xform",
        "finger_articulations",
        "finger_tip_links",
        "finger_settle_task",
        "randomized_finger_properties",
    ):
        globals().pop(handle_name, None)
    gc.collect()


def get_l_object_part_paths(path):
    return [
        f"{path}/VerticalLeg",
        f"{path}/HorizontalLeg",
    ]


def create_l_shaped_object(path, origin, material):
    stage = stage_utils.get_current_stage(backend="usd")
    l_part_paths = get_l_object_part_paths(path)

    #! when creating /World/LObject using this raw USD api here, xformOp:translate
    #! is not created by default. XformPrim below resets xform ops before moving it.
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
    l_geometry.set_collision_approximations([L_COLLISION_APPROXIMATION] * len(l_part_paths))

    #! USD transforms are hierarchical, so moving /World/LObject moves its
    #! children /World/LObject/VerticalLeg and /World/LObject/HorizontalLeg together.
    l_transform = XformPrim(
        paths=path,
        positions=origin,
        reset_xform_op_properties=True,
    )
    l_object = RigidPrim(paths=path, masses=1.0)
    return l_parts, l_geometry, l_transform, l_object


def create_cylinder(path, position, radius, height, material):
    cylinder_shape = Cylinder(
        paths=path,
        positions=position,
        radii=radius,
        heights=height,
        axes="Z",
    )
    cylinder_shape.apply_visual_materials(material)

    cylinder_geometry = GeomPrim(paths=path, apply_collision_apis=True)
    cylinder_geometry.set_collision_approximations(CYLINDER_COLLISION_APPROXIMATION)
    cylinder_object = RigidPrim(paths=path, masses=1.0)
    return cylinder_shape, cylinder_geometry, cylinder_object


def create_source_env(path, l_material, cylinder_material):
    stage = stage_utils.get_current_stage(backend="usd")
    UsdGeom.Xform.Define(stage, ENVS_ROOT_PATH)
    UsdGeom.Xform.Define(stage, path)

    source_env = XformPrim(
        paths=path,
        positions=[0.0, 0.0, 0.0],
        reset_xform_op_properties=True,
    )

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
    source_finger_root_xform = build_finger_articulation(
        root_path=f"{path}/{FINGER_LOCAL_ROOT_PATH}",
        root_position=[0.0, 0.0, 0.0],
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
        source_finger_root_xform,
    )


def clone_envs(source_env_path, num_envs, spacing):
    cloner = GridCloner(spacing=spacing)
    env_paths = cloner.generate_paths(f"{ENVS_ROOT_PATH}/env", num_envs)
    cloner.clone(
        source_prim_path=source_env_path,
        prim_paths=env_paths,
        replicate_physics=True,
        base_env_path=ENVS_ROOT_PATH,
    )
    return cloner, env_paths


def as_numpy(values):
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def sample_planar_positions(rng, count, area_min, area_max, z_height):
    xy_positions = rng.uniform(area_min, area_max, size=(count, 2))
    z_positions = np.full((count, 1), z_height)
    return np.concatenate([xy_positions, z_positions], axis=1)


def sample_yaw_orientations(rng, count, yaw_range):
    yaws = rng.uniform(yaw_range[0], yaw_range[1], size=count)
    orientations = np.zeros((count, 4), dtype=np.float32)
    orientations[:, 0] = np.cos(yaws * 0.5)
    orientations[:, 3] = np.sin(yaws * 0.5)
    return orientations, yaws


def randomize_object_poses(env_roots, l_objects, cylinder_objects, seed=None):
    rng = np.random.default_rng(seed)

    env_positions, _ = env_roots.get_world_poses()
    env_positions = as_numpy(env_positions)
    num_envs = len(env_positions)

    l_local_positions = sample_planar_positions(
        rng,
        num_envs,
        L_POSITION_AREA_MIN,
        L_POSITION_AREA_MAX,
        z_height=0.0,
    )
    cylinder_local_positions = sample_planar_positions(
        rng,
        num_envs,
        CYLINDER_POSITION_AREA_MIN,
        CYLINDER_POSITION_AREA_MAX,
        z_height=OBJECT_HEIGHT * 0.5,
    )

    l_orientations, l_yaws = sample_yaw_orientations(rng, num_envs, L_YAW_RANGE)
    cylinder_orientations, cylinder_yaws = sample_yaw_orientations(rng, num_envs, CYLINDER_YAW_RANGE)

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


def randomize_finger_properties(finger_articulations, finger_tip_links, seed=None):
    rng = np.random.default_rng(seed)
    num_fingers = len(finger_articulations)
    num_dofs = finger_articulations.num_dofs

    stiffnesses = rng.uniform(
        DRIVE_STIFFNESS_RANGE[0],
        DRIVE_STIFFNESS_RANGE[1],
        size=(num_fingers, num_dofs),
    ).astype(np.float32)
    dampings = rng.uniform(
        DRIVE_DAMPING_RANGE[0],
        DRIVE_DAMPING_RANGE[1],
        size=(num_fingers, num_dofs),
    ).astype(np.float32)
    fingertip_masses = rng.uniform(
        FINGERTIP_MASS_RANGE[0],
        FINGERTIP_MASS_RANGE[1],
        size=(num_fingers, 1),
    ).astype(np.float32)

    finger_tip_links.set_masses(fingertip_masses)

    return {
        "stiffnesses": stiffnesses,
        "dampings": dampings,
        "fingertip_masses": fingertip_masses,
    }


#! need to step the app in async mode as the script editor is already in this mode
#! so app_utils.update_app(steps=DELAY_TO_SETTLE) gives erros
def move_fingers_to(finger_articulations, xyz, num_steps=120):
    dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
    finger_articulations.set_dof_position_targets(
        positions=list(xyz),
        dof_indices=dof_indices,
    )
    return run_coroutine(settle_scene_async(num_steps))


async def settle_scene_async(
    steps,
    finger_articulations=None,
    finger_target=None,
    finger_properties=None,
):
    app_utils.play()
    await app_utils.update_app_async(steps=1)
    if finger_articulations is not None:
        if finger_properties is None:
            configure_drives(finger_articulations)
        else:
            configure_drives(
                finger_articulations,
                stiffnesses=finger_properties["stiffnesses"],
                dampings=finger_properties["dampings"],
            )
        if finger_target is not None:
            dof_indices = finger_articulations.get_dof_indices(finger_articulations.dof_names)
            finger_articulations.set_dof_position_targets(
                positions=list(finger_target),
                dof_indices=dof_indices,
            )
    if steps > 1:
        await app_utils.update_app_async(steps=steps - 1)
    print(f"Settled scene for {steps} app update steps.")


#! run_coroutine is needed as async function cannot be called directly from the
#! main(). run_coroutine() tells isaac to run the async task safely through Kit's
#! event loop
def schedule_settle_scene(
    steps,
    finger_articulations=None,
    finger_target=None,
    finger_properties=None,
):
    return run_coroutine(
        settle_scene_async(steps, finger_articulations, finger_target, finger_properties)
    )


def main():
    clear_previous_handles()
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    ground_plane = GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

    distant_light = DistantLight("/World/DistantLight")
    distant_light.set_intensities(500)

    l_material = PreviewSurfaceMaterial("/VisualMaterials/l_object_blue")
    l_material.set_input_values("diffuseColor", [0.1, 0.35, 1.0])

    cylinder_material = PreviewSurfaceMaterial("/VisualMaterials/cylinder_orange")
    cylinder_material.set_input_values("diffuseColor", [1.0, 0.45, 0.05])


    #! create a source env with L-shaped object, cylinder, and finger robot
    (
        source_env,
        l_parts,
        l_geometry,
        l_transform,
        l_object,
        cylinder_shape,
        cylinder_geometry,
        cylinder_object,
        source_finger_root_xform,
    ) = create_source_env(SOURCE_ENV_PATH, l_material, cylinder_material)

    #! replicate the source env here
    cloner, env_paths = clone_envs(
        source_env_path=SOURCE_ENV_PATH,
        num_envs=NUM_ENVS,
        spacing=ENV_SPACING,
    )

    env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
    l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
    cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
    finger_articulations = Articulation(paths=FINGER_ROOT_PATTERN)
    finger_tip_links = RigidPrim(paths=FINGER_TIP_LINK_PATTERN)

    randomized_poses = None
    randomized_finger_properties = None
    if RANDOMIZE_ON_START:
        randomized_poses = randomize_object_poses(
            env_roots=env_roots,
            l_objects=l_objects,
            cylinder_objects=cylinder_objects,
            seed=RANDOM_SEED,
        )
        randomized_finger_properties = randomize_finger_properties(
            finger_articulations=finger_articulations,
            finger_tip_links=finger_tip_links,
            seed=RANDOM_SEED + 1,
        )
    
    #! asynchronously forward the physics to resolve overlap after randomization
    settle_task = schedule_settle_scene(
        DELAY_TO_SETTLE,
        finger_articulations=finger_articulations,
        finger_target=DEFAULT_XYZ,
        finger_properties=randomized_finger_properties,
    )

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=[2.8, 2.8, 2.0],
        target=[0.0, 0.0, 0.45],
    )


    #! This turns local variables from main() into global variables, so after 
    #! main() finishes, these objects are still accessible. This lets us interact
    #! with these objects after running the scripts
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
            "source_env": source_env,
            "env_roots": env_roots,
            "env_paths": env_paths,
            "cloner": cloner,
            "randomized_poses": randomized_poses,
            "settle_task": settle_task,
            "source_finger_root_xform": source_finger_root_xform,
            "finger_articulations": finger_articulations,
            "finger_tip_links": finger_tip_links,
            "move_fingers_to": move_fingers_to,
            "randomized_finger_properties": randomized_finger_properties,
        }
    )


main()


#! ---------------------------------------------------------------------------
#! The workflow convention here
#! ---------------------------------------------------------------------------
#! one USD stage
#! └── many envs
#!     ├── env_0/FingerRobot  ← articulation instance 0
#!     ├── env_1/FingerRobot  ← articulation instance 1
#!     ├── env_2/FingerRobot  ← articulation instance 2
#! ---------------------------------------------------------------------------
