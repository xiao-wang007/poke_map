"""
Training script for Spatial Actor-Critic with FiLM-conditioned Q-head.

Runs inside the Isaac Sim Script Editor.  Before executing:
  1. Run ``scene_setup_articulated_vectorized.py``
  2. Run this file (may take a few seconds to import)
  3. Call ``train()`` or ``main()``

Architecture
------------
  - SpatialActorCritic with shared U-Net backbone
  - FiLM-conditioned Q-head: action params modulate features before Q
  - Two losses: TD (Huber) for Q-head, DPG for param-head
  - Replay buffer, ε-greedy pixel + Gaussian param noise
  - Target network with soft (polyak) updates
"""

from __future__ import annotations

import __main__
import gc
from collections import deque
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

#* -- project paths --------------------------------------------------------
def find_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path.cwd(),
        Path("/home/xiao/0_codes/poke_map"),
    ]
    for candidate in candidates:
        if ((candidate / "config.yaml").exists() and
            (candidate / "nn").exists() and
            (candidate / "vision").exists() and
            (candidate / "env").exists()):
            return candidate
    raise RuntimeError("Could not find poke_map project root.")


PROJECT_ROOT = find_project_root()
for p in (PROJECT_ROOT, PROJECT_ROOT / "nn", PROJECT_ROOT / "env",
          PROJECT_ROOT / "vision"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

#* -- Isaac Sim ------------------------------------------------------------
try:
    from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from omni.kit.async_engine import run_coroutine
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    _HAS_ISAAC = True
except ModuleNotFoundError:
    Gf = None
    Sdf = None
    UsdGeom = None
    UsdShade = None
    stage_utils = None
    _HAS_ISAAC = False

#* -- project modules ------------------------------------------------------
from config_loader import load_config
from vision.camera import (
    RESOLUTION,
    get_camera_intrinsics,
    get_object_poses_vectorized,
    build_vision_observation,
    pixel_to_world,
)
from nn.networks import SpatialActorCritic
from env.make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives
from env.scene_setup_articulated_vectorized import (
    l_object_cylinder_clearance,
    randomize_object_poses,
)
from env.quintic_poly import (
    quintic_poly_query,
    quintic_poly_derivative,
    quintic_poly_second_derivative,
    compute_strike_params,
)
from rl.rewards import compute_rewards
from rl.termination import EpisodeTermination
from rl.replay_buffer import ReplayBuffer
from rl.prior import heuristic_poke_actions, select_action

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]
WORKSPACE_CONFIG = CONFIG["workspace"]
OBJECT_CONFIG = CONFIG["objects"]

OBJECT_HEIGHT = SCENE_CONFIG["object_height"]
L_ARM_LENGTH = SCENE_CONFIG["l_arm_length"]
L_THICKNESS = SCENE_CONFIG["l_thickness"]
CYLINDER_RADIUS = SCENE_CONFIG["cylinder_radius"]
ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
FINGER_LOCAL_PATH = FINGER_CONFIG["local_root_path"]
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_PATH}"
FINGER_TIP_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
NUM_ENVS = SCENE_CONFIG["num_envs"]
TRAIN_CONFIG = CONFIG["training"]


#* ================================================================
#*  Hyper-parameters  (sourced from config.yaml → training)
#* ================================================================

GAMMA: float = TRAIN_CONFIG["gamma"]
LR: float = TRAIN_CONFIG["lr"]
BATCH_SIZE: int = TRAIN_CONFIG["batch_size"]
BUFFER_CAPACITY: int = TRAIN_CONFIG["buffer_capacity"]
EPS_START: float = TRAIN_CONFIG["eps_start"]
EPS_END: float = TRAIN_CONFIG["eps_end"]
EPS_DECAY: int = TRAIN_CONFIG["eps_decay_episodes"]
SIGMA_START: float = TRAIN_CONFIG["sigma_start"]
SIGMA_END: float = TRAIN_CONFIG["sigma_end"]
TAU: float = TRAIN_CONFIG["tau"]
MAX_STEPS: int = TRAIN_CONFIG["max_steps_per_episode"]
VELOCITY_MAX: float = TRAIN_CONFIG["velocity_max"]
SPEED_XY: float = TRAIN_CONFIG["reposition_speed_xy"]
SPEED_Z: float = TRAIN_CONFIG["reposition_speed_z"]
L_STRIKE: float = TRAIN_CONFIG["l_strike"]
L_STRIKE_MIN: float = TRAIN_CONFIG["l_strike_min"]
STRIKE_CONTROL_FREQ: int = TRAIN_CONFIG["strike_control_freq"]
STRIKE_DT: float = 1.0 / STRIKE_CONTROL_FREQ
STRIKE_MAX_DURATION: float = TRAIN_CONFIG["strike_max_duration"]
STRIKE_MIN_STEPS: int = TRAIN_CONFIG["strike_min_steps"]
POKE_SETTLE_STEPS: int = TRAIN_CONFIG["poke_settle_steps"]
FINGERTIP_RADIUS: float = FINGER_CONFIG["sphere_radius"]
POKE_Z: float = OBJECT_HEIGHT * 0.5
SAFE_Z: float = TRAIN_CONFIG["safe_z"]
TARGET_MIN_CLEARANCE: float = OBJECT_CONFIG["min_initial_clearance"]
TARGET_RESAMPLE_ATTEMPTS: int = OBJECT_CONFIG["pose_resample_attempts"]

# yaw training controls
YAW_TARGET_MODE: str = TRAIN_CONFIG["yaw_target_mode"]
C_YAW: float = TRAIN_CONFIG["c_yaw"]
YAW_REWARD_ENABLED: bool = TRAIN_CONFIG["yaw_reward_enabled"]
YAW_SUCCESS_ENABLED: bool = TRAIN_CONFIG["yaw_success_enabled"]
YAW_CURRICULUM_START: int = TRAIN_CONFIG["yaw_curriculum_start_ep"]
YAW_CURRICULUM_END: int = TRAIN_CONFIG["yaw_curriculum_end_ep"]
YAW_FULL_RANGE: float = np.pi

TRAIN_AFTER: int = TRAIN_CONFIG["train_after"]
TRAIN_EVERY: int = TRAIN_CONFIG["train_every"]
GRAD_UPDATES_PER_STEP: int = TRAIN_CONFIG["grad_updates_per_step"]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_EVERY: int = TRAIN_CONFIG["checkpoint_every"]
CHECKPOINT_KEEP: int = TRAIN_CONFIG["checkpoint_keep"]
SHOW_PLANE_OVERLAY: bool = TRAIN_CONFIG["show_plane_overlay"]
PLANE_OVERLAY_THICKNESS: float = TRAIN_CONFIG["plane_overlay_thickness"]
PLANE_OVERLAY_OPACITY: float = TRAIN_CONFIG["plane_overlay_opacity"]
SHOW_TARGET_OVERLAY: bool = TRAIN_CONFIG["show_target_overlay"]
TARGET_OVERLAY_Z_OFFSET: float = TRAIN_CONFIG["target_overlay_z_offset"]
TARGET_OVERLAY_THICKNESS: float = TRAIN_CONFIG["target_overlay_thickness"]
TARGET_OVERLAY_OPACITY: float = TRAIN_CONFIG["target_overlay_opacity"]
SHOW_ACTION_OVERLAY: bool = TRAIN_CONFIG["show_action_overlay"]
ACTION_OVERLAY_Z_OFFSET: float = TRAIN_CONFIG["action_overlay_z_offset"]


#* ================================================================
#*  Isaac Sim helpers
#* ================================================================

def as_numpy(values):
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


async def _step_physics(steps: int):
    await app_utils.update_app_async(steps=steps)


async def ensure_sim_running_async():
    app_utils.play()
    await _step_physics(1)



#! this gets the handles to all parallel envs, given my vectorized envs are
#! setup in the other script.
def get_finger_handles() -> tuple[Articulation, RigidPrim, XformPrim]:
    """Return finger articulations, tip links, and env roots."""
    fingers = Articulation(paths=FINGER_ROOT_PATTERN)
    tips = RigidPrim(paths=FINGER_TIP_PATTERN)
    env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
    return fingers, tips, env_roots


#* ================================================================
#*  Target sampling
#* ================================================================

def sample_target_poses(
    rng: np.random.Generator,
    num_envs: int,
    workspace_range: tuple = (-0.20, 0.20),
    l_yaws: np.ndarray | None = None,
    min_clearance: float = TARGET_MIN_CLEARANCE,
    resample_attempts: int = TARGET_RESAMPLE_ATTEMPTS,
) -> dict[str, np.ndarray]:
    """Random 2-D target on the table for each object and each env.

    Reject samples where the cylinder target overlaps the L target footprint.
    Returns dict: obj_name → (num_envs, 3)  (z = OBJECT_HEIGHT * 0.5)
    """
    z = np.full((num_envs,), OBJECT_HEIGHT * 0.5, dtype=np.float32)
    l_yaws = np.zeros(num_envs, dtype=np.float32) if l_yaws is None else np.asarray(l_yaws)
    lx = np.zeros(num_envs, dtype=np.float32)
    ly = np.zeros(num_envs, dtype=np.float32)
    cx = np.zeros(num_envs, dtype=np.float32)
    cy = np.zeros(num_envs, dtype=np.float32)

    for env_idx in range(num_envs):
        best_sample = None
        best_clearance = -float("inf")
        for _ in range(max(1, resample_attempts)):
            sample_l = rng.uniform(workspace_range[0], workspace_range[1], size=(2,)).astype(np.float32)
            sample_c = rng.uniform(workspace_range[0], workspace_range[1], size=(2,)).astype(np.float32)
            clearance = l_object_cylinder_clearance(
                sample_l, float(l_yaws[env_idx]), sample_c
            )
            if clearance > best_clearance:
                best_clearance = clearance
                best_sample = (sample_l, sample_c)
            if clearance >= min_clearance:
                break

        sample_l, sample_c = best_sample
        lx[env_idx], ly[env_idx] = sample_l
        cx[env_idx], cy[env_idx] = sample_c

    return {"LObject":  np.stack([lx, ly, z], axis=-1),
            "Cylinder": np.stack([cx, cy, z], axis=-1)}


def compute_adaptive_strike_lengths(
    pixel_ij: np.ndarray,
    poses: dict,
    targets: dict[str, np.ndarray],
    env_root_pos: np.ndarray,
    K: np.ndarray,
    active: np.ndarray | None = None,
    min_length: float = L_STRIKE_MIN,
    max_length: float = L_STRIKE,
) -> np.ndarray:
    """Shrink strike travel when the selected object is close to its target."""
    num_envs = pixel_ij.shape[0]
    lengths = np.full(num_envs, max_length, dtype=np.float32)
    env_offsets = env_root_pos[:, :2]

    contact_xy = np.zeros((num_envs, 2), dtype=np.float32)
    for b in range(num_envs):
        contact_xy[b] = pixel_to_world(tuple(pixel_ij[b]), K)[:2]

    object_names = list(targets.keys())
    object_local_xy = []
    object_target_dist = []
    for obj_name in object_names:
        local_xy = poses[obj_name][0][:, :2].copy() - env_offsets
        object_local_xy.append(local_xy)
        object_target_dist.append(
            np.linalg.norm(local_xy - targets[obj_name][:, :2], axis=1)
        )

    object_local_xy = np.stack(object_local_xy, axis=0)        # (O, B, 2)
    object_target_dist = np.stack(object_target_dist, axis=0)  # (O, B)
    contact_dist = np.linalg.norm(
        object_local_xy - contact_xy[None, :, :], axis=2
    )
    selected_obj_idx = np.argmin(contact_dist, axis=0)
    env_idx = np.arange(num_envs)
    selected_target_dist = object_target_dist[selected_obj_idx, env_idx]

    adaptive_lengths = np.clip(
        2.0 * selected_target_dist,
        min_length,
        max_length,
    ).astype(np.float32)
    if active is None:
        lengths = adaptive_lengths
    else:
        lengths[active] = adaptive_lengths[active]
    return lengths


def min_realizable_contact_velocity_np(strike_lengths: np.ndarray) -> np.ndarray:
    """Minimum midpoint/contact velocity allowed by strike_max_duration."""
    lengths = np.asarray(strike_lengths, dtype=np.float32)
    return (
        float(quintic_poly_derivative(0.5))
        * lengths
        / STRIKE_MAX_DURATION
    ).astype(np.float32)


def clamp_command_velocity_np(
    velocities: np.ndarray,
    strike_lengths: np.ndarray,
) -> np.ndarray:
    """Clamp nominal action velocity to what the quintic executor can realize."""
    v_min = min_realizable_contact_velocity_np(strike_lengths)
    return np.clip(velocities, v_min, VELOCITY_MAX).astype(np.float32)


def clamp_action_velocity_torch(
    params: torch.Tensor,
    strike_lengths: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent for critic/actor conditioning during training.

    The forward value is clamped to the executable range.  Gradients through
    the velocity channel are straight-through so an actor that fell below the
    physical minimum can still learn its way back up.
    """
    v_min = (
        float(quintic_poly_derivative(0.5))
        * strike_lengths.to(params.device, dtype=params.dtype)
        / STRIKE_MAX_DURATION
    )
    raw_velocity = params[:, 2:3]
    executable_velocity = torch.maximum(
        raw_velocity.clamp(min=0.0, max=VELOCITY_MAX),
        v_min,
    )
    velocity = raw_velocity + (executable_velocity - raw_velocity).detach()
    return torch.cat([params[:, :2], velocity], dim=1)


def _ensure_xform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _set_local_transform(path: str, translation, yaw: float = 0.0, scale=None):
    stage = stage_utils.get_current_stage(backend="usd")
    prim = _ensure_xform(stage, path)
    xform = UsdGeom.Xformable(prim)
    transform_attr = prim.GetAttribute("xformOp:transform")
    if transform_attr and transform_attr.IsValid():
        xform_op = UsdGeom.XformOp(transform_attr)
    else:
        xform_op = xform.AddTransformOp()
    xform.SetXformOpOrder([xform_op])

    mat = Gf.Matrix4d(1.0)
    if scale is not None:
        mat.SetScale(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
    rot = Gf.Matrix4d(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(float(yaw))),
                      Gf.Vec3d(0.0, 0.0, 0.0))
    trans = Gf.Matrix4d(1.0)
    trans.SetTranslate(Gf.Vec3d(float(translation[0]),
                                float(translation[1]),
                                float(translation[2])))
    xform_op.Set(mat * rot * trans)


def _set_display_color(prim, color_rgb, opacity: float | None = None):
    gprim = UsdGeom.Gprim(prim)
    color_attr = gprim.GetDisplayColorAttr()
    if not color_attr.IsValid():
        color_attr = gprim.CreateDisplayColorAttr()
    color_attr.Set([Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))])
    if opacity is not None:
        opacity_attr = gprim.GetDisplayOpacityAttr()
        if not opacity_attr.IsValid():
            opacity_attr = gprim.CreateDisplayOpacityAttr()
        opacity_attr.Set([float(opacity)])


def _remove_if_type_mismatch(stage, path: str, type_name: str):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.GetTypeName() != type_name:
        stage.RemovePrim(path)


def _define_colored_cube(stage, path: str, color_rgb, opacity: float | None = None):
    _remove_if_type_mismatch(stage, path, "Cube")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_display_color(cube.GetPrim(), color_rgb, opacity=opacity)
    return cube


def _define_colored_sphere(stage, path: str, radius: float, color_rgb,
                           opacity: float | None = None):
    _remove_if_type_mismatch(stage, path, "Sphere")
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    _set_display_color(sphere.GetPrim(), color_rgb, opacity=opacity)
    return sphere


def _transparent_material(stage, path: str, color_rgb, opacity: float):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _bind_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def update_target_overlay(targets_pos: dict[str, np.ndarray],
                          targets_ori: dict[str, np.ndarray]):
    """Draw non-physics target overlays under each env root."""
    if not SHOW_TARGET_OVERLAY or not _HAS_ISAAC:
        return

    stage = stage_utils.get_current_stage(backend="usd")
    l_target_mat = _transparent_material(
        stage, "/VisualMaterials/target_overlay_l_blue",
        (0.1, 0.35, 1.0), TARGET_OVERLAY_OPACITY
    )
    cyl_target_mat = _transparent_material(
        stage, "/VisualMaterials/target_overlay_cylinder_orange",
        (1.0, 0.45, 0.05), TARGET_OVERLAY_OPACITY
    )
    for env_idx in range(NUM_ENVS):
        overlay_root = f"{ENVS_ROOT_PATH}/env_{env_idx}/TargetOverlay"
        _ensure_xform(stage, overlay_root)

        # L target: blue transparent guide bars, parented under
        # an oriented target xform matching the rendered goal contour.
        l_path = f"{overlay_root}/LObjectTarget"
        _ensure_xform(stage, l_path)
        l_pos = targets_pos["LObject"][env_idx].copy()
        l_pos[2] = TARGET_OVERLAY_Z_OFFSET
        l_quat = targets_ori["LObject"][env_idx]
        l_yaw = 2.0 * np.arctan2(l_quat[3], l_quat[0])
        _set_local_transform(l_path, l_pos, yaw=l_yaw)

        l_parts = {
            "VerticalLeg": (
                [L_THICKNESS * 0.5, L_ARM_LENGTH * 0.5, TARGET_OVERLAY_THICKNESS * 0.5],
                [L_THICKNESS, L_ARM_LENGTH, TARGET_OVERLAY_THICKNESS],
            ),
            "HorizontalLeg": (
                [L_ARM_LENGTH * 0.5, L_THICKNESS * 0.5, TARGET_OVERLAY_THICKNESS * 0.5],
                [L_ARM_LENGTH, L_THICKNESS, TARGET_OVERLAY_THICKNESS],
            ),
        }
        for part_name, (center, scale) in l_parts.items():
            part_path = f"{l_path}/{part_name}"
            cube = _define_colored_cube(
                stage, part_path, (0.1, 0.35, 1.0), opacity=TARGET_OVERLAY_OPACITY
            )
            _bind_material(cube.GetPrim(), l_target_mat)
            _set_local_transform(part_path, center, scale=scale)

        # Cylinder target: orange transparent guide disk at target center.
        cyl_path = f"{overlay_root}/CylinderTarget"
        _remove_if_type_mismatch(stage, cyl_path, "Cylinder")
        cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
        cyl.CreateRadiusAttr(CYLINDER_RADIUS)
        cyl.CreateHeightAttr(TARGET_OVERLAY_THICKNESS)
        _set_display_color(
            cyl.GetPrim(), (1.0, 0.45, 0.05), opacity=TARGET_OVERLAY_OPACITY
        )
        _bind_material(cyl.GetPrim(), cyl_target_mat)
        cyl_pos = targets_pos["Cylinder"][env_idx].copy()
        cyl_pos[2] = TARGET_OVERLAY_Z_OFFSET + TARGET_OVERLAY_THICKNESS * 0.5
        _set_local_transform(cyl_path, cyl_pos)


def update_plane_overlays(K: np.ndarray):
    """Draw workspace and camera-view footprints under each env root."""
    if not SHOW_PLANE_OVERLAY or not _HAS_ISAAC:
        return

    stage = stage_utils.get_current_stage(backend="usd")
    workspace_mat = _transparent_material(
        stage, "/VisualMaterials/workspace_overlay_green",
        (0.0, 1.0, 0.1), PLANE_OVERLAY_OPACITY
    )
    camera_mat = _transparent_material(
        stage, "/VisualMaterials/camera_view_overlay_gray",
        (0.45, 0.45, 0.45), PLANE_OVERLAY_OPACITY
    )

    wx_min = float(WORKSPACE_CONFIG.get("x_min", -0.30))
    wx_max = float(WORKSPACE_CONFIG.get("x_max", 0.30))
    wy_min = float(WORKSPACE_CONFIG.get("y_min", -0.30))
    wy_max = float(WORKSPACE_CONFIG.get("y_max", 0.30))
    workspace_center = [(wx_min + wx_max) * 0.5, (wy_min + wy_max) * 0.5, 0.002]
    workspace_scale = [wx_max - wx_min, wy_max - wy_min, PLANE_OVERLAY_THICKNESS]

    H, W = RESOLUTION
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    camera_x_min = (0.0 - cx) / fx
    camera_x_max = ((W - 1.0) - cx) / fx
    camera_y_min = (0.0 - cy) / fy
    camera_y_max = ((H - 1.0) - cy) / fy
    camera_center = [
        (camera_x_min + camera_x_max) * 0.5,
        (camera_y_min + camera_y_max) * 0.5,
        0.001,
    ]
    camera_scale = [
        camera_x_max - camera_x_min,
        camera_y_max - camera_y_min,
        PLANE_OVERLAY_THICKNESS,
    ]

    for env_idx in range(NUM_ENVS):
        overlay_root = f"{ENVS_ROOT_PATH}/env_{env_idx}/PlaneOverlay"
        _ensure_xform(stage, overlay_root)

        camera_path = f"{overlay_root}/CameraView"
        camera_cube = _define_colored_cube(
            stage, camera_path, (0.45, 0.45, 0.45), opacity=PLANE_OVERLAY_OPACITY
        )
        _bind_material(camera_cube.GetPrim(), camera_mat)
        _set_local_transform(camera_path, camera_center, scale=camera_scale)

        workspace_path = f"{overlay_root}/Workspace"
        workspace_cube = _define_colored_cube(
            stage, workspace_path, (0.0, 1.0, 0.1), opacity=PLANE_OVERLAY_OPACITY
        )
        _bind_material(workspace_cube.GetPrim(), workspace_mat)
        _set_local_transform(workspace_path, workspace_center, scale=workspace_scale)


def update_action_overlay(pixel_ij: np.ndarray,
                          d_xy: np.ndarray,
                          delta_d: np.ndarray,
                          K: np.ndarray,
                          active: np.ndarray | None = None,
                          strike_lengths: np.ndarray | None = None,
                          is_policy_action: np.ndarray | None = None):
    """Draw the selected side-poke action for each env."""
    if not SHOW_ACTION_OVERLAY or not _HAS_ISAAC:
        return

    stage = stage_utils.get_current_stage(backend="usd")
    dir_norm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm < 1e-8, 1.0, dir_norm)
    dirs = d_xy / dir_norm
    strike_lengths = (
        np.full(NUM_ENVS, L_STRIKE, dtype=np.float32)
        if strike_lengths is None
        else np.asarray(strike_lengths, dtype=np.float32)
    )

    marker_radius = 0.01
    line_thickness = 0.004
    z = POKE_Z + ACTION_OVERLAY_Z_OFFSET

    for env_idx in range(NUM_ENVS):
        overlay_root = f"{ENVS_ROOT_PATH}/env_{env_idx}/ActionOverlay"
        _ensure_xform(stage, overlay_root)

        if active is not None and not bool(active[env_idx]):
            _set_local_transform(overlay_root, [0.0, 0.0, -10.0])
            continue
        _set_local_transform(overlay_root, [0.0, 0.0, 0.0])
        old_strike = stage.GetPrimAtPath(f"{overlay_root}/Strike")
        if old_strike.IsValid():
            stage.RemovePrim(f"{overlay_root}/Strike")

        world = pixel_to_world(tuple(pixel_ij[env_idx]), K)
        world_xy = world[:2]
        l_env = float(strike_lengths[env_idx])
        standoff_xy = world_xy - dirs[env_idx] * (l_env / 2.0)
        contact_xy = world_xy - dirs[env_idx] * FINGERTIP_RADIUS   # sphere touches surface
        strike_xy = world_xy + dirs[env_idx] * (l_env / 2.0)

        path_color = (
            (0.95, 0.35, 1.0)
            if is_policy_action is not None and bool(is_policy_action[env_idx])
            else (1.0, 0.85, 0.0)
        )
        points = {
            "Standoff": (standoff_xy, (0.0, 0.25, 1.0)),
            "Contact": (world_xy, (1.0, 0.0, 0.0)),
            "Strike": (strike_xy, (1.0, 1.0, 1.0)),
        }
        for name, (xy, color) in points.items():
            path = f"{overlay_root}/{name}"
            _define_colored_sphere(stage, path, marker_radius, color)
            _set_local_transform(path, [xy[0], xy[1], z])

        line_vec = strike_xy - standoff_xy
        line_len = float(np.linalg.norm(line_vec))
        if line_len > 1e-6:
            line_center = 0.5 * (standoff_xy + strike_xy)
            line_yaw = float(np.arctan2(line_vec[1], line_vec[0]))
            line_path = f"{overlay_root}/Path"
            _define_colored_cube(stage, line_path, path_color)
            _set_local_transform(
                line_path,
                [line_center[0], line_center[1], z],
                yaw=line_yaw,
                scale=[line_len, line_thickness, line_thickness],
            )


def contact_velocity_from_command(
    v_mid: np.ndarray,
    strike_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Actual midpoint/contact speed after strike-duration clamping."""
    values = np.asarray(v_mid, dtype=np.float32)
    lengths = (
        np.full_like(values, L_STRIKE, dtype=np.float32)
        if strike_lengths is None
        else np.asarray(strike_lengths, dtype=np.float32)
    )
    out = np.zeros_like(values, dtype=np.float32)
    for idx, v in np.ndenumerate(values):
        T, _ = compute_strike_params(
            float(v), float(lengths[idx]), STRIKE_DT, STRIKE_MAX_DURATION, STRIKE_MIN_STEPS
        )
        out[idx] = (lengths[idx] / T) * quintic_poly_derivative(0.5)
    return out



def _yaws_to_quats(yaws: np.ndarray) -> np.ndarray:
    """Convert yaw angles to quaternions [w, x, y, z] (rotation about Z)."""
    half = yaws * 0.5
    out = np.zeros((yaws.shape[0], 4), dtype=np.float32)
    out[:, 0] = np.cos(half)
    out[:, 3] = np.sin(half)
    return out


def yaw_target_half_range(episode: int) -> float:
    """Return the sampled target-yaw half range for this episode."""
    if YAW_TARGET_MODE == "preserve":
        return 0.0
    if YAW_TARGET_MODE == "fixed":
        return YAW_FULL_RANGE
    if YAW_TARGET_MODE == "curriculum":
        frac = np.clip((episode - YAW_CURRICULUM_START) /
                       max(1, YAW_CURRICULUM_END - YAW_CURRICULUM_START),
                       0.0, 1.0)
        return float(frac * YAW_FULL_RANGE)
    raise ValueError(
        f"Unknown YAW_TARGET_MODE={YAW_TARGET_MODE!r}; "
        "use 'preserve', 'curriculum', or 'fixed'."
    )


#* ================================================================
#*  Environment step — articulated finger (batched, all envs)
#* ================================================================

async def env_step_async(
    fingers: Articulation,
    pixel_ij: np.ndarray,        # (B, 2)
    d_xy: np.ndarray,            # (B, 2)
    delta_d: np.ndarray,         # (B,)  — velocity magnitude from policy (v_mid)
    K: np.ndarray,
    active: np.ndarray | None = None,  # (B,) bool — which envs to control
    fingertip_masses: np.ndarray | None = None,  # (B,) kg per env (randomized or default)
    strike_lengths: np.ndarray | None = None,  # (B,) adaptive total strike travel
) -> tuple[dict, np.ndarray]:
    """Execute strikes with the prismatic XYZ finger robot.

    Phases:
      0. lift Z to safe height (constant velocity)
      1. move XY to standoff at safe height (constant velocity)
      2. drop Z to poke height (constant velocity)
      3. quintic strike — smooth trajectory over adaptive L; contact at midpoint;
         position-offset feedforward via q_target += m*a / kp
      4. lift Z to safe height (constant velocity, XY stays)
      5. settle objects

    Inactive envs (where active=False) are frozen at their current position
    throughout all phases — no strike, no movement.
    """
    B = len(fingers)
    dof_indices = fingers.get_dof_indices(fingers.dof_names)

    #* ── convert pixel → world XY ───────────────────────────────────
    world_xy = np.zeros((B, 2), dtype=np.float32)
    for b in range(B):
        w = pixel_to_world(tuple(pixel_ij[b]), K)
        world_xy[b] = w[:2]

    #* ── normalise direction ────────────────────────────────────────
    dir_norm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm < 1e-8, 1.0, dir_norm)
    dirs = d_xy / dir_norm
    L_arr = (
        np.full(B, L_STRIKE, dtype=np.float32)
        if strike_lengths is None
        else np.clip(np.asarray(strike_lengths, dtype=np.float32), L_STRIKE_MIN, L_STRIKE)
    )

    xy_low  = np.array(LIMIT_LOWER[:2], dtype=np.float32)
    xy_high = np.array(LIMIT_UPPER[:2], dtype=np.float32)

    #* ── standoff / strike geometry ─────────────────────────────────
    #* world_xy is the contour contact point.  Quintic trajectory places the
    #* contact at the time-midpoint → standoff 0.5L behind, strike 0.5L ahead.
    #* L_arr is the total travel: 0.5L behind contact, 0.5L ahead.
    standoff_xy = world_xy - dirs * (L_arr[:, None] / 2.0)
    strike_xy   = world_xy + dirs * (L_arr[:, None] / 2.0)

    #* ── snap initial positions (inactive envs hold these throughout) ──
    try:
        q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)
    except AssertionError:
        app_utils.play()
        await _step_physics(2)
        q_cur = as_numpy(fingers.get_dof_positions()).copy()

    active_mask = active if active is not None else np.ones(B, dtype=bool)

    def _hold_inactive(targets: np.ndarray):
        targets[~active_mask] = q_cur[~active_mask]

    async def _constant_velocity_phase(targets: np.ndarray, speed: float):
        """Move to *targets* at constant speed with position+velocity trajectory.

        Each env linearly interpolates from its current position to *targets*
        at *speed* (m/s).  Inactive envs are held throughout.
        """
        q_start_phase = as_numpy(fingers.get_dof_positions()).copy()
        displacement = targets - q_start_phase
        distance = np.linalg.norm(displacement, axis=1)
        n_needed = np.maximum(1, (distance / np.maximum(speed, 0.01) / (1.0 / 60)).astype(int) + 1)
        global_n_phase = int(n_needed[active_mask].max()) if active_mask.any() else 1

        for k in range(global_n_phase):
            still = active_mask & (k < n_needed)

            q_des_phase = q_start_phase.copy()
            v_des_phase = np.zeros((B, 3), dtype=np.float32)

            if still.any():
                alpha = (k + 1) / n_needed[still]
                q_des_phase[still] = q_start_phase[still] + alpha[:, None] * displacement[still]
                v_norm = np.linalg.norm(displacement[still], axis=1, keepdims=True) + 1e-8
                v_des_phase[still] = displacement[still] * (speed / v_norm)

            finished = active_mask & (k >= n_needed)
            if finished.any():
                q_des_phase[finished] = targets[finished]

            q_des_phase[:, :2] = np.clip(q_des_phase[:, :2], xy_low, xy_high)
            fingers.set_dof_position_targets(positions=q_des_phase.tolist(), dof_indices=dof_indices)
            fingers.set_dof_velocity_targets(velocities=v_des_phase.tolist(), dof_indices=dof_indices)
            await _step_physics(1)

    #* ── Phase 0: lift to side-poke height above current XY ──────────
    q0 = q_cur.copy()
    q0[:, 2] = SAFE_Z
    _hold_inactive(q0)
    await _constant_velocity_phase(q0, SPEED_Z)

    #* ── Phase 1: move to standoff at side-poke height ──────────────
    q1 = np.zeros((B, 3), dtype=np.float32)
    q1[:, :2] = np.clip(standoff_xy, xy_low, xy_high)
    q1[:, 2] = SAFE_Z
    _hold_inactive(q1)
    await _constant_velocity_phase(q1, SPEED_XY)

    #* ── Phase 2: drop down to poke height ──────────────────────────
    q2 = q1.copy()
    q2[:, 2] = POKE_Z
    _hold_inactive(q2)
    await _constant_velocity_phase(q2, SPEED_Z)

    #* ── Phase 3: quintic strike trajectory ─────────────────────────
    #* Capture position after Phase 2 as trajectory origin.
    #* Each active env follows a quintic polynomial from standoff → through
    #* contour at midpoint → strike_xy, with per-env timing set by v_mid.
    q_start = as_numpy(fingers.get_dof_positions()).copy()  # (B, 3)

    # Per-env trajectory parameters
    active_mask = active if active is not None else np.ones(B, dtype=bool)
    n_steps = np.ones(B, dtype=int)
    T_arr = np.ones(B, dtype=np.float32)

    for b in range(B):
        if active_mask[b]:
            T_arr[b], n_steps[b] = compute_strike_params(
                delta_d[b], L_arr[b], STRIKE_DT, STRIKE_MAX_DURATION, STRIKE_MIN_STEPS)

    global_n = int(n_steps[active_mask].max()) if active_mask.any() else 1

    # Mass & stiffness for feedforward (position-offset: q_target += F_ff / kp)
    if fingertip_masses is not None:
        m_ff = np.asarray(fingertip_masses, dtype=np.float32).ravel()
        m_ff = np.maximum(m_ff, 0.01)
    else:
        m_ff = np.full(B, FINGER_CONFIG["default_fingertip_mass"], dtype=np.float32)
    kp_default = float(FINGER_CONFIG["drive_stiffness"])

    for k in range(global_n):
        still_running = active_mask & (k < n_steps)

        q_des = q_start.copy()  # default: hold at Phase 2 position
        v_des = np.zeros((B, 3), dtype=np.float32)

        if still_running.any():
            u = (k + 1) / n_steps[still_running]  # normalised time [1/N, 1]

            s_nd = quintic_poly_query(u)                   # position scale [0, 1]
            v_nd = quintic_poly_derivative(u)               # velocity derivative
            a_nd = quintic_poly_second_derivative(u)        # acceleration derivative

            # Position target along direction
            L_running = L_arr[still_running]
            displacement = L_running * s_nd
            q_des[still_running, :2] = (
                q_start[still_running, :2]
                + dirs[still_running] * displacement[:, np.newaxis]
            )

            # Velocity target
            velocity = (L_running / T_arr[still_running]) * v_nd
            v_des[still_running, :2] = dirs[still_running] * velocity[:, np.newaxis]

            # Feedforward: F_ff = m * a, injected via position offset
            acceleration = (L_running / T_arr[still_running] ** 2) * a_nd
            F_ff = m_ff[still_running] * acceleration
            ff_offset = F_ff / kp_default
            q_des[still_running, :2] += dirs[still_running] * ff_offset[:, np.newaxis]

            # Z stays at poke height
            q_des[still_running, 2] = POKE_Z

        # Hold already-finished envs at strike endpoint
        finished = active_mask & (k >= n_steps)
        if finished.any():
            q_des[finished, :2] = np.clip(strike_xy[finished], xy_low, xy_high)
            q_des[finished, 2] = POKE_Z

        # Clip XY to joint limits
        q_des[:, :2] = np.clip(q_des[:, :2], xy_low, xy_high)

        fingers.set_dof_position_targets(positions=q_des.tolist(), dof_indices=dof_indices)
        fingers.set_dof_velocity_targets(velocities=v_des.tolist(), dof_indices=dof_indices)
        await _step_physics(1)

    #* ── Phase 4: lift straight up — keep XY, raise Z to safe height ──
    q4 = as_numpy(fingers.get_dof_positions()).copy()
    q4[:, 2] = SAFE_Z
    _hold_inactive(q4)
    await _constant_velocity_phase(q4, SPEED_Z)

    #* ── Phase 5: settle briefly, then clear residual object velocities ─
    l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
    cyl_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")

    await _step_physics(POKE_SETTLE_STEPS)
    zero_vel = np.zeros((B, 3), dtype=np.float32)
    l_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_vel)
    cyl_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_vel)
    await _step_physics(1)

    #* ── query new object poses ─────────────────────────────────────
    poses, env_root_pos = get_object_poses_vectorized()
    return poses, env_root_pos


#* ================================================================
#*  Training step
#* ================================================================

def train_step(
    actor_critic: SpatialActorCritic,
    target_net: SpatialActorCritic,
    optimizer_q: torch.optim.Optimizer,
    optimizer_mu: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    gamma: float = GAMMA,
):
    """One DQN + DPG training step on a minibatch."""
    x       = batch["x"]
    pixel   = batch["pixel"]       # (B, 2)
    d_xy_st = batch["d_xy"]        # (B, 2)  stored (with noise)
    v_st    = batch["velocity"]      # (B, 1)  stored velocity (m/s)
    L_st    = batch["strike_length"]  # (B, 1)  executed strike travel (m)
    r       = batch["r"]           # (B, 1)
    x_next  = batch["x_next"]
    done    = batch["done"]        # (B, 1)
    mask_next = batch["mask_next"]  # (B, H, W) — pre-computed in sample()

    B = x.shape[0]
    batch_idx = torch.arange(B, device=x.device)
    actor_critic.train()
    target_net.eval()

    #* -- stored action params (with exploration noise, as executed) -----
    a_stored = clamp_action_velocity_torch(
        torch.cat([d_xy_st, v_st], dim=1),
        L_st,
    )  # (B, 3)

    #* =================================================================
    #*  LOSS 1 — Q-Loss (Huber TD)
    #* =================================================================

    #* target Q-value — per-pixel FiLM conditioned, same selection as acting
    with torch.no_grad():
        f_next, params_next, _ = target_net.features_and_params(x_next)
        pixel_next, q_max_next = target_net.greedy_pixel(
            f_next, params_next, mask_next)

        # zero future value for batch items with no valid next-state contour
        q_max_next[~mask_next.any(dim=(-2, -1))] = 0.0

        target = r.squeeze() + gamma * q_max_next * (~done.squeeze())

    #* online Q with stored action
    q_map = actor_critic.q_map_with_params(x, a_stored)
    q_val = q_map[batch_idx, 0, pixel[:, 0], pixel[:, 1]]   # (B,)

    L_Q = F.huber_loss(q_val, target)

    optimizer_q.zero_grad()
    L_Q.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 10.0)
    optimizer_q.step()

    #* =================================================================
    #*  LOSS 2 — DPG (Deterministic Policy Gradient)
    #* =================================================================

    #* freeze critic modules so DPG only updates actor
    for p in actor_critic.q_head.parameters():
        p.requires_grad = False
    for p in actor_critic.unet.parameters():
        p.requires_grad = False
    for p in actor_critic.film.parameters():
        p.requires_grad = False

    #* recompute encoder features (detached) and actor params
    f, params_act, _ = actor_critic.features_and_params(x)
    f_detached = f.detach()

    #* actor params at the stored pixel
    params_actor_at = actor_critic.params_at_pixel(params_act, pixel)  # (B, 3)
    params_actor_at = clamp_action_velocity_torch(params_actor_at, L_st)

    #* FiLM from actor params (forward only — film weights frozen, no gradients)
    film_gamma, film_beta = actor_critic.film(params_actor_at)
    f_mod_pg = film_gamma[:, :, None, None] * f_detached + film_beta[:, :, None, None]
    q_map_pg = actor_critic.q_head(f_mod_pg)   # q_head frozen, just forward
    q_val_pg = q_map_pg[batch_idx, 0, pixel[:, 0], pixel[:, 1]]  # (B,)

    L_mu = -q_val_pg.mean()

    optimizer_mu.zero_grad()
    L_mu.backward()

    # grad clip scoped to param_head only — film/q_head/unet are frozen
    torch.nn.utils.clip_grad_norm_(actor_critic.param_head.parameters(), 10.0)
    optimizer_mu.step()

    #* unfreeze
    for p in actor_critic.q_head.parameters():
        p.requires_grad = True
    for p in actor_critic.unet.parameters():
        p.requires_grad = True
    for p in actor_critic.film.parameters():
        p.requires_grad = True

    return L_Q.item(), L_mu.item()


#* ================================================================
#*  Soft update target network
#* ================================================================

def soft_update(target: SpatialActorCritic, online: SpatialActorCritic, tau: float = TAU):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


#* ================================================================
#*  Main training loop
#* ================================================================

class Trainer:
    """Orchestrates the training loop inside the Isaac Sim Script Editor."""

    def __init__(self):
        self.actor_critic = SpatialActorCritic(velocity_max=VELOCITY_MAX).to(DEVICE)
        self.target_net   = SpatialActorCritic(velocity_max=VELOCITY_MAX).to(DEVICE)
        soft_update(self.target_net, self.actor_critic, tau=1.0)  # copy

        #* param groups for separate optimizers
        q_params = list(self.actor_critic.q_head.parameters()) + \
                   list(self.actor_critic.film.parameters()) + \
                   list(self.actor_critic.unet.parameters())
        mu_params = list(self.actor_critic.param_head.parameters())

        self.optimizer_q = torch.optim.Adam(q_params, lr=LR)
        self.optimizer_mu = torch.optim.Adam(mu_params, lr=LR)

        self.buffer = ReplayBuffer()
        self.rng = np.random.default_rng(42)
        self.epsilon = EPS_START
        self.noise_std = SIGMA_START
        self.global_step = 0

        #* metrics
        self.ep_returns: deque[float] = deque(maxlen=100)
        self.ep_lengths: deque[int]   = deque(maxlen=100)

    async def setup(self):
        """Async init — starts sim, gets handles, configures drives."""
        await ensure_sim_running_async()
        self.refresh_scene_handles()

        finger_props = globals().get("randomized_finger_properties",
                        getattr(__main__, "randomized_finger_properties", None))
        if finger_props is not None:
            configure_drives(self.fingers,
                             stiffnesses=finger_props["stiffnesses"],
                             dampings=finger_props["dampings"])
            self._finger_masses = finger_props["fingertip_masses"].ravel().astype(np.float32)
        else:
            configure_drives(self.fingers)
            self._finger_masses = None

        self.K = get_camera_intrinsics()
        update_plane_overlays(self.K)

    def refresh_scene_handles(self):
        """Recreate Isaac wrappers after stage/timeline reinitialization."""
        self.fingers, self.tips, self.env_roots = get_finger_handles()

    def _decay_schedule(self, episode: int):
        frac = min(1.0, episode / EPS_DECAY)
        self.epsilon  = EPS_START + (EPS_END - EPS_START) * frac
        self.noise_std = SIGMA_START + (SIGMA_END - SIGMA_START) * frac


    #! Reset 
    async def _reset_episode(self, episode: int) -> tuple[torch.Tensor, list, torch.Tensor, dict]:
        """Randomise objects + targets, return first observation."""
        self.refresh_scene_handles()

        #* RigidPrim handles (use same patterns as scene setup)
        l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
        cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")

        try:
            randomize_object_poses(
                self.env_roots, l_objects, cylinder_objects,
                seed=None,
            )
        except AssertionError:
            await ensure_sim_running_async()
            self.refresh_scene_handles()
            l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
            cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")
            randomize_object_poses(
                self.env_roots, l_objects, cylinder_objects,
                seed=None,
            )
        # clear residual velocities from previous episode
        zero_vel = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        zero_ang = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        l_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_ang)
        cylinder_objects.set_velocities(linear_velocities=zero_vel, angular_velocities=zero_ang)
        await _step_physics(SCENE_CONFIG["settle_steps"])

        #* ── reposition finger at object level (centre of each env) ──
        dof_idx = self.fingers.get_dof_indices(self.fingers.dof_names)
        init_pos = np.tile([0.0, 0.0, POKE_Z], (NUM_ENVS, 1)).astype(np.float32)
        init_vel = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        self.fingers.set_dof_position_targets(positions=init_pos.tolist(), dof_indices=dof_idx)
        self.fingers.set_dof_velocity_targets(velocities=init_vel.tolist(), dof_indices=dof_idx)
        await _step_physics(30)  # let PD converge to centre

        # query current object poses for yaw curriculum
        poses_before, env_root_pos = get_object_poses_vectorized()

        # target orientations — Cylinder is identity, LObject uses curriculum
        targets_ori = {"Cylinder": np.zeros((NUM_ENVS, 4), dtype=np.float32)}
        targets_ori["Cylinder"][:, 0] = 1.0

        l_quats = poses_before["LObject"][1]
        l_yaws = 2.0 * np.arctan2(l_quats[:, 3], l_quats[:, 0])

        yaw_half_range = yaw_target_half_range(episode)
        l_target_yaws = l_yaws + self.rng.uniform(-yaw_half_range, yaw_half_range,
                                                   size=NUM_ENVS)
        targets_pos = sample_target_poses(
            self.rng, NUM_ENVS, l_yaws=l_target_yaws.astype(np.float32)
        )
        targets_ori["LObject"] = _yaws_to_quats(l_target_yaws.astype(np.float32))
        update_target_overlay(targets_pos, targets_ori)

        # vision
        x, seg_maps = build_vision_observation(
            poses_before, targets_pos, targets_ori, self.K,
            env_root_pos=env_root_pos,
        )
        contour_masks = (x[:, 0] > 0).to(torch.bool)  # match network observation exactly

        self._targets_pos = targets_pos
        self._targets_ori = targets_ori
        self._env_root_pos = env_root_pos

        return x.to(DEVICE), contour_masks.to(DEVICE), poses_before

    async def train_episode(self, episode: int):
        self._decay_schedule(episode)
        x, contour_masks, poses_before = await self._reset_episode(episode)

        ep_return = 0.0
        ep_len = 0
        active_counts = []
        ep_active_steps = 0
        final_has_contour = np.ones(NUM_ENVS, dtype=bool)
        selected_velocity_samples = []
        contact_velocity_samples = []
        greedy_velocity_samples = []
        strike_length_samples = []
        policy_selected_count = 0
        policy_reward_sum = 0.0
        heuristic_reward_sum = 0.0
        policy_reward_count = 0
        heuristic_reward_count = 0
        stop_reason = "max_steps"
        termination = EpisodeTermination(
            NUM_ENVS, self._targets_pos, self._env_root_pos)

        for step in range(MAX_STEPS):
            was_active, has_contour = termination.begin_step(
                poses_before, contour_masks)
            final_has_contour = has_contour
            active_counts.append(int(was_active.sum()))
            ep_active_steps += int(was_active.sum())

            #* — select actions ——————————————————————————————
            heuristic_actions = heuristic_poke_actions(
                contour_masks,
                poses_before,
                self._targets_pos,
                self._env_root_pos,
                self.K,
            )
            pixel_ij, d_xy, delta_d, is_policy_action, greedy_velocity = select_action(
                self.actor_critic, x, contour_masks,
                self.epsilon, self.noise_std,
                heuristic_actions=heuristic_actions,
            )
            strike_lengths = compute_adaptive_strike_lengths(
                pixel_ij,
                poses_before,
                self._targets_pos,
                self._env_root_pos,
                self.K,
                active=was_active,
            )
            delta_d = clamp_command_velocity_np(delta_d, strike_lengths)
            if was_active.any():
                selected_velocity_samples.append(delta_d[was_active])
                contact_velocity_samples.append(
                    contact_velocity_from_command(
                        delta_d[was_active],
                        strike_lengths[was_active],
                    )
                )
                greedy_velocity_samples.append(greedy_velocity[was_active])
                strike_length_samples.append(strike_lengths[was_active])
                policy_selected_count += int((is_policy_action & was_active).sum())
            update_action_overlay(
                pixel_ij, d_xy, delta_d, self.K,
                active=was_active,
                strike_lengths=strike_lengths,
                is_policy_action=is_policy_action,
            )

            #* — execute (only active envs move; inactive held frozen) —
            poses_after, _ = await env_step_async(
                self.fingers, pixel_ij, d_xy, delta_d, self.K,
                active=was_active,
                fingertip_masses=self._finger_masses,
                strike_lengths=strike_lengths,
            )
            rewards, dones, self._done_once = compute_rewards(
                poses_before, poses_after,
                self._targets_pos, self._env_root_pos, termination.done_once,
                target_oris=self._targets_ori,
                yaw_reward_enabled=YAW_REWARD_ENABLED,
                yaw_success_enabled=YAW_SUCCESS_ENABLED)
            termination.apply_reward_done(self._done_once)
            rewards[~was_active] = 0.0
            policy_active = is_policy_action & was_active
            heuristic_active = ~is_policy_action & was_active
            if policy_active.any():
                policy_reward_sum += float(rewards[policy_active].sum())
                policy_reward_count += int(policy_active.sum())
            if heuristic_active.any():
                heuristic_reward_sum += float(rewards[heuristic_active].sum())
                heuristic_reward_count += int(heuristic_active.sum())

            #* — observe next state ——————————————————————————
            x_next, seg_maps_next = build_vision_observation(
                poses_after, self._targets_pos, self._targets_ori, self.K,
                env_root_pos=self._env_root_pos,
            )
            contour_masks_next = x_next[:, 0] > 0  # match network observation exactly
            final_has_contour, remaining_active_next, early_truncated = (
                termination.finish_step(poses_after, contour_masks_next)
            )
            self._done_once = termination.done_once

            #* — store transitions for envs active at step start ——————
            for b in range(NUM_ENVS):
                if was_active[b]:
                    self.buffer.push(
                        x[b:b+1], pixel_ij[b], d_xy[b],
                        float(delta_d[b]), float(strike_lengths[b]),
                        float(rewards[b]), x_next[b:b+1],
                        termination.replay_done(b, step, early_truncated),
                    )

            ep_return += rewards[was_active].sum()
            ep_len += 1

            #* — advance state ————————————————————————————————
            x = x_next.to(DEVICE)
            contour_masks = contour_masks_next.to(DEVICE)
            poses_before = poses_after
            self.global_step += 1

            #* — training —————————————————————————————————————
            if self.buffer.size >= TRAIN_AFTER and self.global_step % TRAIN_EVERY == 0:
                for _ in range(GRAD_UPDATES_PER_STEP):
                    batch = self.buffer.sample(BATCH_SIZE)
                    lq, lmu = train_step(
                        self.actor_critic, self.target_net,
                        self.optimizer_q, self.optimizer_mu,
                        batch,
                    )
                    soft_update(self.target_net, self.actor_critic, TAU)

            next_stop_reason = termination.stop_reason(early_truncated)
            if next_stop_reason is not None:
                stop_reason = next_stop_reason
                break

        self.ep_returns.append(ep_return)
        self.ep_lengths.append(ep_len)
        self._last_active_counts = active_counts
        self._last_ep_active_steps = ep_active_steps
        self._last_done_count = int(self._done_once.sum())
        self._last_lost_count = int((~final_has_contour).sum())
        self._last_stop_reason = stop_reason
        self._last_velocity_stats = self._summarize_velocity_stats(
            selected_velocity_samples,
            contact_velocity_samples,
            greedy_velocity_samples,
            strike_length_samples,
            policy_selected_count,
            ep_active_steps,
            policy_reward_sum,
            policy_reward_count,
            heuristic_reward_sum,
            heuristic_reward_count,
        )

        return ep_return, ep_len

    @staticmethod
    def _mean_max(samples: list[np.ndarray]) -> tuple[float, float]:
        if not samples:
            return 0.0, 0.0
        values = np.concatenate(samples)
        if values.size == 0:
            return 0.0, 0.0
        return float(values.mean()), float(values.max())

    def _summarize_velocity_stats(
        self,
        selected_samples: list[np.ndarray],
        contact_samples: list[np.ndarray],
        greedy_samples: list[np.ndarray],
        strike_length_samples: list[np.ndarray],
        policy_selected_count: int,
        active_steps: int,
        policy_reward_sum: float,
        policy_reward_count: int,
        heuristic_reward_sum: float,
        heuristic_reward_count: int,
    ) -> dict[str, float]:
        sel_mean, sel_max = self._mean_max(selected_samples)
        contact_mean, contact_max = self._mean_max(contact_samples)
        greedy_mean, greedy_max = self._mean_max(greedy_samples)
        strike_l_mean, strike_l_max = self._mean_max(strike_length_samples)
        return {
            "selected_mean": sel_mean,
            "selected_max": sel_max,
            "contact_mean": contact_mean,
            "contact_max": contact_max,
            "greedy_mean": greedy_mean,
            "greedy_max": greedy_max,
            "strike_l_mean": strike_l_mean,
            "strike_l_max": strike_l_max,
            "policy_frac": policy_selected_count / max(1, active_steps),
            "policy_reward": policy_reward_sum / max(1, policy_reward_count),
            "heuristic_reward": heuristic_reward_sum / max(1, heuristic_reward_count),
        }

    def log(self, episode: int, ep_return: float, ep_len: int, elapsed: float):
        avg_r = np.mean(self.ep_returns) if self.ep_returns else 0.0
        avg_l = np.mean(self.ep_lengths) if self.ep_lengths else 0.0
        active_counts = getattr(self, "_last_active_counts", [])
        active_steps = getattr(self, "_last_ep_active_steps", 0)
        done_count = getattr(self, "_last_done_count", 0)
        lost_count = getattr(self, "_last_lost_count", 0)
        stop_reason = getattr(self, "_last_stop_reason", "max_steps")
        velocity_stats = getattr(self, "_last_velocity_stats", {})
        r_per_active = ep_return / max(1, active_steps)
        active_summary = ""
        if active_counts:
            active_summary = f"  active={active_counts[0]}→{active_counts[-1]}"
        velocity_summary = ""
        if velocity_stats:
            velocity_summary = (
                f"  sel_v={velocity_stats['selected_mean']:.3f}/"
                f"{velocity_stats['selected_max']:.3f}"
                f"  contact_v={velocity_stats['contact_mean']:.3f}/"
                f"{velocity_stats['contact_max']:.3f}"
                f"  greedy_v={velocity_stats['greedy_mean']:.3f}/"
                f"{velocity_stats['greedy_max']:.3f}"
                f"  strike_L={velocity_stats['strike_l_mean']:.3f}/"
                f"{velocity_stats['strike_l_max']:.3f}"
                f"  policy={100.0 * velocity_stats['policy_frac']:.1f}%"
                f"  r_pol={velocity_stats['policy_reward']:.4f}"
                f"  r_heur={velocity_stats['heuristic_reward']:.4f}"
            )
        print(
            f"[ep {episode:5d}] "
            f"return={ep_return:7.2f}  len={ep_len:3d}  "
            f"avg100_r={avg_r:7.2f}  avg100_len={avg_l:5.1f}  "
            f"r/active={r_per_active:7.4f}  "
            f"ε={self.epsilon:.3f}  σ={self.noise_std:.3f}  "
            f"buf={self.buffer.size:6d}  dt={elapsed:.1f}s  "
            f"done={done_count:2d}  lost={lost_count:2d}  stop={stop_reason}"
            f"{velocity_summary}"
            f"{active_summary}"
        )

    def save_checkpoint(self, path: str | Path, episode: int):
        """Save full training state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "actor_critic": self.actor_critic.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer_q":  self.optimizer_q.state_dict(),
            "optimizer_mu": self.optimizer_mu.state_dict(),
            "global_step":  self.global_step,
            "epsilon":      self.epsilon,
            "noise_std":    self.noise_std,
            "episode":      episode,
            "rng_state":    self.rng.bit_generator.state,
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "ep_returns":   list(self.ep_returns),
            "ep_lengths":   list(self.ep_lengths),
            "buffer_x":       self.buffer.x[:self.buffer.size],
            "buffer_x_next":  self.buffer.x_next[:self.buffer.size],
            "buffer_pixel":   self.buffer.pixel[:self.buffer.size],
            "buffer_d_xy":    self.buffer.d_xy[:self.buffer.size],
            "buffer_velocity": self.buffer.velocity[:self.buffer.size],
            "buffer_strike_length": self.buffer.strike_length[:self.buffer.size],
            "buffer_r":       self.buffer.r[:self.buffer.size],
            "buffer_done":    self.buffer.done[:self.buffer.size],
            "buffer_ptr":     self.buffer.ptr,
            "buffer_size":    self.buffer.size,
        }
        if DEVICE == "cuda":
            ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(ckpt, path)
        print(f"[checkpoint] saved ep {episode} to {path} "
              f"(buf={self.buffer.size})")

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore training state from disk.  Returns the last saved episode."""
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer_q.load_state_dict(ckpt["optimizer_q"])
        self.optimizer_mu.load_state_dict(ckpt["optimizer_mu"])
        self.global_step = ckpt["global_step"]
        self.epsilon     = ckpt["epsilon"]
        self.noise_std   = ckpt["noise_std"]
        self.rng.bit_generator.state = ckpt["rng_state"]
        if "np_rng_state" in ckpt:
            np.random.set_state(ckpt["np_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "cuda_rng_state" in ckpt and DEVICE == "cuda":
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        self.ep_returns = deque(ckpt["ep_returns"], maxlen=100)
        self.ep_lengths = deque(ckpt["ep_lengths"], maxlen=100)

        size = ckpt["buffer_size"]
        self.buffer.ptr  = ckpt["buffer_ptr"] % self.buffer.capacity
        self.buffer.size = min(size, self.buffer.capacity)
        s = self.buffer.size
        self.buffer.x[:s].copy_(ckpt["buffer_x"][:s])
        self.buffer.x_next[:s].copy_(ckpt["buffer_x_next"][:s])
        self.buffer.pixel[:s].copy_(ckpt["buffer_pixel"][:s])
        self.buffer.d_xy[:s].copy_(ckpt["buffer_d_xy"][:s])
        self.buffer.velocity[:s].copy_(ckpt["buffer_velocity"][:s])
        if "buffer_strike_length" in ckpt:
            self.buffer.strike_length[:s].copy_(ckpt["buffer_strike_length"][:s])
        else:
            self.buffer.strike_length[:s].fill_(L_STRIKE)
        self.buffer.r[:s].copy_(ckpt["buffer_r"][:s])
        self.buffer.done[:s].copy_(ckpt["buffer_done"][:s])

        print(f"[checkpoint] loaded ep {ckpt['episode']} from {path} "
              f"(buf={s}, step={self.global_step})")
        return ckpt["episode"]

    def load_checkpoint_weights_only(
        self,
        path: str | Path,
        load_optimizers: bool = True,
    ) -> int:
        """Load model weights for a new stage without restoring replay."""
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if load_optimizers:
            self.optimizer_q.load_state_dict(ckpt["optimizer_q"])
            self.optimizer_mu.load_state_dict(ckpt["optimizer_mu"])
        self.buffer = ReplayBuffer()
        self.global_step = 0
        self.epsilon = EPS_START
        self.noise_std = SIGMA_START
        self.ep_returns.clear()
        self.ep_lengths.clear()
        print(f"[checkpoint] loaded weights from ep {ckpt['episode']} at {path} "
              f"(cleared replay buffer)")
        return ckpt["episode"]


async def main_async(
    num_episodes: int = 10_000,
    resume: str | None = None,
    resume_mode: str = "full",
):
    trainer = Trainer()
    start_ep = 1
    if resume:
        await trainer.setup()
        if resume_mode == "full":
            start_ep = trainer.load_checkpoint(resume) + 1
        elif resume_mode == "weights":
            trainer.load_checkpoint_weights_only(resume)
            start_ep = 1
        else:
            raise ValueError("resume_mode must be 'full' or 'weights'")
    else:
        await trainer.setup()

    print(f"[train] device={DEVICE}  envs={NUM_ENVS}  buffer={BUFFER_CAPACITY}")
    print(f"[train] max_steps/ep={MAX_STEPS}  gamma={GAMMA}  lr={LR}")
    print(f"[train] yaw_mode={YAW_TARGET_MODE}  yaw_reward={YAW_REWARD_ENABLED}  "
          f"yaw_success={YAW_SUCCESS_ENABLED}  c_yaw={C_YAW}")
    if resume:
        print(f"[train] resume_mode={resume_mode}  starting episode {start_ep}")

    for ep in range(start_ep, num_episodes + 1):
        t0 = time.time()
        ep_r, ep_len = await trainer.train_episode(ep)
        trainer.log(ep, ep_r, ep_len, time.time() - t0)

        if ep % CHECKPOINT_EVERY == 0:
            trainer.save_checkpoint(CHECKPOINT_DIR / f"ep{ep:06d}.pt", ep)
            # keep last N checkpoints
            saved = sorted(CHECKPOINT_DIR.glob("ep*.pt"))
            for old in saved[:-CHECKPOINT_KEEP]:
                old.unlink()

    print("[train] done.")
    return trainer


def main(
    num_episodes: int = 10_000,
    resume: str | None = None,
    resume_mode: str = "full",
):
    return run_coroutine(main_async(num_episodes, resume, resume_mode))


if __name__ == "__main__":
    trainer_task = main()
