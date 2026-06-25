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
    get_object_masks_2d,
    mask_to_contour,
    pixel_to_world,
)
from nn.networks import SpatialActorCritic
from env.make_finger_robot import LIMIT_LOWER, LIMIT_UPPER, configure_drives
from env.scene_setup_articulated_vectorized import randomize_object_poses

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]
WORKSPACE_CONFIG = CONFIG["workspace"]

OBJECT_HEIGHT = SCENE_CONFIG["object_height"]
L_ARM_LENGTH = SCENE_CONFIG["l_arm_length"]
L_THICKNESS = SCENE_CONFIG["l_thickness"]
CYLINDER_RADIUS = SCENE_CONFIG["cylinder_radius"]
ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
FINGER_LOCAL_PATH = FINGER_CONFIG["local_root_path"]
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/{FINGER_LOCAL_PATH}"
FINGER_TIP_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
NUM_ENVS = SCENE_CONFIG["num_envs"]


#* ================================================================
#*  Hyper-parameters
#* ================================================================

GAMMA: float = 0.95
LR: float = 3e-4
BATCH_SIZE: int = 128
BUFFER_CAPACITY: int = 100_000
EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: int = 250             # episodes over which ε decays
SIGMA_START: float = 0.3
SIGMA_END: float = 0.05
TAU: float = 0.005               # polyak averaging coefficient
MAX_STEPS: int = 30              # max pokes per episode
C_STEP: float = 0.01             # per-step penalty
C_SUCCESS: float = 10.0          # terminal success bonus
SUCCESS_THRESHOLD: float = 0.02  # metres — tolerance to target
STOP_POKE_THRESHOLD: float = 0.035  # metres — stop touching near-target envs
DELTA_D_MAX: float = 0.2          # max standoff (m)
IMPACT_STEPS: int = 80            # physics steps holding the strike command
FINGERTIP_RADIUS: float = FINGER_CONFIG["sphere_radius"]
STANDOFF_CLEARANCE: float = 0.01  # gap between fingertip sphere and object
OVERTRAVEL: float = 0.1          # extra sphere penetration past first contact
STRIKE_SPEED: float = 0.15        # m/s velocity target during impact
POKE_Z: float = OBJECT_HEIGHT * 0.5  # side-poke height at object midline
SAFE_Z: float = POKE_Z
FINGER_TRACK_TOL: float = 0.003   # metres — phase target convergence tolerance

# yaw training controls
# Stage A: keep target yaw equal to current yaw, softly discourage yaw drift,
#          but let success depend on translation only.
# Stage B: load weights-only, clear replay, set YAW_TARGET_MODE="curriculum"
#          or "fixed", and enable yaw success.
YAW_TARGET_MODE: str = "preserve"       # "preserve", "curriculum", or "fixed"
C_YAW: float = 0.1                      # weight on orientation progress reward
YAW_REWARD_ENABLED: bool = True         # soft yaw preservation/control reward
YAW_SUCCESS_ENABLED: bool = False       # include yaw threshold in done/success
YAW_THRESHOLD: float = np.deg2rad(10)   # radians — tolerance for yaw success
YAW_CURRICULUM_START: int = 0           # episode to begin expanding yaw range
YAW_CURRICULUM_END: int = 500           # episode to reach full yaw range
YAW_FULL_RANGE: float = np.pi           # radians — ±180° full orientation

TRAIN_AFTER: int = 1_024         # start training after this many transitions
TRAIN_EVERY: int = 1             # train every N env steps
GRAD_UPDATES_PER_STEP: int = 4   # replay updates per env step
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_EVERY: int = 100       # episodes between saves
CHECKPOINT_KEEP: int = 3          # how many recent checkpoints to keep
SHOW_PLANE_OVERLAY: bool = True   # draw workspace/camera view footprints
PLANE_OVERLAY_THICKNESS: float = 0.003
PLANE_OVERLAY_OPACITY: float = 0.18
SHOW_TARGET_OVERLAY: bool = True  # draw per-env target poses in the USD scene
TARGET_OVERLAY_Z_OFFSET: float = 0.015
TARGET_OVERLAY_THICKNESS: float = 0.006
TARGET_OVERLAY_OPACITY: float = 0.2
SHOW_ACTION_OVERLAY: bool = True  # draw selected poke standoff/contact/strike
ACTION_OVERLAY_Z_OFFSET: float = 0.01


#* ================================================================
#*  Replay buffer
#* ================================================================

class ReplayBuffer:
    """Ring buffer storing per-transition data (one env, one step).

    Each element is a flat dict so we can mix across envs and episodes.
    x / x_next are stored as uint8 (binary contours) to save memory.
    Mask is reconstructed on-the-fly from x in sample() / train_step().
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY, device: str = DEVICE):
        self.capacity = capacity
        self.device = device

        H, W = RESOLUTION
        self.x        = torch.zeros(capacity, 2, H, W, dtype=torch.uint8)
        self.pixel    = torch.zeros(capacity, 2, dtype=torch.long)
        self.d_xy     = torch.zeros(capacity, 2)
        self.delta_d  = torch.zeros(capacity, 1)
        self.r        = torch.zeros(capacity, 1)
        self.x_next   = torch.zeros(capacity, 2, H, W, dtype=torch.uint8)
        self.done     = torch.zeros(capacity, 1, dtype=torch.bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        x: torch.Tensor,             # (1, 2, H, W) float32 binary contours
        pixel_ij: np.ndarray,        # (2,) int
        d_xy: np.ndarray,            # (2,)
        delta_d: float,
        reward: float,
        x_next: torch.Tensor,        # (1, 2, H, W) float32
        done: bool,
    ):
        idx = self.ptr
        # float32 → uint8 (binary 0/1 contours)
        _x = (x.squeeze(0) * 255).to(torch.uint8)
        _xn = (x_next.squeeze(0) * 255).to(torch.uint8)
        self.x[idx].copy_(_x.cpu() if _x.device.type != "cpu" else _x)
        self.pixel[idx] = torch.tensor(pixel_ij)
        self.d_xy[idx] = torch.tensor(d_xy)
        self.delta_d[idx] = torch.tensor([delta_d])
        self.r[idx] = torch.tensor([reward])
        self.x_next[idx].copy_(_xn.cpu() if _xn.device.type != "cpu" else _xn)
        self.done[idx] = torch.tensor([done])

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        x      = self.x[indices].to(torch.float32).to(self.device) / 255.0
        x_next = self.x_next[indices].to(torch.float32).to(self.device) / 255.0
        return {
            "x":         x,
            "pixel":     self.pixel[indices].to(self.device),
            "d_xy":      self.d_xy[indices].to(self.device),
            "delta_d":   self.delta_d[indices].to(self.device),
            "r":         self.r[indices].to(self.device),
            "x_next":    x_next,
            "done":      self.done[indices].to(self.device),
            "mask_next": x_next[:, 0] > 0,          # derived from channel 0
        }

    def __len__(self) -> int:
        return self.size


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
) -> dict[str, np.ndarray]:
    """Random 2-D target on the table for each object and each env.

    Returns dict: obj_name → (num_envs, 3)  (z = OBJECT_HEIGHT * 0.5)
    """
    z = np.full((num_envs,), OBJECT_HEIGHT * 0.5, dtype=np.float32)
    lx = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    ly = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    cx = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    cy = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    return {"LObject":  np.stack([lx, ly, z], axis=-1),
            "Cylinder": np.stack([cx, cy, z], axis=-1)}


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
                          active: np.ndarray | None = None):
    """Draw the selected side-poke action for each env."""
    if not SHOW_ACTION_OVERLAY or not _HAS_ISAAC:
        return

    stage = stage_utils.get_current_stage(backend="usd")
    dir_norm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm < 1e-8, 1.0, dir_norm)
    dirs = d_xy / dir_norm
    min_standoff = FINGERTIP_RADIUS + STANDOFF_CLEARANCE
    delta_d_clipped = np.clip(delta_d, min_standoff, DELTA_D_MAX)

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
        contact_xy = world[:2]
        standoff_xy = contact_xy - dirs[env_idx] * delta_d_clipped[env_idx]
        sphere_contact_xy = contact_xy - dirs[env_idx] * FINGERTIP_RADIUS
        strike_xy = sphere_contact_xy + dirs[env_idx] * OVERTRAVEL

        points = {
            "Standoff": (standoff_xy, (0.0, 0.25, 1.0)),
            "Contact": (contact_xy, (1.0, 0.0, 0.0)),
            "Overtravel": (strike_xy, (1.0, 1.0, 1.0)),
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
            _define_colored_cube(stage, line_path, (0.0, 1.0, 1.0))
            _set_local_transform(
                line_path,
                [line_center[0], line_center[1], z],
                yaw=line_yaw,
                scale=[line_len, line_thickness, line_thickness],
            )


#* ================================================================
#*  Action selection
#* ================================================================

def _mask_center(mask: torch.Tensor) -> np.ndarray | None:
    pixels = torch.nonzero(mask, as_tuple=False)
    if pixels.numel() == 0:
        return None
    return pixels.to(torch.float32).mean(dim=0).cpu().numpy()


def _pixels_to_local_xy(pixel_ij: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixel_ij, dtype=np.float32)
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    x = (cols - K[0, 2]) / K[0, 0]
    y = (rows - K[1, 2]) / K[1, 1]
    return np.stack([x, y], axis=1).astype(np.float32)


def _normalise_xy(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        if fallback is not None:
            return fallback.astype(np.float32)
        angle = np.random.uniform(-np.pi, np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    return (v / norm).astype(np.float32)


def heuristic_poke_actions(
    contour_masks: torch.Tensor,
    poses_before: dict,
    targets_pos: dict[str, np.ndarray],
    env_root_pos: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physically plausible exploration pokes for epsilon actions.

    Pick the object that is farther from its target, choose a contour pixel on
    the backside relative to target displacement, and push toward the target.
    """
    B = contour_masks.shape[0]
    pixel_ij = np.zeros((B, 2), dtype=np.int32)
    d_xy = np.zeros((B, 2), dtype=np.float32)
    delta_d = np.full(B, 0.05, dtype=np.float32)

    masks_cpu = contour_masks.detach().cpu()
    env_offsets = env_root_pos[:, :2]

    for b in range(B):
        valid_pixels_t = torch.nonzero(masks_cpu[b], as_tuple=False)
        if valid_pixels_t.numel() == 0:
            continue

        object_infos = []
        for obj_name, target in targets_pos.items():
            current_xy = poses_before[obj_name][0][b, :2] - env_offsets[b]
            target_xy = target[b, :2]
            to_target = target_xy - current_xy
            dist = float(np.linalg.norm(to_target))
            object_infos.append((dist, obj_name, current_xy, target_xy, to_target))

        movable_infos = [info for info in object_infos if info[0] >= STOP_POKE_THRESHOLD]
        if not movable_infos:
            movable_infos = object_infos
        _, obj_name, current_xy, _, to_target = max(movable_infos, key=lambda item: item[0])
        goal_dir = _normalise_xy(to_target)

        local_position = poses_before[obj_name][0][b:b + 1].copy()
        local_position[:, :2] -= env_offsets[b]
        object_mask = get_object_masks_2d(
            object_name=obj_name,
            object_positions=local_position,
            object_orientations=poses_before[obj_name][1][b:b + 1],
            K=K,
            resolution=RESOLUTION,
        )[0]
        object_contour, _ = mask_to_contour(object_mask)
        valid_pixels = np.argwhere(object_contour).astype(np.int32)
        if valid_pixels.shape[0] == 0:
            valid_pixels = valid_pixels_t.numpy().astype(np.int32)
        contact_xys = _pixels_to_local_xy(valid_pixels, K)

        if obj_name == "Cylinder":
            desired_contact_xy = current_xy - goal_dir * CYLINDER_RADIUS
            nearest = int(np.argmin(np.linalg.norm(contact_xys - desired_contact_xy[None, :], axis=1)))
            pixel_ij[b] = valid_pixels[nearest]
            noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.05
            d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
            delta_d[b] = np.float32(0.045 + abs(np.random.randn()) * 0.015)
            continue

        inward_scores = (current_xy[None, :] - contact_xys) @ goal_dir
        candidate_mask = inward_scores > 0.0
        candidate_pixels = valid_pixels[candidate_mask]
        candidate_xys = contact_xys[candidate_mask]

        if candidate_pixels.shape[0] == 0:
            candidate_pixels = valid_pixels
            candidate_xys = contact_xys

        backside_scores = (candidate_xys - current_xy[None, :]) @ (-goal_dir)
        k = min(25, candidate_pixels.shape[0])
        top_idx = np.argpartition(backside_scores, -k)[-k:]
        chosen = int(np.random.choice(top_idx))

        pixel_ij[b] = candidate_pixels[chosen]
        noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.10
        d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
        delta_d[b] = np.float32(0.045 + abs(np.random.randn()) * 0.015)

    return pixel_ij, d_xy, delta_d


def select_action(
    actor_critic: SpatialActorCritic,
    x: torch.Tensor,                       # (B, 2, H, W)
    contour_masks: torch.Tensor,           # (B, H, W) bool
    epsilon: float,
    noise_std: float,
    heuristic_actions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select actions for a batch of observations.

    Greedy envs use per-pixel FiLM-conditioned pixel selection (top-K
    re-evaluation).  ε-greedy envs use the provided heuristic action prior
    when available, so early exploration produces meaningful pushes instead
    of untrained param-head thrashing.

    Returns
    -------
    pixel_ij : (B, 2) int   — (row, col) per env
    d_xy     : (B, 2) float — normalised unit vector
    delta_d  : (B,)  float — standoff in [0, delta_d_max]
    """
    B = x.shape[0]
    device = x.device
    actor_critic.eval()

    pixel_ij = np.zeros((B, 2), dtype=np.int32)
    d_xy_arr = np.zeros((B, 2), dtype=np.float32)
    delta_d_arr = np.zeros(B, dtype=np.float32)

    with torch.no_grad():
        # one U-Net pass — features + per-pixel params
        f, params, _ = actor_critic.features_and_params(x)

        # batched greedy pixel selection (per-pixel FiLM conditioned)
        greedy_pix, _ = actor_critic.greedy_pixel(f, params, contour_masks, top_k=top_k)

        for b in range(B):
            if contour_masks[b].sum() == 0:
                delta_d_arr[b] = 0.0           # no-op poke (empty contour)
                continue

            if np.random.random() < epsilon:
                if heuristic_actions is not None:
                    h_pixel, h_dxy, h_delta = heuristic_actions
                    pixel_ij[b] = h_pixel[b]
                    d_noisy = h_dxy[b]
                    dd_noisy = h_delta[b]
                else:
                    valid = torch.nonzero(contour_masks[b], as_tuple=True)
                    r = np.random.randint(len(valid[0]))
                    pixel_ij[b, 0] = valid[0][r].item()
                    pixel_ij[b, 1] = valid[1][r].item()
                    d_noisy = np.random.randn(2).astype(np.float32)
                    dd_noisy = 0.05 + abs(np.float32(np.random.randn())) * 0.02
            else:
                # greedy: pre-computed per-pixel-conditioned best pixel
                pixel_ij[b, 0] = greedy_pix[b, 0].item()
                pixel_ij[b, 1] = greedy_pix[b, 1].item()
                # read params at selected pixel
                p = params[b, :, pixel_ij[b, 0], pixel_ij[b, 1]].cpu().numpy()  # (3,)
                d_noisy = p[:2] + np.random.randn(2).astype(np.float32) * noise_std
                dd_noisy = p[2] + abs(np.float32(np.random.randn())) * noise_std

            # clamp & normalise direction
            norm = np.linalg.norm(d_noisy) + 1e-8
            d_xy_arr[b] = d_noisy / norm
            delta_d_arr[b] = np.clip(dd_noisy, 0.0, actor_critic.delta_d_max)

    return pixel_ij, d_xy_arr, delta_d_arr



#* ================================================================
#*  Reward & termination
#* ================================================================

def _yaw_error(quats: np.ndarray, target_quats: np.ndarray) -> np.ndarray:
    """Wrapped angle difference between quaternion yaws (radians, [0, π])."""
    yaw = 2.0 * np.arctan2(quats[:, 3], quats[:, 0])
    t_yaw = 2.0 * np.arctan2(target_quats[:, 3], target_quats[:, 0])
    diff = yaw - t_yaw
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


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


def compute_rewards(
    poses_before: dict,         # obj_name → (positions, quaternions) in world coords
    poses_after: dict,
    targets: dict,              # obj_name → (N_envs, 3)  in env-local coords
    env_root_pos: np.ndarray | None = None,  # (N_envs, 3)
    done_once: np.ndarray | None = None,     # (N_envs,) bool
    target_oris: dict | None = None,         # obj_name → (N_envs, 4) quat targets
    yaw_reward_enabled: bool = YAW_REWARD_ENABLED,
    yaw_success_enabled: bool = YAW_SUCCESS_ENABLED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-env reward + done flag.

    Progress reward: signed distance + signed yaw reduction.
    C_SUCCESS awarded only the first time an env achieves the threshold.
    """
    num_envs = list(poses_before.values())[0][0].shape[0]
    r = np.zeros(num_envs, dtype=np.float32)
    done = np.ones(num_envs, dtype=bool)
    if done_once is None:
        done_once = np.zeros(num_envs, dtype=bool)

    offset = env_root_pos[:, :2] if env_root_pos is not None else np.zeros((num_envs, 2))

    for obj_name in poses_before:
        pos_before = poses_before[obj_name][0].copy()
        pos_after  = poses_after[obj_name][0].copy()
        pos_before[:, :2] -= offset
        pos_after[:, :2]  -= offset
        d_before = np.linalg.norm(pos_before[:, :2] - targets[obj_name][:, :2], axis=1)
        d_after  = np.linalg.norm(pos_after[:, :2] - targets[obj_name][:, :2], axis=1)
        r += d_before - d_after
        done &= (d_after < SUCCESS_THRESHOLD)

        if obj_name == "LObject" and target_oris is not None and obj_name in target_oris:
            y_err_before = _yaw_error(poses_before[obj_name][1], target_oris[obj_name])
            y_err_after  = _yaw_error(poses_after[obj_name][1], target_oris[obj_name])
            if yaw_reward_enabled:
                r += C_YAW * (y_err_before - y_err_after)
            if yaw_success_enabled:
                done &= (y_err_after < YAW_THRESHOLD)

    r -= C_STEP

    first_done = done & ~done_once
    r[first_done] += C_SUCCESS
    done_once = done_once | done

    return r, done, done_once


def near_translation_targets(
    poses: dict,
    targets: dict,
    env_root_pos: np.ndarray,
    threshold: float = STOP_POKE_THRESHOLD,
) -> np.ndarray:
    """Return envs where all object centers are close enough to targets."""
    num_envs = list(poses.values())[0][0].shape[0]
    near = np.ones(num_envs, dtype=bool)
    offset = env_root_pos[:, :2]
    for obj_name, (positions, _) in poses.items():
        local_xy = positions[:, :2].copy() - offset
        dist = np.linalg.norm(local_xy - targets[obj_name][:, :2], axis=1)
        near &= dist < threshold
    return near


#* ================================================================
#*  Environment step — articulated finger (batched, all envs)
#* ================================================================

async def env_step_async(
    fingers: Articulation,
    pixel_ij: np.ndarray,        # (B, 2)
    d_xy: np.ndarray,            # (B, 2)
    delta_d: np.ndarray,         # (B,)
    K: np.ndarray,
    active: np.ndarray | None = None,  # (B,) bool — which envs to control
) -> tuple[dict, np.ndarray]:
    """Execute strikes with the prismatic XYZ finger robot.

    Phases:
      0. move to side-poke height
      1. move to standoff_xy at side-poke height
      2. settle briefly at standoff
      3. strike — push OVERTRAVEL past contact
      4. retract back to standoff_xy at side-poke height
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

    zero_v = np.zeros((B, 3), dtype=np.float32)
    xy_low  = np.array(LIMIT_LOWER[:2], dtype=np.float32)
    xy_high = np.array(LIMIT_UPPER[:2], dtype=np.float32)

    #* ── standoff from Δd ────────────────────────────────────────────
    #* world_xy is a contour point. The controlled point is the sphere centre,
    #* so keep at least one fingertip radius plus clearance outside the object.
    min_standoff = FINGERTIP_RADIUS + STANDOFF_CLEARANCE
    delta_d_clipped = np.clip(delta_d, min_standoff, DELTA_D_MAX)
    standoff_xy = world_xy - dirs * delta_d_clipped[:, None]  # (B, 2) run-up
    contact_xy = world_xy - dirs * FINGERTIP_RADIUS
    strike_xy = contact_xy + dirs * OVERTRAVEL

    #* ── snap initial positions (inactive envs hold these throughout) ──
    try:
        q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)
    except AssertionError:
        app_utils.play()
        await _step_physics(2)
        q_cur = as_numpy(fingers.get_dof_positions()).copy()

    def _hold_inactive(targets: np.ndarray):
        if active is not None:
            targets[~active] = q_cur[~active]

    async def _drive_to(targets: np.ndarray, max_steps: int,
                        tol: float = FINGER_TRACK_TOL):
        fingers.set_dof_position_targets(positions=targets.tolist(), dof_indices=dof_indices)
        fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
        if active is not None and not np.any(active):
            await _step_physics(1)
            return
        for step_idx in range(max_steps):
            await _step_physics(1)
            try:
                q_now = as_numpy(fingers.get_dof_positions())
            except AssertionError:
                app_utils.play()
                await _step_physics(2)
                try:
                    q_now = as_numpy(fingers.get_dof_positions())
                except AssertionError:
                    await _step_physics(max(0, max_steps - step_idx - 1))
                    break
            err = np.linalg.norm(q_now - targets, axis=1)
            if active is not None:
                err = err[active]
            if err.size > 0 and float(np.max(err)) <= tol:
                break

    #* ── Phase 0: move to side-poke height above current XY ──────────
    q0 = q_cur.copy()
    q0[:, 2] = POKE_Z
    _hold_inactive(q0)
    await _drive_to(q0, max_steps=60)

    #* ── Phase 1: move to standoff at side-poke height ──────────────
    q1 = np.zeros((B, 3), dtype=np.float32)
    q1[:, :2] = np.clip(standoff_xy, xy_low, xy_high)
    q1[:, 2] = POKE_Z
    _hold_inactive(q1)
    await _drive_to(q1, max_steps=120)

    #* ── Phase 2: hold standoff before impact ───────────────────────
    q2 = q1.copy()
    q2[:, 2] = POKE_Z
    _hold_inactive(q2)
    await _drive_to(q2, max_steps=30)

    #* ── Phase 3: strike — push OVERTRAVEL past contact ──────────────
    #* Position-only PD (v_target=0) — OVERTRAVEL creates a steady
    #* pushing force kp*OVERTRAVEL.  No velocity target avoids oscillation
    #* at the contact point (velocity tracking fights the constraint).
    q3 = np.zeros((B, 3), dtype=np.float32)
    q3[:, :2] = np.clip(strike_xy, xy_low, xy_high)
    q3[:, 2] = POKE_Z
    _hold_inactive(q3)

    strike_v = np.zeros((B, 3), dtype=np.float32)
    strike_v[:, :2] = dirs * STRIKE_SPEED
    if active is not None:
        strike_v[~active] = 0.0
    fingers.set_dof_position_targets(positions=q3.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=strike_v.tolist(), dof_indices=dof_indices)
    await _step_physics(IMPACT_STEPS)
    fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)

    #* ── Phase 4: retract back to standoff at side-poke height ───────
    q4 = q2.copy()
    q4[:, 2] = POKE_Z
    _hold_inactive(q4)
    await _drive_to(q4, max_steps=80)

    #* ── Phase 5: settle briefly, then clear residual object velocities ─
    SETTLE_STEPS = 30
    l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
    cyl_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")

    await _step_physics(SETTLE_STEPS)
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
    dd_st   = batch["delta_d"]     # (B, 1)
    r       = batch["r"]           # (B, 1)
    x_next  = batch["x_next"]
    done    = batch["done"]        # (B, 1)
    mask_next = batch["mask_next"]  # (B, H, W) — pre-computed in sample()

    B = x.shape[0]
    batch_idx = torch.arange(B, device=x.device)
    actor_critic.train()
    target_net.eval()

    #* -- stored action params (with exploration noise, as executed) -----
    a_stored = torch.cat([d_xy_st, dd_st], dim=1)  # (B, 3)

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
        self.actor_critic = SpatialActorCritic(delta_d_max=DELTA_D_MAX).to(DEVICE)
        self.target_net   = SpatialActorCritic(delta_d_max=DELTA_D_MAX).to(DEVICE)
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
        else:
            configure_drives(self.fingers)

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

        # random position targets
        targets_pos = sample_target_poses(self.rng, NUM_ENVS)

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
        self._done_once = np.zeros(NUM_ENVS, dtype=bool)

        return x.to(DEVICE), contour_masks.to(DEVICE), poses_before

    async def train_episode(self, episode: int):
        self._decay_schedule(episode)
        x, contour_masks, poses_before = await self._reset_episode(episode)

        ep_return = 0.0
        ep_len = 0
        active_counts = []
        ep_active_steps = 0
        final_has_contour = np.ones(NUM_ENVS, dtype=bool)

        for step in range(MAX_STEPS):
            has_contour = contour_masks.any(dim=(-2, -1)).cpu().numpy()
            final_has_contour = has_contour
            near_target = near_translation_targets(
                poses_before, self._targets_pos, self._env_root_pos
            )
            self._done_once |= near_target
            was_active = ~self._done_once & has_contour  # done OR invisible → frozen
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
            pixel_ij, d_xy, delta_d = select_action(
                self.actor_critic, x, contour_masks,
                self.epsilon, self.noise_std,
                heuristic_actions=heuristic_actions,
            )
            update_action_overlay(pixel_ij, d_xy, delta_d, self.K, active=was_active)

            #* — execute (only active envs move; inactive held frozen) —
            poses_after, _ = await env_step_async(
                self.fingers, pixel_ij, d_xy, delta_d, self.K,
                active=was_active,
            )
            rewards, dones, self._done_once = compute_rewards(
                poses_before, poses_after,
                self._targets_pos, self._env_root_pos, self._done_once,
                target_oris=self._targets_ori,
                yaw_reward_enabled=YAW_REWARD_ENABLED,
                yaw_success_enabled=YAW_SUCCESS_ENABLED)
            self._done_once |= near_translation_targets(
                poses_after, self._targets_pos, self._env_root_pos
            )
            rewards[~was_active] = 0.0

            #* — observe next state ——————————————————————————
            x_next, seg_maps_next = build_vision_observation(
                poses_after, self._targets_pos, self._targets_ori, self.K,
                env_root_pos=self._env_root_pos,
            )
            contour_masks_next = x_next[:, 0] > 0  # match network observation exactly

            #* — store transitions for envs active at step start ——————
            truncated = (step == MAX_STEPS - 1)
            for b in range(NUM_ENVS):
                if was_active[b]:
                    self.buffer.push(
                        x[b:b+1], pixel_ij[b], d_xy[b], float(delta_d[b]),
                        float(rewards[b]), x_next[b:b+1],
                        bool(self._done_once[b] or truncated),
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

            if self._done_once.all():
                break

        self.ep_returns.append(ep_return)
        self.ep_lengths.append(ep_len)
        self._last_active_counts = active_counts
        self._last_ep_active_steps = ep_active_steps
        self._last_done_count = int(self._done_once.sum())
        self._last_lost_count = int((~final_has_contour).sum())

        return ep_return, ep_len

    def log(self, episode: int, ep_return: float, ep_len: int, elapsed: float):
        avg_r = np.mean(self.ep_returns) if self.ep_returns else 0.0
        avg_l = np.mean(self.ep_lengths) if self.ep_lengths else 0.0
        active_counts = getattr(self, "_last_active_counts", [])
        active_steps = getattr(self, "_last_ep_active_steps", 0)
        done_count = getattr(self, "_last_done_count", 0)
        lost_count = getattr(self, "_last_lost_count", 0)
        r_per_active = ep_return / max(1, active_steps)
        active_summary = ""
        if active_counts:
            active_summary = f"  active={active_counts[0]}→{active_counts[-1]}"
        print(
            f"[ep {episode:5d}] "
            f"return={ep_return:7.2f}  len={ep_len:3d}  "
            f"avg100_r={avg_r:7.2f}  avg100_len={avg_l:5.1f}  "
            f"r/active={r_per_active:7.4f}  "
            f"ε={self.epsilon:.3f}  σ={self.noise_std:.3f}  "
            f"buf={self.buffer.size:6d}  dt={elapsed:.1f}s  "
            f"done={done_count:2d}  lost={lost_count:2d}"
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
            "buffer_delta_d": self.buffer.delta_d[:self.buffer.size],
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
        self.buffer.delta_d[:s].copy_(ckpt["buffer_delta_d"][:s])
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
