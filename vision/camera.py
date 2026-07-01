"""
Vision pipeline for poke_map — Components 1–4:
  Component 1: Overhead camera intrinsics + ground-truth segmentation via projection
  Component 2: Contour extraction (segmentation → contour_current, contour_goal)
  Component 3: Input stacking (contour_current + contour_goal → (2, H, W))
  Component 4: Q-Network (U-Net → per-pixel Q-values)

Works in the Isaac Sim Script Editor after scene_setup_vectorized.py has run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#* Isaac Sim imports — guarded so the file can be unit-tested outside the editor
try:
    from isaacsim.core.experimental.prims import RigidPrim, XformPrim
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom
    _HAS_ISAAC = True
except ModuleNotFoundError:
    RigidPrim = None
    XformPrim = None
    stage_utils = None
    Gf = None
    UsdGeom = None
    _HAS_ISAAC = False

#* Project imports — match the convention from poke_executor_vectorized.py
PROJECT_ENV_DIR = Path("/home/xiao/0_codes/poke_map/env")
if PROJECT_ENV_DIR.exists() and str(PROJECT_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ENV_DIR))
from config_loader import load_config

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]
WORKSPACE = CONFIG["workspace"]

#* ---------------------------------------------------------------------------
#* Constants
#* ---------------------------------------------------------------------------
RESOLUTION = (128, 128)  # (H, W) — must match Q-network input
CAMERA_PATH = "/World/OverheadCamera"
CAMERA_HEIGHT = 1.2      # metres above table surface
ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
L_OBJECT_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/LObject"
CYLINDER_PATTERN = f"{ENVS_ROOT_PATH}/env_.*/Cylinder"
OBJECT_HEIGHT = SCENE_CONFIG["object_height"]


#* ================================================================
#*  Component 1: Overhead camera + ground-truth segmentation
#* ================================================================

def _require_isaac():
    if not _HAS_ISAAC:
        raise RuntimeError(
            "This function requires Isaac Sim. Run it inside the Isaac Sim Script Editor."
        )


def create_overhead_camera(
    height: float = CAMERA_HEIGHT,
    position: tuple[float, float] = (0.0, 0.0),
    resolution: tuple[int, int] = RESOLUTION,
    camera_path: str = CAMERA_PATH,
    focal_length: float = 24.0,
):
    """Create a top-down camera prim looking at the workspace.

    The camera faces straight down (-Z in Isaac's coordinate frame).
    After creation the intrinsics are stored so that image pixels map
    to world XY coordinates on the table surface.

    Parameters
    ----------
    height : distance from table to camera along Z.
    position : (x, y) of the camera in world frame.
    resolution : (H, W) pixel dimensions.
    camera_path : USD prim path.
    focal_length : horizontal aperture * focal length (hint for renderer).

    Returns
    -------
    UsdGeom.Camera prim.
    """
    _require_isaac()
    stage = stage_utils.get_current_stage(backend="usd")

    camera = UsdGeom.Camera.Define(stage, camera_path)
    xform = UsdGeom.Xformable(camera)

    #! Isaac Sim expects meters; the camera looks along +Y locally, so
    #! we rotate the prim to look down (-Z in world).
    transform_attr = camera.GetPrim().GetAttribute("xformOp:transform")
    if transform_attr and transform_attr.IsValid():
        xform_op = UsdGeom.XformOp(transform_attr)
    else:
        xform_op = xform.AddTransformOp()
    xform.SetXformOpOrder([xform_op])
    xform_op.Set(
        Gf.Matrix4d(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90.0),  # rotate to look -Z
            Gf.Vec3d(position[0], position[1], height),
        )
    )

    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetVerticalApertureAttr().Set(20.955 * resolution[0] / resolution[1])
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))

    print(
        f"Created overhead camera at {camera_path} "
        f"(h={height} m, {resolution[0]}×{resolution[1]})"
    )
    return camera


def get_env_paths_vectorized(envs_root_path: str = ENVS_ROOT_PATH) -> list[str]:
    """Return vectorized env root paths, e.g. /World/envs/env_0, env_1, ..."""
    _require_isaac()
    stage = stage_utils.get_current_stage(backend="usd")
    envs_root = stage.GetPrimAtPath(envs_root_path)
    if not envs_root or not envs_root.IsValid():
        raise RuntimeError(
            f"No vectorized env root found at {envs_root_path}. "
            "Run env/scene_setup_vectorized.py first."
        )

    def env_sort_key(path: str):
        name = path.rsplit("/", 1)[-1]
        if name.startswith("env_") and name[4:].isdigit():
            return int(name[4:])
        return name

    env_paths = [
        str(child.GetPath())
        for child in envs_root.GetChildren()
        if child.GetName().startswith("env_")
    ]
    return sorted(env_paths, key=env_sort_key)


def create_overhead_cameras_vectorized(
    height: float = CAMERA_HEIGHT,
    resolution: tuple[int, int] = RESOLUTION,
    camera_name: str = "OverheadCamera",
    focal_length: float = 24.0,
    envs_root_path: str = ENVS_ROOT_PATH,
) -> tuple[list, list[str]]:
    """Create one overhead camera under each vectorized environment.

    Cameras are parented under each env root, so their transform is env-local:
    /World/envs/env_0/OverheadCamera at local (0, 0, height), etc.
    """
    env_paths = get_env_paths_vectorized(envs_root_path=envs_root_path)
    cameras = []
    camera_paths = []

    for env_path in env_paths:
        camera_path = f"{env_path}/{camera_name}"
        cameras.append(
            create_overhead_camera(
                height=height,
                position=(0.0, 0.0),
                resolution=resolution,
                camera_path=camera_path,
                focal_length=focal_length,
            )
        )
        camera_paths.append(camera_path)

    print(f"Created {len(cameras)} vectorized overhead cameras.")
    return cameras, camera_paths


def get_camera_intrinsics(
    resolution: tuple[int, int] = RESOLUTION,
    height: float = CAMERA_HEIGHT,
    workspace_x_range: tuple[float, float] = None,
    workspace_y_range: tuple[float, float] = None,
) -> np.ndarray:
    """Return (3, 3) camera intrinsics matrix K for the overhead camera.

    When the camera is perfectly overhead, the intrinsics reduce to a
    simple scaling between pixels and world XY on the table plane.
    The principal point is at the image centre.

    Parameters
    ----------
    resolution : (H, W) in pixels.
    height : camera height above table (m).
    workspace_x_range : (x_min, x_max) visible in the image.  If None,
        computed from the config workspace bounds, centred at origin.
    workspace_y_range : (y_min, y_max).  Same treatment.

    Returns
    -------
    K : (3, 3) float32 array
        [fx,  0, cx]
        [ 0, fy, cy]
        [ 0,  0,  1]
    """
    H, W = resolution

    if workspace_x_range is None and workspace_y_range is None:
        wx_min = WORKSPACE.get("x_min", -0.30)
        wx_max = WORKSPACE.get("x_max", 0.30)
        wy_min = WORKSPACE.get("y_min", -0.30)
        wy_max = WORKSPACE.get("y_max", 0.30)
    else:
        wx_min, wx_max = workspace_x_range
        wy_min, wy_max = workspace_y_range

    world_width = wx_max - wx_min
    world_height = wy_max - wy_min

    fx = W / world_width
    fy = H / world_height
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def pixel_to_world(
    pixel_ij: tuple[int, int],
    K: np.ndarray,
    z_height: float = OBJECT_HEIGHT,
) -> np.ndarray:
    """Project a pixel (i, j) to a 3-D world coordinate on the table.

    Parameters
    ----------
    pixel_ij : (row, col).
    K : (3, 3) intrinsics.
    z_height : height of the contact surface (object top).

    Returns
    -------
    world_xyz : (3,) float32 — (x, y, z) in world frame.
    """
    i, j = pixel_ij
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (j - cx) / fx
    y = (i - cy) / fy
    return np.array([x, y, z_height], dtype=np.float32)


def world_to_pixel(
    world_xy: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> np.ndarray:
    """Project world (x, y) coordinates to image-space pixels.

    Parameters
    ----------
    world_xy : (..., 2) or (..., 3) — only x, y are used.
    K : (3, 3) intrinsics.
    resolution : (H, W).

    Returns
    -------
    pixels : (..., 2) int array — (row, col).
    """
    world_xy = np.atleast_2d(world_xy)[:, :2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    j = world_xy[:, 0] * fx + cx
    i = world_xy[:, 1] * fy + cy
    pixels = np.stack([i, j], axis=-1)
    return np.clip(pixels, 0, np.array(resolution) - 1).astype(np.int32)


#* ---------------------------------------------------------------------------
#* Ground-truth segmentation (no rendering — uses known object poses)
#* ---------------------------------------------------------------------------

def get_object_poses_vectorized(
    env_paths: list[str] | None = None,
) -> tuple[dict, np.ndarray]:
    """Query world poses for every object in every environment.

    Returns
    -------
    poses : dict mapping object category name →
        (positions (N, 3), quaternions (N, 4)).
    env_root_pos : (N_envs, 3) world positions of env root prims.
    """
    _require_isaac()
    env_roots = XformPrim(paths=f"{ENVS_ROOT_PATH}/env_.*")
    r_pos, _ = env_roots.get_world_poses()
    env_root_pos = as_numpy(r_pos)

    l_objects = RigidPrim(paths=L_OBJECT_PATTERN)
    cylinder_objects = RigidPrim(paths=CYLINDER_PATTERN)

    l_pos, l_quat = l_objects.get_world_poses()
    c_pos, c_quat = cylinder_objects.get_world_poses()

    poses = {
        "LObject": (as_numpy(l_pos), as_numpy(l_quat)),
        "Cylinder": (as_numpy(c_pos), as_numpy(c_quat)),
    }
    return poses, env_root_pos


def get_object_bounding_boxes_2d(
    object_positions: np.ndarray,       # (N, 3)
    object_orientations: np.ndarray,    # (N, 4) quat wxyz
    half_extents: tuple[float, float],   # (half_x, half_y) local
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> np.ndarray:
    """Project oriented bounding-box corners onto the image plane.

    Each object's footprint is a rectangle on the table; we compute
    the 4 corners, project with K, and return the convex hull (as
    a filled polygon) per object.

    Parameters
    ----------
    object_positions : (N, 3)
    object_orientations : (N, 4) quaternion [w, x, y, z]
    half_extents : (hx, hy) half-extents of the object footprint.
    K : (3, 3) intrinsics.
    resolution : (H, W).

    Returns
    -------
    masks : list of (H, W) bool ndarrays, one per object.
    """
    H, W = resolution
    N = object_positions.shape[0]
    masks = []

    hx, hy = half_extents
    #* local corners (half-extent box centred at origin)
    local_corners = np.array([
        [-hx, -hy],
        [ hx, -hy],
        [ hx,  hy],
        [-hx,  hy],
    ], dtype=np.float32)  # (4, 2)

    for n in range(N):
        #* rotation from quaternion → 2×2 (only yaw matters for footprint)
        qw, qx, qy, qz = object_orientations[n].astype(np.float32)
        rot = np.array([
            [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qw*qz],
            [2*qx*qy + 2*qw*qz,      1 - 2*qx*qx - 2*qz*qz],
        ], dtype=np.float32)  # rotation about Z

        world_corners = object_positions[n, :2] + (rot @ local_corners.T).T  # (4, 2)
        pixels = world_to_pixel(world_corners, K, resolution)  # (4, 2)

        #* fill convex polygon via scan-line (simple approach)
        mask = np.zeros((H, W), dtype=bool)
        r_min, r_max = pixels[:, 0].min(), pixels[:, 0].max()
        for r in range(max(0, r_min), min(H, r_max + 1)):
            crossings = []
            for k in range(4):
                p0 = pixels[k]
                p1 = pixels[(k + 1) % 4]
                if min(p0[0], p1[0]) <= r < max(p0[0], p1[0]):
                    t = (r - p0[0]) / (p1[0] - p0[0] + 1e-8)
                    c = p0[1] + t * (p1[1] - p0[1])
                    crossings.append(c)
            crossings.sort()
            for k in range(0, len(crossings), 2):
                if k + 1 < len(crossings):
                    c0 = max(0, int(crossings[k]))
                    c1 = min(W - 1, int(crossings[k + 1]))
                    if c1 > c0:
                        mask[r, c0:c1 + 1] = True
        masks.append(mask)

    return masks


def quaternion_to_yaw_rotation_2d(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion.astype(np.float32)
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qw*qz],
        [2*qx*qy + 2*qw*qz,      1 - 2*qx*qx - 2*qz*qz],
    ], dtype=np.float32)


def get_circle_masks_2d(
    object_positions: np.ndarray,
    radius: float,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> list[np.ndarray]:
    """Rasterize circular object footprints in top-down env-local coordinates."""
    H, W = resolution
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    rows, cols = np.ogrid[:H, :W]
    masks = []

    for position in object_positions:
        center_col = position[0] * fx + cx
        center_row = position[1] * fy + cy
        radius_cols = radius * fx
        radius_rows = radius * fy
        mask = (
            ((cols - center_col) / radius_cols) ** 2
            + ((rows - center_row) / radius_rows) ** 2
            <= 1.0
        )
        masks.append(mask)

    return masks


def get_circle_masks_2d_batched(
    object_positions: np.ndarray,  # (N, 3)
    radius: float,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> np.ndarray:
    """Rasterize N circular footprints at once via broadcasting → (N, H, W)."""
    H, W = resolution
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    center_col = object_positions[:, 0] * fx + cx  # (N,)
    center_row = object_positions[:, 1] * fy + cy  # (N,)
    radius_cols = radius * fx
    radius_rows = radius * fy

    rows, cols = np.ogrid[:H, :W]
    masks = (
        ((cols[None, :, :] - center_col[:, None, None]) / radius_cols) ** 2
        + ((rows[None, :, :] - center_row[:, None, None]) / radius_rows) ** 2
        <= 1.0
    )
    return masks  # (N, H, W) bool


def get_l_shape_masks_2d(
    object_positions: np.ndarray,
    object_orientations: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> list[np.ndarray]:
    """Rasterize the L object as the union of its two rectangular arms.

    The L root is the inner corner used in scene_setup_vectorized.py.  The two
    cube children are offset from that root, so a single centered square is the
    wrong footprint.
    """
    masks = []
    local_centers = [
        np.array([SCENE_CONFIG["l_thickness"] * 0.5, SCENE_CONFIG["l_arm_length"] * 0.5], dtype=np.float32),
        np.array([SCENE_CONFIG["l_arm_length"] * 0.5, SCENE_CONFIG["l_thickness"] * 0.5], dtype=np.float32),
    ]
    half_extents = [
        (SCENE_CONFIG["l_thickness"] * 0.5, SCENE_CONFIG["l_arm_length"] * 0.5),
        (SCENE_CONFIG["l_arm_length"] * 0.5, SCENE_CONFIG["l_thickness"] * 0.5),
    ]

    for position, quaternion in zip(object_positions, object_orientations):
        rot = quaternion_to_yaw_rotation_2d(quaternion)
        mask = np.zeros(resolution, dtype=bool)
        for local_center, extent in zip(local_centers, half_extents):
            arm_center = position[:2] + rot @ local_center
            arm_position = np.array([[arm_center[0], arm_center[1], position[2]]], dtype=np.float32)
            arm_orientation = quaternion[None, :]
            mask |= get_object_bounding_boxes_2d(
                object_positions=arm_position,
                object_orientations=arm_orientation,
                half_extents=extent,
                K=K,
                resolution=resolution,
            )[0]
        masks.append(mask)

    return masks


def get_object_masks_2d(
    object_name: str,
    object_positions: np.ndarray,
    object_orientations: np.ndarray,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
) -> list[np.ndarray]:
    if object_name == "LObject":
        return get_l_shape_masks_2d(
            object_positions=object_positions,
            object_orientations=object_orientations,
            K=K,
            resolution=resolution,
        )
    if object_name == "Cylinder":
        return get_circle_masks_2d(
            object_positions=object_positions,
            radius=SCENE_CONFIG["cylinder_radius"],
            K=K,
            resolution=resolution,
        )
    raise ValueError(f"Unknown object type: {object_name}")


def render_segmentation_ground_truth(
    env_poses: dict,
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
    env_root_pos: np.ndarray | None = None,
    per_env_half_extents: list[dict] | None = None,
) -> list[np.ndarray]:
    """Build instance segmentation for every environment.

    Each object is filled with a unique integer ID (1-indexed, 0 = bg).
    Object world positions are shifted by *env_root_pos* so that each
    env is treated as having its own local origin at the env root.

    Parameters
    ----------
    env_poses : dict from object name → (positions, quaternions).
    K : (3, 3) intrinsics.
    resolution : (H, W).
    env_root_pos : (N_envs, 3) world positions of env roots.  If None,
        objects are assumed to be in the camera's local frame already.

    Returns
    -------
    seg_maps : list of (H, W) int arrays, one per environment.
    """
    num_envs = list(env_poses.values())[0][0].shape[0]
    obj_names = list(env_poses.keys())  # deterministic iteration order
    H, W = resolution

    offset = env_root_pos[:, :2].copy() if env_root_pos is not None else np.zeros((num_envs, 2), dtype=np.float32)

    # Pre-render Cylinder masks in one batched pass
    cyl_key = "Cylinder"
    cyl_local = np.zeros((num_envs, 3), dtype=np.float32)
    if cyl_key in env_poses:
        cyl_world = env_poses[cyl_key][0]
        cyl_local[:, :2] = cyl_world[:, :2] - offset
        cyl_local[:, 2] = cyl_world[:, 2]
    cyl_masks = get_circle_masks_2d_batched(
        cyl_local, SCENE_CONFIG["cylinder_radius"], K, resolution
    )  # (N, H, W) bool

    seg_maps = []
    for env_idx in range(num_envs):
        seg = np.zeros((H, W), dtype=np.int32)
        obj_id = 0

        for obj_name in obj_names:
            obj_id += 1
            if obj_name == cyl_key:
                mask = cyl_masks[env_idx]
            else:
                positions = env_poses[obj_name][0]
                quaternions = env_poses[obj_name][1]
                local_pos = positions[env_idx:env_idx + 1].copy()
                local_pos[:, :2] -= offset[env_idx]
                masks = get_object_masks_2d(
                    object_name=obj_name,
                    object_positions=local_pos,
                    object_orientations=quaternions[env_idx:env_idx + 1],
                    K=K,
                    resolution=resolution,
                )
                mask = masks[0]
            seg[mask] = obj_id

        seg_maps.append(seg)

    return seg_maps


#* ================================================================
#*  Component 2: Contour extraction
#* ================================================================

def mask_to_contour(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one-pixel-thick boundary and its image-space centre of mass.

    contour = mask XOR erode(mask).

    Parameters
    ----------
    binary_mask : (H, W) bool.
    kernel_size : erosion kernel size.

    Returns
    -------
    contour : (H, W) bool.
    contour_com_pixel : (2,) float32
        Centre of mass of contour pixels in image coordinates (row, col).
        If the contour is empty, both values are NaN.
    """
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(binary_mask, iterations=1, structure=np.ones((kernel_size, kernel_size)))
    contour = binary_mask & ~eroded
    contour_pixels = np.argwhere(contour)
    if contour_pixels.size == 0:
        contour_com_pixel = np.array([np.nan, np.nan], dtype=np.float32)
    else:
        contour_com_pixel = contour_pixels.mean(axis=0).astype(np.float32)
    return contour, contour_com_pixel


def segmentation_to_per_object_contours(
    seg: np.ndarray,
    kernel_size: int = 3,
) -> dict[int, np.ndarray]:
    """Convert instance segmentation to per-object binary contour images.

    Parameters
    ----------
    seg : (H, W) int — pixel = object ID, 0 = background.
    kernel_size : erosion kernel.

    Returns
    -------
    dict[int, np.ndarray] — obj_id → (H, W) float32 contour in [0, 1].
    """
    contours: dict[int, np.ndarray] = {}
    for obj_id in np.unique(seg):
        if obj_id == 0:
            continue
        contour, _ = mask_to_contour(seg == obj_id, kernel_size=kernel_size)
        contours[int(obj_id)] = contour.astype(np.float32)
    return contours


def segmentation_to_contour_current(
    seg: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Merged object contours (for backward compat / debugging)."""
    combined = np.zeros_like(seg, dtype=np.float32)
    for contour in segmentation_to_per_object_contours(seg, kernel_size).values():
        combined = np.maximum(combined, contour)
    return combined


#* ---------------------------------------------------------------------------
#* Goal contour rendering (projects objects at target poses)
#* ---------------------------------------------------------------------------

def render_goal_contour(
    target_positions: dict[str, np.ndarray],   # object_name → (N, 3)
    target_orientations: dict[str, np.ndarray],  # object_name → (N, 4) quat
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
    kernel_size: int = 3,
) -> np.ndarray:
    """Render contour_goal: project each object at its target pose onto the
    image plane, fill, extract contour.

    Parameters
    ----------
    target_positions : per-object target world positions.
    target_orientations : per-object target orientations.
    K : (3, 3) intrinsics.
    resolution : (H, W).
    kernel_size : erosion kernel for contour.

    Returns
    -------
    contour_goal : (H, W) float32.
    """
    H, W = resolution
    combined = np.zeros((H, W), dtype=bool)

    for obj_name in target_positions:
        masks = get_object_masks_2d(
            object_name=obj_name,
            object_positions=target_positions[obj_name],
            object_orientations=target_orientations[obj_name],
            K=K,
            resolution=resolution,
        )
        for mask in masks:
            combined |= mask

    if not combined.any():
        return np.zeros((H, W), dtype=np.float32)

    contour, _ = mask_to_contour(combined, kernel_size=kernel_size)
    return contour.astype(np.float32)


def render_goal_contours_batched(
    object_name: str,
    target_positions: np.ndarray,        # (N, 3)
    target_orientations: np.ndarray,     # (N, 4) quat
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
    kernel_size: int = 3,
) -> np.ndarray:
    """Render per-env goal contours for one object type in a single batched pass.

    Returns (N, H, W) float32 contour array.
    """
    N = target_positions.shape[0]
    H, W = resolution

    if object_name == "Cylinder":
        radius = SCENE_CONFIG["cylinder_radius"]
        masks = get_circle_masks_2d_batched(target_positions, radius, K, resolution)  # (N, H, W)
    elif object_name == "LObject":
        # L-shape polygon fill is still per-env internally, but called once
        mask_list = get_l_shape_masks_2d(target_positions, target_orientations, K, resolution)
        masks = np.stack(mask_list, axis=0)  # (N, H, W)
    else:
        raise ValueError(f"Unknown object type: {object_name}")

    contours = np.zeros((N, H, W), dtype=np.float32)
    for n in range(N):
        if masks[n].any():
            contour, _ = mask_to_contour(masks[n], kernel_size=kernel_size)
            contours[n] = contour.astype(np.float32)
    return contours


def render_goal_contour_scene(
    target_positions: np.ndarray,       # (N_obj, 3) — all targets merged
    target_orientations: np.ndarray,    # (N_obj, 4)
    K: np.ndarray,
    half_extents: list[tuple[float, float]],
    resolution: tuple[int, int] = RESOLUTION,
    kernel_size: int = 3,
) -> np.ndarray:
    """Simplified version: single (N_obj, ...) arrays, not dicts.

    Use this when you have a flat list of target poses and their
    corresponding half-extents.
    """
    H, W = resolution
    combined = np.zeros((H, W), dtype=bool)

    for n in range(target_positions.shape[0]):
        hx, hy = half_extents[n]
        masks = get_object_bounding_boxes_2d(
            object_positions=target_positions[n:n + 1],
            object_orientations=target_orientations[n:n + 1],
            half_extents=(hx, hy),
            K=K,
            resolution=resolution,
        )
        combined |= masks[0]

    if not combined.any():
        return np.zeros((H, W), dtype=np.float32)

    contour, _ = mask_to_contour(combined, kernel_size=kernel_size)
    return contour.astype(np.float32)


#* ================================================================
#*  Component 3: Input stacking
#* ================================================================

def stack_input(
    contour_current: np.ndarray,  # (H, W) or (N, H, W)
    contour_goal: np.ndarray,     # (H, W) or (N, H, W)
) -> torch.Tensor:
    """Stack current and goal contours into a (2, H, W) or (N, 2, H, W) tensor.

    Parameters
    ----------
    contour_current : binary contour (float, [0, 1]).
    contour_goal : binary goal contour (float, [0, 1]).

    Returns
    -------
    x : torch.float32 tensor, shape (..., 2, H, W) where ... is the batch dim(s).
    """
    current = torch.as_tensor(contour_current, dtype=torch.float32)
    goal = torch.as_tensor(contour_goal, dtype=torch.float32)

    #* insert a channel dimension just before H, W
    #* (H, W) → (1, H, W); (N, H, W) → (N, 1, H, W)
    ndim = current.dim()
    current = current.unsqueeze(max(0, ndim - 2))
    goal = goal.unsqueeze(max(0, ndim - 2))

    return torch.cat([current, goal], dim=-3)  # cat along channel dim




#* ================================================================
#*  High-level pipeline
#* ================================================================

def build_vision_observation(
    env_poses: dict,
    target_positions: dict[str, np.ndarray],
    target_orientations: dict[str, np.ndarray],
    K: np.ndarray,
    resolution: tuple[int, int] = RESOLUTION,
    env_root_pos: np.ndarray | None = None,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Full observation pipeline: seg → per-object contours → stacked tensor.

    Parameters
    ----------
    env_poses : dict of object name → (positions (N,3), quaternions (N,4)).
    target_positions : per-object target poses (in world frame).
    target_orientations : per-object target orientations.
    K : (3, 3) camera intrinsics.
    resolution : (H, W).
    env_root_pos : (N_envs, 3) env root positions for per-env local coords.

    Returns
    -------
    x : (N_envs, 4, H, W) stacked contour tensor.
        Channels: [LObject_contour, Cylinder_contour, LObject_goal, Cylinder_goal]
    seg_maps : list of (H, W) int arrays, one per env (useful for debug).
    """
    num_envs = list(env_poses.values())[0][0].shape[0]
    obj_names = list(env_poses.keys())  # deterministic iteration order
    n_objects = len(obj_names)
    H, W = resolution

    seg_maps = render_segmentation_ground_truth(env_poses, K, resolution, env_root_pos)

    # Pre-render all goal contours in batched passes (one per object type)
    goal_contours = {}  # obj_name → (N, H, W) float32
    for obj_name in obj_names:
        goal_contours[obj_name] = render_goal_contours_batched(
            obj_name,
            target_positions[obj_name],
            target_orientations[obj_name],
            K, resolution,
        )

    all_channels = []
    for env_idx in range(num_envs):
        obj_contours = segmentation_to_per_object_contours(seg_maps[env_idx])

        current_channels = []
        for i in range(n_objects):
            obj_id = i + 1
            contour = obj_contours.get(obj_id, np.zeros((H, W), dtype=np.float32))
            current_channels.append(contour)

        goal_channels = [goal_contours[name][env_idx] for name in obj_names]
        env_channels = np.stack(current_channels + goal_channels, axis=0)  # (2*n_objects, H, W)
        all_channels.append(env_channels)

    x = torch.from_numpy(np.stack(all_channels, axis=0)).float()  # (N, 2*n_objects, H, W)
    return x, seg_maps


#* ================================================================
#*  Utility
#* ================================================================

def as_numpy(values):
    """Convert torch tensor / Isaac Sim wp.array / list-of-Vec3 to numpy.

    Isaac Sim's ``get_world_poses()`` may return a ``wp.array`` (warp GPU
    array), a ``torch.Tensor``, or a list of ``Gf.Vec3d``.  This normalises
    all of those to a ``(N, 3)`` float32 ndarray.
    """
    #* warp array (no direct indexing, but has .numpy() or .tolist())
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    elif hasattr(values, "tolist") and callable(values.tolist):
        values = np.asarray(values.tolist(), dtype=np.float32)
    try:
        result = np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError):
        #* fallback — e.g. list of Gf.Vec3d that NumPy can't infer shape from
        result = np.array([[float(c) for c in v] for v in values], dtype=np.float32)
    return result
