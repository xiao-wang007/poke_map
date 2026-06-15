"""
Run this in the Isaac Sim Script Editor AFTER scene_setup_vectorized.py has
finished.  It walks through the vision pipeline step by step so you can
inspect every intermediate output.

Usage (Script Editor):
    1. Window → Script Editor
    2. Open this file  (Ctrl-O)
    3. Run  (Ctrl-Enter)
    4. Inspect the globals:  K, seg_maps, contour_current, q_map, ...
"""

from __future__ import annotations

import sys
from pathlib import Path
import importlib

import numpy as np
from PIL import Image

#* We need the camera module from the vision package
VISION_DIR = Path("/home/xiao/0_codes/poke_map/vision")
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))
import camera as camera_module
camera_module = importlib.reload(camera_module)
from camera import (          # vision/camera.py
    create_overhead_camera,
    get_camera_intrinsics,
    get_object_poses_vectorized,
    render_segmentation_ground_truth,
    segmentation_to_contour_current,
    render_goal_contour,
    stack_input,
    QNetwork,
    pixel_to_world,
    world_to_pixel,
)
import torch


SAVE_DIR = Path("/home/xiao/0_codes/poke_map/vision_outputs")


def save_binary_image(array, path):
    image = (np.asarray(array) > 0).astype(np.uint8) * 255
    Image.fromarray(image).save(path)


def main():
    resolution = (128, 128)

    #* ------------------------------------------------------------------
    #* Step 0 — DEBUG: dump object positions, including env root offsets
    #* ------------------------------------------------------------------
    print(">>> Step 0: Debug — get env root + object poses ...")
    from isaacsim.core.experimental.prims import RigidPrim, XformPrim
    ENVS_ROOT = "/World/envs"

    def _to_numpy(v):
        if hasattr(v, "numpy") and callable(v.numpy):
            return v.numpy()
        if hasattr(v, "tolist") and callable(v.tolist):
            return np.asarray(v.tolist(), dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    env_roots = XformPrim(paths=f"{ENVS_ROOT}/env_.*")
    env_root_pos, _ = env_roots.get_world_poses()
    env_root_pos = _to_numpy(env_root_pos)

    l_prims = RigidPrim(paths=f"{ENVS_ROOT}/env_.*/LObject")
    c_prims = RigidPrim(paths=f"{ENVS_ROOT}/env_.*/Cylinder")
    l_world = _to_numpy(l_prims.get_world_poses()[0])
    c_world = _to_numpy(c_prims.get_world_poses()[0])

    print(f"    env_0 root     : {env_root_pos[0]}")
    print(f"    LObject world  : {l_world[0]}")
    l_rel = l_world[0, :2] - env_root_pos[0, :2]
    c_rel = c_world[0, :2] - env_root_pos[0, :2]
    print(f"    LObject rel    : {l_rel}")
    print(f"    Cylinder rel   : {c_rel}")

    K_test = get_camera_intrinsics(resolution)
    l_pix = world_to_pixel(l_rel, K_test, resolution)
    c_pix = world_to_pixel(c_rel, K_test, resolution)
    l_ok = 0 <= l_pix[0, 0] < 128 and 0 <= l_pix[0, 1] < 128
    c_ok = 0 <= c_pix[0, 0] < 128 and 0 <= c_pix[0, 1] < 128
    print(f"    LObject pixel  : ({l_pix[0, 0]}, {l_pix[0, 1]})  in_view={l_ok}")
    print(f"    Cylinder pixel : ({c_pix[0, 0]}, {c_pix[0, 1]})  in_view={c_ok}")

    #! one camera looking at vectorized envs
    #* ------------------------------------------------------------------
    #* Step 1 — Create the overhead camera 
    #* ------------------------------------------------------------------
    print(">>> Step 1: Creating overhead camera ...")
    camera = create_overhead_camera(resolution=resolution)
    K = get_camera_intrinsics(resolution)
    print(f"    K =\n{K}")

    #* ------------------------------------------------------------------
    #* Step 2 — Get object poses from the vectorised environments
    #* ------------------------------------------------------------------
    print(">>> Step 2: Fetching object poses ...")
    env_poses, env_root_pos = get_object_poses_vectorized()
    num_envs = list(env_poses.values())[0][0].shape[0]
    print(f"    {num_envs} environments, root_shape={env_root_pos.shape}")
    for name, (pos, _) in env_poses.items():
        print(f"    {name}: {pos.shape}")

    #* ------------------------------------------------------------------
    #* Step 3 — Build ground-truth segmentation maps
    #* ------------------------------------------------------------------
    print(">>> Step 3: Building segmentation maps ...")
    seg_maps = render_segmentation_ground_truth(env_poses, K, resolution, env_root_pos=env_root_pos)

    #* ------------------------------------------------------------------
    #* Step 4 — Extract contour_current for env_0
    #* ------------------------------------------------------------------
    print(">>> Step 4: Extracting contour_current (env_0) ...")
    contour_current = segmentation_to_contour_current(seg_maps[0])
    print(f"    contour pixels : {int(contour_current.sum())}")

    #* ------------------------------------------------------------------
    #* Step 5 — Render contour_goal in env-local coords
    #* ------------------------------------------------------------------
    print(">>> Step 5: Rendering contour_goal ...")
    #! Targets in env-local coords (within the [-0.25, 0.25] workspace).
    #! The LObject currently sits at ~(-0.12, -0.10) — set its target to (0.20, -0.15).
    #! The Cylinder is at ~(-0.09, 0.21) — set its target to (-0.15, 0.05).
    target_positions = {
        "LObject": np.array([[0.20, -0.15, 0.04]], dtype=np.float32),
        "Cylinder": np.array([[-0.15, 0.05, 0.04]], dtype=np.float32),
    }
    target_orientations = {
        "LObject": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "Cylinder": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    contour_goal = render_goal_contour(
        target_positions, target_orientations, K, resolution
    )
    print(f"    goal contour pixels : {int(contour_goal.sum())}")
    print(f"    target LObject : {target_positions['LObject'][0]}")
    print(f"    target Cylinder: {target_positions['Cylinder'][0]}")

    #* ------------------------------------------------------------------
    #* Step 5.5 — Save contours for inspection
    #* ------------------------------------------------------------------
    print(">>> Step 5.5: Saving contours ...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    contour_current_png = SAVE_DIR / "contour_current_env0.png"
    contour_goal_png = SAVE_DIR / "contour_goal_env0.png"
    contour_current_npy = SAVE_DIR / "contour_current_env0.npy"
    contour_goal_npy = SAVE_DIR / "contour_goal_env0.npy"
    save_binary_image(contour_current, contour_current_png)
    save_binary_image(contour_goal, contour_goal_png)
    np.save(contour_current_npy, contour_current)
    np.save(contour_goal_npy, contour_goal)
    print(f"    saved current PNG : {contour_current_png}")
    print(f"    saved goal PNG    : {contour_goal_png}")

    #* ------------------------------------------------------------------
    #* Step 6 — Stack into input tensor
    #* ------------------------------------------------------------------
    print(">>> Step 6: Stacking input ...")
    x = stack_input(contour_current, contour_goal)
    print(f"    input tensor shape : {tuple(x.shape)}   # (2, H, W)")

    #* ------------------------------------------------------------------
    #* Step 7 — Forward pass through a fresh Q-network
    #* ------------------------------------------------------------------
    print(">>> Step 7: Q-network forward pass ...")
    qnet = QNetwork(in_channels=2, base_channels=16)
    qnet.eval()
    with torch.no_grad():
        q_map = qnet(x.unsqueeze(0)).squeeze().numpy()  # (H, W)
    print(f"    Q-map shape : {q_map.shape}")
    print(f"    Q-map range : [{q_map.min():.3f}, {q_map.max():.3f}]")

    #* ------------------------------------------------------------------
    #* Step 8 — Show which pixel the policy would pick
    #* ------------------------------------------------------------------
    print(">>> Step 8: Policy pick (argmax) ...")
    best_pixel = np.unravel_index(np.argmax(q_map), q_map.shape)  # (row, col)
    world_xyz = pixel_to_world(best_pixel, K)
    print(f"    argmax pixel  : {best_pixel}  (row={best_pixel[0]}, col={best_pixel[1]})")
    print(f"    world contact : ({world_xyz[0]:.3f}, {world_xyz[1]:.3f}, {world_xyz[2]:.3f})")

    #* ------------------------------------------------------------------
    #* Expose everything so you can inspect interactively
    #* ------------------------------------------------------------------
    globals().update(
        {
            "camera": camera,
            "K": K,
            "resolution": resolution,
            "env_poses": env_poses,
            "env_root_pos": env_root_pos,
            "seg_maps": seg_maps,
            "contour_current": contour_current,
            "contour_goal": contour_goal,
            "save_dir": SAVE_DIR,
            "contour_current_png": contour_current_png,
            "contour_goal_png": contour_goal_png,
            "contour_current_npy": contour_current_npy,
            "contour_goal_npy": contour_goal_npy,
            "x": x,
            "qnet": qnet,
            "q_map": q_map,
            "best_pixel": best_pixel,
            "world_xyz": world_xyz,
        }
    )

    print("\nDone.  Inspect the globals above — e.g.  q_map, contour_current, seg_maps[0].")
    print("Use 'import matplotlib.pyplot as plt; plt.imshow(q_map); plt.show()' to visualise.")


main()
