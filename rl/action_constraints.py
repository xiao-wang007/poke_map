from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vision.camera import pixel_to_world


def selected_object_target_dirs(
    pixel_ij: np.ndarray,
    poses: dict,
    targets: dict[str, np.ndarray],
    env_root_pos: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the target-facing direction for the object nearest each pixel."""
    num_envs = pixel_ij.shape[0]
    target_dirs = np.zeros((num_envs, 2), dtype=np.float32)
    valid = np.zeros(num_envs, dtype=bool)
    if not targets:
        return target_dirs, valid

    env_offsets = env_root_pos[:, :2]
    contact_xy = np.zeros((num_envs, 2), dtype=np.float32)
    for b in range(num_envs):
        contact_xy[b] = pixel_to_world(tuple(pixel_ij[b]), K)[:2]

    object_names = list(targets.keys())
    object_local_xy = []
    for obj_name in object_names:
        object_local_xy.append(poses[obj_name][0][:, :2].copy() - env_offsets)
    object_local_xy = np.stack(object_local_xy, axis=0)  # (O, B, 2)

    contact_dist = np.linalg.norm(
        object_local_xy - contact_xy[None, :, :], axis=2
    )
    selected_obj_idx = np.argmin(contact_dist, axis=0)

    for b in range(num_envs):
        obj_idx = int(selected_obj_idx[b])
        obj_name = object_names[obj_idx]
        to_target = targets[obj_name][b, :2] - object_local_xy[obj_idx, b]
        norm = float(np.linalg.norm(to_target))
        if norm > 1e-6:
            target_dirs[b] = to_target / norm
            valid[b] = True

    return target_dirs, valid


def constrain_target_half_plane(
    pixel_ij: np.ndarray,
    d_xy: np.ndarray,
    poses: dict,
    targets: dict[str, np.ndarray],
    env_root_pos: np.ndarray,
    K: np.ndarray,
    active: np.ndarray | None = None,
    min_dot: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror/project action directions into the selected object's target half-plane.

    ``min_dot=0`` means exactly the target-facing half-plane:
    dot(action_dir, object_to_target_dir) >= 0.  Set ``min_dot=-1`` to disable.
    """
    dirs = np.asarray(d_xy, dtype=np.float32).copy()
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    zero_dir = norms[:, 0] < 1e-8
    dirs = dirs / np.where(norms < 1e-8, 1.0, norms)

    target_dirs, valid_target = selected_object_target_dirs(
        pixel_ij, poses, targets, env_root_pos, K
    )
    active_mask = (
        np.ones(dirs.shape[0], dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool)
    )
    valid = valid_target & active_mask
    dir_po = np.zeros(dirs.shape[0], dtype=bool)
    min_dot = float(np.clip(min_dot, -1.0, 0.99))

    for b in range(dirs.shape[0]):
        if not valid[b] or min_dot <= -1.0:
            continue

        goal_dir = target_dirs[b]
        if zero_dir[b]:
            dirs[b] = goal_dir
            dir_po[b] = True
            continue

        dot = float(np.dot(dirs[b], goal_dir))
        if dot >= min_dot:
            continue

        candidate = dirs[b]
        if dot < 0.0:
            candidate = candidate - 2.0 * dot * goal_dir

        cand_norm = float(np.linalg.norm(candidate))
        if cand_norm < 1e-8:
            candidate = goal_dir
        else:
            candidate = candidate / cand_norm

        cand_dot = float(np.dot(candidate, goal_dir))
        if cand_dot < min_dot:
            tangent = candidate - cand_dot * goal_dir
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-8:
                tangent = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)
            else:
                tangent = tangent / tangent_norm
            forward = max(min_dot, 0.0)
            lateral = float(np.sqrt(max(0.0, 1.0 - forward * forward)))
            candidate = forward * goal_dir + lateral * tangent

        dirs[b] = candidate / max(float(np.linalg.norm(candidate)), 1e-8)
        dir_po[b] = True

    return dirs.astype(np.float32), target_dirs.astype(np.float32), dir_po


def constrain_params_target_half_plane_torch(
    params: torch.Tensor,
    target_dirs: torch.Tensor,
    min_dot: float = 0.0,
) -> torch.Tensor:
    """Torch version for actor/Q conditioning in the replay update."""
    min_dot = float(np.clip(min_dot, -1.0, 0.99))
    if min_dot <= -1.0:
        return params

    dirs = F.normalize(params[:, :2], dim=1, eps=1e-6)
    velocities = params[:, 2:3]

    goal_norm = target_dirs.norm(dim=1, keepdim=True)
    valid = goal_norm > 1e-6
    goal_dirs = target_dirs / goal_norm.clamp_min(1e-6)

    dot = (dirs * goal_dirs).sum(dim=1, keepdim=True)

    #! flips only the component along g, while keeping 
    #! the sideways component the same.
    reflected = F.normalize(dirs - 2.0 * dot * goal_dirs, dim=1, eps=1e-6)

    #! use the reflected direction only when the policy direction 
    #! is in the wrong half-plane. Otherwise keep the original direction.
    constrained = torch.where((dot < 0.0) & valid, reflected, dirs)

    if min_dot > 0.0:
        constrained_dot = (constrained * goal_dirs).sum(dim=1, keepdim=True)
        tangent = constrained - constrained_dot * goal_dirs
        tangent_norm = tangent.norm(dim=1, keepdim=True)
        default_tangent = torch.stack(
            [-goal_dirs[:, 1], goal_dirs[:, 0]], dim=1
        )
        tangent = torch.where(
            tangent_norm > 1e-6,
            tangent / tangent_norm.clamp_min(1e-6),
            default_tangent,
        )
        forward = torch.as_tensor(
            min_dot, dtype=params.dtype, device=params.device
        )
        lateral = torch.sqrt(torch.clamp(1.0 - forward * forward, min=0.0))
        cone_dir = F.normalize(
            forward * goal_dirs + lateral * tangent,
            dim=1,
            eps=1e-6,
        )
        constrained = torch.where(
            (constrained_dot < min_dot) & valid,
            cone_dir,
            constrained,
        )

    return torch.cat([constrained, velocities], dim=1)
