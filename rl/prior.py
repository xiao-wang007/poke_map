from __future__ import annotations

import numpy as np
import torch

from config_loader import load_config
from nn.networks import SpatialActorCritic
from vision.camera import (
    RESOLUTION,
    get_object_masks_2d,
    mask_to_contour,
)


CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
TRAIN_CONFIG = CONFIG["training"]

CYLINDER_RADIUS = SCENE_CONFIG["cylinder_radius"]
STOP_POKE_THRESHOLD: float = TRAIN_CONFIG["stop_poke_threshold"]
VELOCITY_MAX: float = TRAIN_CONFIG["velocity_max"]
HEURISTIC_VELOCITY_RANGE: tuple[float, float] = tuple(
    TRAIN_CONFIG["heuristic_velocity_range"])


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


def sample_heuristic_velocity() -> np.float32:
    low, high = HEURISTIC_VELOCITY_RANGE
    low = max(0.0, float(low))
    high = min(float(high), VELOCITY_MAX)
    if high < low:
        high = low
    return np.float32(np.random.uniform(low, high))


def heuristic_poke_actions(
    contour_masks: torch.Tensor,
    poses_before: dict,
    targets_pos: dict[str, np.ndarray],
    env_root_pos: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physically plausible exploration pokes for epsilon actions."""
    batch_size = contour_masks.shape[0]
    pixel_ij = np.zeros((batch_size, 2), dtype=np.int32)
    d_xy = np.zeros((batch_size, 2), dtype=np.float32)
    delta_d = np.full(batch_size, HEURISTIC_VELOCITY_RANGE[0], dtype=np.float32)

    masks_cpu = contour_masks.detach().cpu()
    env_offsets = env_root_pos[:, :2]

    for b in range(batch_size):
        valid_pixels_t = torch.nonzero(masks_cpu[b], as_tuple=False)
        if valid_pixels_t.numel() == 0:
            continue

        object_infos = []
        for obj_name, target in targets_pos.items():
            current_xy = poses_before[obj_name][0][b, :2] - env_offsets[b]
            target_xy = target[b, :2]
            to_target = target_xy - current_xy
            dist = float(np.linalg.norm(to_target))
            object_infos.append((dist, obj_name, current_xy, to_target))

        movable_infos = [
            info for info in object_infos if info[0] >= STOP_POKE_THRESHOLD
        ]
        if not movable_infos:
            movable_infos = object_infos
        _, obj_name, current_xy, to_target = max(
            movable_infos, key=lambda item: item[0])
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
            nearest = int(np.argmin(
                np.linalg.norm(contact_xys - desired_contact_xy[None, :], axis=1)
            ))
            pixel_ij[b] = valid_pixels[nearest]
            noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.05
            d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
            delta_d[b] = sample_heuristic_velocity()
            continue

        inward_scores = (current_xy[None, :] - contact_xys) @ goal_dir
        candidate_mask = inward_scores > 0.0
        candidate_pixels = valid_pixels[candidate_mask]
        candidate_xys = contact_xys[candidate_mask]

        if candidate_pixels.shape[0] == 0:
            candidate_pixels = valid_pixels
            candidate_xys = contact_xys

        backside_scores = (candidate_xys - current_xy[None, :]) @ (-goal_dir)
        top_k = min(25, candidate_pixels.shape[0])
        top_idx = np.argpartition(backside_scores, -top_k)[-top_k:]
        chosen = int(np.random.choice(top_idx))

        pixel_ij[b] = candidate_pixels[chosen]
        noisy_dir = goal_dir + np.random.randn(2).astype(np.float32) * 0.10
        d_xy[b] = _normalise_xy(noisy_dir, fallback=goal_dir)
        delta_d[b] = sample_heuristic_velocity()

    return pixel_ij, d_xy, delta_d


def select_action(
    actor_critic: SpatialActorCritic,
    x: torch.Tensor,
    contour_masks: torch.Tensor,
    epsilon: float,
    noise_std: float,
    heuristic_actions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select epsilon-prior or greedy policy actions for a batch."""
    batch_size = x.shape[0]
    device = x.device
    actor_critic.eval()

    pixel_ij = np.zeros((batch_size, 2), dtype=np.int32)
    d_xy_arr = np.zeros((batch_size, 2), dtype=np.float32)
    velocity_arr = np.zeros(batch_size, dtype=np.float32)
    is_policy = np.zeros(batch_size, dtype=bool)
    greedy_velocity_arr = np.zeros(batch_size, dtype=np.float32)

    with torch.no_grad():
        features, params, _ = actor_critic.features_and_params(x)
        greedy_pix, _ = actor_critic.greedy_pixel(
            features, params, contour_masks, top_k=top_k)
        batch_idx = torch.arange(batch_size, device=device)
        greedy_velocity_arr = (
            params[batch_idx, 2, greedy_pix[:, 0], greedy_pix[:, 1]]
            .detach().cpu().numpy().astype(np.float32)
        )

        for b in range(batch_size):
            if contour_masks[b].sum() == 0:
                velocity_arr[b] = 0.0
                continue

            if np.random.random() < epsilon:
                if heuristic_actions is not None:
                    h_pixel, h_dxy, h_delta = heuristic_actions
                    pixel_ij[b] = h_pixel[b]
                    d_noisy = h_dxy[b]
                    dd_noisy = h_delta[b]
                else:
                    valid = torch.nonzero(contour_masks[b], as_tuple=True)
                    chosen = np.random.randint(len(valid[0]))
                    pixel_ij[b, 0] = valid[0][chosen].item()
                    pixel_ij[b, 1] = valid[1][chosen].item()
                    d_noisy = np.random.randn(2).astype(np.float32)
                    dd_noisy = sample_heuristic_velocity()
            else:
                pixel_ij[b, 0] = greedy_pix[b, 0].item()
                pixel_ij[b, 1] = greedy_pix[b, 1].item()
                selected_params = params[b, :, pixel_ij[b, 0], pixel_ij[b, 1]].cpu().numpy()
                d_noisy = selected_params[:2] + (
                    np.random.randn(2).astype(np.float32) * noise_std
                )
                dd_noisy = selected_params[2] + (
                    abs(np.float32(np.random.randn())) * noise_std
                )
                is_policy[b] = True

            norm = np.linalg.norm(d_noisy) + 1e-8
            d_xy_arr[b] = d_noisy / norm
            velocity_arr[b] = np.clip(dd_noisy, 0.0, actor_critic.velocity_max)

    return pixel_ij, d_xy_arr, velocity_arr, is_policy, greedy_velocity_arr
