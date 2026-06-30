from __future__ import annotations

import numpy as np
import torch

from config_loader import load_config


CONFIG = load_config()
WORKSPACE_CONFIG = CONFIG["workspace"]
TRAIN_CONFIG = CONFIG["training"]

STOP_POKE_THRESHOLD: float = TRAIN_CONFIG["stop_poke_threshold"]
MAX_STEPS: int = TRAIN_CONFIG["max_steps_per_episode"]
MIN_ACTIVE_ENVS: int = TRAIN_CONFIG["min_active_envs"]


def objects_in_workspace(
    poses: dict,
    env_root_pos: np.ndarray,
    wx_min: float = float(WORKSPACE_CONFIG["x_min"]),
    wx_max: float = float(WORKSPACE_CONFIG["x_max"]),
    wy_min: float = float(WORKSPACE_CONFIG["y_min"]),
    wy_max: float = float(WORKSPACE_CONFIG["y_max"]),
) -> np.ndarray:
    """Return True for envs where all object centers are inside the workspace."""
    num_envs = list(poses.values())[0][0].shape[0]
    inside = np.ones(num_envs, dtype=bool)
    offset = env_root_pos[:, :2]
    for _, (positions, _) in poses.items():
        local_xy = positions[:, :2].copy() - offset
        inside &= (local_xy[:, 0] >= wx_min) & (local_xy[:, 0] <= wx_max)
        inside &= (local_xy[:, 1] >= wy_min) & (local_xy[:, 1] <= wy_max)
    return inside


def near_translation_targets(
    poses: dict,
    targets: dict,
    env_root_pos: np.ndarray,
    threshold: float = STOP_POKE_THRESHOLD,
) -> np.ndarray:
    """Return envs where all object centers are close enough to their targets."""
    num_envs = list(poses.values())[0][0].shape[0]
    near = np.ones(num_envs, dtype=bool)
    offset = env_root_pos[:, :2]
    for obj_name, (positions, _) in poses.items():
        local_xy = positions[:, :2].copy() - offset
        dist = np.linalg.norm(local_xy - targets[obj_name][:, :2], axis=1)
        near &= dist < threshold
    return near


class EpisodeTermination:
    """Centralize per-env and whole-episode stop bookkeeping."""

    def __init__(
        self,
        num_envs: int,
        targets: dict,
        env_root_pos: np.ndarray,
        max_steps: int = MAX_STEPS,
        min_active_envs: int = MIN_ACTIVE_ENVS,
    ):
        self.done_once = np.zeros(num_envs, dtype=bool)
        self.targets = targets
        self.env_root_pos = env_root_pos
        self.max_steps = max_steps
        self.min_active_envs = min_active_envs

    @staticmethod
    def contour_visible(contour_masks: torch.Tensor) -> np.ndarray:
        return contour_masks.any(dim=(-2, -1)).cpu().numpy()

    def begin_step(
        self,
        poses: dict,
        contour_masks: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        has_contour = self.contour_visible(contour_masks)
        self.done_once |= near_translation_targets(
            poses, self.targets, self.env_root_pos)
        self.done_once |= ~objects_in_workspace(poses, self.env_root_pos)
        was_active = ~self.done_once & has_contour
        return was_active, has_contour

    def apply_reward_done(self, done_once: np.ndarray):
        self.done_once = done_once

    def finish_step(
        self,
        poses: dict,
        contour_masks_next: torch.Tensor,
    ) -> tuple[np.ndarray, int, bool]:
        self.done_once |= near_translation_targets(
            poses, self.targets, self.env_root_pos)
        self.done_once |= ~objects_in_workspace(poses, self.env_root_pos)
        has_contour_next = self.contour_visible(contour_masks_next)
        remaining_active = int((~self.done_once & has_contour_next).sum())
        early_truncated = (
            not self.done_once.all()
            and remaining_active <= self.min_active_envs
        )
        return has_contour_next, remaining_active, early_truncated

    def replay_done(self, env_idx: int, step: int, early_truncated: bool) -> bool:
        truncated = (step == self.max_steps - 1) or early_truncated
        return bool(self.done_once[env_idx] or truncated)

    def stop_reason(self, early_truncated: bool) -> str | None:
        if self.done_once.all():
            return "all_done"
        if early_truncated:
            return "min_active"
        return None
