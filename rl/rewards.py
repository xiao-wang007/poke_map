from __future__ import annotations

import numpy as np

from config_loader import load_config
from rl.termination import objects_in_workspace


CONFIG = load_config()
TRAIN_CONFIG = CONFIG["training"]

C_STEP: float = TRAIN_CONFIG["c_step"]
C_SUCCESS: float = TRAIN_CONFIG["c_success"]
C_OOB: float = TRAIN_CONFIG["c_oob"]
C_TARGET_SEPARATION: float = TRAIN_CONFIG["c_target_separation"]
SUCCESS_THRESHOLD: float = TRAIN_CONFIG["success_threshold"]
C_YAW: float = TRAIN_CONFIG["c_yaw"]
YAW_REWARD_ENABLED: bool = TRAIN_CONFIG["yaw_reward_enabled"]
YAW_SUCCESS_ENABLED: bool = TRAIN_CONFIG["yaw_success_enabled"]
YAW_THRESHOLD: float = np.deg2rad(TRAIN_CONFIG["yaw_threshold_deg"])


def _yaw_error(quats: np.ndarray, target_quats: np.ndarray) -> np.ndarray:
    """Wrapped angle difference between quaternion yaws in radians."""
    yaw = 2.0 * np.arctan2(quats[:, 3], quats[:, 0])
    target_yaw = 2.0 * np.arctan2(target_quats[:, 3], target_quats[:, 0])
    diff = yaw - target_yaw
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def compute_rewards(
    poses_before: dict,
    poses_after: dict,
    targets: dict,
    env_root_pos: np.ndarray | None = None,
    done_once: np.ndarray | None = None,
    target_oris: dict | None = None,
    yaw_reward_enabled: bool = YAW_REWARD_ENABLED,
    yaw_success_enabled: bool = YAW_SUCCESS_ENABLED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-env reward, done flags, and the updated done-once mask."""
    num_envs = list(poses_before.values())[0][0].shape[0]
    reward = np.zeros(num_envs, dtype=np.float32)
    done = np.ones(num_envs, dtype=bool)
    if done_once is None:
        done_once = np.zeros(num_envs, dtype=bool)

    offset = (
        env_root_pos[:, :2]
        if env_root_pos is not None
        else np.zeros((num_envs, 2))
    )

    local_before_xy: dict[str, np.ndarray] = {}
    local_after_xy: dict[str, np.ndarray] = {}

    for obj_name in poses_before:
        pos_before = poses_before[obj_name][0].copy()
        pos_after = poses_after[obj_name][0].copy()
        pos_before[:, :2] -= offset
        pos_after[:, :2] -= offset
        local_before_xy[obj_name] = pos_before[:, :2]
        local_after_xy[obj_name] = pos_after[:, :2]

        d_before = np.linalg.norm(
            pos_before[:, :2] - targets[obj_name][:, :2], axis=1)
        d_after = np.linalg.norm(
            pos_after[:, :2] - targets[obj_name][:, :2], axis=1)
        reward += d_before - d_after
        done &= d_after < SUCCESS_THRESHOLD

        if obj_name == "LObject" and target_oris is not None and obj_name in target_oris:
            y_err_before = _yaw_error(poses_before[obj_name][1], target_oris[obj_name])
            y_err_after = _yaw_error(poses_after[obj_name][1], target_oris[obj_name])
            if yaw_reward_enabled:
                reward += C_YAW * (y_err_before - y_err_after)
            if yaw_success_enabled:
                done &= y_err_after < YAW_THRESHOLD

    #! target-relative separation shaping
    #! So if the targets are close together, the reward encourages the objects to 
    #! become close too. If the targets are separated, it rewards separating them. 
    #! This avoids the bad “always push objects apart” behavior.
    obj_names = [name for name in poses_before if name in targets]
    for i, name_i in enumerate(obj_names):
        for name_j in obj_names[i + 1:]:
            sep_before = np.linalg.norm(
                local_before_xy[name_i] - local_before_xy[name_j], axis=1)
            sep_after = np.linalg.norm(
                local_after_xy[name_i] - local_after_xy[name_j], axis=1)
            target_sep = np.linalg.norm(
                targets[name_i][:, :2] - targets[name_j][:, :2], axis=1)
            sep_err_before = np.abs(sep_before - target_sep)
            sep_err_after = np.abs(sep_after - target_sep)
            reward += C_TARGET_SEPARATION * (sep_err_before - sep_err_after)

    reward -= C_STEP

    oob_after = ~objects_in_workspace(poses_after, env_root_pos)
    reward[oob_after] += C_OOB

    # success bonus: only for genuine task completion, not OOB
    first_success = done & ~done_once & ~oob_after
    reward[first_success] += C_SUCCESS

    done |= oob_after
    done_once = done_once | done

    return reward, done, done_once
