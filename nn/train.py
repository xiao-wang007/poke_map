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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "nn", PROJECT_ROOT / "env",
          PROJECT_ROOT / "vision"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

#* -- Isaac Sim ------------------------------------------------------------
try:
    from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
    import isaacsim.core.experimental.utils.app as app_utils
    from omni.kit.async_engine import run_coroutine
    _HAS_ISAAC = True
except ModuleNotFoundError:
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
from env.scene_setup_articulated_vectorized import randomize_object_poses

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]

OBJECT_HEIGHT = SCENE_CONFIG["object_height"]
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
BATCH_SIZE: int = 64
BUFFER_CAPACITY: int = 25_000
EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: int = 2_000           # episodes over which ε decays
SIGMA_START: float = 0.3
SIGMA_END: float = 0.05
TAU: float = 0.005               # polyak averaging coefficient
MAX_STEPS: int = 30              # max pokes per episode
C_STEP: float = 0.01             # per-step penalty
C_SUCCESS: float = 10.0          # terminal success bonus
SUCCESS_THRESHOLD: float = 0.02  # metres — tolerance to target
DELTA_D_MAX: float = 0.2          # max standoff (m)
IMPACT_STEPS: int = 40            # physics steps per strike (k_p=200, m≈2 kg)
OVERTRAVEL: float = 0.05          # finger pushes this far past contact (m)
SAFE_Z: float = FINGER_CONFIG["default_xyz"][2]  # safe height (m)

# yaw curriculum — translation-first, orientation added gradually
C_YAW: float = 0.5                     # weight on orientation progress reward
YAW_THRESHOLD: float = np.deg2rad(10)  # radians — tolerance for success
YAW_CURRICULUM_START: int = 0          # episode to begin expanding yaw range
YAW_CURRICULUM_END: int = 500          # episode to reach full yaw range
YAW_FULL_RANGE: float = np.pi          # radians — ±180° full orientation

TRAIN_AFTER: int = 256           # start training after this many transitions
TRAIN_EVERY: int = 4             # train every N env steps
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_EVERY: int = 100       # episodes between saves
CHECKPOINT_KEEP: int = 3          # how many recent checkpoints to keep


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


#* ================================================================
#*  Action selection
#* ================================================================

def select_action(
    actor_critic: SpatialActorCritic,
    x: torch.Tensor,                       # (B, 2, H, W)
    contour_masks: torch.Tensor,           # (B, H, W) bool
    epsilon: float,
    noise_std: float,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select actions for a batch of observations.

    Greedy envs use per-pixel FiLM-conditioned pixel selection (top-K
    re-evaluation).  ε-greedy envs pick a random valid pixel.

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
                # ε-greedy: random valid pixel
                valid = torch.nonzero(contour_masks[b], as_tuple=True)
                r = np.random.randint(len(valid[0]))
                pixel_ij[b, 0] = valid[0][r].item()
                pixel_ij[b, 1] = valid[1][r].item()
            else:
                # greedy: pre-computed per-pixel-conditioned best pixel
                pixel_ij[b, 0] = greedy_pix[b, 0].item()
                pixel_ij[b, 1] = greedy_pix[b, 1].item()

            # read params at selected pixel
            p = params[b, :, pixel_ij[b, 0], pixel_ij[b, 1]].cpu().numpy()  # (3,)

            d_noisy = p[:2] + np.random.randn(2).astype(np.float32) * noise_std
            dd_noisy = p[2] + abs(np.random.randn().astype(np.float32)) * noise_std

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


def compute_rewards(
    poses_before: dict,         # obj_name → (positions, quaternions) in world coords
    poses_after: dict,
    targets: dict,              # obj_name → (N_envs, 3)  in env-local coords
    env_root_pos: np.ndarray | None = None,  # (N_envs, 3)
    done_once: np.ndarray | None = None,     # (N_envs,) bool
    target_oris: dict | None = None,         # obj_name → (N_envs, 4) quat targets
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
            r += C_YAW * (y_err_before - y_err_after)
            done &= (y_err_after < YAW_THRESHOLD)

    r -= C_STEP

    first_done = done & ~done_once
    r[first_done] += C_SUCCESS
    done_once = done_once | done

    return r, done, done_once


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
      0. lift to safe height above current XY
      1. move to standoff_xy at safe height (collision-free)
      2. descend to standoff_xy at object height
      3. strike — push OVERTRAVEL past contact
      4. retract to safe height
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
    delta_d_clipped = np.clip(delta_d, 0.001, DELTA_D_MAX)
    standoff_xy = world_xy - dirs * delta_d_clipped[:, None]  # (B, 2) run-up

    #* ── snap initial positions (inactive envs hold these throughout) ──
    q_cur = as_numpy(fingers.get_dof_positions()).copy()     # (B, 3)

    def _hold_inactive(targets: np.ndarray):
        if active is not None:
            targets[~active] = q_cur[~active]

    #* ── Phase 0: lift to safe height above current position ─────────
    q0 = q_cur.copy()
    q0[:, 2] = SAFE_Z
    _hold_inactive(q0)
    fingers.set_dof_position_targets(positions=q0.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
    await _step_physics(25)

    #* ── Phase 1: move to standoff at safe height (above objects) ───
    q1 = np.zeros((B, 3), dtype=np.float32)
    q1[:, :2] = np.clip(standoff_xy, xy_low, xy_high)
    q1[:, 2] = SAFE_Z
    _hold_inactive(q1)
    fingers.set_dof_position_targets(positions=q1.tolist(), dof_indices=dof_indices)
    await _step_physics(40)

    #* ── Phase 2: descend to standoff at object height ───────────────
    q2 = q1.copy()
    q2[:, 2] = OBJECT_HEIGHT
    _hold_inactive(q2)
    fingers.set_dof_position_targets(positions=q2.tolist(), dof_indices=dof_indices)
    await _step_physics(20)

    #* ── Phase 3: strike — push OVERTRAVEL past contact ──────────────
    #* Position-only PD (v_target=0) — OVERTRAVEL creates a steady
    #* pushing force kp*OVERTRAVEL.  No velocity target avoids oscillation
    #* at the contact point (velocity tracking fights the constraint).
    q3 = np.zeros((B, 3), dtype=np.float32)
    q3[:, :2] = np.clip(world_xy + dirs * OVERTRAVEL, xy_low, xy_high)
    q3[:, 2] = OBJECT_HEIGHT
    _hold_inactive(q3)

    fingers.set_dof_position_targets(positions=q3.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
    await _step_physics(IMPACT_STEPS)

    #* ── Phase 4: retract to safe height ─────────────────────────────
    q4 = q3.copy()
    q4[:, 2] = SAFE_Z
    _hold_inactive(q4)
    fingers.set_dof_position_targets(positions=q4.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
    await _step_physics(20)

    #* ── Phase 5: settle objects — wait until all stopped ────────────
    VEL_THRESH = 0.005
    MAX_SETTLE = 100
    l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
    cyl_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")

    for _ in range(MAX_SETTLE):
        await _step_physics(1)
        lv = as_numpy(l_objects.get_linear_velocities())
        cv = as_numpy(cyl_objects.get_linear_velocities())
        if (bool(np.all(np.linalg.norm(lv, axis=1) < VEL_THRESH)) and
            bool(np.all(np.linalg.norm(cv, axis=1) < VEL_THRESH))):
            break

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
        self.fingers, self.tips, self.env_roots = get_finger_handles()

        finger_props = globals().get("randomized_finger_properties",
                        getattr(__main__, "randomized_finger_properties", None))
        if finger_props is not None:
            configure_drives(self.fingers,
                             stiffnesses=finger_props["stiffnesses"],
                             dampings=finger_props["dampings"])
        else:
            configure_drives(self.fingers)

        self.K = get_camera_intrinsics()

    def _decay_schedule(self, episode: int):
        frac = min(1.0, episode / EPS_DECAY)
        self.epsilon  = EPS_START + (EPS_END - EPS_START) * frac
        self.noise_std = SIGMA_START + (SIGMA_END - SIGMA_START) * frac


    #! Reset 
    async def _reset_episode(self, episode: int) -> tuple[torch.Tensor, list, torch.Tensor, dict]:
        """Randomise objects + targets, return first observation."""
        #* RigidPrim handles (use same patterns as scene setup)
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
        init_pos = np.tile([0.0, 0.0, SAFE_Z], (NUM_ENVS, 1)).astype(np.float32)
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

        # yaw curriculum: linear ramp from 0 to full range
        frac = np.clip((episode - YAW_CURRICULUM_START) /
                       max(1, YAW_CURRICULUM_END - YAW_CURRICULUM_START), 0.0, 1.0)
        yaw_half_range = frac * YAW_FULL_RANGE
        l_target_yaws = l_yaws + self.rng.uniform(-yaw_half_range, yaw_half_range,
                                                   size=NUM_ENVS)
        targets_ori["LObject"] = _yaws_to_quats(l_target_yaws.astype(np.float32))

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

        for step in range(MAX_STEPS):
            has_contour = contour_masks.any(dim=(-2, -1)).cpu().numpy()
            was_active = ~self._done_once & has_contour  # done OR invisible → frozen

            #* — select actions ——————————————————————————————
            pixel_ij, d_xy, delta_d = select_action(
                self.actor_critic, x, contour_masks,
                self.epsilon, self.noise_std,
            )

            #* — execute (only active envs move; inactive held frozen) —
            poses_after, _ = await env_step_async(
                self.fingers, pixel_ij, d_xy, delta_d, self.K,
                active=was_active,
            )
            rewards, dones, self._done_once = compute_rewards(
                poses_before, poses_after,
                self._targets_pos, self._env_root_pos, self._done_once,
                target_oris=self._targets_ori)
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

        return ep_return, ep_len

    def log(self, episode: int, ep_return: float, ep_len: int, elapsed: float):
        avg_r = np.mean(self.ep_returns) if self.ep_returns else 0.0
        avg_l = np.mean(self.ep_lengths) if self.ep_lengths else 0.0
        print(
            f"[ep {episode:5d}] "
            f"return={ep_return:7.2f}  len={ep_len:3d}  "
            f"avg100_r={avg_r:7.2f}  avg100_len={avg_l:5.1f}  "
            f"ε={self.epsilon:.3f}  σ={self.noise_std:.3f}  "
            f"buf={self.buffer.size:6d}  dt={elapsed:.1f}s"
        )

    def save_checkpoint(self, path: str | Path, episode: int):
        """Save full training state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer_q":  self.optimizer_q.state_dict(),
            "optimizer_mu": self.optimizer_mu.state_dict(),
            "global_step":  self.global_step,
            "epsilon":      self.epsilon,
            "noise_std":    self.noise_std,
            "episode":      episode,
            "rng_state":    self.rng.bit_generator.state,
            "ep_returns":   list(self.ep_returns),
            "ep_lengths":   list(self.ep_lengths),
            "buffer_x":       self.buffer.x[:self.buffer.size].clone(),
            "buffer_x_next":  self.buffer.x_next[:self.buffer.size].clone(),
            "buffer_pixel":   self.buffer.pixel[:self.buffer.size].clone(),
            "buffer_d_xy":    self.buffer.d_xy[:self.buffer.size].clone(),
            "buffer_delta_d": self.buffer.delta_d[:self.buffer.size].clone(),
            "buffer_r":       self.buffer.r[:self.buffer.size].clone(),
            "buffer_done":    self.buffer.done[:self.buffer.size].clone(),
            "buffer_ptr":     self.buffer.ptr,
            "buffer_size":    self.buffer.size,
        }, path)
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

async def main_async(num_episodes: int = 10_000, resume: str | None = None):
    trainer = Trainer()
    start_ep = 1
    if resume:
        await trainer.setup()
        start_ep = trainer.load_checkpoint(resume) + 1
    else:
        await trainer.setup()

    print(f"[train] device={DEVICE}  envs={NUM_ENVS}  buffer={BUFFER_CAPACITY}")
    print(f"[train] max_steps/ep={MAX_STEPS}  gamma={GAMMA}  lr={LR}")
    if resume:
        print(f"[train] resuming from episode {start_ep}")

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


def main(num_episodes: int = 10_000, resume: str | None = None):
    return run_coroutine(main_async(num_episodes, resume))


if __name__ == "__main__":
    trainer_task = main()
