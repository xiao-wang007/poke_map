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

CONFIG = load_config()
SCENE_CONFIG = CONFIG["scene"]
FINGER_CONFIG = CONFIG["finger"]

OBJECT_HEIGHT = SCENE_CONFIG["object_height"]
ENVS_ROOT_PATH = SCENE_CONFIG["envs_root_path"]
FINGER_LOCAL_PATH = FINGER_CONFIG["local_root_path"]
FINGER_ROOT_PATTERN = f"{ENVS_ROOT_PATH}/env_*/{FINGER_LOCAL_PATH}"
FINGER_TIP_PATTERN = f"{FINGER_ROOT_PATTERN}/z_link"
NUM_ENVS = SCENE_CONFIG["num_envs"]


#* ================================================================
#*  Hyper-parameters
#* ================================================================

GAMMA: float = 0.95
LR: float = 3e-4
BATCH_SIZE: int = 64
BUFFER_CAPACITY: int = 100_000
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
A_MAX: float = 5.0               # max finger acceleration (m/s²) 13 m/s² for panda ee
DELTA_D_MAX: float = 0.2          # max standoff (m)
IMPACT_STEPS: int = 30            # physics steps per strike
OVERTRAVEL: float = 0.05          # finger pushes this far past contact (m)
SAFE_Z: float = FINGER_CONFIG["default_xyz"][2]  # safe height (m)

TARGET_UPDATE_EVERY: int = 5000   # steps between soft updates
#! TARGET_UPDATE_EVERY = 100 with 12 envs means the target network 
#! updates every ~8 episode steps (each step adds 12 to global_step). 
#! Standard DQN updates the target every 2000-10000 global steps. 
#! This will cause the TD target to chase a moving target, 
#! destabilizing training. Change to 5000.

TRAIN_AFTER: int = 256           # start training after this many transitions
TRAIN_EVERY: int = 4             # train every N env steps
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


#* ================================================================
#*  Replay buffer
#* ================================================================

class ReplayBuffer:
    """Ring buffer storing per-transition data (one env, one step).

    Each element is a flat dict so we can mix across envs and episodes.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY, device: str = DEVICE):
        self.capacity = capacity
        self.device = device

        H, W = RESOLUTION
        self.x      = torch.zeros(capacity, 2, H, W)
        self.pixel  = torch.zeros(capacity, 2, dtype=torch.long)  # (row, col)
        self.d_xy   = torch.zeros(capacity, 2)                     # unit direction
        self.delta_d = torch.zeros(capacity, 1)                    # standoff
        self.r      = torch.zeros(capacity, 1)
        self.x_next = torch.zeros(capacity, 2, H, W)
        self.done   = torch.zeros(capacity, 1, dtype=torch.bool)
        self.mask   = torch.zeros(capacity, H, W, dtype=torch.bool)       # contour
        self.mask_next = torch.zeros(capacity, H, W, dtype=torch.bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        x: torch.Tensor,             # (1, 2, H, W)
        pixel_ij: np.ndarray,        # (2,) int
        d_xy: np.ndarray,            # (2,)
        delta_d: float,
        reward: float,
        x_next: torch.Tensor,        # (1, 2, H, W)
        done: bool,
        mask: np.ndarray,            # (H, W) bool
        mask_next: np.ndarray,       # (H, W) bool
    ):
        idx = self.ptr
        self.x[idx].copy_(x.cpu() if x.device.type != "cpu" else x)
        self.pixel[idx] = torch.tensor(pixel_ij)
        self.d_xy[idx] = torch.tensor(d_xy)
        self.delta_d[idx] = torch.tensor([delta_d])
        self.r[idx] = torch.tensor([reward])
        self.x_next[idx].copy_(x_next.cpu() if x_next.device.type != "cpu" else x_next)
        self.done[idx] = torch.tensor([done])
        self.mask[idx] = torch.tensor(mask)
        self.mask_next[idx] = torch.tensor(mask_next)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "x":         self.x[indices].to(self.device),
            "pixel":     self.pixel[indices].to(self.device),
            "d_xy":      self.d_xy[indices].to(self.device),
            "delta_d":   self.delta_d[indices].to(self.device),
            "r":         self.r[indices].to(self.device),
            "x_next":    self.x_next[indices].to(self.device),
            "done":      self.done[indices].to(self.device),
            "mask":      self.mask[indices].to(self.device),
            "mask_next": self.mask_next[indices].to(self.device),
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


def step_physics(steps: int = 1):
    run_coroutine(_step_physics(steps))


def ensure_sim_running():
    app_utils.play()
    step_physics(1)


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
    x = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    y = rng.uniform(workspace_range[0], workspace_range[1], size=(num_envs,))
    z = np.full((num_envs,), OBJECT_HEIGHT * 0.5, dtype=np.float32)
    return {"LObject": np.stack([x, y, z], axis=-1),
            "Cylinder": np.stack([x, y, z], axis=-1)}


#* ================================================================
#*  Action selection
#* ================================================================

def select_action(
    actor_critic: SpatialActorCritic,
    x: torch.Tensor,                       # (B, 2, H, W)
    contour_masks: torch.Tensor,           # (B, H, W) bool
    epsilon: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select actions for a batch of observations.

    Returns
    -------
    pixel_ij : (B, 2) int   — (row, col) per env
    d_xy     : (B, 2) float — normalised unit vector
    delta_d  : (B,)  float — standoff in [0, delta_d_max]
    """
    B = x.shape[0]
    H, W = x.shape[-2:]
    device = x.device
    actor_critic.eval()

    with torch.no_grad():
        q_map, params = actor_critic(x)         # (B, 1, H, W), (B, 3, H, W)
        q_map_sq = q_map.squeeze(1)             # (B, H, W)

    #* mask non-contour pixels
    q_map_sq[~contour_masks] = -float("inf")

    #* ε-greedy pixel selection
    pixel_ij = np.zeros((B, 2), dtype=np.int32)
    d_xy_arr = np.zeros((B, 2), dtype=np.float32)
    delta_d_arr = np.zeros(B, dtype=np.float32)

    for b in range(B):
        if np.random.random() < epsilon:
            # random valid pixel
            valid = torch.nonzero(contour_masks[b], as_tuple=True)
            if len(valid[0]) == 0:
                continue  # no contour → action stays zero
            r = np.random.randint(len(valid[0]))
            row, col = valid[0][r].item(), valid[1][r].item()
            pixel_ij[b] = [row, col]
        else:
            idx = q_map_sq[b].argmax().item()
            pixel_ij[b, 0] = idx // W
            pixel_ij[b, 1] = idx % W

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
#*  Contour mask extraction
#* ================================================================

def extract_contour_masks(seg_maps: list[np.ndarray]) -> torch.Tensor:
    """Build binary contour masks from instance segmentation maps."""
    from scipy.ndimage import binary_erosion
    B = len(seg_maps)
    H, W = seg_maps[0].shape
    masks = np.zeros((B, H, W), dtype=bool)
    for b, seg in enumerate(seg_maps):
        for oid in np.unique(seg):
            if oid == 0:
                continue
            obj_mask = seg == oid
            eroded = binary_erosion(obj_mask, iterations=1)
            masks[b] |= obj_mask & ~eroded
    return torch.tensor(masks, dtype=torch.bool)


#* ================================================================
#*  Reward & termination
#* ================================================================

def compute_rewards(
    poses_before: dict,         # obj_name → (N_envs, 3)
    poses_after: dict,
    targets: dict,              # obj_name → (N_envs, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-env reward + done flag.

    #! TODO: may incorporate velocity reward, orientation error, 
    #!       possibly contact/force signals, negative shaping, etc

    r = Σ max(0, d_before - d_after) - c_step  (+ c_success if all at target)
    """
    num_envs = list(poses_before.values())[0].shape[0]
    r = np.zeros(num_envs, dtype=np.float32)
    done = np.ones(num_envs, dtype=bool)

    for obj_name in poses_before:
        d_before = np.linalg.norm(poses_before[obj_name][:, :2] - targets[obj_name][:, :2], axis=1)
        d_after  = np.linalg.norm(poses_after[obj_name][:, :2] - targets[obj_name][:, :2], axis=1)
        r += np.maximum(0.0, d_before - d_after)
        done &= (d_after < SUCCESS_THRESHOLD)

    r -= C_STEP
    r[done] += C_SUCCESS
    return r, done


#* ================================================================
#*  Environment step — articulated finger (batched, all envs)
#* ================================================================

async def env_step_async(
    fingers: Articulation,
    pixel_ij: np.ndarray,        # (B, 2)
    d_xy: np.ndarray,            # (B, 2)
    delta_d: np.ndarray,         # (B,)
    K: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Execute strikes with the prismatic XYZ finger robot.

    Phases: position at standoff → strike (velocity-controlled) → settle.
    Approach/descent/retract are offloaded to a trajectory optimisation policy.
    """
    B = len(fingers)
    dof_indices = fingers.get_dof_indices(fingers.dof_names)

    #* ── convert pixel → world XY + strike speeds ──────────────────
    world_xy = np.zeros((B, 2), dtype=np.float32)
    speeds = np.zeros(B, dtype=np.float32)
    for b in range(B):
        w = pixel_to_world(tuple(pixel_ij[b]), K)
        world_xy[b] = w[:2]
        speeds[b] = np.sqrt(2.0 * A_MAX * delta_d[b])  # g(Δd)

    #* ── normalise direction ───────────────────────────────────────
    dir_norm = np.linalg.norm(d_xy, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm < 1e-8, 1.0, dir_norm)
    dirs = d_xy / dir_norm

    zero_v = np.zeros((B, 3), dtype=np.float32)

    #* standoff: start delta_d behind contact point so finger has
    #* room to accelerate before impact
    delta_d_clipped = np.clip(delta_d, 0.001, DELTA_D_MAX)
    standoff_xy = world_xy - dirs * delta_d_clipped[:, None]  # (B, 2)

    #* ── Phase 1: position at standoff, object height ─────────────
    #*    approach/descent handled by trajectory optimisation policy
    q1 = np.zeros((B, 3), dtype=np.float32)
    q1[:, 0] = standoff_xy[:, 0]
    q1[:, 1] = standoff_xy[:, 1]
    q1[:, 2] = OBJECT_HEIGHT
    fingers.set_dof_position_targets(positions=q1.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=zero_v.tolist(), dof_indices=dof_indices)
    await _step_physics(20)   # let PD converge at standoff

    #* ── Phase 2: strike — accelerate through delta_d then push
    #*    OVERTRAVEL past contact ──────────────────────────────────
    q2 = np.zeros((B, 3), dtype=np.float32)
    q2[:, 0] = world_xy[:, 0] + dirs[:, 0] * OVERTRAVEL
    q2[:, 1] = world_xy[:, 1] + dirs[:, 1] * OVERTRAVEL
    q2[:, 2] = OBJECT_HEIGHT

    v2 = np.zeros((B, 3), dtype=np.float32)
    v2[:, 0] = dirs[:, 0] * speeds
    v2[:, 1] = dirs[:, 1] * speeds

    fingers.set_dof_position_targets(positions=q2.tolist(), dof_indices=dof_indices)
    fingers.set_dof_velocity_targets(velocities=v2.tolist(), dof_indices=dof_indices)
    await _step_physics(IMPACT_STEPS)

    #* ── Phase 3: settle objects — wait until all stopped ─────────
    VEL_THRESH = 0.005   # m/s
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

    #* ── query new object poses ───────────────────────────────────
    poses, env_root_pos = get_object_poses_vectorized()
    return poses, env_root_pos


def env_step(
    fingers: Articulation,
    pixel_ij: np.ndarray,
    d_xy: np.ndarray,
    delta_d: np.ndarray,
    K: np.ndarray,
) -> tuple[dict, np.ndarray]:
    return run_coroutine(env_step_async(fingers, pixel_ij, d_xy, delta_d, K))


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
    mask_next = batch["mask_next"]  # (B, H, W)

    B = x.shape[0]
    batch_idx = torch.arange(B, device=x.device)
    actor_critic.train()
    target_net.eval()

    #* -- stored action params (with exploration noise, as executed) -----
    a_stored = torch.cat([d_xy_st, dd_st], dim=1)  # (B, 3)

    #* =================================================================
    #*  LOSS 1 — Q-Loss (Huber TD)
    #* =================================================================

    #* target Q-value
    with torch.no_grad():
        q_next, params_next = target_net(x_next)
        q_next_sq = q_next.squeeze(1)
        q_next_sq[~mask_next] = -float("inf")
        pixel_next = q_next_sq.view(B, -1).argmax(dim=1)
        q_max_next = q_next_sq[batch_idx, pixel_next // q_next.shape[-1],
                               pixel_next % q_next.shape[-1]]
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

    #* recompute encoder features (detached) and actor params
    f, params_act, _ = actor_critic.features_and_params(x)
    f_detached = f.detach()

    #* actor params at the stored pixel
    params_actor_at = actor_critic.params_at_pixel(params_act, pixel)  # (B, 3)

    #* FiLM from actor params (GRADS flow through film MLPs → param_head)
    gamma, beta = actor_critic.film(params_actor_at)
    f_mod_pg = gamma[:, :, None, None] * f_detached + beta[:, :, None, None]
    q_map_pg = actor_critic.q_head(f_mod_pg)   # q_head frozen, just forward
    q_val_pg = q_map_pg[batch_idx, 0, pixel[:, 0], pixel[:, 1]]  # (B,)

    L_mu = -q_val_pg.mean()

    optimizer_mu.zero_grad()
    L_mu.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 10.0)
    optimizer_mu.step()

    #* unfreeze
    for p in actor_critic.q_head.parameters():
        p.requires_grad = True
    for p in actor_critic.unet.parameters():
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
        ensure_sim_running()

        self.fingers, self.tips, self.env_roots = get_finger_handles()
        configure_drives(self.fingers)
        self.K = get_camera_intrinsics()

        self.actor_critic = SpatialActorCritic(delta_d_max=DELTA_D_MAX).to(DEVICE)
        self.target_net   = SpatialActorCritic(delta_d_max=DELTA_D_MAX).to(DEVICE)
        soft_update(self.target_net, self.actor_critic, tau=1.0)  # copy

        #* param groups for separate optimizers
        q_params = list(self.actor_critic.q_head.parameters()) + \
                   list(self.actor_critic.film.parameters()) + \
                   list(self.actor_critic.unet.parameters())
        mu_params = list(self.actor_critic.param_head.parameters()) + \
                    list(self.actor_critic.film.parameters())

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

    def _decay_schedule(self, episode: int):
        frac = min(1.0, episode / EPS_DECAY)
        self.epsilon  = EPS_START + (EPS_END - EPS_START) * frac
        self.noise_std = SIGMA_START + (SIGMA_END - SIGMA_START) * frac

    def _reset_episode(self) -> tuple[torch.Tensor, list, torch.Tensor, dict]:
        """Randomise objects + targets, return first observation."""
        #* RigidPrim handles (use same patterns as scene setup)
        l_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/LObject")
        cylinder_objects = RigidPrim(paths=f"{ENVS_ROOT_PATH}/env_.*/Cylinder")

        from env.scene_setup_articulated_vectorized import randomize_object_poses
        randomize_object_poses(
            self.env_roots, l_objects, cylinder_objects,
            seed=None,
        )
        step_physics(SCENE_CONFIG["settle_steps"])

        # random targets
        targets_pos = sample_target_poses(self.rng, NUM_ENVS)
        targets_ori = {"LObject": np.zeros((NUM_ENVS, 4), dtype=np.float32),
                       "Cylinder": np.zeros((NUM_ENVS, 4), dtype=np.float32)}
        targets_ori["LObject"][:, 0] = 1.0
        targets_ori["Cylinder"][:, 0] = 1.0

        # vision
        poses_before, env_root_pos = get_object_poses_vectorized()
        x, seg_maps = build_vision_observation(
            poses_before, targets_pos, targets_ori, self.K,
            env_root_pos=env_root_pos,
        )
        contour_masks = extract_contour_masks(seg_maps)

        # self._poses = poses_before
        self._targets_pos = targets_pos
        self._targets_ori = targets_ori
        self._env_root_pos = env_root_pos

        return x.to(DEVICE), contour_masks.to(DEVICE), poses_before

    def train_episode(self, episode: int):
        self._decay_schedule(episode)
        x, contour_masks, poses_before = self._reset_episode()

        ep_return = 0.0
        ep_len = 0

        for step in range(MAX_STEPS):
            #* — select actions ——————————————————————————————
            pixel_ij, d_xy, delta_d = select_action(
                self.actor_critic, x, contour_masks,
                self.epsilon, self.noise_std,
            )

            #* — execute —————————————————————————————————————
            poses_after, _ = env_step(
                self.fingers, pixel_ij, d_xy, delta_d, self.K,
            )
            rewards, dones = compute_rewards(poses_before, poses_after,
                                             self._targets_pos)

            #* — observe next state ——————————————————————————
            x_next, seg_maps_next = build_vision_observation(
                poses_after, self._targets_pos, self._targets_ori, self.K,
                env_root_pos=self._env_root_pos,
            )
            contour_masks_next = extract_contour_masks(seg_maps_next)

            #* — store per-env transitions ————————————————————
            for b in range(NUM_ENVS):
                self.buffer.push(
                    x[b:b+1], pixel_ij[b], d_xy[b], float(delta_d[b]),
                    float(rewards[b]), x_next[b:b+1], bool(dones[b]),
                    contour_masks[b].cpu().numpy(),
                    contour_masks_next[b].cpu().numpy(),
                )

            ep_return += rewards.sum()
            ep_len += 1

            #* — advance state ————————————————————————————————
            x = x_next.to(DEVICE)
            contour_masks = contour_masks_next.to(DEVICE)
            poses_before = poses_after
            self.global_step += NUM_ENVS

            #* — training —————————————————————————————————————
            if self.buffer.size >= TRAIN_AFTER and self.global_step % TRAIN_EVERY == 0:
                batch = self.buffer.sample(BATCH_SIZE)
                lq, lmu = train_step(
                    self.actor_critic, self.target_net,
                    self.optimizer_q, self.optimizer_mu,
                    batch,
                )

            #* — target update ———————————————————————————————
            if self.global_step % TARGET_UPDATE_EVERY == 0:
                soft_update(self.target_net, self.actor_critic, TAU)

            if dones.all():
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


#* ================================================================
#*  Entry point
#* ================================================================

def main(num_episodes: int = 10_000):
    trainer = Trainer()
    print(f"[train] device={DEVICE}  envs={NUM_ENVS}  buffer={BUFFER_CAPACITY}")
    print(f"[train] max_steps/ep={MAX_STEPS}  gamma={GAMMA}  lr={LR}")

    for ep in range(1, num_episodes + 1):
        t0 = time.time()
        ep_r, ep_len = trainer.train_episode(ep)
        trainer.log(ep, ep_r, ep_len, time.time() - t0)

    print("[train] done.")
    return trainer


if __name__ == "__main__":
    trainer = main()
