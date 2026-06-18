import numpy as np

from vision.camera import (
    RESOLUTION,
    mask_to_contour,
    pixel_to_world,
    world_to_pixel,
)

#* Isaac Sim imports — guarded so the prior module can be imported
#* outside the Script Editor for syntax-checking / linting.
try:
    from isaacsim.core.experimental.prims import RigidPrim
    import isaacsim.core.experimental.utils.app as app_utils
    from omni.kit.async_engine import run_coroutine
    _HAS_ISAAC = True
except ModuleNotFoundError:
    RigidPrim = None
    app_utils = None
    run_coroutine = None
    _HAS_ISAAC = False


def _require_isaac():
    if not _HAS_ISAAC:
        raise RuntimeError(
            "poke_prior requires Isaac Sim. Run it inside the Isaac Sim Script Editor."
        )


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def as_numpy(values):
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    elif hasattr(values, "tolist") and callable(values.tolist):
        values = np.asarray(values.tolist(), dtype=np.float32)
    try:
        return np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError):
        return np.array([[float(c) for c in v] for v in values], dtype=np.float32)


def sample_pixel_on_farside(contour_pixel_coords, obj_init_CoM_pixel, direction,
                            num_samples=20, w=0.1):
    weights = []
    for coord in contour_pixel_coords:
        offset = coord - obj_init_CoM_pixel
        far_side_score = max(0.0, -float(np.dot(offset, direction)))
        weights.append(1.0 + w * far_side_score)

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    sampled_indices = np.random.choice(
        len(contour_pixel_coords),
        size=min(num_samples, len(contour_pixel_coords)),
        p=weights,
        replace=False,
    )
    return contour_pixel_coords[sampled_indices]


#* ================================================================
#*  simulate_poke — physics probe for the prior
#* ================================================================

async def _step_physics(steps: int):
    """Step the PhysX simulation forward by *steps* frames."""
    await app_utils.update_app_async(steps=steps)


def simulate_poke(
    rigid_prim: RigidPrim,
    world_xy: tuple[float, float],
    direction: np.ndarray,
    impulse_speed: float = 2.0,
    sim_steps: int = 30,
) -> float:
    """Apply a short impulse to *rigid_prim* at *world_xy* and measure
    the resulting displacement projected onto *direction*.

    The impulse is decomposed into a linear component (translation) and an
    angular component (rotation) to model the effect of striking the object
    at an offset from its centre of mass.  Poking the far edge produces a
    different motion than poking the centre.

    The probe is self-contained: it saves the object state before the poke,
    runs a brief physics forward pass, measures displacement, then restores
    the original state.

    Parameters
    ----------
    rigid_prim : RigidPrim handle (single prim, not a batch).
    world_xy : (x, y) contact location in world frame (z is object top).
    direction : (2,) unit vector — push direction in XY plane.
    impulse_speed : reference velocity magnitude (m/s) for the linear part.
    sim_steps : number of PhysX steps to run (1 step ≈ 1/60 s).

    Returns
    -------
    score : float in [0, 1] — how much the object moved toward the target.
    """
    _require_isaac()

    #* 1. Save initial state -------------------------------------------------
    init_pos, init_quat = rigid_prim.get_world_poses()
    init_pos = as_numpy(init_pos)[0].copy()  # (3,)
    init_lin_vel = as_numpy(rigid_prim.get_linear_velocities())[0].copy()
    init_ang_vel = as_numpy(rigid_prim.get_angular_velocities())[0].copy()

    #* 2. Decompose impulse into linear + angular velocity -------------------
    com = init_pos[:2]                            # centre of mass, XY
    contact = np.array(world_xy, dtype=np.float32)  # (2,)
    r = contact - com                              # offset (2,)

    impulse = direction * impulse_speed             # (2,)
    lin_vel_2d = impulse                           # linear part

    #* angular: ω_z ∝ r × F  (2-D cross product)
    torque_z = r[0] * impulse[1] - r[1] * impulse[0]
    #* rough scaling so rotation is visible (I_z ~ 0.1 kg·m² for a small
    #* cube, scaled down so the effect is noticeable in a short sim window)
    ang_vel_z = torque_z * 5.0                      # tunable gain

    lin_vel_3d = np.array([lin_vel_2d[0], lin_vel_2d[1], 0.0], dtype=np.float32)
    ang_vel_3d = np.array([0.0, 0.0, ang_vel_z], dtype=np.float32)

    rigid_prim.set_linear_velocities([lin_vel_3d.tolist()])
    rigid_prim.set_angular_velocities([ang_vel_3d.tolist()])

    #* 3. Step physics forward inside the Kit event loop --------------------
    run_coroutine(_step_physics(sim_steps))

    #* 4. Measure displacement -----------------------------------------------
    final_pos, _ = rigid_prim.get_world_poses()
    final_pos = as_numpy(final_pos)[0]
    displacement_2d = final_pos[:2] - init_pos[:2]

    #* 5. Restore object state -----------------------------------------------
    rigid_prim.set_world_poses(
        positions=[init_pos.tolist()],
        orientations=[init_quat.tolist()],
    )
    rigid_prim.set_linear_velocities([init_lin_vel.tolist()])
    rigid_prim.set_angular_velocities([init_ang_vel.tolist()])

    #* 6. Score: projection of displacement onto target direction ------------
    proj = float(np.dot(displacement_2d, direction))
    max_mv = impulse_speed * sim_steps * (1.0 / 60.0)
    score = np.clip(proj / max(max_mv, 1e-6), 0.0, 1.0)
    return score


#* ================================================================
#*  compute_prior
#* ================================================================

def compute_prior(
    seg_map: np.ndarray,
    env_poses: dict,
    target_positions: dict,
    K: np.ndarray,
    rigid_prim_map: dict,  # obj_name → RigidPrim (single env, not vectorized batch)
    resolution: tuple[int, int] = RESOLUTION,
) -> np.ndarray:
    """Build a per-pixel heatmap that scores candidate poke locations for
    every object visible in *seg_map*.

    The heatmap is **not** a learned component — it is a cheap, one-shot,
    model-based heuristic that biases early RL exploration toward pixels
    where physics says a poke can actually move an object toward its goal.

    Parameters
    ----------
    seg_map : (H, W) int — instance segmentation (0 = background).
    env_poses : dict  obj_name → (positions (1,3), quaternions (1,4)).
    target_positions : dict  obj_name → (1, 3) target world pose.
    K : (3, 3) camera intrinsics.
    rigid_prim_map : dict  obj_name → RigidPrim for the **current** env.
                     Each prim must be a single-element handle (not a
                     pattern that matches across multiple envs).

    Returns
    -------
    prior : (H, W) float32 in [0, 1].
    """
    prior = np.zeros(resolution, dtype=np.float32)
    obj_id = 0

    for obj_name in env_poses:
        positions = env_poses[obj_name][0]   # (1, 3)
        targets = target_positions[obj_name]  # (1, 3)
        obj_id += 1

        #* ── per-object mask & contour ────────────────────────────
        obj_mask = (seg_map == obj_id)
        if not obj_mask.any():
            continue

        obj_contour = mask_to_contour(obj_mask)  # (H, W) bool

        #* centroid pixel — used as reference for far-side weighting
        mask_pixels = np.argwhere(obj_mask)
        centre_pixel = mask_pixels.mean(axis=0).astype(np.int32)

        #* ── push direction (object → target, XY plane) ──────────
        obj_pos = positions[0, :2]
        target_pos = targets[0, :2]
        direction = normalize(target_pos - obj_pos)
        if np.allclose(direction, 0.0):
            continue  # already at target — no poke needed

        #* ── sample contour pixels (far-side bias) ────────────────
        contour_pixels = np.argwhere(obj_contour)
        if len(contour_pixels) == 0:
            continue
        sampled_pixels = sample_pixel_on_farside(
            contour_pixels, centre_pixel, direction,
        )

        #* ── physics probes ───────────────────────────────────────
        rigid_prim = rigid_prim_map.get(obj_name)
        if rigid_prim is None:
            continue

        for pixel in sampled_pixels:
            world_xy = pixel_to_world(tuple(pixel), K)[:2]
            score = simulate_poke(rigid_prim, tuple(world_xy), direction)
            prior[tuple(pixel)] = max(prior[tuple(pixel)], score)

    return prior


#* Sub-modules ====================================================

