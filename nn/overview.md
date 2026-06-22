I should recommend implementing manually rather than using an off-the-shelf RL library. The custom action space (spatial pixel + continuous params), U-Net architecture, and Isaac Sim integration make stock libraries a poor fit. Manual implementation is ~300-400 lines and gives full control.

Let me now formulate a comprehensive plan.

Here's my analysis and plan.

## Manual implementation — strongly recommended

Off-the-shelf RL libraries (SB3, RLlib, SKRL) all assume:
- Flat vector observation/action spaces
- Standard `gym.Env` interface
- Synchronous `env.step()` calls

Your setup breaks all three: spatial 2D Q-map outputs, hybrid discrete-continuous actions per pixel, and async Isaac Sim physics stepping. The abstraction overhead of adapting a library exceeds the cost of writing the loop yourself. HACMan and VPG both use manual training loops for the same reason.

## Training script plan

### File: `/home/xiao/0_codes/poke_map/nn/train.py`

The script needs 5 components:

---

### 1. Replay Buffer

```python
class ReplayBuffer:
    store(x, pixel_ij, d_xy, delta_d, r, x_next, done, contour_mask)
    sample(batch_size) → dict of tensors
    # Ring buffer, fixed capacity (100K transitions)
    # Each env produces one transition per step
    # x shape: (B, 2, H, W); pixel_ij: (B, 2); d_xy: (B, 2); delta_d: (B,)
```

---

### 2. Action Selection

```python
def select_action(x, contour_masks, epsilon, noise_std, device):
    """
    x:              (B, 2, H, W) batch of observations
    contour_masks:  (B, H, W) bool — which pixels are valid
    
    Returns:
        pixel_ij:   (B, 2) int
        d_xy:       (B, 2) float, normalized unit vectors
        delta_d:    (B,)  float in [0, delta_d_max]
        q_vals:     (B,)  for logging
    """
    1. q_map, params = actor_critic(x)           # forward pass
    2. q_map[~contour_masks] = -inf              # mask non-contour pixels
    3. For each batch item with prob ε: random contour pixel
       else: argmax q_map
    4. Read params at selected pixels
    5. Add Gaussian noise to d_xy, delta_d; clamp
    6. Normalize d_xy → unit vector
    7. Return (pixel_ij, d_xy, delta_d, q_vals)
```

---

### 3. Environment Step

```python
def env_step(env_idx, pixel_ij, d_xy, delta_d, strikers, K):
    """
    1. pixel_to_world(pixel_ij, K) → world_xy
    2. direction = d_xy  (already normalized)
    3. speed = sqrt(2 * a_max * delta_d)      # g(Δd) mapping
    4. strike(strikers, env_idx, world_xy, direction, speed)
    5. get new object poses → compute reward
    6. render new contours → x_next
    7. return (x_next, reward, done)
    """
```

**Reward:**
```python
r = Σ_i max(0, dist_before_i - dist_after_i)  # improvement only
r -= c_step                                    # 0.01 per step
if all objects within threshold of targets:
    r += c_success                             # 10.0
    done = True
```

---

### 4. Training Step

```python
def train_step(actor_critic, target_net, buffer, optimizer_q, optimizer_mu):
    batch = buffer.sample(64)

    # ── Q-loss (TD) ────────────────────────
    with torch.no_grad():
        q_next, params_next = target_net(batch.x_next)
        q_next[~batch.mask_next] = -inf
        pixel_next = argmax(q_next)  # per batch item

    target = batch.r + γ * q_next[arange, 0, pixel_next.row, pixel_next.col] * (1-batch.done)

    q_vals = actor_critic.get_q_at_pixels(batch.x, batch.pixel)
    L_Q = Huber(q_vals, target)

    optimizer_q.zero_grad()
    L_Q.backward()
    optimizer_q.step()

    # ── Policy gradient (DPG) ──────────────
    # Freeze Q-head during actor update
    for p in actor_critic.q_head.parameters():
        p.requires_grad = False

    _, params_selected = actor_critic.get_values_at_pixel(batch.x, batch.pixel)
    # params_selected: (B, 3) — (d_xy_raw, delta_d) at selected pixel
    # We need Q(s, pixel, params) → need Q to accept params as input
    # This is Option A: Q is spatial only, params improve via task reward
    # → skip explicit DPG, use REINFORCE-style: log-prob weight = advantage

    # OR Option B: Q conditioned on params via FiLM/concat
    # → compute Q(s, pixel, params_selected), backprop through Q into params

    optimizer_mu.zero_grad()
    L_mu.backward()
    optimizer_mu.step()

    # Unfreeze
    for p in actor_critic.q_head.parameters():
        p.requires_grad = True
```

This is where a decision is needed — see below.

---

### 5. Main Training Loop

```
Initialize:
    actor_critic, target_net (polyak copy)
    optimizer_q (Adam, lr=3e-4)
    optimizer_mu (Adam, lr=3e-4)
    replay buffer (100K)
    ε = 1.0, σ_noise = 0.3

For episode in 1..M:
    reset objects (random poses) + random targets
    x, seg_maps = build_vision_observation(poses, targets, K)
    contour_masks = extract_contour_masks(seg_maps)

    For step in 1..max_steps:
        # Forward pass + action selection
        pixel_ij, d_xy, delta_d, q_vals = select_action(x, contour_masks, ε, σ)

        # Execute (vectorized across envs)
        results = execute_strike_all(strikers, pixel_ij, d_xy, delta_d, K)

        # Observe
        x_next, seg_maps_next = build_vision_observation(new_poses, targets, K)
        rewards, dones = compute_rewards(new_poses, targets)

        # Store
        for env_i in range(num_envs):
            buffer.add(x[env_i], pixel_ij[env_i], d_xy[env_i],
                       delta_d[env_i], rewards[env_i], x_next[env_i],
                       dones[env_i], contour_masks[env_i])

        # Train if buffer has enough samples
        if len(buffer) >= batch_size:
            train_step(...)

        x, contour_masks = x_next, contour_masks_next

        if all(dones): break

    # Decay ε, σ
    # Soft-update target network every C episodes
    # Log metrics
```

---

### The critical design question: Option A vs B for param-head training

| | Option A: Spatial-only Q | Option B: Q conditioned on params |
|---|---|---|
| Q(s, pixel, d̂, Δd)? | No — Q ignores params | Yes — Q evaluates the full action |
| Param-head training | RL reward signal only (slower) | DPG: Q-gradient through params (efficient) |
| Implementation | Simpler — no Q(param) dependency | Q-head takes extra 3-channel param input |
| Initial choice | **Start here** — simpler, easier to debug | Add later as ablation if param learning is slow |

**Recommendation**: Implement Option A first. The param-head improves through the same task reward that the Q-head uses — both heads contribute to getting objects to targets. The spatial Q-value at a pixel is correlated with "good actions are selected here" which includes param quality. Add Option B (DPG with Q-conditioned-on-params) as a follow-up once the basic loop works.

---

### What NOT to implement yet

| Keep out of v1 | Reason |
|---|---|
| Prioritized replay | Uniform sampling is fine for initial debugging |
| Multi-step returns (n-step TD) | Adds complexity, marginal gain early |
| Double Q-learning | Add after basic loop converges |
| DPG with Q-conditioned-on-params | Option B — v2 |
| Logging/metrics infrastructure | Print-based debugging is enough initially |

---

### Key integration points with Isaac Sim

1. All physics stepping must use `run_coroutine()` — the training loop itself can be sync, but each `strike()` call wraps an async coroutine
2. `build_vision_observation()` needs `env_poses` from `get_object_poses_vectorized()` which queries Isaac
3. Network forward pass runs on GPU (CUDA, torch device) — Isaac tensors may need CPU→GPU transfer
4. The `strike_all()` function already handles batching — feed it per-env targets and directions