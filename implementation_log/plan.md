Training: modified DDPG/DQN hybrid, like HACMan's approach:
- Critic (Q): takes state + discrete action (pixel) + continuous params → Q-value
- Actor (param-head): outputs continuous params for each pixel
- Pixel selection: argmax over Q-map

Let me generate a comprehensive plan covering:
1. Action space formulation
2. Network architecture
3. Training algorithm (losses, exploration)
4. Reward design
5. Training loop structure (Isaac Sim integration)
6. Hyperparameters
7. Evaluation plan
8. Implementation phases

Since I'm in plan mode, I should NOT make any edits. I'll just outline the plan.

Here's the full Level 1 RL plan, no prior, using the HACMan-style hybrid discrete-continuous framework.

---

## Action Space

```
a = (pixel, d̂_x, d̂_y, Δd)

pixel ∈ {0, …, H×W-1}    discrete, masked to contour pixels only
d̂_x, d̂_y ∈ [-1, 1]        continuous, normalized to unit vector at execution
Δd      ∈ [0, Δd_max]     continuous, standoff distance
```

At execution: `d̂ = normalize([d̂_x, d̂_y, 0])`, `v_impact = √(2·a_max·Δd)`.

Masking: only contour pixels are valid actions. Before argmax, set Q-values of non-contour pixels to `-∞`.

---

## Network Architecture

```
Input: x = (2, H, W)          contour_current + contour_goal

┌─────────────────────────────┐
│      Shared Encoder          │
│      U-Net (as in camera.py)│
│      output: f (C, H, W)    │
└──────────┬──────────────────┘
           │
   ┌───────┴───────┐
   ▼               ▼
┌──────────┐  ┌──────────────┐
│ Q-head    │  │  Param-head   │
│ Conv2d   │  │  Conv2d       │
│ (C→1)    │  │  (C→3)        │
│           │  │               │
│ Output:  │  │  Output:      │
│ Q(s,p)  │  │  μ(s,p) =     │
│ (1,H,W)  │  │  (d̂_x,d̂_y,Δd) │
│           │  │  (3,H,W)      │
└──────────┘  └──────────────┘
```

- **Q-head**: per-pixel state-action value for pixel selection
- **Param-head**: per-pixel continuous parameters (direction + standoff)
- Both heads share the U-Net encoder

---

## Action Selection (forward pass)

```python
def select_action(x, contour_mask, epsilon, noise_std):
    Q_map = Q_net(x)           # (1, H, W)
    params = param_net(x)       # (3, H, W)

    # Mask non-contour pixels
    Q_map[~contour_mask] = -inf

    if random() < epsilon:
        pixel = random_choice(contour_pixels)
    else:
        pixel = argmax(Q_map)   # (row, col)

    # Read params at selected pixel
    d̂_x, d̂_y, Δd = params[:, pixel.row, pixel.col]

    # Add exploration noise to continuous params
    d̂_x += normal(0, noise_std)
    d̂_y += normal(0, noise_std)
    Δd  += abs(normal(0, noise_std))  # positive only

    # Clamp
    Δd = clip(Δd, 0, Δd_max)
    d̂_x, d̂_y = normalize([d̂_x, d̂_y])

    return pixel, d̂_x, d̂_y, Δd, Q_map[pixel]
```

---

## Training Algorithm: Modified DDPG + DQN Hybrid

### Networks

| Network | Role | Updates |
|---|---|---|
| `Q(s, pixel)` | Critic — Q-values per pixel | TD loss |
| `μ(s, pixel)` | Actor — per-pixel (d̂_x, d̂_y, Δd) | Deterministic policy gradient |
| `Q_target`, `μ_target` | Lagged copies, polyak averaging | Soft update: θ' ← τ·θ + (1-τ)·θ' |

### Loss 1: Q-loss (TD error — standard DQN)

```
pixel*_next = argmax_{pixel} Q_target(s', pixel)

target = r + γ · Q_target(s', pixel*_next) · (1 - done)

L_Q = Huber( Q(s)[pixel] - target )
```

Gradients update the Q-head + shared encoder.

### Loss 2: Policy gradient (DDPG-style)

```
For the selected pixel only:
    params = μ(s)[pixel]
    L_μ = - Q(s, pixel, params)       # maximize Q for the actor's own params
```

Gradients flow from `Q(s, pixel, params)` through `Q_head` into `params`, then back into `μ(s)`.

**Key detail**: Q must accept continuous params as input. Two options:

**Option A (simpler): Q(s, pixel) ignores params**
The Q-value is purely spatial — "how good is poking here regardless of direction?" The param-head is trained separately via a task reward baseline. This is an approximation but avoids the Q(params) dependency.

**Option B (faithful): Q(s, pixel, dir, Δd)**
Add continuous params as global conditioning to Q-head (e.g., tile params across spatial dimensions or feed through FiLM layers). This makes Q conditioned on the full action. More accurate, slightly more complex.

**Recommendation**: Start with Option A (Q is spatial only). The param-head improves via the same task reward — both heads contribute to success. Option B can be an ablation.

---

## Reward Design

```python
def compute_reward(scene_before, scene_after, targets):
    r = 0.0
    done = True

    for obj in objects:
        d_before = dist(obj.pose, targets[obj])
        d_after  = dist(obj.pose_after_strike, targets[obj])
        r += max(0.0, d_before - d_after)   # only reward improvement

        if d_after > threshold:
            done = False

    r -= c_step                               # small step penalty (~0.01)
    if done:
        r += c_success                        # large bonus (~10.0)

    return r, done
```

Key properties:
- `max(0, d_before - d_after)` — no penalty for objects moving away (cascading collisions)
- Step penalty encourages efficiency (fewer pokes)
- Success bonus provides sparse terminal signal

---

## Exploration

```
ε-greedy for pixel:
    ε: 1.0 → 0.05  over first 2000 episodes
    (later episodes keep small ε for continued exploration)

Gaussian noise for continuous params:
    σ: 0.3 → 0.05  over same horizon
    Added to d̂_x, d̂_y, Δd before clamping

Decay schedule: linear or exponential
```

---

## Training Loop (per episode)

```
1. Reset simulator (random objects, poses, targets)
2. Capture: seg → contour_current + contour_goal → x
3. Extract contour_mask

4. For step in 1..max_steps:
    a. Forward pass: Q_map, params = networks(x)
    b. Select action with ε-greedy + noise
    c. Execute: approach → strike → retract (NN posture + fixed controller)
    d. Observe: next_x, reward, done
    e. Store (x, pixel, d̂_x, d̂_y, Δd, r, x', done) in replay buffer
    f. If buffer size > batch_size:
        - Sample minibatch
        - Compute Q-loss → update Q-head + encoder
        - Compute μ-loss → update param-head
    g. If done: break

5. Soft-update target networks every C episodes
6. Decay ε, σ
```

---

## Hyperparameters (starting point)

| Param | Value | Notes |
|---|---|---|
| H, W | 128×128 | Image resolution |
| γ | 0.95 | Discount |
| Lr (Q, μ) | 3e-4 | Adam |
| Batch size | 64 | |
| Replay buffer | 100K | |
| ε start→end | 1.0→0.05 over 2000 episodes | |
| σ start→end | 0.3→0.05 over 2000 episodes | |
| τ (polyak) | 0.005 | Target network update rate |
| max_steps | 30 | Max pokes per episode |
| c_step | 0.01 | Per-step penalty |
| c_success | 10.0 | Terminal bonus |
| Δd_max | 0.15 | Max standoff (m) |

---

## Sim2Real Strategy

1. Train with domain randomization: friction, mass, object dimensions, initial poses
2. Input is binary contours — invariant to lighting/color
3. At deployment: same camera calibration → same contour pipeline → same Q-network forward pass
4. No system ID needed for v1

---

## Implementation Phases (no prior)

| Phase | What | Verifies |
|---|---|---|
| 1 | Isaac Sim env: objects + finger robot + camera | Reset/step loop works |
| 2 | Vision: seg → contours → stacked tensor | `x` looks correct (debug render) |
| 3 | Single-object, fixed strike (hardcoded pixel, fixed dir) | Physics: object moves on poke |
| 4 | Full action: pixel + dir + Δd, random policy | Episodes run without crashes |
| 5 | DQN + DDPG training, single object | Policy learns to poke toward target |
| 6 | Multi-object training | Policy learns sequential, adaptive strikes |
| 7 | Domain randomization + evaluation | Generalization metrics |
| 8 | Posture NN for Level 2 (amortized optimization) | Adaptive m_dir improves strike efficiency |

---

Should I flesh out any specific phase in more detail, or does this give you enough to start implementing?