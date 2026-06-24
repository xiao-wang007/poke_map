import torch
import torch.nn as nn
import torch.nn.functional as F


#* ================================================================
#*  Component networks
#* ================================================================

class DoubleConv(nn.Module):
    """(Conv → GN → ReLU) × 2.  Padding=1 keeps spatial dims unchanged."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample → concat(skip) → DoubleConv."""

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(prev_ch + skip_ch, out_ch, (prev_ch + skip_ch) // 2)

    def forward(self, x_prev, x_skip):
        x_prev = self.up(x_prev)
        diff_y = x_skip.size(-2) - x_prev.size(-2)
        diff_x = x_skip.size(-1) - x_prev.size(-1)
        x_prev = F.pad(x_prev, [diff_x // 2, diff_x - diff_x // 2,
                                 diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x_skip, x_prev], dim=1))


#* ================================================================
#*  U-Net (feature extractor only, no output head)
#* ================================================================

class UNet(nn.Module):
    """Lightweight U-Net backbone.

    Input  (B, C_in, H, W) — binary contour channels.
    Output (B, C, H, W)     — feature map at original resolution.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc   = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.up2   = Up(c * 8, c * 4, c * 4)
        self.up3   = Up(c * 4, c * 2, c * 2)
        self.up4   = Up(c * 2, c, c)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)         # (B, C, H, W)
        x2 = self.down1(x1)      # (B, 2C, H/2, W/2)
        x3 = self.down2(x2)      # (B, 4C, H/4, W/4)
        x4 = self.down3(x3)      # (B, 8C, H/8, W/8)
        x  = self.up2(x4, x3)    # (B, 4C, H/4, W/4)
        x  = self.up3(x, x2)     # (B, 2C, H/2, W/2)
        x  = self.up4(x, x1)     # (B, C, H, W)
        return x


#* ================================================================
#*  FiLM (Feature-wise Linear Modulation)
#* ================================================================

class FiLM(nn.Module):
    """Map action params (B, 3) → per-channel scale γ and shift β.

    γ, β have the same channel count as the feature map being modulated.
    """

    def __init__(self, param_dim: int = 3, feature_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2),  # γ || β
        )

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (γ, β), each shape (B, feature_dim)."""
        out = self.mlp(params)                     # (B, feature_dim * 2)
        gamma = out[:, :out.shape[-1] // 2]         # (B, feature_dim)
        beta  = out[:, out.shape[-1] // 2:]          # (B, feature_dim)
        return gamma, beta


#* ================================================================
#*  SpatialActorCritic
#* ================================================================

class SpatialActorCritic(nn.Module):
    """Actor-Critic with shared U-Net backbone and FiLM-conditioned Q-head.

    Input  (B, 2, H, W) — contour_current + contour_goal
    Output q_map  (B, 1, H, W) — per-pixel Q-values (critic)
           params (B, 3, H, W) — (d̑_x, d̑_y, Δd) per pixel, activated

    Architecture
    ------------
      x → U-Net → f (B, C, H, W) ─┬─ param_head → params_raw → (tanh,sigmoid) → params
                                  │
                                  │  params pooled or indexed → (B,3)
                                  │       │
                                  │       ▼
                                  │  FiLM(γ,β) ──────────┐
                                  │                       │
                                  └── f ──────────────────┤
                                                          ▼
                                                    f_mod = γ·f + β
                                                          │
                                                          ▼
                                                     q_head → Q-map

    Training:
      Q-loss    updates q_head + FiLM + shared U-Net
      DPG loss  updates param_head + FiLM    (q_head and U-Net frozen)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        delta_d_max: float = 0.15,
    ):
        super().__init__()
        c = base_channels

        #! shared backbone for both critic and actor
        self.unet = UNet(in_channels=in_channels, base_channels=c)
        
        #! critic 
        self.q_head = nn.Conv2d(c, 1, 1)              # per-pixel Q
        self.film = FiLM(param_dim=3, feature_dim=c, hidden_dim=64)

        #! actor
        self.param_head = nn.Conv2d(c, 3, 1)          # raw (d̑_x, d̑_y, Δd)
        self.delta_d_max = delta_d_max

    #* -- helpers -------------------------------------------------------

    #! bound the action params to valid ranges
    def _activate_params(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply tanh/sigmoid to raw param output; normalise direction."""
        d_xy  = torch.tanh(raw[:, :2])                    # (B, 2, H, W) in [-1, 1]
        d_xy  = F.normalize(d_xy, dim=1, eps=1e-6)       # unit length per pixel
        delta = torch.sigmoid(raw[:, 2:]) * self.delta_d_max  # (B, 1, H, W)
        return torch.cat([d_xy, delta], dim=1)          # (B, 3, H, W)

    def params_at_pixel(
        self, params: torch.Tensor, pixel_ij: torch.Tensor
    ) -> torch.Tensor:
        """Extract (d̑, Δd) at specific pixels.  pixel_ij: (B, 2)."""
        B = pixel_ij.shape[0]
        batch_idx = torch.arange(B, device=params.device)
        return params[batch_idx, :, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B, 3)

    def q_vals_at_pixel(
        self, q_map: torch.Tensor, pixel_ij: torch.Tensor
    ) -> torch.Tensor:
        """Extract Q-values at specific pixels."""
        B = pixel_ij.shape[0]
        batch_idx = torch.arange(B, device=q_map.device)
        return q_map[batch_idx, 0, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B,)

    #* -- forward -------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass; FiLM uses contour-masked spatial mean of params.
        Used for TD target pixel selection in train_step (cheap ranking before
        per-pixel re-evaluation) and get_values_at_pixel convenience."""
        f = self.unet.get_features(x)              # (B, C, H, W)
        params_raw = self.param_head(f)            # (B, 3, H, W)
        params = self._activate_params(params_raw) # (B, 3, H, W)

        # contour-masked mean — only object pixels contribute to FiLM
        contour_mask = (x[:, 0:1] > 0.0).float()   # (B, 1, H, W)
        counts = contour_mask.sum(dim=[-2, -1]).clamp(min=1)  # (B, 1)
        params_global = (params * contour_mask).sum(dim=[-2, -1]) / counts  # (B, 3)
        gamma, beta = self.film(params_global)
        f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
        q_map = self.q_head(f_mod)                 # (B, 1, H, W)

        return q_map, params

    def q_map_with_params(
        self,
        x: torch.Tensor,
        params_at_pixel: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:
        """Q-map conditioned on explicit per-sample action params.

        Used during training to compute Q(s, pixel*, a) where a comes
        from the replay buffer (Q-loss) or from the current actor (DPG).
        The shared encoder is recomputed because x may differ in each call.

        Returns: q_map (B, 1, H, W)
        """
        f = self.unet.get_features(x)          # (B, C, H, W)
        gamma, beta = self.film(params_at_pixel)
        f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
        return self.q_head(f_mod)

    def features_and_params(self, x: torch.Tensor) -> tuple:
        """Return U-Net features + activated params (used in DPG step)."""
        f = self.unet.get_features(x)
        params_raw = self.param_head(f)
        params = self._activate_params(params_raw)
        return f, params, params_raw

    def get_values_at_pixel(
        self,
        x: torch.Tensor,
        pixel_ij: torch.Tensor,     # (B, 2) int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return (Q_val, params) at chosen pixels."""
        q_map, params = self.forward(x)
        B = pixel_ij.shape[0]
        batch_idx = torch.arange(B, device=x.device)

        q_vals = q_map[batch_idx, 0, pixel_ij[:, 0], pixel_ij[:, 1]]  # (B,)
        p_vals = self.params_at_pixel(params, pixel_ij)                # (B, 3)
        return q_vals, p_vals

    def greedy_pixel(
        self,
        f: torch.Tensor,                 # (B, C, H, W) — pre-computed U-Net features
        params: torch.Tensor,            # (B, 3, H, W) — per-pixel action params
        contour_mask: torch.Tensor,      # (B, H, W) bool
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel FiLM-conditioned greedy pixel selection.

        Ranks pixels by cheap mean-conditioned Q, then re-evaluates the
        top-K with each pixel's own action params via per-pixel FiLM.
        One U-Net pass is assumed (caller provides f).

        Note: this is exact only when top_k ≥ all contour pixels.
        With typical top_k=5 it is an approximation — the globally best
        per-pixel conditioned pixel is found if it ranks in the top-K
        mean-conditioned candidates, which is empirically nearly always true.

        Returns:
          pixels: (B, 2) long — (row, col) of best pixel per batch item
          q_vals: (B,)  float — per-pixel conditioned Q at winning pixel
        """
        B, C, H, W = f.shape
        batch_idx = torch.arange(B, device=f.device)

        # cheap mean-conditioned Q for candidate ranking
        mask_f = contour_mask.unsqueeze(1).float()
        counts = mask_f.sum(dim=[-2, -1]).clamp(min=1)
        params_mean = (params * mask_f).sum(dim=[-2, -1]) / counts  # (B, 3)
        gamma_m, beta_m = self.film(params_mean)
        f_mean = gamma_m[:, :, None, None] * f + beta_m[:, :, None, None]
        q_mean = self.q_head(f_mean).squeeze(1)                    # (B, H, W)
        q_mean[~contour_mask] = -float("inf")

        # top-K candidates per batch item
        q_flat = q_mean.view(B, -1)                               # (B, H*W)
        candidates = min(top_k, contour_mask.sum(dim=(-2, -1)).max().item())
        _, topk_idx = torch.topk(q_flat, candidates, dim=1)       # (B, K)
        topk_row = topk_idx // W                                   # (B, K)
        topk_col = topk_idx % W                                    # (B, K)

        best_q = torch.full((B,), -float("inf"), device=f.device)
        best_row = torch.zeros(B, dtype=torch.long, device=f.device)
        best_col = torch.zeros(B, dtype=torch.long, device=f.device)

        for k in range(candidates):
            row = topk_row[:, k]                                    # (B,)
            col = topk_col[:, k]                                    # (B,)
            pixel_params = params[batch_idx, :, row, col]           # (B, 3)
            gamma, beta = self.film(pixel_params)                   # (B, C)
            f_mod = gamma[:, :, None, None] * f + beta[:, :, None, None]
            q_cand = self.q_head(f_mod)                             # (B, 1, H, W)
            q_val = q_cand[batch_idx, 0, row, col]                 # (B,)
            valid = contour_mask[batch_idx, row, col]
            q_val = q_val.masked_fill(~valid, -float("inf"))

            better = q_val > best_q
            best_q[better] = q_val[better]
            best_row[better] = row[better]
            best_col[better] = col[better]

        pixels = torch.stack([best_row, best_col], dim=1)          # (B, 2)
        return pixels, best_q


#! FiLM (Feature-wise Linear Modulation) is a conditioning technique from paper:
#! Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
#?
#? Instead of concatenating extra input channels, we modulate existing features with
#? a learned affine transform per channel. Here is the flow: 
#?
#?    f ------------(B, 32, H, W)    ← U-net output features
#?    params -------(B, 3)           ← (d̂_x, d̂_y, Δd)
#?    γ  =  MLP_γ(params) ----- (B, 32)           ← learned scale per channel
#?    β  =  MLP_β(params) ----- (B, 32)           ← learned shift per channel
#?    f_mod = γ * f + β   ----- (B, 32, H, W)     ← same shape, modulated
#?    Q_map = Conv2d(32, 1)(f_mod)                ← Q-head sees conditioned features
#?
#? γ and β are produced by a tiny MLP (3 → 32 → 32 neurons, ~2K parameters). 
#? They scale and shift each of the 32 feature channels based on the action params.



