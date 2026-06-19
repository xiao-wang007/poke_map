import torch
import torch.nn as nn
import torch.nn.functional as F

#* ================================================================
#*  Component 4: Q-Network (U-Net)
#* ================================================================

#! instance: DoubleConv(2, 32), input (B, 2, H, W) --> output (B, 32, H, W)

class DoubleConv(nn.Module):
    """(Conv → GN → ReLU) × 2."""
    #! old-school:
    #! kernel_size=3, padding=1, (stride=1 by default)
    #! output = floor((input + 2*padding - kernel_size) / stride) + 1
    #! so padding=1 keeps the spatial dimensions the same.
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),         
            
            #! split the mid_ch channels into 8 groups, normalize each group
            #! independently, then learn scale/shift parameters
            nn.GroupNorm(8, mid_ch), #* 2nd arg = input channels
            
            #! in-place ops can sometimes break autograd if the original tensor
            #! value is needed later for gradient computation. But in normal
            #! sequential conv blocks is very common and usually safe.
            nn.ReLU(inplace=True), #* negative activations are zeroed in-place to save memory
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
    """Upsample → concat(skip) → DoubleConv.

    Parameters
    ----------
    prev_ch : channels from the level below (after upsampling).
    skip_ch : channels from the encoder skip connection.
    out_ch  : channels after the double conv.
    """

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #* total channels after concatenation = prev_ch + skip_ch
        self.conv = DoubleConv(prev_ch + skip_ch, out_ch, (prev_ch + skip_ch) // 2)

    def forward(self, x_prev, x_skip):
        x_prev = self.up(x_prev)
        #! handle odd-size inputs
        diff_y = x_skip.size(-2) - x_prev.size(-2)
        diff_x = x_skip.size(-1) - x_prev.size(-1)
        x_prev = F.pad(x_prev, [diff_x // 2, diff_x - diff_x // 2,
                                 diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x_skip, x_prev], dim=1))


class UNet(nn.Module):
    """Lightweight U-Net for spatial Q-value prediction.

    Input  (B, C_in, H, W) — binary contour channels.
    Output (B, 1, H, W)     — per-pixel Q-values, same spatial dims.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        #* up2: prev from x4 (c*8), skip from x3 (c*4) → out c*4
        self.up2 = Up(c * 8, c * 4, c * 4)
        #* up3: prev from up2 (c*4), skip from x2 (c*2) → out c*2
        self.up3 = Up(c * 4, c * 2, c * 2)
        #* up4: prev from up3 (c*2), skip from x1 (c) → out c
        self.up4 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)        # (B, C, H, W)
        x2 = self.down1(x1)     # (B, 2C, H/2, W/2)
        x3 = self.down2(x2)     # (B, 4C, H/4, W/4)
        x4 = self.down3(x3)     # (B, 8C, H/8, W/8)

        x = self.up2(x4, x3)    # (B, 4C, H/4, W/4)
        x = self.up3(x, x2)     # (B, 2C, H/2, W/2)
        x = self.up4(x, x1)     # (B, C, H, W)
        return self.outc(x)     # (B, 1, H, W)


class QNetwork(nn.Module):
    """Q-network wrapper: (B, 2, H, W) → (B, 1, H, W).

    Provides forward() to produce Q-values and get_q_values() to extract
    per-pixel Q at a specific batch of pixel locations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce per-pixel Q-values.

        Parameters
        ----------
        x : (B, C_in, H, W) float32, values in [0, 1].

        Returns
        -------
        Q : (B, 1, H, W) float32 — one Q-value per pixel.
        """
        return self.unet(x)

    def get_q_values(
        self,
        x: torch.Tensor,
        pixel_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-values at specific pixel locations.

        Parameters
        ----------
        x : (B, C_in, H, W).
        pixel_indices : (B, 2) int — (row, col) per batch item.

        Returns
        -------
        q_values : (B,) float32.
        """
        q_map = self.forward(x).squeeze(1)  # (B, H, W)
        B = q_map.shape[0]
        return q_map[torch.arange(B), pixel_indices[:, 0], pixel_indices[:, 1]]