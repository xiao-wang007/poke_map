from __future__ import annotations

import numpy as np
import torch

from config_loader import load_config
from vision.camera import RESOLUTION


CONFIG = load_config()
TRAIN_CONFIG = CONFIG["training"]

BUFFER_CAPACITY: int = TRAIN_CONFIG["buffer_capacity"]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Ring buffer storing one active env transition per entry."""

    def __init__(self, capacity: int = BUFFER_CAPACITY, device: str = DEVICE):
        self.capacity = capacity
        self.device = device

        height, width = RESOLUTION
        self.x = torch.zeros(capacity, 2, height, width, dtype=torch.uint8)
        self.pixel = torch.zeros(capacity, 2, dtype=torch.long)
        self.d_xy = torch.zeros(capacity, 2)
        self.velocity = torch.zeros(capacity, 1)
        self.strike_length = torch.zeros(capacity, 1)
        self.r = torch.zeros(capacity, 1)
        self.x_next = torch.zeros(capacity, 2, height, width, dtype=torch.uint8)
        self.done = torch.zeros(capacity, 1, dtype=torch.bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        x: torch.Tensor,
        pixel_ij: np.ndarray,
        d_xy: np.ndarray,
        velocity: float,
        strike_length: float,
        reward: float,
        x_next: torch.Tensor,
        done: bool,
    ):
        idx = self.ptr
        x_uint8 = (x.squeeze(0) * 255).to(torch.uint8)
        x_next_uint8 = (x_next.squeeze(0) * 255).to(torch.uint8)

        self.x[idx].copy_(
            x_uint8.cpu() if x_uint8.device.type != "cpu" else x_uint8)
        self.pixel[idx] = torch.tensor(pixel_ij)
        self.d_xy[idx] = torch.tensor(d_xy)
        self.velocity[idx] = torch.tensor([velocity])
        self.strike_length[idx] = torch.tensor([strike_length])
        self.r[idx] = torch.tensor([reward])
        self.x_next[idx].copy_(
            x_next_uint8.cpu()
            if x_next_uint8.device.type != "cpu"
            else x_next_uint8
        )
        self.done[idx] = torch.tensor([done])

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        x = self.x[indices].to(torch.float32).to(self.device) / 255.0
        x_next = self.x_next[indices].to(torch.float32).to(self.device) / 255.0
        return {
            "x": x,
            "pixel": self.pixel[indices].to(self.device),
            "d_xy": self.d_xy[indices].to(self.device),
            "velocity": self.velocity[indices].to(self.device),
            "strike_length": self.strike_length[indices].to(self.device),
            "r": self.r[indices].to(self.device),
            "x_next": x_next,
            "done": self.done[indices].to(self.device),
            "mask_next": x_next[:, 0] > 0,
        }

    def __len__(self) -> int:
        return self.size
