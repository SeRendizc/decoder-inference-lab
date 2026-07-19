from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        mean_square = x.square().mean(dim=-1, keepdim=True)
        # mean_square: [B, T, 1]

        inverse_rms = torch.rsqrt(mean_square + self.eps)
        # inverse_rms: [B, T, 1]
        # rsqrt: 1/√x

        normalized = x * inverse_rms
        # normalized: [B, T, D]

        out = normalized * self.weight
        # out: [B, T, D]

        return out
