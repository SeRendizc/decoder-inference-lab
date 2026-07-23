from __future__ import annotations

import torch
from torch import nn

from decoder_inference_lab.config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_dim = config.d_model * config.mlp_ratio
        self.up_proj = nn.Linear(config.d_model, self.hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(self.hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        hidden = self.up_proj(x)
        # hidden: [B, T, hidden_dim]

        hidden = self.activation(hidden)

        out = self.down_proj(hidden)
        # out: [B, T, D]

        out = self.dropout(out)

        return out
