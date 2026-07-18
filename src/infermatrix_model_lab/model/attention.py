from __future__ import annotations

import math

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device,
        ).tril()

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        context = weights @ v
        context = context.transpose(1, 2)

        out = context.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out