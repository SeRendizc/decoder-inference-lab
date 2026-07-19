from __future__ import annotations

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.attention import CausalSelfAttention
from infermatrix_model_lab.model.mlp import FeedForward
from infermatrix_model_lab.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        attention_input = self.norm_1(x)
        attention_delta = self.attention(attention_input)

        x = x + attention_delta

        mlp_input = self.norm_2(x)
        mlp_delta = self.mlp(mlp_input)

        x = x + mlp_delta

        return x