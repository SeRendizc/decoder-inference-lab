from __future__ import annotations

import math

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig


KVCache = tuple[torch.Tensor, torch.Tensor]


def _build_causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = past_length + torch.arange(
        query_length,
        device=device,
    )
    key_positions = torch.arange(
        key_length,
        device=device,
    )

    query_positions = query_positions.unsqueeze(1)
    key_positions = key_positions.unsqueeze(0)

    causal_mask = key_positions <= query_positions

    return causal_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # x: [B, Q, D]
        batch_size, query_length, _ = x.shape

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)

        key = key.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        key = key.transpose(1, 2)

        value = value.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        value = value.transpose(1, 2)

        # query/key/value: [B, H, Q, Dₕ]

        if past_key_value is None:
            past_length = 0
        else:
            past_key, past_value = past_key_value
            past_length = past_key.size(2)

            key = torch.cat(
                [past_key, key],
                dim=2,
            )
            value = torch.cat(
                [past_value, value],
                dim=2,
            )

        # key/value: [B, H, K, Dₕ]
        key_length = key.size(2)

        causal_mask = _build_causal_mask(
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            device=x.device,
        )

        # causal_mask: [1, 1, Q, K]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        scores = query @ key.transpose(-2, -1)
        # scores: [B, H, Q, K]

        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        context = weights @ value
        # context: [B, H, Q, Dₕ]

        context = context.transpose(1, 2)
        context = context.contiguous().view(
            batch_size,
            query_length,
            self.num_heads * self.head_dim,
        )

        output = self.out_proj(context)
        # output: [B, Q, D]

        if use_cache:
            present_key_value = (
                key,
                value,
            )
            return output, present_key_value

        return output