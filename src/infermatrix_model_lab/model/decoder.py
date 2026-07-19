from __future__ import annotations

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.block import TransformerBlock
from infermatrix_model_lab.model.norm import RMSNorm


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids 必须是 [B, T]，实际 shape 为 {tuple(input_ids.shape)}")

        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} 超过 max_seq_len={self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        hidden = token_vectors + position_vectors

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)

        logits = self.lm_head(hidden)

        return logits
