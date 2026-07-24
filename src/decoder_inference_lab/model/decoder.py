from __future__ import annotations

import torch
from torch import nn

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import KVCache
from decoder_inference_lab.model.block import TransformerBlock
from decoder_inference_lab.model.norm import RMSNorm

PastKeyValues = tuple[KVCache, ...]


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(
            config.d_model,
            config.norm_eps,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PastKeyValues]:
        # input_ids: [B, Q]
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是 [B, T]，"
                f"实际 shape 为 {tuple(input_ids.shape)}"
            )

        if (
            past_key_values is not None
            and len(past_key_values) != len(self.blocks)
        ):
            raise ValueError(
                "past_key_values 的层数必须与 "
                "TransformerBlock 数量一致"
            )

        _, query_length = input_ids.shape

        if past_key_values is None:
            past_length = 0
        else:
            first_key, _ = past_key_values[0]
            past_length = first_key.size(2)

        total_length = past_length + query_length

        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"total_length={total_length} "
                f"超过 max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)
        hidden = token_vectors + position_vectors

        present_key_values: list[KVCache] = []

        for layer_index, block in enumerate(self.blocks):
            if past_key_values is None:
                layer_past = None
            else:
                layer_past = past_key_values[layer_index]

            block_result = block(
                hidden,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                hidden, layer_present = block_result
                present_key_values.append(layer_present)
            else:
                hidden = block_result

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        if use_cache:
            return logits, tuple(present_key_values)

        return logits