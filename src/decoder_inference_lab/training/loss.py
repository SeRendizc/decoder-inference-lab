from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import cross_entropy


class NextTokenCrossEntropy(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # logits: [B, T, V]
        # targets: [B, T]
        if logits.ndim != 3:
            raise ValueError(f"logits 必须是 [B, T, V]，实际为 {tuple(logits.shape)}")
        if targets.ndim != 2:
            raise ValueError(f"targets 必须是 [B, T]，实际为 {tuple(targets.shape)}")
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"logits 和 targets 的 [B, T] 必须一致，"
                f"实际为 {tuple(logits.shape[:2])} 和 {tuple(targets.shape)}"
            )

        vocab_size = logits.size(-1)

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)

        loss = cross_entropy(flat_logits, flat_targets)

        return loss
