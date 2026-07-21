from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from infermatrix_model_lab.training.step import train_step


def overfit_tiny_batch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> list[float]:
    # input_ids: [B, T]
    # targets: [B, T]
    losses = []

    for _ in range(steps):
        loss = train_step(
            model,
            loss_fn,
            optimizer,
            input_ids,
            targets,
        )
        losses.append(loss.item())

    return losses
