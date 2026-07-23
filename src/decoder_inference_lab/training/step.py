from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # input_ids: [B, T]
    # targets: [B, T]
    model.train()

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    return loss
