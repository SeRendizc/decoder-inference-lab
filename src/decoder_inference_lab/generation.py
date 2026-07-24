from __future__ import annotations

import torch
from torch import nn


def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    # input_ids: [B, T]
    generated = input_ids

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -max_seq_len:]

            logits = model(context)
            next_token_logit = logits[:, -1, :]

            next_token = next_token_logit.argmax(
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

    return generated
