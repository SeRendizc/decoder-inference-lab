from __future__ import annotations

from collections.abc import Sequence

import torch


class ByteTokenizer:
    @property
    def eos_token_id(self) -> int:
        return 256

    @property
    def vocab_size(self) -> int:
        return 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        byte_ids = []

        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break

            byte_ids.append(token_id)

        return bytes(byte_ids).decode(
            "utf-8",
            errors="replace",
        )


def make_next_token_examples(
    token_ids: Sequence[int],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数")
    if len(token_ids) <= sequence_length:
        raise ValueError("token 数量必须大于 sequence_length")

    tokens = torch.tensor(token_ids, dtype=torch.long)
    windows = tokens.unfold(0, sequence_length + 1, 1)
    inputs = windows[:, :-1].contiguous()
    targets = windows[:, 1:].contiguous()
    return inputs, targets
