import torch
from torch import nn

from infermatrix_model_lab.generation import generate_greedy


class IncrementModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError("输入超过 max_seq_len")

        batch_size, sequence_length = input_ids.shape

        logits = torch.full(
            (
                batch_size,
                sequence_length,
                self.vocab_size,
            ),
            fill_value=-1000.0,
        )

        next_token_ids = (input_ids + 1) % self.vocab_size

        logits.scatter_(
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
            value=1000.0,
        )

        return logits


def test_generate_greedy_appends_predicted_tokens() -> None:
    model = IncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[1, 2]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        max_seq_len=4,
    )

    expected = torch.tensor([[1, 2, 3, 4, 5]])

    assert torch.equal(generated, expected)


def test_generate_greedy_respects_context_length() -> None:
    model = IncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[0, 1, 2, 3]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=4,
        max_seq_len=4,
    )

    assert generated.shape == torch.Size([1, 8])
    assert torch.equal(
        generated,
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )
