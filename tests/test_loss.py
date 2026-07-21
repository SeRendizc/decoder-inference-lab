import pytest
import torch
from torch.nn import functional as F

from infermatrix_model_lab.training.loss import NextTokenCrossEntropy


def test_next_token_cross_entropy_matches_pytorch_reference() -> None:
    loss_fn = NextTokenCrossEntropy()

    torch.manual_seed(7)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor(
        [
            [3, 1, 4],
            [0, 2, 3],
        ]
    )

    actual = loss_fn(logits, targets)
    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    torch.testing.assert_close(actual, expected)


def test_next_token_cross_entropy_returns_scalar() -> None:
    loss_fn = NextTokenCrossEntropy()

    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))

    loss = loss_fn(logits, targets)

    assert loss.shape == torch.Size([])


def test_loss_decreases_when_correct_logit_increases() -> None:
    loss_fn = NextTokenCrossEntropy()

    targets = torch.tensor([[1]])

    weak_logits = torch.tensor(
        [
            [
                [0.2, 2.0, 1.8, 0.1],
            ]
        ]
    )
    strong_logits = torch.tensor(
        [
            [
                [0.2, 4.0, 1.8, 0.1],
            ]
        ]
    )

    weak_loss = loss_fn(weak_logits, targets)
    strong_loss = loss_fn(strong_logits, targets)

    assert strong_loss < weak_loss


def test_loss_backward_reaches_logits() -> None:
    loss_fn = NextTokenCrossEntropy()

    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3))

    loss_fn(logits, targets).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (
            torch.randn(6, 5),
            torch.zeros(2, 3, dtype=torch.long),
            "logits",
        ),
        (
            torch.randn(2, 3, 5),
            torch.zeros(6, dtype=torch.long),
            "targets",
        ),
    ],
)
def test_loss_rejects_invalid_tensor_ranks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    loss_fn = NextTokenCrossEntropy()

    with pytest.raises(ValueError, match=message):
        loss_fn(logits, targets)


def test_loss_rejects_mismatched_batch_sequence_shape() -> None:
    loss_fn = NextTokenCrossEntropy()

    logits = torch.randn(4, 8, 5)
    targets = torch.zeros(8, 4, dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[B, T\]"):
        loss_fn(logits, targets)
