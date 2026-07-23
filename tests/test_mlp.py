import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.mlp import FeedForward


def _make_config(*, dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=dropout,
    )


def test_feed_forward_preserves_model_shape() -> None:
    mlp = FeedForward(_make_config())
    x = torch.randn(2, 10, 64)

    out = mlp(x)

    assert out.shape == x.shape


def test_feed_forward_backward_reaches_both_projections() -> None:
    mlp = FeedForward(_make_config())
    x = torch.randn(2, 10, 64, requires_grad=True)

    mlp(x).square().mean().backward()

    assert x.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None


def test_feed_forward_eval_disables_dropout() -> None:
    mlp = FeedForward(_make_config(dropout=0.5))
    mlp.eval()
    x = torch.randn(2, 10, 64)

    first = mlp(x)
    second = mlp(x)

    torch.testing.assert_close(first, second)
