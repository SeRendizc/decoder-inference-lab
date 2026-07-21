import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.attention import CausalSelfAttention


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
    )


def test_attention_supports_backward() -> None:
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 5, 64, requires_grad=True)

    attention(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [
        parameter.grad for parameter in attention.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
