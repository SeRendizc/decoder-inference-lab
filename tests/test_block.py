import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.block import TransformerBlock


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_block_preserves_shape_and_fp32_dtype() -> None:
    block = TransformerBlock(_config())
    x = torch.randn(2, 10, 64, dtype=torch.float32)

    out = block(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_block_is_identity_when_sublayer_parameters_are_zero() -> None:
    block = TransformerBlock(_config())
    for parameter in block.attention.parameters():
        torch.nn.init.zeros_(parameter)
    for parameter in block.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 10, 64)

    out = block(x)

    torch.testing.assert_close(out, x)


def test_block_preserves_attention_residual_when_mlp_is_zero() -> None:
    block = TransformerBlock(_config())
    for parameter in block.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 10, 64)

    expected = x + block.attention(block.norm_1(x))
    actual = block(x)

    torch.testing.assert_close(actual, expected)


def test_block_backward_reaches_input_attention_and_mlp() -> None:
    block = TransformerBlock(_config())
    x = torch.randn(2, 10, 64, requires_grad=True)

    block(x).square().mean().backward()

    assert x.grad is not None
    assert block.attention.out_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None
