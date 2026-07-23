import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import CausalSelfAttention


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
    )


def test_attention_preserves_shape() -> None:
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 5, 64)

    output = attention(x)

    assert output.shape == (2, 5, 64)
