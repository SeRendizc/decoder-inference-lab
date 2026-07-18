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


def test_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(_config()).eval()

    shared_prefix = torch.randn(1, 3, 64)
    first_future = torch.randn(1, 2, 64)
    second_future = torch.randn(1, 2, 64) + 10.0

    first_input = torch.cat([shared_prefix, first_future], dim=1)
    second_input = torch.cat([shared_prefix, second_future], dim=1)

    first_output = attention(first_input)
    second_output = attention(second_input)

    torch.testing.assert_close(
        first_output[:, :3],
        second_output[:, :3],
        rtol=0.0,
        atol=1e-6,
    )
