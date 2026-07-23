import pytest
import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import CausalSelfAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_attention_runs_cuda_forward_backward() -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
    )
    attention = CausalSelfAttention(config).cuda()
    x = torch.randn(2, 5, 64, device="cuda", requires_grad=True)

    output = attention(x)
    output.square().mean().backward()

    assert output.shape == (2, 5, 64)
    assert output.device.type == "cuda"
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
