import pytest
import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.block import TransformerBlock


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_block_runs_on_cuda_when_explicitly_moved() -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    block = TransformerBlock(config).cuda()
    x = torch.randn(2, 10, 64, device="cuda")

    out = block(x)

    assert out.shape == x.shape
    assert out.device.type == "cuda"
