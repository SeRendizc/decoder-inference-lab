import pytest
import torch

from infermatrix_model_lab.model.norm import RMSNorm


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_rms_norm_runs_on_cuda_when_explicitly_moved() -> None:
    norm = RMSNorm(d_model=64).cuda()
    x = torch.randn(2, 10, 64, device="cuda")

    out = norm(x)

    assert out.shape == x.shape
    assert out.device.type == "cuda"
