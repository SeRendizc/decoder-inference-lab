import torch

from decoder_inference_lab.model.norm import RMSNorm


def test_rms_norm_preserves_shape_and_fp32_dtype() -> None:
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 10, 64, dtype=torch.float32)

    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_rms_norm_outputs_unit_mean_square() -> None:
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 10, 64, dtype=torch.float32)

    out = norm(x)
    mean_square = out.square().mean(dim=-1)

    torch.testing.assert_close(
        mean_square,
        torch.ones_like(mean_square),
        atol=2e-5,
        rtol=2e-5,
    )


def test_rms_norm_zero_input_is_finite_zero() -> None:
    norm = RMSNorm(d_model=64)
    x = torch.zeros(2, 10, 64, dtype=torch.float32)

    out = norm(x)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0


def test_rms_norm_backward_reaches_input_and_weight() -> None:
    norm = RMSNorm(d_model=64)
    x = torch.randn(
        2,
        10,
        64,
        dtype=torch.float32,
        requires_grad=True,
    )

    norm(x).square().mean().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
