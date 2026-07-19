import pytest
import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.decoder import DecoderOnlyTransformer


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


def test_decoder_returns_logits_with_expected_shape_and_dtype() -> None:
    model = DecoderOnlyTransformer(_config())
    input_ids = torch.randint(0, 256, (2, 6))

    logits = model(input_ids)

    assert logits.shape == (2, 6, 256)
    assert logits.dtype == torch.float32


def test_decoder_rejects_non_matrix_input() -> None:
    model = DecoderOnlyTransformer(_config())
    input_ids = torch.randint(0, 256, (2, 3, 4))

    with pytest.raises(ValueError, match=r"\[B, T\]"):
        model(input_ids)


def test_decoder_rejects_sequences_longer_than_maximum() -> None:
    model = DecoderOnlyTransformer(_config())
    input_ids = torch.randint(0, 256, (2, 17))

    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids)


def test_decoder_is_causal_at_model_level() -> None:
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_config()).eval()
    original = torch.tensor([[3, 5, 8, 13, 21, 34]])
    changed_future = torch.tensor([[3, 5, 8, 13, 55, 89]])

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed_future)

    torch.testing.assert_close(
        original_logits[:, :4],
        changed_logits[:, :4],
    )


def test_decoder_backward_reaches_embeddings_blocks_and_head() -> None:
    model = DecoderOnlyTransformer(_config())
    input_ids = torch.randint(0, 256, (2, 6))

    model(input_ids).square().mean().backward()

    assert model.token_embedding.weight.grad is not None
    assert model.position_embedding.weight.grad is not None
    assert model.blocks[0].attention.out_proj.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_decoder_uses_every_configured_block() -> None:
    model = DecoderOnlyTransformer(_config())

    assert len(model.blocks) == 2
