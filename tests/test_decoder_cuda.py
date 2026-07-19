import pytest
import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.decoder import DecoderOnlyTransformer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_decoder_keeps_logits_on_cuda() -> None:
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(config).cuda()
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 6),
        device="cuda",
    )

    logits = model(input_ids)

    assert logits.is_cuda
    assert logits.device == input_ids.device
    assert model.position_embedding.weight.is_cuda
