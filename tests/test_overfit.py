import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.decoder import DecoderOnlyTransformer
from infermatrix_model_lab.training.loss import NextTokenCrossEntropy
from infermatrix_model_lab.training.overfit import overfit_tiny_batch


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        max_seq_len=4,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_overfit_tiny_batch_records_every_loss() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_config())
    loss_fn = NextTokenCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 0]])

    losses = overfit_tiny_batch(
        model,
        loss_fn,
        optimizer,
        input_ids,
        targets,
        steps=5,
    )

    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)


def test_overfit_tiny_batch_reduces_loss() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_config())
    loss_fn = NextTokenCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 0]])

    losses = overfit_tiny_batch(
        model,
        loss_fn,
        optimizer,
        input_ids,
        targets,
        steps=200,
    )

    assert losses[-1] < losses[0]
