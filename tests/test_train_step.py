import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.decoder import DecoderOnlyTransformer
from infermatrix_model_lab.training.loss import NextTokenCrossEntropy
from infermatrix_model_lab.training.step import train_step


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_train_step_returns_finite_scalar_loss() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_config())
    loss_fn = NextTokenCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    input_ids = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))

    loss = train_step(
        model,
        loss_fn,
        optimizer,
        input_ids,
        targets,
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_train_step_updates_model_parameters() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_config())
    loss_fn = NextTokenCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    input_ids = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))
    before = model.lm_head.weight.detach().clone()

    train_step(
        model,
        loss_fn,
        optimizer,
        input_ids,
        targets,
    )

    after = model.lm_head.weight.detach()

    assert not torch.equal(before, after)
