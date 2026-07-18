from pathlib import Path

import pytest

from infermatrix_model_lab.config import ModelConfig, load_model_config

PROJECT_ROOT = Path(__file__).parents[1]


def test_load_smoke_config() -> None:
    config = load_model_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    assert config == ModelConfig(
        vocab_size=256,
        max_seq_len=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.0,
    )
    assert config.head_dim == 32


def test_rejects_d_model_not_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="d_model.*num_heads"):
        ModelConfig(
            vocab_size=256,
            max_seq_len=64,
            d_model=130,
            num_heads=4,
            num_layers=2,
        )


@pytest.mark.parametrize(
    "field", ["vocab_size", "max_seq_len", "d_model", "num_heads", "num_layers"]
)
def test_rejects_non_positive_dimensions(field: str) -> None:
    values = {
        "vocab_size": 256,
        "max_seq_len": 64,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
    }
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        ModelConfig(**values)
