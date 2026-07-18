from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive_dimensions = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} 必须为正整数")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须满足 0.0 <= dropout < 1.0")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def load_model_config(path: Path) -> ModelConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("配置文件必须包含 model mapping")
    return ModelConfig(**raw["model"])
