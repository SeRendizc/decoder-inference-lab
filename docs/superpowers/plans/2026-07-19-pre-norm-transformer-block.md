# Pre-Norm Transformer Block Implementation Plan

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not delegate or pre-fill the user-owned `forward()` residual logic.

**Goal:** Compose the verified RMSNorm, causal self-attention, and feed-forward modules into one FP32 pre-norm residual Transformer Block.

**Architecture:** `TransformerBlock` owns two independent RMSNorms plus the existing Attention and MLP. Its only behavior is sequential `norm -> sublayer -> residual add` composition, preserving `[B,T,D]`; causal masking and MLP internals remain delegated to their existing modules.

**Tech Stack:** Python 3.10+, PyTorch, pytest, Ruff, WSL2 Ubuntu-22.04.

**Repository rule:** Continue on `d4-rmsnorm`; preserve all existing MLP and RMSNorm work; do not auto-commit.

---

### Task 1: Create the Block interface shell

**Files:**
- Create: `src/infermatrix_model_lab/model/block.py`

- [ ] **Step 1: Wire the four owned submodules**

Create:

```python
from __future__ import annotations

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.attention import CausalSelfAttention
from infermatrix_model_lab.model.mlp import FeedForward
from infermatrix_model_lab.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        raise NotImplementedError("由用户实现两次 pre-norm residual 更新")
```

- [ ] **Step 2: Verify the constructor registers distinct norms**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -c "from infermatrix_model_lab.config import ModelConfig; from infermatrix_model_lab.model.block import TransformerBlock; c=ModelConfig(256,16,64,4,1); b=TransformerBlock(c); print(list(b._modules)); print(b.norm_1.weight is b.norm_2.weight)"
```

Expected:

```text
['norm_1', 'attention', 'norm_2', 'mlp']
False
```

### Task 2: Add failing CPU behavior tests

**Files:**
- Create: `tests/test_block.py`

- [ ] **Step 1: Add shape/dtype and residual identity tests**

Create:

```python
import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.block import TransformerBlock


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_block_preserves_shape_and_fp32_dtype() -> None:
    block = TransformerBlock(_config())
    x = torch.randn(2, 10, 64, dtype=torch.float32)

    out = block(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_block_is_identity_when_sublayer_parameters_are_zero() -> None:
    block = TransformerBlock(_config())
    for parameter in block.attention.parameters():
        torch.nn.init.zeros_(parameter)
    for parameter in block.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 10, 64)

    out = block(x)

    torch.testing.assert_close(out, x)


def test_block_backward_reaches_input_attention_and_mlp() -> None:
    block = TransformerBlock(_config())
    x = torch.randn(2, 10, 64, requires_grad=True)

    block(x).square().mean().backward()

    assert x.grad is not None
    assert block.attention.out_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest tests/test_block.py -q
```

Expected: three failures at the intentional `NotImplementedError`.

### Task 3: Add failing CUDA integration test

**Files:**
- Create: `tests/test_block_cuda.py`

- [ ] **Step 1: Add explicit-device coverage**

Create:

```python
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
```

- [ ] **Step 2: Verify CUDA RED**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest tests/test_block_cuda.py -q
```

Expected on the configured WSL GPU: one failure at the intentional `NotImplementedError`.

### Task 4: User implements the two residual updates

**Files:**
- Modify: `src/infermatrix_model_lab/model/block.py`

- [ ] **Step 1: Predict the data flow**

Record before editing:

```text
x                                      [B,T,D]
norm_1(x)                              [B,T,D]
attention(norm_1(x))                   [B,T,D]
x + attention_delta                    [B,T,D]
norm_2(updated_x)                      [B,T,D]
mlp(norm_2(updated_x))                 [B,T,D]
updated_x + mlp_delta                  [B,T,D]
```

- [ ] **Step 2: Replace only the intentional exception**

The user writes these exact semantic operations without adding unrelated behavior:

```text
1. normalize current x with norm_1
2. compute attention_delta from the normalized value
3. add attention_delta to the original current x
4. normalize the updated x with norm_2
5. compute mlp_delta from that normalized value
6. add mlp_delta to the updated x and return it
```

Do not run Attention and MLP in parallel; do not add a final normalization, detach, clone, in-place `+=`, loss, or device conversion.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest tests/test_block.py tests/test_block_cuda.py -q
```

Expected: four tests pass, including CUDA.

### Task 5: Full review and regression verification

**Files:**
- Review: `src/infermatrix_model_lab/model/block.py`
- Review: `tests/test_block.py`
- Review: `tests/test_block_cuda.py`

- [ ] **Step 1: Run full tests**

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest -q
```

Expected: all Attention, MLP, RMSNorm, config, data, Block CPU, and Block CUDA tests pass.

- [ ] **Step 2: Run lint and focused format checks**

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m ruff check .
/root/.venvs/infermatrix-model-lab/bin/python -m ruff format --check src/infermatrix_model_lab/model/block.py tests/test_block.py tests/test_block_cuda.py
```

Expected: all checks pass. Existing formatting debt in unrelated Attention/MLP files remains reported separately.

- [ ] **Step 3: Verify scope**

```bash
git diff --check
git status --short --branch
```

Expected: `d4-rmsnorm`, no generated artifacts, no `.idea/` or `.venv/`, and no unrelated InferMatrix changes.

- [ ] **Step 4: Learning gate**

The user explains without reading code:

1. why the MLP must consume the residual stream after Attention rather than the original `x`;
2. why `norm_1` and `norm_2` do not share parameters;
3. why zero sublayer outputs make the complete block an identity function;
4. why residual addition requires both sublayers to return `[B,T,D]`.

