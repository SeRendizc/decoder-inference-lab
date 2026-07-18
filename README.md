# InferMatrix Model Lab

面向 AI Infra 学习和实验的最小 PyTorch 仓库。目标不是训练有竞争力的大模型，而是亲手理解 Decoder-only Transformer、KV Cache、prefill/decode 和推理优化，并产生可被 InferMatrix 固化的实验与故障证据。

## 当前状态

- D1：WSL2 GPU/PyTorch 环境通过技术与 Ownership Gate。
- D2：配置、byte tokenizer、next-token examples、测试和工程骨架已完成技术验收。
- D3：手写 multi-head causal self-attention 已完成，shape、causal behavior、CPU/CUDA backward 测试通过。

## 环境

- WSL2 Ubuntu 22.04
- Python 3.10
- PyTorch 2.12.1 + CUDA 13.0 wheel
- RTX 3060 Laptop 6GB，compute capability 8.6
- venv：`/root/.venvs/infermatrix-model-lab`

验证环境：

```bash
python scripts/verify_environment.py \
  --check-compile \
  --output artifacts/d1_environment.json
```

## 目录职责

| 路径                                    | 职责                                 | Ownership |
| ------------------------------------- | ---------------------------------- | --------- |
| `configs/`                            | smoke/lab 超参数，不包含模型逻辑              | P/C       |
| `src/infermatrix_model_lab/config.py` | YAML 读取、维度约束、`head_dim`            | C         |
| `src/infermatrix_model_lab/data.py`   | UTF-8 byte tokenizer、next-token 样本 | C         |
| `src/infermatrix_model_lab/model/`    | attention、block、decoder 核心模型       | U         |
| `tests/`                              | 正确性与边界条件                           | U/C       |
| `scripts/`                            | 可重复执行入口                            | C         |
| `artifacts/`                          | 环境、指标、trace 与证据                    | P/C       |
| `.planning/`                          | 路线、学习 Gate 与进度                     | P         |

核心模型的 ownership 约束见 [`src/infermatrix_model_lab/model/OWNERSHIP.md`](src/infermatrix_model_lab/model/OWNERSHIP.md)。

## 安装与验证

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check .
```

## 两级配置

- `configs/smoke.yaml`：2 layers、`d_model=128`、4 heads，用于秒级 correctness。
- `configs/lab.yaml`：6 layers、`d_model=384`、6 heads，约 8M～15M 目标区间；最终参数量在模型实现后实测。

## 当前验证

- 14 项 pytest 通过；无 CUDA 的机器会自动跳过 CUDA 专项测试。
- Ruff 全绿。
- Attention 保持朴素实现，用于后续与 PyTorch SDPA 对比。

## 下一步

实现 normalization、MLP 和 residual decoder block，再进入最小 decoder-only language model。
