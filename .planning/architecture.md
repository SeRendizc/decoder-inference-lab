# Model Lab 计划架构

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

## 1. 目录设计

```text
infermatrix_model_lab/
├── README.md
├── pyproject.toml                         # C
├── configs/
│   ├── smoke.yaml                        # C：极小模型，秒级正确性验证
│   └── lab.yaml                          # P：约 8M～15M 参数
├── src/infermatrix_model_lab/
│   ├── config.py                         # C
│   ├── data.py                           # C
│   ├── model/
│   │   ├── attention.py                  # U
│   │   ├── block.py                      # U/P
│   │   └── decoder.py                    # U
│   ├── generation/
│   │   ├── baseline.py                   # U
│   │   └── kv_cache.py                   # U
│   ├── training/
│   │   ├── loop.py                       # U/P
│   │   └── checkpoint.py                 # C
│   └── experiments/
│       ├── metrics.py                    # P
│       ├── benchmark.py                  # C
│       ├── profiler.py                   # P/C
│       └── artifacts.py                  # C
├── tests/
│   ├── test_attention.py                 # U/C：性质由本人定义，样板可代写
│   ├── test_model.py                     # C
│   ├── test_overfit.py                   # P
│   └── test_kv_cache.py                  # U/C
├── scripts/
│   ├── train.py                          # C
│   ├── generate.py                       # C
│   ├── benchmark.py                      # C
│   └── profile.py                        # C
├── artifacts/                            # 不提交大文件
└── .planning/
```

`U/P/C` 定义见 `D:\infermatrix\.planning\ai-infra-dual-track\ownership_matrix.md`。

## 2. 设计边界

- 核心模型只依赖 PyTorch，不先用 `transformers` 代替实现。
- 数据和 tokenizer 保持简单，因为本阶段不研究 tokenizer 或数据工程。
- 训练质量只用于证明实现正确，不追求语言生成效果。
- benchmark 与 model 分离，避免计时代码污染 forward。
- 所有优化必须先通过 reference correctness，再比较性能。
- artifacts 是两个仓库的边界；Model Lab 不 import InferMatrix，InferMatrix 也不修改 Lab 原始结果。

## 3. 建议的两级配置

### smoke

- 2 layers；
- `d_model=128`；
- 4 heads；
- 短 sequence；
- 用于单测、overfit smoke 和 CPU fallback。

### lab

- 约 6 layers；
- `d_model≈384`；
- 6 heads；
- sequence length 从 128/256 起；
- 规模根据 6GB 显存实测调整，不以参数量为目标。

精确数值在 D2 由参数量计算和显存 smoke 决定，避免提前锁死错误配置。

## 4. 指标分层

### Model Lab microbenchmark

- prefill latency；
- per-token decode latency；
- tokens/s；
- peak allocated/reserved memory；
- compile time 与 steady-state；
- correctness tolerance。

### 真实 serving

- TTFT；
- TPOT；
- ITL；
- end-to-end latency；
- request throughput；
- KV/prefix cache 相关引擎指标。

两类数据必须分别命名和解释。不能用 microbenchmark 的单进程 tokens/s 冒充 serving throughput。
