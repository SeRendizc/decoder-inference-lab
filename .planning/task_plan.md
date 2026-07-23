# Model Lab 四周执行计划

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

## Goal

用尽可能少的外围工程，完成从 Transformer 正确性、KV Cache 到 PyTorch 推理优化和真实 serving 行为的学习闭环。核心机制必须本人可实现、可测量、可解释。

## Phase 0｜环境 Gate（D1）

Codex 负责：

- 给出隔离环境命令和版本检查脚本；
- 检查 PyTorch、CUDA runtime、GPU architecture 和基础算子；
- 记录可复现环境信息。

本人负责：

- 亲自执行并解释系统 NVIDIA driver、环境内 CUDA runtime、PyTorch build 的边界；
- 验证 CUDA tensor、forward/backward 和显存查询。

验收：

- `torch.cuda.is_available()` 为真；
- RTX 3060 Laptop 被正确识别；
- 一个矩阵乘 forward/backward 成功；
- 生成 `environment.json`；
- 不复用当前已知不兼容的旧 PyTorch 1.12 + CUDA 10.2 环境。

## Phase 1｜Minimal Decoder（D2–D7）

先用 `smoke` 配置验证正确性，再用约 8M～15M 参数的 `lab` 配置训练；参数规模不是成绩指标。

本人核心任务：

- scaled dot-product causal attention；
- multi-head projection/reshape；
- RMSNorm 或 LayerNorm、MLP、residual block 的数据流；
- Decoder-only forward；
- next-token loss 与 train step。

Codex 支持任务：

- byte/char tokenizer 和 tiny corpus 管道；
- config、CLI、checkpoint、seed、logger；
- shape、gradient、overfit tests；
- loss curve 和环境记录。

验收：

- 所有 `forward()` 标注 shape；
- causal mask property test 通过；
- 固定 tiny batch 可过拟合；
- CPU/CUDA device 均不写死；
- 本人能计算模型参数量的主要组成。

## Phase 2｜Generation 与 KV Cache（D8–D14）

本人核心任务：

- 无 cache greedy generation；
- 每层 K/V cache 的 shape、更新和生命周期；
- prefill 与单 token decode 两条路径；
- cached/uncached logits、token 序列一致性判断。

Codex 支持任务：

- benchmark harness、warmup/repeat/synchronize；
- 参数化 correctness tests；
- metrics JSONL、汇总和绘图；
- OOM/非法 shape 等边界测试。

验收：

- cache 前后 greedy tokens 完全一致；
- logits 在合理 tolerance 内一致；
- 记录不同 prompt/generated length 的 prefill latency、decode latency、tokens/s 和 peak memory；
- 本人能推导无 cache 与 cache decode 的计算差异，且不把“理论复杂度”直接等同于实际加速比。

## Phase 3｜PyTorch 优化与 Profiler（D15–D21）

实验轴保持窄而可解释：

1. naive attention vs SDPA；
2. FP32 vs 硬件实际支持的 FP16/BF16；
3. eager vs `torch.compile`；
4. 不同 batch/prompt length 中选一组观察。

Codex 负责生成实验编排，但本人必须在运行前写预测，在运行后解释：

- 正确性 tolerance；
- 冷启动与稳态；
- GPU kernel 时间与 CPU launch gap；
- 显存、吞吐和延迟 trade-off；
- graph break 或 fallback 是否存在。

验收：

- 生成至少一个 Chrome trace；
- 保存 raw measurements，而不只有柱状图；
- 找到一个有 trace 支持的瓶颈；
- 写出至少一个“优化没有变快或只在特定 shape 变快”的诚实结论也视为成功。

## Phase 4｜Serving 对接与故障闭环（D22–D28）

- 将 Lab 中对 prefill/decode/KV Cache 的理解用于设计 vLLM 实验；
- Model Lab 不直接包装 vLLM，而由 InferMatrix 执行真实服务 case；
- 选择一个真实异常做最小复现、根因分析、修复或 workaround；
- 将 experiment artifacts 交给 InferMatrix evidence index。

验收：

- 真实服务数据和 Lab microbenchmark 不混为同一指标；
- failure 连续复现 3 次；
- 有分层排查记录和 regression；
- 本人完成一次 30 分钟无稿答辩。

## Backlog（四周后再决定）

- GQA/MQA；
- RoPE scaling；
- quantization internals；
- speculative decoding；
- continuous batching simulator；
- CUDA/Triton kernel；
- 多 GPU parallelism。

这些方向重要，但不得在当前四周抢占 Transformer、KV Cache、Profiler 和真实 failure chain 的时间。
