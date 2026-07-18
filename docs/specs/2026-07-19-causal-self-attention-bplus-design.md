# Causal Self-Attention B+ 练习设计

## 目标

用户亲手完成第一版 `CausalSelfAttention`，能够解释从 `[B, T, D]` 到多头布局、注意力分数、causal mask、上下文聚合再回到 `[B, T, D]` 的完整数据流。Codex 提供接口、行为测试、shape 路线卡和代码审查，但不在用户第一次尝试前生成完整核心实现。

## Ownership

- 用户负责：`CausalSelfAttention.__init__()` 中的核心层定义，以及 `forward()` 的 Q/K/V 投影、拆分 head、scaled dot-product、causal mask、softmax、聚合与合并 head。
- Codex 负责：公开接口、失败测试、测试命令、局部 API 解释、错误定位和逐行 review。
- 如果用户在同一局部步骤连续卡住两次，Codex 只把该局部降级为填空式提示，不公布整份实现。

## 接口契约

- 新建 `src/infermatrix_model_lab/model/attention.py`。
- 类名为 `CausalSelfAttention(nn.Module)`，构造参数为现有 `ModelConfig`。
- `forward(x)` 接收 `[B, T, D]` 浮点 Tensor，返回同 shape、同 device 的 Tensor。
- `D == config.d_model`，`head_dim == d_model // num_heads`。
- 第一版仅实现标准 multi-head causal self-attention；不加入 dropout、RoPE、KV Cache、GQA/MQA 或 SDPA。

## 数据流路线卡

1. `Q/K/V: [B,T,D]`。
2. 拆分并交换维度为 `[B,H,T,Dh]`。
3. `Q @ K^T / sqrt(Dh)` 得到 `[B,H,T,T]`。
4. 使用下三角 causal mask 遮住未来位置。
5. 对最后一维 softmax，得到每行和为 1 的权重。
6. `weights @ V` 得到 `[B,H,T,Dh]`。
7. 合并 head，恢复 `[B,T,D]`，再经过输出投影。

## 测试设计

RED 阶段先创建测试，确认因 `CausalSelfAttention` 尚不存在或未实现而失败：

1. shape 测试：输入 `[2,5,64]`，输出必须为 `[2,5,64]`。
2. causal 行为测试：两组输入共享相同前缀、只修改未来位置；共享前缀对应的输出必须保持一致。
3. 梯度测试在上述两项通过后再加入，确保输出可反向传播到输入和投影参数。

测试验证行为，不读取内部 attention weights，也不添加测试专用生产接口。

## 验收标准

- 用户能够逐行说明每次 `view/transpose/matmul` 前后的 shape。
- shape、causal behavior 和 backward 测试通过。
- CPU 与当前 CUDA 环境均不依赖硬编码 device。
- Ruff 与 Lab 全量 pytest 通过。
- 用户第一次实现完成并接受 review 前，Codex 不代写完整 attention 核心。
