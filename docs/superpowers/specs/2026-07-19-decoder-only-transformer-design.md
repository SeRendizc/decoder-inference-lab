# Decoder-Only Transformer Design

## Goal

组装一个最小、可测试的 Decoder-only Transformer，把已有的 Embedding 概念、Pre-Norm TransformerBlock 和词表预测连接成完整 forward，同时保持训练、生成和 KV Cache 在本轮范围之外。

## Scope

本轮实现 `DecoderOnlyTransformer.forward(input_ids) -> logits`：

- 输入：整数 token id，shape 为 `[B, T]`。
- 输出：每个位置对完整词表的未归一化分数，shape 为 `[B, T, V]`。
- 位置编码：learned absolute position embedding。
- 主干：`num_layers` 个已有的 `TransformerBlock`。
- 输出端：Final RMSNorm 和独立的无 bias LM Head。

本轮不实现 loss、训练循环、generation、RoPE、KV Cache、weight tying 或采样。

## Components

`DecoderOnlyTransformer` 包含：

1. `token_embedding: nn.Embedding(vocab_size, d_model)`：表示 token 身份。
2. `position_embedding: nn.Embedding(max_seq_len, d_model)`：表示绝对位置。
3. `blocks: nn.ModuleList`：按顺序堆叠 `num_layers` 个 `TransformerBlock`。
4. `final_norm: RMSNorm`：在词表投影前规范化最终 residual stream。
5. `lm_head: nn.Linear(d_model, vocab_size, bias=False)`：把每个位置的 hidden vector 映射为词表 logits。

Token Embedding 与 LM Head 暂不共享权重，便于分别观察参数、梯度和职责。

## Data Flow

```text
input_ids [B,T]
  ├─ token_embedding                 -> [B,T,D]
  └─ arange(T) -> position_embedding -> [T,D]
                  相加并广播          -> [B,T,D]
  -> blocks[0]
  -> ...
  -> blocks[N-1]                    -> [B,T,D]
  -> final_norm                     -> [B,T,D]
  -> lm_head                        -> logits [B,T,V]
```

位置 tensor 必须创建在 `input_ids.device` 上，模型不得假设 CUDA。每个 Block 保持 shape 不变，只有 LM Head 把最后一维从 `D` 攒射到 `V`。

## Validation and Errors

- `input_ids` 必须是二维 `[B, T]`，否则抛出带 shape 信息的 `ValueError`。
- `T` 不得超过 `config.max_seq_len`，否则抛出提及 `max_seq_len` 的 `ValueError`。
- token id 的 dtype 和取值范围交给 `nn.Embedding` 的 PyTorch 原生错误处理，避免重复维护框架校验。

## Tests

行为测试覆盖：

- `[B,T] -> [B,T,V]` 的 shape 和 FP32 dtype。
- 输入超过 `max_seq_len` 时明确失败。
- 修改未来 token 不改变更早位置 logits 的模型级 causal 性质。
- backward 后梯度到达 Token Embedding、Position Embedding、首个 Block 和 LM Head。
- CUDA 可用时，输入、参数和输出留在 CUDA device。

## Learning Ownership

Codex 负责 spec、计划、测试、组件空壳、局部提示和验收。用户负责首次实现 Decoder `forward()`，并能解释：

- 为什么 Token Embedding 与 Position Embedding 可以相加；
- 为什么 Position Embedding 的 `[T,D]` 能与 `[B,T,D]` 相加；
- 为什么 Final RMSNorm 位于全部 Block 之后；
- 为什么 LM Head 输出的是 logits 而不是 token id。

## Acceptance

