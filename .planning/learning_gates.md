# Model Lab 学习门禁

## Gate A｜Transformer

本人必须能回答：

- 为什么 attention score 要除以 `sqrt(head_dim)`？
- causal mask 施加在 softmax 前还是后，为什么？
- `[B,T,C]` 如何变成 `[B,H,T,D]`，哪里最容易 transpose 错？
- residual、normalization、MLP 分别解决什么问题？
- next-token loss 的输入与 target 为什么要 shift？

实操：修改 sequence length/head count，先预测 shape 和显存变化，再运行验证。

## Gate B｜KV Cache 与推理阶段

本人必须能回答：

- prefill 和 decode 的输入 shape、并行度和瓶颈有何不同？
- 为什么 K/V 可以 cache，而 Q 通常针对当前 token 重算？
- 每层 cache 的主要 shape 和显存增长项是什么？
- cache 为什么应改变性能而不改变 greedy 语义？
- batch、prompt length、generated length 如何分别影响结果？

实操：在一个故意写错 cache position 的版本中定位首个 logits divergence。

## Gate C｜测量与 Profiler

本人必须能回答：

- CUDA 异步执行为什么会让朴素计时错误？
- 为什么需要 warmup 和重复实验？
- TTFT、TPOT、ITL、throughput 分别适合描述什么？
- `torch.compile` 为什么可能首轮更慢，稳态也不保证更快？
- Profiler 中 CPU launch gap、kernel、graph break 各说明什么？

实操：给一份未知 trace，提出瓶颈假设、证据和下一步实验。

## Gate D｜真实 serving

本人必须能回答：

- Model Lab 的 KV Cache 与 vLLM 的 paged KV 管理处于哪两个抽象层？
- prefix caching 的命中需要哪些输入保持一致？
- continuous batching 为什么改变吞吐/延迟 trade-off？
- streaming chunk 正常到达是否等价于语义结果正确？
- 如何区分 client parser、HTTP/SSE protocol、engine 和 model behavior 的问题？

实操：拿一个 InferMatrix failure report，给出最小复现和分层排查顺序。

## 最终答辩门禁

不看代码，在 30 分钟内完成：

1. 5 分钟画模型 forward 与 KV Cache；
2. 5 分钟解释一次优化实验；
3. 5 分钟解释 InferMatrix 的分层证据链；
4. 10 分钟讲 failure、诊断和修复；
5. 5 分钟说明项目限制与下一步。

答不上来的部分进入下一轮学习 backlog，不包装进简历。
