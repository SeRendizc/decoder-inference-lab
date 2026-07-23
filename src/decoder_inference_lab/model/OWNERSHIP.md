# Core model ownership

本目录的 attention、block 和 decoder 核心实现属于 `U`（User-owned core）。

Codex 可以提供接口、测试、局部提示和 Code Review，但不会在用户第一次尝试前生成完整 attention 实现。

