# Progress

## 2026-07-18｜Planning

- 确认 Model Lab 为双项目学习主线，完成四周 Phase、目录和 Learning Gates。

## 2026-07-18｜D1 完成

- 用户理解“GPU 可见性不等于 PyTorch wheel/kernel 兼容性”，Ownership Gate 通过。
- 新 WSL venv 的 CUDA forward/backward、SDPA 和 `torch.compile` 通过。
- 环境证据保存至 `artifacts/d1_environment.json`。

## 2026-07-19｜D2 技术 Gate 通过

- 新 Git 仓库初始化在 `d2-scaffold` feature branch；项目此前无 Git history，无法创建 worktree。
- TDD RED：10 个测试因明确的 `NotImplementedError` 失败。
- TDD GREEN：实现 `ModelConfig`、YAML loader、UTF-8 `ByteTokenizer` 和 next-token examples 后，10 tests passed。
- Ruff lint、Ruff format check、GPU/SDPA/compile 回归、`pip check` 全部通过。
- 建立 smoke/lab 配置、`src` layout、README、开发计划和核心模型 Ownership 声明。
- 未实现 attention、block、decoder 或训练循环。
- D2 当前状态：技术 Gate 通过；等待用户说明主要目录职责后进入 D3。

## Errors Encountered

- setuptools 误发现 `configs`/`artifacts` 为 flat-layout package；显式声明从 `src` 发现 package。
- editable install 在 `src` 不存在时停止；建立 NotImplemented 接口壳后进入有效 RED。
- 首轮 Ruff 发现 import 顺序和 E501；先 format 再 lint 后完整验证通过。
- 外层 PowerShell 曾掩盖 WSL 子命令的非零退出码；最终验证让 WSL 命令成为直接终止条件并读取完整输出。
