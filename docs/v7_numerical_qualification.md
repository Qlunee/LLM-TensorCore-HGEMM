# V7 模型级数值验收与安全回退

## 已复现的现象

Qwen3-4B，RTX 3090，FP16，替换 `model.layers.0.mlp.down_proj`，batch=1，prompt=128，decode=16，seed=2026：

- V3 WMMA、V4 PTX、V5 向量化、V6 大 tile/紧凑 tile 均产生同样的 logits 偏差：max_abs=0.03125、max_relative_l2=0.004031672。
- 绝对误差通过 0.05 门槛，但相对 L2 未通过 0.001 门槛。
- 捕获原生路径的真实输入，比较第一组 Prefill/Decode Linear 输出：自定义后端相对 FP32 GEMM 后转 FP16 参考的 L2 偏差分别约 0.0104%/0.0127%；原生路径约 0.0315%/0.0297%。这是这组输入的结果，不代表所有输入。
- 仅关闭 PyTorch 的 FP16 reduced-precision reduction 后，自定义后端的模型级相对误差仍为约 0.3436%，并未通过门槛。

证据支持“后端之间的有限精度累加差异在后续层传播放大”，而不是 V6 独有的异步流水错误；这些检查不能证明所有 kernel/输入都没有错误。FP32 accumulation 也不意味着不同实现的舍入结果必须逐位一致。

## 修改行为

`benchmarks/bench_integration.py` 先对完整固定 token 序列进行非计时验证，再开始计时；所有阈值保持不变。

- `--on-logits-failure fallback`（默认）：当 best_available/best_custom 未通过门槛，显式换为 cuBLASLt，再对完整序列验收。只有回退也通过时才计时。
- `--on-logits-failure error`：不回退；保存失败信息，不计时失败路径，最后返回非零退出码。
- 原始候选结果记录在 `requested_correctness`、`requested_logits_*`、`requested_executed_backends`；实际路径记录在 `accuracy_fallback`、`executed_backends`。
- `logits_per_step_errors` 与 `requested_logits_per_step_errors` 保存 Prefill/各 Decode 步的误差。
- NaN/Inf 不可通过验收；失败路径不得产生吞吐指标。
- `custom_integration_qualified=false` 表示自定义模型集成仍未验收成功。

这是此 benchmark 对当前模型、目标层和固定 token 工作负载的保护，不是部署模块的通用数值认证，也不修改全局 Shape 调度表。验证逻辑不进入计时循环。

## 简历口径

`report_v7.py` 明确拒绝把模型级数值失败后 cuBLASLt 回退的吞吐统计为自定义后端吞吐。此案例可以报告 kernel-level 性能和误差传播分析，但不能声称自定义 kernel 已通过该模型级精度门槛。

## 重跑

只改 Python，无需重新编译 extension。调度表严格绑定代码指纹，修改 benchmark 后旧表会失效；不要手工替换指纹来复用旧性能结果。先重跑三轮直接后端测量并生成新表，再运行集成验收。保留旧 CSV 和旧调度表作为历史记录。

回归测试：`python -m pytest tests/test_integration_gate.py tests/test_v7.py -q`。
