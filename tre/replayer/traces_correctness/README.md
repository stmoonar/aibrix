# 正确性验证 Trace

本目录包含 4 条用于调度正确性验证的 trace。所有 trace 都使用 model-keyed JSON 格式，每个模型从 0 秒到 trace 结束时间都有显式段覆盖，段之间无空洞、无重叠，并按时间排序。

重放建议：

```bash
run_trace.py --seed 42
```

通用约束：

- baseline：`dsqwen-7b` 与 `dsllama-8b` 为 `rps=0.5, input_tokens=256, max_tokens=128`；`dsqwen-14b` 为 `rps=0.2, input_tokens=256, max_tokens=128`。
- 饱和或高负载段统一使用 `input_tokens=512, max_tokens=256`。
- 所有相位时长均 >=50s，除 `c4_spike_glitch` 中刻意设置的 17s 窄脉冲。
- 相位长度使用了 233/287 类与 5s、10s、30s 控制周期互素或错位的值，降低边界与控制周期固定对齐造成的偶然性。

## c1_staged_saturation

目的：覆盖 A1 需求转移与 serving floor 不缩到 0。先让 `dsqwen-7b` 饱和，再切换到 `dsllama-8b` 饱和，`dsqwen-14b` 保持 baseline。

预期系统行为：`dsqwen-7b` 应在 180-467s 扩容，并在 467s 后按策略缩回但不缩到 0；`dsllama-8b` 应在 653-940s 扩容，并在 940s 后缩回但保持 serving floor；`dsqwen-14b` 不应因 baseline 负载发生明显扩容。

验收判据：高负载窗口内目标模型容量或并发资源增加；负载转移后原高载模型释放多余资源但仍保留服务实例；无负载模型不被误扩容；切换边界附近无长时间空窗或错误迁移。

## c2_anticorrelated

目的：覆盖 A2 反相关再平衡。`dsqwen-7b` 与 `dsllama-8b` 在 120-1052s 之间以 233s 为相长反相交替。

预期系统行为：奇数相 `dsqwen-7b` 扩容、`dsllama-8b` 保持 baseline；偶数相 `dsllama-8b` 扩容、`dsqwen-7b` 保持 baseline；`dsqwen-14b` 全程 baseline 且不应参与再平衡。

验收判据：资源随高载模型在每相边界后迁移；低载模型按策略缩回且不缩到 0；反相切换不应造成两个 7b/8b 模型同时长期高配，也不应让高载模型长期资源不足。

## c3_tp_pressure

目的：覆盖 A5 TP 异构与槽位压力。`dsqwen-14b` 逐级升压，同时 `dsqwen-7b` 和 `dsllama-8b` 全程 baseline，保持它们 awake 以占住单卡槽位，制造 14b 扩容压力。

预期系统行为：`dsqwen-14b` 应在 150-443s 扩容，并在 443-697s 面对更高 rps 继续扩容或提升 TP 配置；697s 后逐步缩回 baseline 所需容量。`dsqwen-7b` 与 `dsllama-8b` 应保持低配服务，不应被错误清零。

验收判据：系统能为 `dsqwen-14b` 找到满足 TP/异构约束的可行放置；槽位紧张时不会破坏 7b/8b 的 serving floor；高压段 14b 没有因错误槽位选择导致持续不可服务。

## c4_spike_glitch

目的：覆盖 A4 毛刺 vs 真突发与 SafeScale。trace 同时包含 17s 窄脉冲和 122s 宽突发，用来区分短毛刺与真实突发。

预期系统行为：`dsqwen-7b` 在 150-167s 的 17s 窄脉冲不应触发完整扩容，345-467s 的 122s 宽突发应触发扩容；`dsllama-8b` 在 643-660s 的 17s 窄脉冲不应触发完整扩容；`dsqwen-14b` 全程 baseline 且不应扩容。

验收判据：SafeScale 或等价保护机制过滤 17s 毛刺；122s 宽突发被识别为真实需求并触发扩容；短脉冲结束后系统不应残留长时间过量资源。
