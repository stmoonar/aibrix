# 第二轮标定预注册：已被 D 线决策（D2–D14）取代

> 状态说明文件。被说明的文件是 [`preregistration-20260923-calibration-run2.md`](./preregistration-20260923-calibration-run2.md)（下称"预注册"）。

## 1. 结论

2026-09-23 用户裁决：09-22 工作线（`fix/tss-rate-20260922`，plan 2026-09-21 §6.11 的 D2–D14）与 09-23 工作线（main，含预注册）冲突之处**一律以 D 线为准**。两条线在分支 `merge/tss-main-20260923` 上合并（合并提交 `121daa1a`）。从这次合并起：

- 预注册的**决策规则**（§2–§4）不再是默认分析；它保留为可选对照：`python -m scripts.analysis.calibration_decision <dataset> --regime-groups ... --rule preregistered`。默认 `--rule dline`。
- 预注册的**标签与窗口**（§6）被 D 线的标签与窗口取代（见 §3 对照表）。
- 预注册的**采集设计**（§5 数据划分、§7 阶梯/阶段/哨兵/排空/独立种子）保留：`--design ladder` 仍按它采集、按它的 split 划分训练/留出/辅助，但判定 probe 和 cell 用的"尺子"换成了 D 线的（§3）。

## 2. 预注册原文不改

预注册文件**一个字节都不改**：第二轮每个模型的 `run_manifest.json` 记录了它的路径、sha256（`2c4cfc95d87c06209ba8570e70f8de74853750649aaa9136dd41308283796979`）和 commit（`9ee7dcb5`）。

- 追溯一次 run 按什么规则采集，看 manifest 里记录的 **commit**，不看工作区里的文件。
- 代码里没有任何路径读取预注册文件的**内容**来做判断。`calibration_ladder.preregistration_provenance` 只记录它的路径、哈希、commit 和"是否改过"。`calibration_decision` 只在报告里引用本文件的路径。
- 所以即使预注册文件内容变了，也不会有代码因此失败。只是新 run 的 manifest 会如实记下新的哈希。

## 3. 逐项对照

| 项 | 预注册（§） | D 线（现行） | 实现 |
|---|---|---|---|
| TSS 分子 | — | 窗口 token 总量（30 s 窗）（D2） | `tre_common.tss` |
| TTFT 判据 | 窗口 p95 TTFT > 500 ms（§6） | **主标签** `max(500 ms, 5·(c_m + b_m·L))`，按请求算，窗口项取各请求比值的 p95（D6′）；固定 500 ms 作对照列，k=3、下限 150 ms 作消融 | `tre_common.slo_labels`：`slo_label` / `slo_label_fixed` / `slo_label_k3` |
| TPOT 判据 | 客户端 p95 > 75 ms（§6） | 同 | 同上 |
| 窗口 | 30 s 宽、5 s 步长、自由相位（§6） | 30 s 宽、10 s 步长，窗尾落在网关 10 s 格上，区间 `(start, end]`（D8） | `rewindow_from_raw --window-align grid --step-ms 10000`；在线 probe 窗口与之相同 |
| 窗口最少请求数 | 10（p95 守卫即标签守卫）（§6） | 完成请求 **≥ 20**，否则 `unlabeled`（min-n）；10 仍作为 p95 守卫 | `LabelDefinition.min_completed_requests` |
| 未服务请求 | 三类计入违规（§6） | 同；**只读三个计数列** | `slo_labels.row_unserved` |
| 无延迟样本的未服务窗口 | θ 计违规；δ 拟合中跳过（§6） | θ 计违规；δ 拟合中按 ratio = 2.0 计严重度（不跳过） | `tre_calibration.fit.fit_delta_margins` + `slo_labels.UNSERVED_MIN_RATIO` |
| probe / cell 判定 | 固定 500/75、5 s 滑窗（§6、§7） | 主标签（D6′）、D8 窗口——与拟合同一把尺子 | `adaptive_boundary.probe_verdict(label=...)`，`calibration_campaign.primary_label` |
| λ_wait | 3（§2） | **1**，另跑 λ 检查：只有最优 λ 的 BA 比 λ=1 高 ≥ 0.02 才换 | `dline_refit` wp 阶段 |
| w_p | 基线 0，三条件替换规则（§4） | 受约束 1-SE 规则（D3）：候选须同时满足①训练 BA 与 w_p=0 相差不超过 1 个 SE ②族间差 ≤ CI 半宽 ③族规则发布合并 θ；满足者中取最大 w_p | `scripts.dline_refit.d3_select` |
| EMA α | —（registry 值） | D4′：同窗 LOSO BA、误报 ≤ 5 %，1 SE 内取稳态健康 cell 上虚假 CRITICAL 最少者，再取较大 α | `scripts.alpha_fit`（稳态 cell 从 `cells.jsonl` 读） |
| θ 发布 | 单一 θ（§2） | 发布合并 θ（D5），族 θ 只作诊断 | `dline_refit` final 阶段 |
| 停止规则 | — | CI 半宽 ≤ 15 %（D13），10 % 作附录目标 | `adaptive_boundary.stop_rule` |
| 边界 | 修复后的三态判定（§7.3） | 同，另加向外扩展阶段、ρ* 状态标注、`--reprobe-shapes` 重测（D14）；静态网格（D11）仅在 ρ* 为 measured 时可用 | `adaptive_boundary`、`calibration_campaign` |

## 4. 第二轮数据怎么用

第二轮原始数据（每请求 JSONL、sidecar、guard、ledger）与标签定义无关，**全部可用**。

- 标准数据集按修订 2 重建：`python -m scripts.calibration_dataset <run>`。重建时对每个 attempt 做在线 CSV 一致性检查：按第二轮自己的网格（30 s / 5 s）和它记录的标签重开窗，逐窗对比在线 CSV 的 `slo_label`，必须全部相同。结果见 `manifest.json → online_parity` 与 `docs/DATASET.md`。
- D 线拟合：fit_plan 的 `rewindow`（`--ledger cells.jsonl --only-split train|holdout`）→ `alpha`（D4′）→ `dline`（`scripts.dline_refit` alpha / wp / final × primary / fixed，再 summary）。
- 预注册规则对照：`calibration_decision --rule preregistered`。它自身不变：λ=3/10、三条件规则、固定 500/75 且无 min-n。要复现预注册的窗口（30 s / 5 s），先用 `--step-ms 5000 --window-align none` 重建数据集。

第二轮在线阶段（边界搜索、阶梯判定、补点决策）是在旧尺子下做出的。在新尺子下，某些 probe 的判定会不同，这些差异会列在数据集的 `discrepancies` 里。它影响的是"当时把 cell 放在哪个 ρ"，不影响 cell 本身数据的有效性：标签来自实测，不来自 ρ。
