# 标定标准数据集（`<run>/dataset/`）

本文件描述 `scripts.calibration_dataset` 产出的标准数据集：每张表每一列的含义、单位、来源和计算方式。仓库里的权威副本在 `tre/docs/DATASET.md`，每次生成数据集时会复制一份到 `<run>/dataset/DATASET.md`。

## 1. 为什么有这个格式

2026-09-21 那轮标定暴露了两个问题：边界搜索把"窗口不够、无法判断"的 probe 记成了"健康"；边界搜索判违规用的是**服务端** vLLM 直方图的 p95 TPOT，拟合用的是**客户端**每请求口径——两把尺子。修复后：

- 全链路只有一个窗口标签函数 `tre_common.slo_labels.window_slo_label`，只有一条开窗 + 打标路径 `scripts.rewindow_from_raw.label_cell`。边界搜索（在线）、拟合用的 re-window（离线）、本数据集都走它。
- 任何表里的延迟列都在列名里写明来源（`_client_ms` / `_server_ms`）和单位。

## 2. 目录布局

一个 run 一个目录。run 目录要么是一次 campaign 的 `--out-dir`（里面有 `plan.json`），要么是多个 campaign（每个模型一个）的父目录；后者生成**跨模型合并**的一份数据集。

```
<run>/
  <model>/                      # 每个模型一次 campaign（run_calibration.sh 的 OUT_ROOT/<model>）
    plan.json                   # 计划 + provenance（代码 commit、registry 哈希、标签定义）
    fit_plan.json               # 拟合用 re-window / refit 命令
    campaign_status.json        # campaign 结束状态（complete / stopped / failed / interrupted）
    capacity/ boundary/ schedules/ prompts/
    raw/<stem>/                 # 每个 attempt 一个目录（原始数据，只读）
      <cell_id>.jsonl           # 每请求一行（作废 attempt 会被改名为 .jsonl.void）
      <cell_id>.instant.jsonl   # 1 Hz 队列 sidecar（带 on_live_grid 标记）
      <cell_id>.failures.jsonl  # 失败请求的完整分类证据
      <cell_id>.guard.json      # cell 守卫判定（void 原因、start_ms/end_ms、goodput…）
      <cell_id>.rps.csv         # 计划 vs 实际到达率
    <stem>.csv                  # 在线窗口 CSV（r3_grid 写的，列同 windows.csv 的窗口列）
    dataset/                    # 本模型自己的标准数据集（campaign 结束时自动生成）
  dataset/                      # 跨模型合并的标准数据集（最后一个结束的 campaign 自动生成）
    manifest.json
    windows.csv
    requests.csv
    cells.csv
    DATASET.md
```

`<stem>` 的命名：计划内的 cell 是 `<model>_<shape>_<primitive>`（重跑为 `..._a2`）；边界搜索 probe 是 `<model>_<shape>_<shape>_hold<code>_a<attempt>`，`code = 1000 + round(100·rho)`。

**第二轮（预注册 ladder 设计，`plan.json` 里 `"design": "ladder"`）**的布局不同，数据集按它自己的台账读取，不再解析目录名：

```
<model>/
  run_manifest.json   # 发车前写一次、只读：预注册全部固定参数、regime 分组与 ρ 先验（全文 + sha256）、
                      # 设计种子、预注册文件 commit、静态计划（每个 cell 的 id 与种子）
  plan.json           # 含 run_manifest_sha256（发车时的哈希；数据集会复核文件是否被改过）
  cells.jsonl         # 台账：每个驱动过的 attempt 一行（role/stage/rho/rho_factor/replicate/种子/
                      # 排空记录/possibly_contaminated/判定/文件路径）
  design_result.json  # 各 shape 的 ρ* 锚点、补点决策、哨兵漂移、疑似污染 cell 列表
  boundary/<model>_<shape>.json  schedules/<model>/<stem>.json  prompts/<stem>/
  raw/<stem>/ ...     # 同上
```

此时 `<stem>` = `<model>_<shape>_<role>_c<code>_a<attempt>`，`cell_id` = `i<in>_o<out>_c<code>`，`code = 1000000 + 模型序号×100000 + 序号`（模型序号按 dsqwen-7b / dsllama-8b / dsqwen-14b）：**每个 cell 全局唯一**，同 shape 同 ρ 的两个重复也不同；每个 cell 还有自己的到达种子和 prompt key（`run_manifest.json → seed_derivation`）。

**只读保证**：转换工具只读 run 目录，唯一写入的是它自己拥有的 `dataset/`（先写到同级临时目录 `.dataset.building`，完成后整体换入）。

生成 / 重建：

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre/deploy
PYTHONPATH=../common:.:../controller:../service-manager:../calibration:../replayer:../ui \
  python3 -m scripts.calibration_dataset <run_dir>
```

读取：

```python
import pandas as pd
w = pd.read_csv("dataset/windows.csv")
train = w[(w.split == "train") & (w.slo_label != "unlabeled")]
```

## 3. 标签定义（所有表共用）

延迟口径 = **客户端每请求**（与 E1 计分 V_req 同口径）。

| 量 | 定义 |
|---|---|
| TTFT | 第一个流式 token 到达时刻 − 请求真正上线（写 socket）时刻，ms |
| TPOT | `(e2e_ms − ttft_ms) / (completion_tokens − 1)`，单个请求的平均 token 间隔，ms；completion_tokens ≤ 1 时为空 |
| 窗口 p95 | 窗口内**完成**（`done_ts_ms ∈ [start, end)`）且**成功**（outcome = ok）的请求，精确样本上的 `histogram_percentile(…, 0.95, bucket_upper)`；样本数 < `min_latency_samples`（=10，与线上 N1 守卫一致）时为空，不是 0 |
| 未服务请求 | **发送时刻**（`send_ts_ms`）落在窗口内、结局为 model_error / proxy_transient / client_timeout 的请求，分三列计数 |

窗口标签 `slo_label`，按顺序判定：

1. `violated`：窗口内有任何未服务请求（三类任一）；
2. `unlabeled`：SLO 需要的某个客户端 p95 为空（完成数不足）——**不是健康**，拟合与 probe 判定都不把它算作任何一边；
3. `violated`：任一客户端 p95 超过其 SLO；
4. `healthy`：其余。

`slo_violated` 是它的布尔视图：violated → True，healthy → False，unlabeled → 空。两列由同一个函数（`slo_labels.apply_label`）同时写入。SLO 值记录在 `manifest.json → label.slo_ms`（本轮 TTFT p95 500 ms，TPOT p95 75 ms）。

`shed`（网关准入溢出）不进窗口标签：它由 shed 策略处理——标定 cell 一旦出现即整格作废（void）。

## 4. 开窗

- 窗口 = 控制器实际消费的窗口：**30 s 宽、5 s 步长的滑动窗口**（`TRE_METRICS_WINDOW_MODE=sliding`，`TRE_METRICS_WINDOW_MS=30000`，`TRE_METRICS_REFRESH_INTERVAL_SECONDS=5`），参数记录在 `manifest.json → windowing`。
- 边界：每个 attempt 自己 guard 里的 `[start_ms, end_ms]`，从 `start_ms` 起每 5 s 一个窗口，直到窗口尾超过 `end_ms`。
- 队列列（`avg_*`、`queue_control`）只用 sidecar 中 `on_live_grid = true` 的样本（网关 10 s 桶的那一点），除数为 10 s——即控制器看到的信号。
- 在截断（admission overflow truncation）之后开始的窗口被删除。
- 相邻滑动窗口重叠 5/6：**统计"有多少独立证据"时只能数互不重叠的窗口**（见 `cells.csv → independent_windows`）。

## 5. `windows.csv`

一行一个窗口；三个模型、所有**非 void** attempt 合在一张表。列顺序固定如下。

| 列 | 单位 | 含义 / 来源 |
|---|---|---|
| `model` | – | 模型名 |
| `shape` | – | token 形状（S1–S5、T8、T9、M；M 为混合形状，held-out） |
| `primitive` | – | `steps` / `ramp` / `bursts` / `hold`（边界搜索 probe） |
| `stage` | – | probe 所处阶段 `coarse` / `bisect` / `dwell`；第二轮另有 `ladder` / `adaptive` / `sentinel`（分析按它切训练集，见 `calibration_design.TRAINING_HOLD_STAGES`）；ramp 等为空 |
| `rho` | 无量纲 | hold cell 的相对负载（相对 steps 测得或先验容量；第二轮相对 `rho_priors.json` 的 C_s）；ramp 为峰值；其余为空 |
| `cell_id` | – | `i<输入tokens>_o<输出tokens>_c<负载码>` |
| `attempt` | – | 第几次驱动（1 起）；void 或 inconclusive 后重跑会递增 |
| `split` | – | `train` / `holdout`（held-out 形状 M，绝不进拟合）；第二轮按预注册 §5：ladder + adaptive（M 除外）为 `train`，M 的全部 cell 与所有 ramp 为 `holdout`，训练 shape 的 probe 与哨兵为 `auxiliary` |
| `cell_status` | – | `valid` / `inconclusive`（void 的 attempt 不在本表） |
| `role` | – | 仅第二轮：`boundary`（阶段 0 probe）/ `ladder` / `adaptive`（阶段 3 补点）/ `ramp` / `sentinel`；第一轮为空 |
| `rho_factor` | 无量纲 | 仅第二轮：相对该 shape 实测 ρ* 的负载倍数（ladder 0.70–1.30；ramp 记峰值 1.4）；probe 与哨兵为空 |
| `replicate` | – | 仅第二轮：同 (shape, rho_factor) 的第几个重复（1 起）；哨兵为第几次 |
| `possibly_contaminated` | – | 仅第二轮：True = 这个 cell 开始前引擎在 90 s 内没排空（前一个 cell 的积压可能还在） |
| `in_warmup` | – | 仅第二轮：True = 窗口起点早于 cell `start_ms` + 60 s（hold cell 的队列建立期，预注册 §7.1 要求弃用；数据保留，由分析过滤——判定与 `calibration_decision.build_cells` 相同）。ramp 恒为 False；第一轮为空 |
| `scenario_id` | – | 同 `cell_id`（窗口 CSV 原有列） |
| `scenario_family` | – | `i<in>_o<out>` |
| `input_tokens` | tokens | cell 名义输入长度（M 为 0） |
| `output_tokens` | tokens | cell 名义输出长度（M 为 0） |
| `concurrency` | – | cell_id 里的负载码（历史列名，**不是**并发数） |
| `window_start_ms` | ms（epoch） | 窗口起点，含 |
| `window_end_ms` | ms（epoch） | 窗口终点，不含 |
| `prompt_tokens_total` | tokens | 窗口内完成请求的 usage prompt tokens 之和 |
| `generation_tokens_total` | tokens | 窗口内完成请求的 usage completion tokens 之和 |
| `avg_waiting` | 请求数 | 窗口内 live-grid 样本的 vLLM waiting 之和 ÷ (窗口/10 s) |
| `avg_running` | 请求数 | 同上，running |
| `avg_swapping` | 请求数 | 同上，swapped |
| `queue_control` | 请求数 | TRS 计算器输出的控制队列 Q_ctl（由 waiting/running/swapping 按 registry 的 trs 参数组合，见 `tre_controller.signals.trs`） |
| `trs` | 信号值 | TSS（旧名 TRS，加权吞吐 ÷ 控制队列，带时间常数 EMA），用**本次转换所用 registry** 的 trs 参数重算（见 manifest `registry_used_for_signal_columns`） |
| `completed_requests` | 请求数 | 窗口内完成且成功的请求数（= 延迟样本数） |
| `p95_ttft_client_ms` | ms | 客户端 TTFT 窗口 p95（见 §3） |
| `p95_tpot_client_ms` | ms | 客户端 TPOT 窗口 p95 |
| `p95_e2e_client_ms` | ms | 客户端端到端延迟窗口 p95 |
| `model_errors` | 请求数 | 窗口内发送、引擎失败的请求 |
| `proxy_transient_errors` | 请求数 | 窗口内发送、连接被断（如 `reset reason: connection termination`）的请求 |
| `client_timeouts` | 请求数 | 窗口内发送、客户端超时（≥30 s）放弃的请求 |
| `slo_label` | – | `violated` / `healthy` / `unlabeled`（§3） |
| `slo_violated` | – | True / False / 空（§3） |
| `p95_ttft_server_ms` | ms | **服务端** vLLM 直方图 p95（redis 读回，分桶上界）。仅诊断，任何标签/拟合都不读它；由数据集转换重算时为空（只有在线 CSV 里有） |
| `p95_tpot_server_ms` | ms | 同上，TPOT（按 token 间隔计，含单 token 卡顿） |
| `p95_e2e_server_ms` | ms | 同上，端到端 |

## 6. `requests.csv`

一行一个请求；**所有 attempt（包括 void）**。这是最原始的证据：换标签定义时可以从它重算一切。

| 列 | 单位 | 含义 / 来源 |
|---|---|---|
| `model` … `split` | – | 同 windows.csv |
| `cell_status` | – | `valid` / `inconclusive` / `void` —— 过滤 void 请用它 |
| `role` … `possibly_contaminated` | – | 同 windows.csv |
| `in_warmup` | – | 仅第二轮：发送时刻早于 cell `start_ms` + warm-up |
| `request_id` | – | 发送器请求 id（`<model>-<序号>`；第二轮为 `<prompt key>-<model>-<序号>`，prompt 由它做种子）。2026-09-23 之前的采集只有失败请求有 |
| `scheduled_send_ts_ms` | ms（epoch） | 调度表里这个请求**应该**发出的时刻 = `send_ts_ms − on_wire_delay_ms`。2026-09-23 之前的采集为空 |
| `send_ts_ms` | ms（epoch） | 请求**真正上线**（写 socket 前一刻）的时刻 |
| `first_token_ts_ms` | ms（epoch） | 第一个流式 token 到达时刻 |
| `done_ts_ms` | ms（epoch） | 请求结束时刻（`send_ts_ms + e2e_ms`） |
| `on_wire_delay_ms` | ms | 计划时刻 → 真正上线的延迟（发压端自身的迟到）。旧采集为空 |
| `ttft_ms` | ms | 客户端 TTFT |
| `tpot_ms` | ms | 客户端 TPOT（§3） |
| `e2e_ms` | ms | 客户端端到端时长（失败请求为失败所用时长） |
| `input_tokens` | tokens | vLLM usage 的 prompt tokens（失败请求通常为空） |
| `output_tokens` | tokens | vLLM usage 的 completion tokens |
| `http_status` | – | HTTP 状态码；0 = 无响应（超时/传输失败） |
| `outcome` | – | `ok` / `shed`（网关准入溢出）/ `proxy_transient`（连接被断）/ `model_error`（引擎失败）/ `client_timeout`（客户端放弃），分类规则见 `openloop.classify_failure` |
| `proxy_reason` | – | 代理类失败的原文原因（如 `connection termination`、`overflow`） |
| `in_flight_at_send` | 请求数 | 该请求发出时发送端在途请求数。旧采集只有失败请求有 |
| `request_timeout_s` | s | 客户端超时设置（`max(30, output_tokens/4)`） |
| `target_pod` | – | 服务该请求的 pod（仅经 AIBrix 路由时有） |

## 7. `cells.csv`

一行一个 attempt（void、inconclusive、计划了但没跑的 `missing` 都登记）。

| 列 | 单位 | 含义 |
|---|---|---|
| `model` … `split` | – | 同上 |
| `status` | – | `valid` / `void` / `inconclusive` / `missing` |
| `void_reasons` | – | 守卫作废原因，`; ` 分隔 |
| `probe_verdict` | – | probe 在**当前规则**下的判定：`violated` / `healthy` / `inconclusive` / `void`（`adaptive_boundary.probe_verdict`，在 windows.csv 的窗口上算） |
| `probe_verdict_recorded` | – | 边界搜索当时**实际采用**的判定（来自 `boundary/*.json`）。两者不同 = 当时的决策在新规则下不成立 |
| `windows` | 窗口数 | 本 attempt 的窗口行数（滑动，重叠） |
| `labeled_windows` | 窗口数 | 其中非 unlabeled 的 |
| `independent_windows` | 窗口数 | 已打标窗口中互不重叠的最大集合大小——"够不够判断"只看它（≥ 3 才能下结论） |
| `violating_windows` | 窗口数 | violated 的窗口行数 |
| `start_ms` / `end_ms` | ms（epoch） | 驱动起止（guard） |
| `planned_duration_s` | s | 调度计划时长 |
| `capacity_rps` | 请求/s | 该形状采用的容量（probe：steps 测量或先验；ramp：重生成时所用） |
| `offered_rps` | 请求/s | probe 的计划到达率 = rho × capacity_rps |
| `requests` | 请求数 | 实际发出的请求 |
| `requests_ok` / `_shed` / `_model_error` / `_proxy_transient` / `_client_timeout` | 请求数 | 按 outcome 计数 |
| `goodput` | 比例 | 守卫记录的 goodput = (已服务且满足 SLO) ÷ 发出 |
| `raw_path` / `guard_path` / `online_csv_path` / `schedule_path` | – | 相对 run 根目录的路径 |
| `role` / `rho_factor` / `replicate` | – | 仅第二轮，同 windows.csv |
| `warmup_s` | s | 仅第二轮：该 cell 的 warm-up（hold 60，ramp 0） |
| `arrival_seed` / `prompt_key` | – | 仅第二轮：该 cell 自己的到达种子与 prompt key |
| `possibly_contaminated` / `drained_before` / `drain_waited_s` | – | 仅第二轮：开始前的排空结果与等待时长 |
| `backlog_stopped` | – | 仅第二轮：probe 因客户端在途数达到上限（1024）被提前停发；这样的 probe 判为 violated |

第二轮的 `probe_verdict` 对每个 hold cell（probe、ladder、adaptive、哨兵）都给出，在 **warm-up 之后**的窗口上计算；只有 probe 会因证据不足而 `status = inconclusive`，其余 cell 证据不足仍是 `valid`。

## 8. `manifest.json`

| 键 | 内容 |
|---|---|
| `format_revision` | 本格式的修订号（整数） |
| `builder` | 转换工具的代码 commit 和工作树是否有改动 |
| `campaigns[]` | 每个 campaign 的采集 provenance：代码 commit、registry 路径与 sha256（2026-09-23 之前的 run 没记录，为 null）、`campaign_status.json`；第二轮另有 `run_manifest`（当前 sha256、发车时 sha256、`unchanged_since_start`）与 `design_result` |
| `registry_used_for_signal_columns` | 重算 `trs`/`queue_control` 所用的 registry 及其 sha256 |
| `label` | 标签定义全文（口径 = 客户端每请求、SLO 值、N1、规则、实现函数） |
| `windowing` | 窗口宽度 / 步长 / 类型 / 边界 / 队列来源 |
| `probe_rule` | probe 判定规则（最少独立窗口数、违规窗口比例） |
| `tables` | 每张表的行数与列 |
| `cells[]` | 每个 attempt：标识、`status`、`void_reasons`、两种 probe 判定、`files`（该 attempt 所有文件的相对路径） |
| `boundary_searches[]` | 每个 (model, shape) 的边界搜索原始记录 + 每个 probe 的 `verdict_recorded` 与 `verdict_current_rule` |
| `discrepancies[]` | 转换时发现的所有对不上的地方（原样列出，不做修补） |
