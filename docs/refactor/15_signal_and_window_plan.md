# TRE 信号新鲜度 / 标定公平性 / 实验工具 专项计划（Signal & Tooling Plan v1）

> 文档编号：`docs/refactor/15_signal_and_window_plan.md`
> 写作：2026-07-06 架构师第四轮审计后（源自 QA.md 的 Q1–Q5 深挖）
> 性质：这是一份**独立专项计划**，与 `14_endgame_plan.md` 并行/交织执行。它解决的是
> "指标滞后、标定范围、消融公平性、数据可重拟、实验工具"这五类**系统质量问题**，不是
> N4b/N5 的功能推进。
>
> **给执行模型的重要说明（先读这段）**：
> - 本文档假设你**不理解**背后的控制理论意图，所有步骤都写成"改哪个文件、加什么函数、
>   RED 测什么、GREEN 判据、真机验收看什么数字"。**遇到任何一步你觉得"大概是这个意思"
>   的地方，停下来记 Blocked，不要自行发挥。**
> - 本文档里 **S1 是最高优先级**（TSS 窗口滞后），因为它直接影响控制器的反应速度，也是
>   后面 S2/S4 的前置。**但 S1 有一个致命耦合：改了计算窗口，`theta_m` 和 `ema_alpha`
>   这两个继承值就全部失效**（原因见 S1.0）。所以 S1 必须和"在新窗口上重拟 theta / 重定
>   EMA 时间常数"一起做，**不允许只改窗口不重拟**。这条是整份文档最容易被做错的地方。
> - 执行顺序建议见第 7 章。S1 应在 `14_endgame_plan.md` 的 N5-R3（重标定）**之前**完成，
>   因为 R3 必须在"最终确定的控制窗口"上采集和拟合，否则 R3 白跑。

---

## 0. 术语与当前事实基线（照抄，勿凭记忆）

以下都是 2026-07-06 在 76 `/data/nfs_shared_data/xxy/aibrix` 实查的代码事实，写步骤时以此为准：

- **TRS（论文里叫 TSS，代码里叫 TRS，是同一个东西）** 公式在
  `tre/controller/tre_controller/signals/trs.py`：
  `Y = prompt·w_p·(1-kv) + generation·w_d`；`Q_ctl = max(qmin, running+swapping+lambda_wait·waiting)`；
  `TRS_raw = (Y/Q_ctl)·(assigned_replicas/routable_pods)`；`TRS = EMA(TRS_raw)`；
  `Z_m = TRS/theta_m`。分档：`Z_m<tau_crit`→CRITICAL，`<tau_low(=1.0)`→LOW，
  `<=tau_high`→HEALTHY，`>tau_high`→HIGH（`planning/classify.py`）。
- **计算窗口**：`config.py` 的 `metrics_window_ms=60_000`（60s），由环境变量
  `TRE_METRICS_WINDOW_MS` 控制。
- **窗口取法**：`loops/metrics_task.py:_last_complete_window`——tumbling（不重叠）、
  对齐到 epoch、**只取上一个已完整结束的 60s 块**。这是滞后的主因。
- **刷新装配**：`app.py:build_controller_task_specs` 里，**只有一个** `metrics` 任务，
  循环体 `metrics_task`（metrics_task.py）每 `monitor_interval_s`（默认 20s）刷新一次，
  把快照写进**唯一的** `snapshot_box`。
- **三个消费环都读同一个 box**：`rescue_task`（5s）、`fairness_task`（10s）、
  `safescale_task` 都是 `snapshot_box.get()` 读同一份，**不自己重算窗口**。
- **EMA 推进时机**：`trs.py` 的 `_update_ema` 每处理一个新快照推进一步。因为快照的窗口
  每 60s 才换一次（tumbling + `metrics_store` 进程内 `_window_cache` 缓存），所以
  **EMA 实际每 ~60s 才推进一步**。`ema_alpha` 现值（如 dsqwen-7b=0.2485）就是在"每 60s
  一步"这个前提下调的。
  > **【2026-07-06 更正｜执行模型实查 + 架构师 Fable5 复核，见 ADR-0011】** 上面这段关于
  > EMA 的描述**对 live 路径是错的**。live 路径每 tick 都**新建** `TRSComputer`
  > （`tick.py:_model_contexts`、`safescale_task.py:_observation_from_metrics`），且**任何
  > 地方都不 restore EMA 状态**（`state_store` 只存 SafeScale 探针；`app.py` 只 `safescale.restore()`）。
  > 所以 `_update_ema` 每次都见 `_trs_ema is None` → 返回 raw：**live `TRS == TRS_raw` 恒成立，
  > `ema_alpha` 从来对 live 零作用**。生产里唯一的平滑就是 60s tumbling 窗口本身。
  > 因此**耦合 B 的正确表述**不是"提速刷新会过度平滑 4×"（本来就没有可被过度平滑的 EMA），
  > 而是"S1.2 缩短滑动窗口会去掉窗口的隐式平滑，所以必须**先建一个真正的、按墙钟 τ、跨 tick
  > 持久**的 EMA"——这正是 S1.3 现在做的（Option A-minimal：共享 per-model 内存态 EMA，无 Redis
  > 持久，重启后 ~τ 内从 raw 收敛）。"从 alpha_old 反推 τ"仅作 τ 的设计起点，实际 τ 在 S1.2 真机验收冻结。
  > 另记两条连带事实：① `SaturationGuard`/gamma 在 live 路径同样是死代码（从不实例化，tick.py 用
  > `Q_ctl >= qsat` 直判）——是否 live 化是 S1.3 范围外的单独决定；② `r3_grid.py` 的 `trs` CSV 列
  > 是**逐格内 EMA 平滑过**的，而 live 是 raw——**R3/S1.4 必须让 r3_grid 复刻 live 的 EMA 语义
  > （τ + window_end 增量）**，否则 theta 是在一个控制器从未见过的信号上拟合的。
- **写入周期**：AIBrix 网关每 `RequestTraceWriteInterval=10s`（`pkg/cache/trace.go:69`）
  把 vLLM 指标写进 Redis，TTL 10 分钟。
- **拟合流水线**：`tre/calibration/`，CLI（`cli.py`）**只拟合 `theta_m`**，其余
  w_p/lambda_wait/qmin 是入参回显、ema/tau 不经手（详见 QA.md 的 A1）。
  `signals.py:grid_search_parameters` 实现了 w_p/lambda_wait/qmin 的网格搜索但 CLI 没调用。
- **grid 采集脚本**：`tre/deploy/scripts/r3_grid.py`，有 cell 级 checkpoint（可断点续跑），
  但输出的是**窗口级预聚合 CSV**，原始逐请求数据不落盘（详见 QA.md 的 A4）。
- **UI**：`tre/ui/tre_ui/app.py`，113 行、全 GET、纯只读（详见 QA.md 的 A5）。

---

## 1. S1：TSS 计算窗口滞后修复【最高优先级】

### 1.0 先理解问题与耦合（这段必须读懂再动手）

**先纠正架构认知（架构师 2026-07-06 澄清）**：控制器只有**两个决策环**——rescue（救援，
每 5s）与 fairness（公平，每 10s）。它们是**两种决策"模式"，不是两种指标口径**：
**它们必须使用同一个 TSS/Z_m 计算窗口**（本来就读同一个 `snapshot_box`）。**不要给它们
分成不同长度的窗口**（我方案早期版本设想过"快慢双窗口"，已作废，别照那个做）。

**问题（三层滞后叠加）**：
1. **窗口太长**：TSS 用 60s 窗口（`metrics_window_ms=60_000`）。
2. **窗口不动（tumbling）**：`_last_complete_window` 取"上一个已完整结束的 60s 块"，块末尾
   最新数据点已是 60–120s 前，且每 60s 才换一格。
3. **刷新更慢**：快照由 `metrics_task` 每 `monitor_interval_s`=**20s** 才刷新一次
   （metrics_task.py:78），**比 rescue(5s)、fairness(10s) 两个决策环都慢**。所以 5s 的
   rescue 在 20s 内反复读同一份、这份还是 60s tumbling 算出来的——三层滞后叠一起。

**目标**：**一个共享的滑动窗口 W（比 60s 短），且刷新频率 ≤ 最快决策环（5s）**，让 rescue
和 fairness 每次触发读到的都是同一份、新鲜的 TSS/Z_m。把最坏滞后从 60–120s 降到
~W + 一次刷新 + 写入粒度。

**新鲜 vs 稳定怎么兼顾（关键设计思想）**：不是靠"给 fairness 一个长窗口"来求稳，而是
**窗口只负责新鲜（短 + 滑动 + 快刷新），平滑/稳定交给时间常数 EMA（τ）**（见 S1.3）。
两个环读同一个 EMA 后的 TSS：rescue 直接用它救急，fairness 的稳定性由 τ 提供的时间平滑
保证。**一个窗口、一个 theta、一个 EMA，rescue 和 fairness 共用。** 这是本次的目标架构。

**两个致命耦合（务必理解，否则做出错误结果）**：
- **耦合 A：theta_m 依赖窗口。** `theta_m` 是"在 60s 窗口上算出的 TRS 的 SLO 违约边界值"。
  把窗口改短，同样负载下 TRS 的数值分布会变（TRS 分子是窗口内 token 求和、正比于窗口长度；
  分母 Q_ctl 是时间平均、与窗口无关；所以 TRS 整体随窗口线性变），旧 `theta_m=738` 就不再
  对应 SLO 边界了。**结论：改了窗口 W，就必须在同一个 W 上重新用数据拟合 theta_m。**
  **架构师已定：不采纳任何"走捷径"的替代**——不做按窗口比例解析缩放 theta、也不把 TRS
  归一化成"每秒吞吐"来消除耦合（那会改 TRS 量纲，论文量纲保持"每窗口 token"不变）。
  **唯一正确做法就是在最终窗口上老老实实重拟**（我们有时间，重拟成本可接受）。别自作聪明缩放。
- **耦合 B：ema_alpha 依赖推进频率。** 现在 EMA 每 20s 一步（=刷新周期）、`ema_alpha=0.2485`。
  刷新提到每 5s 一步后，同样的 alpha 会让平滑强度在**墙钟时间**上暴增约 4 倍，信号被过度
  平滑反而更迟钝。**结论：EMA 必须从"每步固定 alpha"改成"按墙钟时间常数 τ 计算的时间衰减
  EMA"**（见 S1.3），这样平滑强度只由 τ 决定、与刷新频率解耦。

因此 S1 分四步（严格按序）：
S1.3 先把 EMA 改成时间常数式（独立可验证，是提高刷新率的前提）→ S1.1 把窗口改滑动 →
S1.2 缩短这个**单一共享**窗口并把刷新提到 5s → S1.4 在冻结后的窗口上重拟**一个** theta。

### 1.1 滑动窗口（去掉 tumbling 的 60–120s 滞后）

**改动文件**：`tre/controller/tre_controller/loops/metrics_task.py`

1. 新增窗口函数 `_sliding_window(now_ms, window_ms) -> (start, end)`：
   `end = now_ms`；`start = max(0, now_ms - window_ms)`。**不做 epoch 对齐、不取上一个完整块。**
2. `refresh_metrics_once` 增加参数 `window_mode: str`（`"tumbling"` | `"sliding"`，默认
   `"tumbling"` 保持旧行为不破坏现有测试）。当 `sliding` 时用 `_sliding_window`，否则用
   现有 `_last_complete_window`。
3. `config.py` 增加 `metrics_window_mode`（env `TRE_METRICS_WINDOW_MODE`，默认 `sliding`
   —— 目标状态设为 sliding；为不破坏 golden/离线对拍，`ControllerConfig.from_env` 默认给
   `sliding`，而所有**已有测试**里显式依赖 tumbling 的，构造时传 `tumbling`）。合法值校验
   照 `PERCENTILE_MODES` 的写法加一个集合。
4. `metrics_task` 异步循环把 `window_mode=cfg.metrics_window_mode` 透传下去。

**关键连带修改（否则会内存泄漏）**：`tre/controller/tre_controller/store/metrics_store.py`
的 `_window_cache`（`dict[(schema,model,start,end)]`）在 tumbling 下 key 有限，但 sliding
下每次 `end=now` 都不同 → 缓存无限增长。**必须处理**：
- 最简单正确的做法：sliding 模式下**不写这个缓存**（`read_model_window` 增加一个
  `use_cache: bool` 参数，sliding 路径传 False）。RED 测试：连续用 100 个不同的
  `[start,end]` 调用，断言 `_window_cache` 大小不增长。

**RED/GREEN**：
- RED：`test_metrics_task.py` 加 `_sliding_window(now=125_000, window=60_000)` 应返回
  `(65_000, 125_000)`（对比 tumbling 返回 `(60_000, 120_000)`）。
- RED：`refresh_metrics_once(..., window_mode="sliding")` 用一个 fake store 断言它请求的是
  `[now-W, now]`。
- GREEN 后 `cd tre && make check` 全绿。

**真机验收（S1.1 单独）**：滚动 controller 镜像（env 设 `TRE_METRICS_WINDOW_MODE=sliding`，
`TRE_METRICS_WINDOW_MS` 暂**保持 60000 不动**），给一个模型加负载，从 decision 快照里
观察：Z_m 现在**每次刷新都在变**（不再 60s 一跳）。把"改造前 vs 改造后"的 Z_m 时间序列
各存一份 JSON 到 `docs/refactor/p11_evidence/s1_sliding_<date>/`。

### 1.2 缩短单一共享窗口 + 把刷新提到 5s（把滞后真正降下来）

**目标架构（记 ADR D12）**：**单一共享 TSS 窗口**，rescue 和 fairness 共用。做两件事：
（a）把刷新周期从 20s 降到 ≤ 最快决策环（5s）；（b）把窗口 W 从 60s 缩短。**不拆双窗口。**

**改动**：
1. `config.py`：新增 `metrics_refresh_interval_s`（env `TRE_METRICS_REFRESH_INTERVAL_SECONDS`，
   默认 **5.0**），**把 metrics 刷新周期从 `monitor_interval_s` 解耦出来**。把
   `metrics_window_ms` 的默认从 60000 改小——但**默认先设 `30000`（30s，理由见 N5：取写入周期 10s 的 3 倍，稳定覆盖 3 个 histogram 点，token delta 与 p95 都有足够样本）**，最终值在真机
   验收后冻结。
2. `metrics_task.py`：`metrics_task` 的 `await asyncio.sleep(...)` 从 `cfg.monitor_interval_s`
   改成 `cfg.metrics_refresh_interval_s`。（`MetricsTaskConfig` Protocol 增加该字段。）
3. **不改** `app.py` 的 box 结构——**仍然只有一个 `snapshot_box`**，rescue/fairness/safescale
   继续都读它。**不新增 fast_snapshot_box、不拆 metrics_slow/metrics_fast。**（这条是对早期
   草案的明确否定：不要双快照。）

**RED/GREEN**：
- RED：`metrics_task` 用 fake clock 断言按 `metrics_refresh_interval_s`(=5s) 循环，而不是
  `monitor_interval_s`(=20s)。
- RED：config 默认 `metrics_refresh_interval_s==5.0`、`metrics_window_ms==30000`、
  `metrics_window_mode=="sliding"`。
- GREEN → `make check` 全绿。

**真机验收**：加阶跃负载，测"负载上升 → rescue 决策里 Z_m 反映出来"的延迟。改造前 60–120s，
改造后应 ≤ W + 刷新 + 写入粒度（30s 窗口时约 ≤ 35s；若想更快就在验收里把 W 试到 20s，
同时看 S1.3 之后的信号噪声/AUROC 是否还能接受——见 N1）。取样多次记 P95，把"最终冻结的
W 值 + 依据"写进 WORKLOG 和 `06_calibration_design.md`。证据存 `s1_shortwindow_<date>/`。
**这个冻结的 W 就是 S1.4 / R3 拟合 theta 要用的唯一窗口。**

### 1.3 EMA 改为墙钟时间常数式（耦合 B，必须先于 S1.2 完成）

**问题**：见 S1.0 耦合 B。现在 `_update_ema`（trs.py:131）是
`ema = alpha*prev + (1-alpha)*raw`，每"步"固定 alpha。步频从 60s 变 5s 会破坏平滑强度。

**改造**：把 EMA 改成"按两次更新之间的墙钟间隔 Δt 计算衰减"：
- 新公式：`decay = exp(-Δt_ms / tau_ms)`；`ema = decay*prev + (1-decay)*raw`。
- 其中 `tau_ms` 是墙钟时间常数（env `TRE_TRS_EMA_TAU_MS`，或每模型 registry 字段
  `ema_tau_ms`）。`Δt_ms` = 本次快照 `window_end_ms` − 上次的 `window_end_ms`。
- **τ 是"平滑时间尺度"，物理含义是"EMA 大约反映过去多久的趋势"**。选一个墙钟 τ（起点见
  下）后，无论刷新是 5s 还是 20s，单位墙钟时间内的平滑量都一致——这就是"解耦刷新频率"。
- **从旧 alpha 反推 tau 做等价初值**：旧的 EMA 每一步（=当时的刷新推进）用 alpha_old=0.2485。
  用 `tau_ms = -step_ms / ln(alpha_old)` 反推：按旧刷新步长 ~20s 算，
  `-20000/ln(0.2485) ≈ 14_000ms`；若你认为旧的有效推进更接近窗口换格 60s，则 ≈43_000ms。
  **这两个值差不小，说明旧 alpha 的"墙钟含义"本来就模糊**——所以 τ 不要机械照搬，**把它当
  作一个要在 S1.2 真机验收里调定的设计参数**：先按 `ema_tau_ms=20000`（20s）起步填进
  registry，真机看 rescue 既不抖也不迟钝就冻结；抖就加大 τ，迟钝就减小 τ。把最终 τ 及依据
  记进 `06_calibration_design.md`。
- `TRSComputer` 需要在 `compute` 时知道"本次与上次更新的墙钟间隔 Δt_ms"（= 两次快照
  `window_end_ms` 之差，现在≈刷新周期 5s）。改 `compute` 签名接受 `now_ms`（或 `dt_ms`）。
  首个样本 `prev is None` 时直接取 raw（同现状）。

**RED/GREEN**：
- RED：`test_trs.py` 构造两次调用间隔 Δt=60s、tau=43s，断言 decay≈0.2485（复现旧行为）；
  再构造 Δt=5s、同 tau，断言 decay≈exp(-5/43)≈0.891（新样本权重≈0.109，即高频下每步动得少，
  但**单位墙钟时间内的总平滑量与 60s 一步一致**）。
- RED：`prev is None` 时返回 raw。
- GREEN → `make check` 全绿。

**注意**：registry 里保留 `ema_alpha` 字段兼容旧离线路径；新增 `ema_tau_ms`。若两者都在，
`TRSComputer` 优先用 `ema_tau_ms`。golden 对拍路径：若 golden 依赖旧的固定 alpha，则 golden
构造 `TRSComputer` 时显式传 `ema_tau_ms=None` 走旧 alpha 分支（保证冻结副本行为不漂移）。

### 1.4 在最终控制窗口上重拟 theta（耦合 A，接到 N5-R3）

**这一步把 S1 和 endgame 的 R3 绑定，务必执行，否则 Z_m 语义错误。**

**排序硬护栏（架构师定，不可违反）**：**R3 重标定不允许在 S1 的控制窗口 W 最终冻结之前
开跑。** 因为 theta（以及 S2/S3 的 w_p/qsat）都必须在"最终确定的那个窗口 W"上拟合；
若窗口还会变，R3 采集的数据和拟合出的 theta 全部作废、10h/模型 白跑。所以流程是：
先完成 S1.1–S1.3 并通过 S1.2 真机验收把**唯一的**窗口长度 W **冻结**，再开 R3；R3 里就在
这个 W 上拟合 theta。**不采纳任何缩放/换量纲的过渡捷径**（见 S1.0）。

> **2026-07-07 架构变更（ADR-0014）**：饱和段（saturation segment）概念已移除，扩缩容与 fairness 受者资格纯由 z_m 阈值分带决定。**上面“theta（以及 S2/S3 的 w_p/qsat）都必须在最终窗口 W 上拟合”中的 qsat 部分作废**：R3/S2/S3 不再拟合 `qsat/epsat/hsat`（三者已弃用；`epsat/hsat` 完全失效，`qsat` 仅作 queue_len 信号的 z_m 归一化常数保留）。R3 在冻结 W 上只拟合 `theta_m`（rescue/fairness 共用一个）。后续 R2/R3/R5 一律按无饱和段语义执行——别按本节旧文去拟合 qsat。详见 `DECISIONS.md` ADR-0014。

**只有一个窗口、一个 theta（rescue 和 fairness 共用，不要双 theta）**：
- R3 的 r3_grid 用冻结的 `--window-ms=W` 采集，对每个模型拟合**一个** `theta_m`，写回
  registry 的 `theta_m` 字段（覆盖旧继承值）。`classify` 里 rescue 和 fairness 用的是同一个
  theta（本来就该如此）。
- **percentile 口径仍要两套**（bucket_upper / interpolated）——这是 R4 消融要用的，与窗口
  无关，用 S4 原始日志离线双聚合即可（见 endgame 6.2）。别把"两套 percentile"和"双窗口"
  搞混：percentile 是两套、窗口只有一个。
- **验收**：在冻结的 W 上，`theta_m` 拟合报告的 AUROC 与 coverage 要达标（沿用 fit 的
  reliability gate）。若 W 太短导致信号打碎、AUROC 明显掉，说明 S1.2 的 W 选小了——回到
  S1.2 把 W 调大重新冻结，再来 R3。把这个实测结论记进 `docs/refactor/06_calibration_design.md`。

### 1.5 S1 gate

- [ ] S1.3 时间常数 EMA（τ 起步值填 registry，与刷新频率解耦）落地，`make check` 绿
- [ ] S1.1 滑动窗口 + 窗口缓存不泄漏，`make check` 绿，真机 Z_m 不再 60s 一跳
- [ ] S1.2 单一共享窗口缩短(默认 30s) + 刷新提到 5s（**不拆双窗口**），rescue/fairness
      共用同一份新鲜快照，真机滞后 P95 ≤ 30s（附改造前后对照 JSON），并**冻结最终 W**
- [ ] S1.4 在冻结的 W 上拟合**一个** theta_m（rescue/fairness 共用）随 R3 完成
- [ ] ADR D12（单一共享短窗口+5s快刷新+时间常数EMA；明确否决双窗口）记入 DECISIONS.md
- [ ] `05_paper_vs_impl.md` 补录"控制窗口为滑动、快慢双窗、EMA 为时间常数"这三条契约

---

## 2. S2：标定范围补齐（回应 Q1）

**目标**：不让论文被质疑"参数是旧 0.4.0 系统调好的直接拿来用"。

**做什么**：在 N5-R3 里，除了拟合 theta_m，**额外把 `grid_search_parameters`（已实现，
signals.py）跑一遍**，对 `w_p / lambda_wait / qmin` 做网格搜索，产出"重拟值 vs 继承值"对照。

**具体步骤**：
1. `r3_grid.py` 采集出的窗口 CSV 已含 token/queue 原料。写一个薄封装脚本
   `tre/deploy/scripts/refit_trs_params.py`：读同一批 CSV，构造 `SignalInputs` 序列，调
   `grid_search_parameters(w_p_candidates=[0.02,0.04,0.06,0.08,0.10],
   lambda_wait_candidates=[1.5,1.875,2.25,2.625,3.0], qmin_candidates=[1.0])`，
   输出每模型 best 参数 + objective/AUROC/spearman。
2. 对每个模型打印"继承值的得分 vs best 的得分"。
3. **判据与决策**：若 best 与继承值得分差 < 5%（AUROC 与 spearman 都是），则**保留继承值**
   并在 `06_calibration_design.md` 记"重拟验证：继承值仍在最优邻域，予以保留"；若差 > 5%，
   则用重拟值更新 registry 并记原因。
4. `w_d / ema_tau_ms / tau_crit / tau_high` 明确**不拟合**（理由见 QA.md A1：w_d 是 gauge、
   tau 是控制迟滞带、ema 是时间常数），但要在论文标定章节**列一张表**：哪些参数拟合、
   哪些固定、固定值来源、为什么固定。

**gate**：三模型的重拟对照表进 `13_experiments_log.md`（R3 条目下）。

---

## 3. S3：消融信号公平性（回应 Q2）

**目标**：让 R5 消融比较的是"信号质量"，不是"谁的阈值被拟合了"。

**当前不公平点**（sources.py）：`zm` 的 theta_m 是数据拟合的，但 `queue_len` 的 qsat=4.0
是手设常数、`kv_cache` 的 0.5 是硬编码。

**做什么**：
1. **给 queue_len 拟合 qsat**：把 `fit_theta_by_reliability` 泛化成"对任意信号列拟合归一化
   阈值"。R3 的 CSV 里增加一列 `queue_control`（= max(qmin, running+swapping+lambda_wait·
   waiting)，用 metrics_store 现成量算）。用和 theta_m 完全相同的口径（信号≥阈值的窗口
   子集 SLO 达标率≥目标）拟合出 `qsat_fit`，写回 registry 每模型的 `qsat`。
   - 注意方向：queue 信号 z_m = qsat/queue，是"越小越健康"反过来的，拟合时要对齐
     `_queue_signal` 的归一化方向（z_m 大=健康）。RED 测试覆盖方向正确性。
2. **latency 信号**：已用真实 SLO 归一化（`min(SLO/observed)`），天然公平，不动。但要
   确认三个 SLO（ttft/tpot/e2e）都用 registry 里的真值。
3. **kv_cache 信号的 0.5**：这是占位，**两个选择**（做之前记 Blocked 问架构师）：
   (a) 同样用数据拟合一个 kv 阈值；(b) 直接把 kv_cache 从消融 arm 里删掉（它本就不是一个
   合理的扩缩信号）。**我倾向 (b)**——kv 命中率不是拥塞信号，放进论文消融会显得凑数。
4. **R5 口径声明**：无论如何，在 `13_experiments_log.md` 的 R5 条目里写死一句：
   "所有被比较信号的归一化阈值均用同一 reliability 拟合流程在同一批 R3 数据上产出"，
   或（若你故意保留默认阈值基线）"消融为 TRS(拟合) vs 朴素默认阈值基线，口径见此"。

**gate**：qsat 拟合落地（RED/GREEN + registry 回填）；kv 处理决策记录；R5 口径声明成文。

---

## 4. S4：R3 grid 原始日志落盘（回应 Q4，让数据可事后重窗口重拟）

**目标**：R3 是最贵的一步（每模型 ~10h）。采集时若只存 60s 窗口 CSV，日后想换窗口
（如 S1 要的 20s）就得重跑 10h。让它顺手把**原始逐请求日志**也落盘，一次采集支撑任意
窗口离线重聚合。

**做什么**（改 `tre/deploy/scripts/r3_grid.py`）：
1. `drive_cell` 在发压时，对**每个请求**记录一行到本地盘 JSONL
   `<output>.raw/<cell_id>.jsonl`：`{send_ts_ms, recv_first_token_ts_ms, done_ts_ms,
   input_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, http_status, cell_id}`。
   （vLLM 的 OpenAI 接口能拿到 usage.prompt_tokens/completion_tokens；ttft 用流式首 token
   时间；拿不到的字段记 null 不要瞎填。）
2. 新增离线脚本 `tre/deploy/scripts/rewindow_from_raw.py`：读 `.raw/*.jsonl`，按传入的
   `--window-ms` 和 `--step-ms`（支持滑动）重新聚合成和现在一样列的窗口 CSV，供 fit CLI 用。
   聚合口径必须和 `metrics_store._aggregate_model` **完全一致**（token 求和、队列量的处理、
   p95 从延迟样本按 bucket_upper/interpolated 两种模式）——**复用或对拍 metrics_store 的
   聚合逻辑，不要另写一套**，否则重聚合的数和在线的对不上。RED 测试：给定一段 raw，
   `rewindow_from_raw` 按 60s 聚合的结果与 r3_grid 在线 60s CSV 对同一段数据一致（容差内）。
3. 磁盘：raw JSONL 写 76 本地盘（`/root/tre-experiments/r3_raw/`），不写 NFS。估算体积
   （每请求 ~200B，10h×若干 QPS），在脚本启动时打印预计占用，超过阈值告警。

**gate**：R3 开跑前 S4 完成；R3 采集同时产出窗口 CSV + 原始 JSONL；`rewindow_from_raw`
能从同一份 raw 产出 20s 和 60s 两套 CSV 且与在线聚合对拍通过。

---

## 5. S5：实验控制台（回应 Q5，独立阶段 F6，与 N5 并行）

**目标**：把当前 113 行只读 UI 扩成"做实验的控制台"。这是体量最大、最不在关键路径上的
一块，**先做后端控制面 API + 时序落库，前端最后做**。建议排在 F4 之后、与 N5 并行
（N5 期间最需要它盯实验）。

**分层落地（每层独立可验收，按序）**：

### 5.1 后端时序落库（先做，其它都依赖它）
- 现在决策快照是 `tre:v2:decision:latest` 单点。新增：controller 每次决策把
  `{ts, model, TRS, Z_m, state, raw_signal, awake, bound, window_ms}` 追加进 Redis
  Sorted Set `tre:v2:decision:hist:{model}`（score=ts_ms，保留最近 N 小时，带 TTL）。
- 复用已有 `DecisionSnapshotWriter`，加一个 append-to-zset 的写法。RED/GREEN。

### 5.2 后端控制面 API（在 controller 或 SM 上加，需鉴权/确认）
按"读→轻写→重操作"分批，每个 endpoint 都要 RED/GREEN + 幂等：
- 读：`GET /api/signal/history?model=&from=&to=` 读 5.1 的时序；`GET /api/redis/keys?prefix=`
  安全地列/取指定前缀 key（**白名单前缀**，禁止任意 key 读写）。
- 轻写：`PUT /api/params/{model}` 热改 registry 的 trs 参数（w_p/theta/tau/ema_tau…），
  **带校验**（范围、类型）+ 写审计日志 + 触发 controller 重载 registry。**危险操作，
  必须在 UI 上二次确认，且记谁改了什么。**
- 控制窗口：`PUT /api/window` 改 `metrics_window_ms/fast_window_ms/window_mode` 并热生效
  （依赖 S1 已把窗口做成可运行时切换——若 S1 只做了 env 级，则此 API 先返回"需重启"，
  不要假装热生效）。
- 触发拟合：`POST /api/calibrate/{model}` 后台跑 calibration CLI（对一份指定 CSV），
  返回 job id；`GET /api/calibrate/{job}` 查结果（AUROC/coverage/reject_reason）。
- 发压：`POST /api/trace/run` 用给定 trace 参数调 r3_grid/replayer 发压，`GET` 查实时进度。

### 5.3 前端面板（最后做）
- Z_m/TRS 时序图（读 5.1）、集群拓扑热力（GPU 真值驻留）、决策流/plan events 实时滚动、
  参数编辑表单（调 5.2 轻写，带确认）、trace 构造器 + 实时 SLO 曲线。
- 前端用轻量方案（单页 + 图表库内联，遵守无外网 CDN 的约束——本地打包）。

### 5.4 F6 gate
- [ ] 5.1 时序落库 + 5.2 只读 API 先上线（做实验立刻能用的最小闭环）
- [ ] 5.2 轻写/控制 API 带校验+审计+确认
- [ ] 5.3 前端面板
- [ ] ADR D13（实验控制台边界：白名单 key、参数热改审计）

---

## 6. 我额外发现、建议一并解决的新问题（架构师主动提出）

这些是我审计 S1–S5 时连带发现的、执行模型大概率不会自己意识到的问题：

- **N1（高）：短窗口下 p95 延迟样本太少。** 30s（或更短）窗口 + 低 QPS 时，某模型窗口内
  可能只有个位数请求，`ttft_p95/tpot_p95` 从直方图 bucket 估出来会很噪。**对策**：给 latency
  类信号加"最小样本数守卫"，样本不足时该窗口延迟量标 None（走 S1.3 的 EMA 保持 + F2 的
  stale hold），不要用少量样本的 p95 直接决策。RED/GREEN 覆盖"样本不足→None 而非 0"。
  这也是 S1.2 里 W 不宜取太小（默认 30s 而非 10–15s）的原因之一。
- **N2（高）：改窗口后 SafeScale 的窗口假设。** `SafeScaleConfig` 里有自己的
  `default_window_ms=60000 / min 15000 / max 300000`（config.py）。SafeScale 也读那**同一个**
  `snapshot_box`（S1.2 后是单一共享短窗口），但它内部另有一套自己的窗口概念。S1 改了共享
  窗口后，要确认二者不冲突。**做之前通读 `safescale_task.py` + `planning/safescale.py`，把
  两套窗口的关系理清并记进 05 文档；不清楚就记 Blocked。**
- **N3（中）：Z_m 分布随窗口变，tau 带宽可能要跟着调。** tau_crit/tau_high 是在 60s 窗口的
  Z_m 分布上调的迟滞带。换窗口后 Z_m 抖动幅度变化，可能需要重定 δ_crit/δ_high（否则短窗口
  下频繁穿越 tau 造成扩缩抖动/振荡）。**对策**：S1.2 真机验收时**额外统计 Z_m 的窗口间
  抖动标准差**，若明显增大，把"重定 tau 带宽或加去抖/滞回计数"作为 follow-up 记录。
- **N4（中）：控制动作的最小间隔 / 反抖。** 快环变灵敏后，扩容→缩容→再扩容的抖动风险上升。
  检查 planner/action_queue 是否有"同模型两次反向动作的最小冷却时间"；若无，加一个
  `TRE_ACTION_COOLDOWN_MS`。这是让快环"快而不抖"的必要护栏。
- **N5（中）：写入对齐边界的相位。** AIBrix 每 10s 对齐边界写（X:00.5s…）。20s 滑动窗口
  `[now-20s, now]` 里的文档数会在 1~2 个之间跳变，导致 token delta 抖动。**对策**：窗口长度
  取写入周期的整数倍（如 30s 稳定覆盖 3 个点），S1.2 里默认已定
  调成 **25000** 并说明理由。
- **N6（低）：离线 golden 对拍会不会被 S1/S3 破坏。** 每做完 S1.3/S3 的公式改动，**先跑一次
  现有 golden 对拍测试**，破了就把 golden 路径钉死到旧口径（构造参数而非 env），确保冻结
  副本永不漂移。这条在每个改公式的 slice 里都要顺手做。

---

## 7. 执行顺序与总 gate

**推荐顺序（与 14_endgame_plan.md 交织）**：
1. 先做 **S1.3（时间常数 EMA）** —— 纯离线、独立可验、是 S1 其余步骤的前提。
2. 做 **S1.1 + S1.2（滑动窗口 + 单一共享短窗+5s快刷新）** + **N1/N5**（短窗口的样本守卫与窗口长度选取）。
   —— 这一批把"指标滞后"问题落地，是本文档的核心目标。
3. **S4（原始日志落盘）** —— 必须在 N5-R3 开跑**之前**完成。
4. 进入 14_endgame 的 **N5-R3** 时，一并做 **S1.4（新窗口重拟 theta，双 theta）**、
   **S2（w_p/λ/qmin 重拟对照）**、**S3（qsat 拟合）** —— 它们都复用 R3 的同一批采集。
5. **N2/N3/N4** 在 S1 真机验收暴露数据后按需处理。
6. **S5（F6 实验控制台）** 排在 F4 之后、与 N5 并行，先后端时序+只读 API。

**总 gate**：
- [ ] S1 全部 gate 通过（单一共享窗口 W≈30s+5s刷新，滞后 P95 ≤ 35s，EMA 时间常数化，单 theta）
- [ ] S2/S3 的重拟与公平性对照进 13 号实验日志，论文标定章节有"拟合/固定参数表"
- [ ] S4 让 R3 数据可事后任意重窗口重拟（rewindow 对拍通过）
- [ ] N1–N6 逐条有结论（解决或明确记录为已知限制）
- [ ] S5 至少完成 5.1+5.2 只读闭环（前端可后置）
- [ ] 新增 ADR：D12（单一共享短窗口+快刷新+时间常数EMA）、D13（控制台边界）
- [ ] `05_paper_vs_impl.md` / `06_calibration_design.md` 相应契约与口径补录完整

---

## 8. 给执行模型的开工指令（原样粘贴）

```text
读 docs/refactor/15_signal_and_window_plan.md 全文。这是一份专项质量计划，和
14_endgame_plan.md 并行执行。核心是 S1（修 TSS 指标滞后），最高优先级。

【最重要的一条，先记牢】改 TSS 计算窗口会同时让继承的 theta_m 和 ema_alpha 失效：
- theta_m 是在 60s 窗口上拟合的 SLO 边界，换窗口必须重拟（S1.4，接到 R3）。
- ema_alpha 是"每 60s 一步"下调的，换成 5s 一步会过度平滑，必须先改成墙钟时间常数
  EMA（S1.3）。
所以 S1 的顺序是：S1.3(时间常数EMA) → S1.1(滑动窗口) → S1.2(单一共享短窗+5s快刷新，不拆双窗口) → S1.4(重拟)。
不允许只改窗口不管 theta/ema。

按第 7 章顺序执行。每个 S 小节里都写了：改哪个文件、加什么函数、RED 测什么、GREEN 判据、
真机验收看什么数字。凡是文档里写了"做之前记 Blocked 问架构师"的地方（S1.4 单/双 theta、
S3 的 kv_cache 取舍、N2 的 safescale 窗口关系），一律停下来问，不要自行二选一。

纪律沿用：TDD 小步提交、每步 cd tre && make check、WORKLOG 记证据路径、真机操作前
nvidia-smi+kubectl 双检、改公式后先跑 golden 对拍确认没破冻结副本、实验输出写本地盘、
脚本收编进 tre/deploy/scripts/。
```
