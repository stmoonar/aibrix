# TRE v2 Console 重构设计（2026-07-25）

## 1. 问题

现有 console（镜像 `tre-v2-ui:20260710-ca61e485`）有三个视图：Live 信号、Fleet 共驻、Control 控制。用户报告的问题是四类叠加，且互相强化：

1. **关键数据压根没显示**。UI 镜像早于 2026-07-15 的模型冷启动治理架构，因此 supervisor 状态、operation journal、GPU lease、desired↔observed 差异、SM audit issues 在 console 里完全不可见——而这些恰好是 07-14 事故之后最需要观测的东西。
2. **没有时间维度**。界面只渲染此刻的值。后端其实一直在存历史（`/api/signal/history` 端点已存在，sampler 有 2000 点/模型的内存环，`tre:v2:controller:signal_log` stream 有 20 万条约 5 天），但前端从未画过曲线。
3. **信息要自己拼**。一个问题（某个决策落到哪个 pod、哪张卡）需要在多个视图之间来回切换并在脑子里对应。
4. **数字没有解释**。满屏原始值和 null，没有阈值参照、单位或正常范围。空载时 `z_m` 为 `nan`、界面渲染成 —，看起来像系统坏了。

补充：图表尺寸过小，无法用于盯盘。

## 2. 目标与非目标

Console 需要同时服务四件事：跑实验时盯盘、出事时排障、日常巡检、调参与执行操作。

**非目标**：不改 controller 与 service-manager 的任何代码；不改变扩缩容行为；不引入外部依赖（Grafana/Prometheus/CDN 框架）。

## 3. 硬约束：不得影响扩缩容性能

这是本设计的第一约束，决定了采集分层。实测各只读端点的开销：

| 端点 | 实际开销 | 结论 |
|---|---|---|
| `/v2/supervisor` | 纯内存 `asdict(snapshot)` | 可 2s 轮询 |
| `/v2/fleet/state` | 2 次 Redis 读 | 可 2s 轮询 |
| `/v2/operations?limit=N` | 1 次 Redis hash 读 | 可 2s 轮询 |
| GPU lease (`tre:v2:sm:gpu_leases`) | 1 次 Redis hash 读 | 可 2s 轮询 |
| **`/v2/audit`** | **k8s list pods + 逐个 HTTP 探测 20 个 vLLM pod 的 `/is_sleeping`** | **绝不自动轮询** |

`audit()` 走 `audit_state(store, k8s_client, prober=_VllmPodProber(...))`。2 秒轮询等于每秒 10 次 HTTP 请求打到模型 pod 上，负载下会直接干扰推理，进而污染实验数据。

因此：
- audit 仅由用户点击触发，页面显示上次审计时间与结果。
- 提供默认关闭的自动审计开关，且下限 5 分钟。
- 以守卫测试固化该约束：断言 audit 不出现在 sampler 的轮询路径中。

其余性能原则：
- 浏览器只读 sampler 的内存快照，永不直接打 SM。N 个标签页 = 1 份后端负载（延续现有架构）。
- SSE 0.5s 推送频率不变，新增内容只是往既有 snapshot 加 key。
- 历史曲线走按需 GET，不塞进 SSE 通道。

## 4. 信息架构：四个页面，一页一任务

### 4.1 Overview（巡检）

目标：10 秒内判断是否需要深入。

- 顶部状态灯行：controller mode（并显示**为什么**是 observe——人工切换还是 supervisor 自动切换）、supervisor running/drift、audit healthy + issues 数（含上次审计时间）、每节点 GPU truth 新鲜度、SSE 连接状态。
- desired↔observed 一致性摘要：desired / observed / mismatch 计数，不一致时直接列出条目。
- 每模型一行：tier、z_m 当前值 + 迷你趋势、awake/target 副本、routable pods。
- 最近 5 条动作/事件。

### 4.2 Signals（盯盘）

三个模型各一张全宽图，绘图区高约 300px。

每张图包含：
- 主轴：Z_m 曲线 + τ_high / τ_crit 阈值带（背景色带）。
- 次轴：queue_len。
- 副本数阶梯线：awake 与 target 两条。二者分叉即想扩但扩不上。
- decode_tps / prefill_tps：默认折叠，按钮展开。
- 动作竖线标注：悬停显示 action 与 reason。

交互：
- 三图共享时间轴，十字准星联动——悬停任一张，三张同时显示同一时刻的值。跨模型比较是 TRE 的核心命题，该联动是本页的关键能力。
- 时间窗 5m / 15m / 1h，默认 15m（单次实验约 12 分钟）。
- 暂停按钮冻结画面以便细看。

### 4.3 Fleet（排障）

2 节点 × 4 GPU 矩阵。每格展开：驻留模型列表（awake 高亮）、显存条 used/total、lease 的 phase + fencing token + owner、pod 名与 UID。desired 与 observed 不一致处标红。

### 4.4 Ops（操作与审计）

- operation journal 时间线：id、类型、状态、耗时、**失败原因**。
- supervisor 面板：running、drift_observations、last_drift、last_recovery_operation_id。
- audit issues 列表 + 手动触发按钮。
- 控制区：mode 切换（带确认）、reconcile、defrag；params 编辑并显示 `params_hash` vs `applied_hash` 与 `pending_restart`，让改了到底生没生效可见。

## 5. 贯穿规则：数字必须自带参照

- z_m 标注其相对 τ 的位置；queue_len 标注阈值；显存标注百分比。
- 所有 age / ttl 显示新鲜度，过期变色。
- **区分空载与信号缺失**：空载时 z_m 为 nan，必须显示为「idle · 无负载」，不得渲染成 —（当前行为看起来像系统故障）。

## 6. 后端改动

限制在 `ui/tre_ui/` 内。

1. `sampler.py`：新增廉价只读源的 2s 轮询（supervisor / fleet state / operations / gpu leases），并入既有快照；audit 不进轮询路径。
2. `app.py`：
   - 新增 `GET /api/signal/timeline`：以 `XREVRANGE` + `COUNT` 上限读 `signal_log` 最近 1h，含 queue_len / decode_tps / prefill_tps / tier / action。sampler 记住上次 stream id 做增量拉取，且仅在有客户端订阅时拉取。**不得使用 XRANGE 全量扫描**（该 stream 有 20 万条）。
   - 新增 `POST /api/ops/audit`：手动触发 SM audit 并缓存结果与时间戳。
3. 前端 `static/`：index.html / app.js / style.css 按上述四页重写。纯 vanilla JS + 内联 SVG 自绘图表，不引任何 CDN 或框架（UI pod 除 docker.io 外无外网，且与现有实现保持一致）。

## 7. 测试

- sampler 分层采集测试，含**守卫测试：audit 不在轮询路径中**（固化第 3 节的性能约束）。
- `/api/signal/timeline`：COUNT 上限、增量 stream id、空 stream、未知模型。
- 快照 schema 测试：新增 key 存在且形状正确。
- 既有 ui/tests 全部保持通过。

## 8. 交付

UI 为独立镜像与独立 Deployment，滚动升级不影响 controller / service-manager，亦不触碰 `aibrix-system`。镜像 tag 需同时更新 `deploy/overlays/tre-v2/ui.yaml` 与守卫测试 `deploy/tests/test_kustomize_overlays.py`。
