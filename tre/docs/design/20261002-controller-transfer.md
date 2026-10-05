# 控制器接入 SM 接力原语（2026-10-02，第二阶段，方案 B）

分支 `feat/ctrl-transfer-20261002`（基于控制器栈 `feat/timer-cleanup-20261002` @f860bcde）。只改控制器、
测试和文档。

SM 采用**方案 B（v1 式整锁）**：每个写请求从头到尾持有一把全局写锁，请求内部仍然并行；接力在一次持锁内
完成"选对 → donor 并行睡 → 确认 → receiver 并行醒 → 提交"。因此 `POST /v2/transfers` 返回时，接力已经
完成或已经失败，没有交给恢复、需要事后跟踪的中间状态。SM 方案 B 在另一个分支实现；本文按下文 §2 列出的
接口编写，字段待 SM 分支完成后核对。早先的多阶段方案（锁外 I/O）见
[`20261002-sm-transfer.md`](./20261002-sm-transfer.md)，其中 `left_to_recovery` / `GET /v2/transfers` 跟踪不再适用。

**必须和方案 B 的 SM 同批上线**：本控制器的立即接力只走 `POST /v2/transfers`，不保留旧 SM 的回退路径（旧 SM
上接力会被记为"未执行"，见 §8）。

## 1. 分工

| 谁 | 决定什么 | 依据 |
|---|---|---|
| 控制器 | **数量**：扩 / 缩多少、从哪个模型、给哪个模型、谁先 | 信号（Z、状态带）、副本数（routable、`floor_headroom`、`max_awake`） |
| SM | **位置**：哪个 pod、哪张卡；锁内守住 floor 和"一卡一醒" | store、GPU lease、全局写锁 |

热切换下 wake / sleep 只要 1–3 s，所以控制器不引入冷启动式的计时器：信号滞后用状态门（O1 view-pending），
正确性交给 SM。

donor 策略不变：CRIT 的 IDLE / HIGH donor 立即释放；公平环的 IDLE / HIGH donor 按 v1；中间地带、HIGH
主动缩容、TP 同槽抢占（`ShrinkForSlotAction`）走 SafeScale，SafeScale 提交本批不改。

## 2. 用到的 SM 接口（方案 B）

- `POST /v2/transfers {donor_model, receiver_model, count, sleep_path}`（`sleep_path` 默认 `urgent`）。
  响应：`transfer_id`、`pairs[]`（`donor` / `donors` / `receiver` / `node` / `gpu_ids` / `status` / `error?` /
  `compensating_sleep?`；`status` 只有 `done` | `donor_sleep_failed` | `receiver_wake_failed`）、`done`、`taken`、
  `unfilled`、`refusals[]`、`skipped{}`、`picked[]`、`clamped_by_floor`、`phases_ms`。至少一对完成或整体被
  floor clamp（`taken: 0`）时 200；一对都没完成时 409 `partial`（带整份响应）。
- 写锁等待超时：409 `writer_busy`（什么都没做，可重试）。`/target` 和 `/v2/transfers` 不再返回 RetryLater：
  并发请求在 SM 的锁上排队。
- 路由视图读不到（Pod LIST / Redis 失败）：结构化 409 `routable_unknown`（什么都没做）。
- `/target` 缩容按 floor clamp，200 带 `taken`、`clamped_by_floor`；`/v2/state` 带每个 binding 的 `routable`、
  每个模型的 `routable` / `floor` / `floor_headroom`、顶层 `floor_enforced`。
- 超时：SM 最坏持锁是 `worst_case_lock_hold_s`（默认值下约 290 s，含串行 Kubernetes 调用各 7 s 上限）；客户端看到的最坏一次调用是
  `worst_case_sleep_call_s` = `writer_lock_wait_s` + 最长持锁 + `io_margin_s`（默认 322 s）。控制器的慢调用超时
  （`service_manager.api_call_timeout_s` / `TRE_SM_SLOW_TIMEOUT_SECONDS`，360 s）必须大于它
  ；如果设了 `TRE_SM_SLOW_TIMEOUT_SECONDS`，必须 > 322 s（`app.resolve_sm_call_timeout_s` 启动时用 `sleep_call_timeout_errors` 检查），覆盖"排队等锁 + 本次操作"。控制器里没有依赖旧的分钟级超时做的计算：
  接力不重试、无后续跟踪；`commit_max_age_ms`、view 新鲜度等都与 SM 调用时长无关。

## 3. SM 视图（`/v2/state`）

- `sm_client.parse_state_routable` 解析新字段；`routable` 缺失（旧 SM）或为 null（带 `routable_error`）时
  `routable_ids = None`。`ClusterView` 新增 `routable_ids`、`model_floors`、`floor_enforced`、`routable_error`。
- tick 的 routable 计数（`_cluster_view_counts`）直接用 SM 口径（与它的 floor 检查同一函数）；读不到时退回
  控制器原算法（awake 且未 hidden），并发事件 `sm_routable_fallback:<原因>`。
- 上下文带 `floor` / `floor_headroom`（只在 SM 口径可读时）；paper-state 缓存命中时也用本 tick 视图的值。
  决策快照的 model_states 在有值时带这两项。
- `fetched_ms` 只作参考（`sm_fetched_ms`），不与控制器时钟比较；view-pending 用控制器本机的请求时刻。

## 4. planner

### 4.1 立即接力 = `TransferIntent`

`critical_donor_immediate`、`low_fairness_donor_immediate` 两种有 receiver 的立即接力，产出一个
`TransferIntent(donor_model, receiver_model, count, reason, source_loop, sleep_path="urgent", pairs, rescue)`，
**不选 pod、不选卡**。`count` 是要交出的 donor 副本数（SM 的 `count`），`pairs` 是预期得到的 receiver 副本数
（TP=2 receiver 吃两个单卡 donor 时 `count=2, pairs=1`）。数量逻辑不变：C1 目标、饱和急救有界翻倍、
`planned_take`、`max_awake` 上限、`_donor_give`。

### 4.2 `pairable_count`

`_SlotOccupancy` 只保留容量计数，新增 `pairable` / `pairable_count(donor, receiver, max_pairs, max_donors)`：
按 SM 的选对规则估算能配成几对——receiver 的 sleeping、未 hidden 的 binding，所在卡没有被本 tick 认领或被
SM 标为不可唤醒，且**每一张卡**都被该 donor 的 awake、未 hidden 副本占着（缺一张即 `uncovered_gpu`；完全
没有占用者是普通唤醒）。一对消耗它的全部占用者，donor 少的对优先；算到的对计入本 tick 的认领。intent 的大小
= 估算结果；一对也配不出时不发 intent，发事件 `donor_no_slot_match:<donor>:<receiver>`。估算出的 pod 不会
发给 SM。

### 4.3 donor 的 floor：`floor_headroom`

`_donor_headroom` = `min(SM floor_headroom, routable − registry min_replicas)`；SM 值缺失时退回
`routable − min_replicas`（SM floor 关闭时报 `floor = 0`，取 min 保留控制器自己的 `min_replicas`）。
headroom 为 0 的模型不会成为 donor（立即接力、中间地带、同槽抢占、HIGH 主动探测、IDLE 主动缩容都用它）。

### 4.4 其他路径

- `idle_proactive_immediate`（没有 receiver）是模型级缩容 `PUT /target`（不点名 pod），依赖 SM 的 floor clamp，
  数量另受 headroom 约束。
- 中间地带仍走 SafeScale（需要点名隐藏的 pod），探测 pod 取自同一个 `pairable` 估算。
- TP 同槽抢占不变；它取走的副本计入 `released_donor_ids`，后面的接力估算不重复计算。
- 容量顺序不变：空闲卡上的 sleeping binding → 空闲槽组创建 → 立即接力 → 中间地带 SafeScale。接力没覆盖
  的需求（`unfilled`、`uncovered_gpu`）下一 tick 从新视图重新规划，空闲卡优先。

## 5. ActionQueue

### 5.1 执行

- `TransferIntent` 是**一个** queued action，资源只有 `model:<donor>` 和 `model:<receiver>`（没有 pod / GPU 键）；
  执行 = 一次 `POST /v2/transfers`，不重试（不幂等；下一 tick 重新规划）。调用期间两个模型在 inflight 中，
  返回后立即释放——方案 B 下没有要等待的在途状态。
- 结果是两条 `DispatchResult`：donor（`taken`）和 receiver（`done`、`picked`），都带 `transfer` 摘要
  （transfer_id、done、taken、unfilled、clamped_by_floor、带 pod 名的 pairs、refusals 数、skipped、phases_ms）。

### 5.2 记账规则

| 情况 | donor | receiver |
|---|---|---|
| 200，`done` / `taken` > 0（含部分完成） | `last_done` 记 "down"（taken > 0） | `last_done` 记 "up"（done > 0）；C1 目标 `gained += done` |
| view_changes / routable_changes | 有任何 pair 或 taken > 0 时记 (now, "down") / −1 | done > 0 或有 pair 为 `receiver_wake_failed` 时记 (now, "up") / +1 |
| 200，`taken: 0`、`clamped_by_floor` | 不记 | 不记；事件 `transfer_clamped_by_floor` |
| 409 `partial`（没有一对完成） | 按 pairs 判断（同上） | 同上 |
| 409 `writer_busy` / `routable_unknown`、400 | "未执行"：不记 | "未执行"：不记 |
| 超时、传输错误、5xx、其他 409 | 结果未知：两边都记 view_changes（保守），不记 last_done | 同左 |
| 404 | "未执行"，error 事件（§8） | 同左 |

一律以响应的 `done` / `taken` / `unfilled` 记账，不按 `count`；C1 目标的 `failures` 在 `done < pairs` 时加 1。

模型级缩容（`/target`，含 `idle_proactive_immediate`）同样读 `taken`：`taken < 请求数` 且 `clamped_by_floor` 时
发 `scale_clamped_by_floor:<model>:taken=..:asked=..`；`taken = 0` 视为没有变化，不更新 view_changes /
routable_changes / last_done。`writer_busy` 和 `routable_unknown` 按"未执行"记账
（`sm_client.ServiceManagerError.not_executed`；`writer_busy` 对模型级调用仍标为可重试，但可重规划的动作本来
就不重试），下一 tick 重新规划，不冷却。其它端点仍可能返回的纯文本 RetryLater（`...; retry`）也按"未执行"
处理，但它不被当作"有接力在途"的信号。

### 5.3 只发事件、不冷却

refusals（`wake_refused:<model>:<node>/<gpus>:<code>`）、`clamped_by_floor`、`taken = 0`、`unfilled`
（`transfer_unfilled:<d>-><r>:<n>:<skipped>`）、409 `floor_violation`（`floor_violation:<model>`）都只是观察
事件，经 `drain_events()` 进入下一 tick 的决策事件，不引起任何计时冷却；下一 tick 用新视图重新规划。

## 6. 删除了什么

| 位置 | 删除 |
|---|---|
| planner | `TransferAction`、`fuse_transfers`、`ScaleAction.transfer_id`、`_slot_targeted_transfer`、`_slot_matched_first`、`_SlotOccupancy.donor_slot_pods` / `take_donor` / `_donor_taken` / `donor_taken_ids`、`_placement_retry_events`、`build_plan` 的 `floor_holds` / `unavailable_gpus` / `refusals` 参数、`_Cooldown` 的 floor hold |
| ActionQueue | 先 sleep 后 wake 的两步执行、`gpu:` 资源键与 `slot_of`（`slot_lookup_from_cluster_view`）、`_busy_gpus` / `avoid_gpus`、`_cool` / `_gpu_cooldowns` / `_node_cooldowns` / `_refusals`、`cooled_gpus()` / `cooled_nodes()` / `recent_refusals()`、30 / 60 s 唤醒冷却（`wake_cooldown_s`）、30 s floor hold（`floor_violation_hold_ms`、`floor_held_models()`） |
| tick | `_cooled_gpus`、`_recent_refusals`、`_floor_held_models` |
| sm_client | `scale_model_hinted` 的 `avoid_gpus`；（方案 B）不使用 `GET /v2/transfers` |
| 事件 | `gpu_cooldown`、`placement_retry`（hint 被替换改为 `placement_substituted`）、`floor_violation_hold` |
| 计数 | `transfer_receiver_dropped_total`、`observe_transfer_stopped_total` |

方案 B 下也不再有接力的事后跟踪（`left_to_recovery`、`pending` / `receiver_waking` 状态、`GET /v2/transfers`
轮询）——它们只存在于多阶段方案。

10-06 起：

- env `TRE_FLOOR_VIOLATION_COOLDOWN_TICKS` 已删除，列入 `config.REMOVED_TIMER_ENV`：设置了只打一次日志，
  不解析、不校验（overlay 已不再设置它）；
- registry `placement.wake_cooldown`：控制器不读；它只是 SM 报出的建议值 `retry_after_s`，所以键本身保留。
  控制器启动时不再对它打日志（`app.log_deprecated_settings` 已删）。

## 7. observe 模式

语义不变：observe 下可重规划的动作不下发。接力只有一次 SM 调用，mode 在发出前（uncached）读一次；已经发出的
接力由 SM 在同一次持锁内做完（donor 睡、receiver 醒），控制器只记录（事件 `observe_entered_during_transfer`、
计数 `observe_transfer_completed_total`），不撤销。原先"donor 已睡、observe 中途切换就不唤醒 receiver"
（`observe_entered_mid_transfer`）不再存在：没有可以停下的中间点。

## 8. 决策日志 / replay / 兼容性

- 决策快照里接力是 `{"kind": "transfer", "donor", "receiver", "count", "pairs", "reason", "source_loop",
  "sleep_path", "rescue"?}`，不带 pod。pod 名来自 SM 响应：queue 事件
  `transfer_pair:<donor>-><receiver>:<donor pods>-><receiver pod>@<node>/<gpus>:<status>`（以及日志
  `transfer_result` 的 pairs / picked），进入之后一个 tick 的决策事件。signal log 按两侧记录（donor −count、
  receiver +pairs，标签 `transfer:<d>-><r>`）。
- 离线 replay（`run_tick_replay`）不依赖 SM；离线集成测试的进程内 SM 适配器
  （`test_safescale_binding_commit.InProcessServiceManager.transfer`）按 SM 的选对规则返回同形的响应。
- 本批 SM 与控制器一起上线，不对旧 SM 做回退。`POST /v2/transfers` 返回 404 时：error 日志
  `transfer_endpoint_missing`、事件 `transfer_unsupported:<d>-><r>`、计数 `transfer_unsupported_total`，按
  "未执行"记账，不崩溃；`/v2/state` 没有 `routable` 字段时退回控制器自己的计数（事件 `sm_routable_fallback`）。

## 9. 与 SM 接口的对应

| SM（方案 B） | 控制器 |
|---|---|
| `POST /v2/transfers` 一次持锁内完成；200 也可能部分完成；409 `partial` | §5.1、§5.2 |
| 选对规则（每张卡都要有 donor、`uncovered_gpu`） | §4.2 `pairable_count` |
| donor floor / receiver cap | §4.3 `floor_headroom`；planner 的 `max_awake` 上限不变 |
| 写锁排队、超时 409 `writer_busy`；`routable_unknown` | §5.2 "未执行" |
| `/v2/state` routable / floor / floor_headroom；`/target` 的 `taken` / `clamped_by_floor` | §3、§5.2 |

## 10. 残留风险

- `pairable_count` 是基于本 tick 视图的估算：SM 锁内可能因 busy、居民探测否决、cap 等配得更少；差额以
  `unfilled` / `refusals` 返回，下一 tick 重新规划。
- 中间地带 SafeScale 仍点名探测 pod（取自估算），提交时 receiver 走 `/target` 的 at_least 唤醒，由 SM 选位置；
  这一路径本批不改。
- 全局写锁下，一次接力（默认值下最坏持锁约 290 s）期间其它模型的 SM 写调用在 SM 侧排队；控制器的 ActionQueue 仍并发
  下发不同模型的动作，排队超时的以 `writer_busy` 返回、下一 tick 重新规划。
- 纯文本 RetryLater：方案 B 的 `/target`、`/v2/transfers` 不再发（只剩 unhide 未确认醒、Pod 启动准入）。
  10-06 起控制器不再识别 `detail` 以 `retry` 结尾的 409：没有 `error` 码的 409 一律按"结果未知"保守处理
  （只多一次 view-pending 等待）。
- 接力 hold（10-06，评审 P2-1）：SM 对一次接力答"什么都没做"（`done == 0`：拒绝、否决、floor clamp、
  unfilled、`writer_busy`、`routable_unknown`、404；不含超时 / 传输错误这类结果未知）时，ActionQueue 记下
  (donor, receiver) 和规划它时的视图输入（`/v2/state` 的 `version` + 两个模型的 floor 视图）；在这些输入不变前
  planner 不再规划同一对（事件 `relay_held:<donor>:<receiver>:<原因>`），不再每 tick 抢一次全局写锁。
  `writer_busy` / `routable_unknown` 以及没有 `version` 的视图，任何更新的视图都会解除。SM 报路由视图读不到时
  （`routable_error`，`routable_missing` 除外）不规划接力（事件 `relay_skipped_routable_unknown`）。只是状态门，
  没有计时器。

## 11. 上线（10-06）

本批控制器与方案 B 的 SM 一起上线，registry 必须先换。顺序：registry ConfigMap → gateway-plugins → SM →
controller；回滚时镜像和 ConfigMap 成对恢复。原因：线上旧 ConfigMap 的超时（`writer_lock_wait_s` 10、
`sleep_call_timeout_s` 45 等）按新公式的最坏调用是 374 s，不小于 360 s，新控制器启动时
（`app.resolve_sm_call_timeout_s`）和新 SM 都会拒绝。细节见 `20261002-sm-wholelock.md` §10。
