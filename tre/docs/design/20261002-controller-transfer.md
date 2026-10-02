# 控制器接入 SM 接力原语（2026-10-02，第二阶段）

分支 `feat/ctrl-transfer-20261002`（基于控制器栈 `feat/timer-cleanup-20261002` @f860bcde）。只改控制器、
测试和文档。SM 侧接口见 [`20261002-sm-transfer.md`](./20261002-sm-transfer.md)（分支
`feat/sm-transfer-20261002`）。

**必须和该 SM 同批上线**：本控制器的立即接力只走 `POST /v2/transfers`，不保留旧 SM 的回退路径（旧 SM 上
接力会被记为"未执行"，见 §8）。

## 1. 分工

| 谁 | 决定什么 | 依据 |
|---|---|---|
| 控制器 | **数量**：扩 / 缩多少、从哪个模型、给哪个模型、谁先 | 信号（Z、状态带）、副本数（routable、`floor_headroom`、`max_awake`） |
| SM | **位置**：哪个 pod、哪张卡；锁内守住 floor 和"一卡一醒" | store、sleep reservation、GPU lease、journal、写锁 |

热切换下 wake / sleep 只要 1–3 s，所以控制器不引入冷启动式的计时器：信号滞后用状态门（O1 view-pending、
接力恢复跟踪），正确性交给 SM。

donor 策略不变：CRIT 的 IDLE / HIGH donor 立即释放；公平环的 IDLE / HIGH donor 按 v1；中间地带、HIGH
主动缩容、TP 同槽抢占（`ShrinkForSlotAction`）走 SafeScale，SafeScale 提交本批不改。

## 2. SM 视图（`/v2/state`）

- `sm_client.parse_state_routable` 解析新字段：每个 binding 的 `routable`、每个模型的
  `routable` / `floor` / `floor_headroom`、顶层 `floor_enforced`；`routable` 缺失（旧 SM）或为 null（SM 读不到
  Pod / reservation / lease，带 `routable_error`）时 `routable_ids = None`。
- `ClusterView` 新增 `routable_ids`、`model_floors`、`floor_enforced`、`routable_error`。
- tick 的 routable 计数（`_cluster_view_counts`）直接用 SM 口径：serve id 在 `routable_ids` 里才算。SM 口径
  比控制器原先的 `awake && !hidden` 少算正在 sleep（prepare 到 commit 之间）的副本，偏差方向与 SM 的 floor
  检查一致。读不到时退回原算法，并发事件 `sm_routable_fallback:<原因>`。
- 上下文带 `floor` / `floor_headroom`（只在 SM 口径可读时）；paper-state 缓存命中时也用本 tick 视图的值，不沿用
  旧值。决策快照的 model_states 在有值时带这两项。
- `fetched_ms` 只作参考（`sm_fetched_ms`），不与控制器时钟比较；view-pending 用控制器本机的请求时刻。

## 3. planner

### 3.1 立即接力 = `TransferIntent`

`critical_donor_immediate`、`low_fairness_donor_immediate` 两种有 receiver 的立即接力，产出一个
`TransferIntent(donor_model, receiver_model, count, reason, source_loop, sleep_path="urgent", pairs, rescue)`，
**不选 pod、不选卡**。`count` 是要交出的 donor 副本数（SM 的 `count`），`pairs` 是 planner 预期得到的
receiver 副本数（TP=2 receiver 吃两个单卡 donor 时 `count=2, pairs=1`）。

数量逻辑不变：C1 目标、饱和急救有界翻倍、`planned_take`、`max_awake` 上限、`_donor_give`（默认每 tick 一步，
`donor_surplus_release` 时给出全部盈余）。

### 3.2 `pairable_count`

`_SlotOccupancy` 只保留容量计数，新增 `pairable(donor, receiver, max_pairs, max_donors)` /
`pairable_count(...)`：按 SM 的选对规则估算能配成几对——receiver 的 sleeping、未 hidden 的 binding，所在卡
没有被本 tick 认领或被 SM 标为不可唤醒，且**每一张卡**都被该 donor 的 awake、未 hidden 副本占着（缺一张即
`uncovered_gpu`，不算；完全没有占用者是普通唤醒，不算）。一对消耗它的全部占用者，donor 少的对优先。算到的
对计入本 tick 的认领（receiver 卡、donor 副本），后面的接力 / 唤醒不重复计算。

intent 的大小 = `pairable_count` 的结果，避免发出注定凑不满的意图；一对也配不出时不发 intent，发事件
`donor_no_slot_match:<donor>:<receiver>`。这里算出的 pod 不会发给 SM。

### 3.3 donor 的 floor：`floor_headroom`

`_donor_headroom` = `min(SM floor_headroom, routable − registry min_replicas)`；SM 值缺失时退回
`routable − min_replicas`。SM floor 关闭时它报 `floor = 0`，所以取 min 保留控制器自己的 `min_replicas`。
headroom 为 0 的模型不会成为 donor（立即接力、中间地带、同槽抢占、HIGH 主动探测、IDLE 主动缩容都用它）。

### 3.4 其他路径

- `idle_proactive_immediate`（没有 receiver）是模型级缩容：`PUT /target`（`scale_model`，不点名 pod），依赖
  SM 的 floor clamp，数量另受 headroom 约束。
- 中间地带仍走 SafeScale：SafeScale 需要点名隐藏的 pod，它的探测 pod 取自同一个 `pairable` 估算（释放它们能
  空出 receiver 的卡）。
- TP 同槽抢占不变；它取走的副本计入 `released_donor_ids`，后面的接力估算不重复计算。
- 容量顺序不变：receiver 先用空闲卡上的 sleeping binding，再用空闲槽组创建，再用立即接力，最后中间地带
  SafeScale。接力没覆盖的需求（`unfilled`、`uncovered_gpu`）下一 tick 从新视图重新规划，空闲卡优先。

## 4. ActionQueue

### 4.1 执行

- `TransferIntent` 是**一个** queued action，资源只有 `model:<donor>` 和 `model:<receiver>`，没有 pod / GPU
  键；observe 模式在发出前检查（§6）。执行 = 一次 `POST /v2/transfers`，不重试（不幂等；下一 tick 重新规划）。
- 结果是两条 `DispatchResult`：donor（`taken`）和 receiver（`done`、`picked`），都带 `transfer` 摘要
  （transfer_id、done、taken、unfilled、clamped_by_floor、带 pod 名的 pairs、refusals 数、skipped、phases_ms）。

### 4.2 记账规则

| 情况 | donor | receiver |
|---|---|---|
| 200，`done`/`taken` > 0（含部分完成） | `last_done` 记 "down"（taken > 0） | `last_done` 记 "up"（done > 0）；C1 目标 `gained += done` |
| view_changes / routable_changes | 有任何 pair 或 taken > 0 时记 (now, "down") / −1 | done > 0 或有 pair 处于 `receiver_wake_failed` / `pending` / `receiver_waking` 时记 (now, "up") / +1 |
| 200，`taken: 0`、`clamped_by_floor` | 不记 | 不记；事件 `transfer_clamped_by_floor` |
| 409 `partial`（没有一对完成） | 同上，按 pairs 判断 | 同上 |
| 409 `routable_unknown` / `writer_busy`、400、`RetryLater`（`...; retry`） | "未执行"：不记 | "未执行"：不记 |
| 超时、传输错误、5xx、其他 409 | 结果未知：两边都记 view_changes（保守），不记 last_done | 同左 |
| 404 | "未执行"，error 事件（§8） | 同左 |

一律以响应的 `done` / `taken` / `unfilled` 记账，不按 `count`。C1 目标的 `failures` 在 `done < pairs` 时加 1。

模型级缩容（`/target`，含 `idle_proactive_immediate`）同样读 `taken`：`taken < 请求数` 且 `clamped_by_floor`
时发 `scale_clamped_by_floor:<model>:taken=..:asked=..`；`taken = 0` 视为没有变化，不更新 view_changes /
routable_changes / last_done。`/target` 返回 `RetryLater` 或 `routable_unknown`（结构化 409）时同样按"未执行"
记账（`sm_client.ServiceManagerError.not_executed`），下一 tick 重新规划。**`RetryLater` 不被当作"有接力在途"
的信号**：在途接力只看本控制器的 `left_to_recovery` 跟踪和 `GET /v2/transfers`。

### 4.3 `left_to_recovery`：状态门

响应里有 `left_to_recovery: true` 的 pair（SM 提交阶段拿不到写锁，pair 交给它的恢复）时，执行这条 intent 的
dispatch 任务不结束：它继续持有两个模型的资源（`inflight_models()` 包含二者，planner 不会把它们规划成 donor 或
receiver，submit 时与之冲突的可重规划动作被丢弃为 `inflight`），每 `transfer_poll_s`（默认 2 s）读一次
`GET /v2/transfers`，直到 `in_progress` 和 `running_here` 都不再列出这个 transfer_id 才放行。这是状态门：
放行条件是 SM 的状态，不是时间；读不到时门保持关闭。结束时给两个模型各记一次 view_changes（恢复可能改变了
两边的 routable）。`recovering_transfers()` 给出当前跟踪中的条目；事件 `transfer_left_to_recovery`、
`transfer_recovered`。

SafeScale 的 one-shot 动作（unhide / commit 回滚）不可重放，仍按原规则排队，不受这个门影响。

### 4.4 只发事件、不冷却

refusals（`wake_refused:<model>:<node>/<gpus>:<code>`）、`clamped_by_floor`、`taken = 0`、`unfilled`
（`transfer_unfilled:<d>-><r>:<n>:<skipped>`）、409 `floor_violation`（`floor_violation:<model>`）都只是观察
事件，经 `drain_events()` 进入下一 tick 的决策事件，不引起任何计时冷却；下一 tick 用新视图重新规划。

## 5. 删除了什么

| 位置 | 删除 |
|---|---|
| planner | `TransferAction`、`fuse_transfers`、`ScaleAction.transfer_id`、`_slot_targeted_transfer`、`_slot_matched_first`、`_SlotOccupancy.donor_slot_pods` / `take_donor` / `_donor_taken` / `donor_taken_ids`、`_placement_retry_events`、`build_plan` 的 `floor_holds` / `unavailable_gpus` / `refusals` 参数、`_Cooldown` 的 floor hold |
| ActionQueue | `_execute_transfer` 的先 sleep 后 wake 两步、`gpu:` 资源键与 `slot_of`（`slot_lookup_from_cluster_view`）、`_busy_gpus` / `avoid_gpus`、`_cool` / `_gpu_cooldowns` / `_node_cooldowns` / `_refusals`、`cooled_gpus()` / `cooled_nodes()` / `recent_refusals()`、30 / 60 s 唤醒冷却（`wake_cooldown_s`）、30 s floor hold（`floor_violation_hold_ms`、`floor_held_models()`） |
| tick | `_cooled_gpus`、`_recent_refusals`、`_floor_held_models` |
| sm_client | `scale_model_hinted` 的 `avoid_gpus` |
| 事件 | `gpu_cooldown`、`placement_retry`（hint 被替换改为 `placement_substituted`）、`floor_violation_hold` |
| 计数 | `transfer_receiver_dropped_total`、`observe_transfer_stopped_total` |

仍能解析但忽略（启动时打 `deprecated_setting_ignored` 日志，`app.log_deprecated_settings`）：

- env `TRE_FLOOR_VIOLATION_COOLDOWN_TICKS`（非法值仍报错；overlay 已不再设置它）；
- registry `placement.wake_cooldown`：控制器不再用；SM 仍把它作为 `retry_after_s` 报出，所以键本身保留。
  取非默认值时打日志。

## 6. observe 模式

语义不变：observe 下可重规划的动作不下发。接力只有一次 SM 调用，mode 在发出前（uncached）读一次；已经发出的
接力由 SM 做完（donor 睡、receiver 醒），控制器只记录（事件 `observe_entered_during_transfer`、计数
`observe_transfer_completed_total`），不撤销。原先"donor 已睡、observe 中途切换就不唤醒 receiver"
（`observe_entered_mid_transfer`）不再存在：SM 在一次请求里完成两步，没有可以停下的中间点。SM 的恢复在 observe
下不发新的 `/wake_up`（SM 文档 §8），那时 receiver 由 wake journal 按物理状态结算，控制器的恢复跟踪照常在
`GET /v2/transfers` 不再列出该条目后放行。

## 7. 决策日志 / replay

- 决策快照里接力是 `{"kind": "transfer", "donor", "receiver", "count", "pairs", "reason", "source_loop",
  "sleep_path", "rescue"?}`，不带 pod。
- pod 名来自 SM 响应：queue 事件 `transfer_pair:<donor>-><receiver>:<donor pods>-><receiver pod>@<node>/<gpus>:<status>`
  （以及日志 `transfer_result` 的 pairs / picked），进入之后一个 tick 的决策事件。
- 离线 replay（`run_tick_replay`）不依赖 SM；离线集成测试用的进程内 SM 适配器
  （`test_safescale_binding_commit.InProcessServiceManager.transfer`）按 SM 的选对规则返回同形的响应。

## 8. 兼容性与上线

- 本批 SM 与控制器一起上线。控制器不对旧 SM 做回退。
- `POST /v2/transfers` 返回 404（SM 太旧）时：error 日志 `transfer_endpoint_missing`、事件
  `transfer_unsupported:<d>-><r>`、计数 `transfer_unsupported_total`，按"未执行"记账，不崩溃；该 tick 的接力
  需求不会被满足（直到 SM 升级）。
- `/v2/state` 没有 `routable` 字段时退回控制器自己的 routable 计数（事件 `sm_routable_fallback`）。
- 控制器重启会丢失内存中的恢复跟踪；SM 侧仍按 journal 完成恢复，SM 的 cap 预算会计入 pending 的 receiver。

## 9. 与 SM 文档的对应

| SM 文档 | 控制器 |
|---|---|
| §2 接口、200 也可能部分完成 | §4.1、§4.2 |
| §2 409 `partial`、`left_to_recovery`、`GET /v2/transfers` | §4.2、§4.3 |
| §3 选对规则（每张卡都要有 donor、`uncovered_gpu`） | §3.2 `pairable_count` |
| §3.6 donor floor / receiver cap | §3.3 `floor_headroom`；planner 的 `max_awake` 上限不变 |
| §8 observe | §6 |
| §10 分工（删除选卡、两步调用、唤醒冷却） | §1、§5 |
| §12 routable 口径、floor clamp、`taken` / `clamped_by_floor` | §2、§4.2 |

## 10. 残留风险

- `pairable_count` 是基于本 tick 视图的估算：SM 锁内可能因 busy、居民探测否决、cap 等配得更少；差额以
  `unfilled` / `refusals` 返回，下一 tick 重新规划，不会重复下发同一对。
- 中间地带 SafeScale 仍点名探测 pod（取自估算），提交时 receiver 走 `/target` 的 at_least 唤醒，由 SM 选位置；
  这一路径本批不改。
- 恢复跟踪期间两个模型都不参与规划；若 SM 恢复长时间不结束（SM 侧有轮次上限），这两个模型会一直被视为
  在途。
- `RetryLater` 的识别依赖 SM 的消息约定（无 `error` 字段、`detail` 以 `retry` 结尾）；识别不到时按"结果未知"
  保守处理（只多一次 view-pending 等待）。
