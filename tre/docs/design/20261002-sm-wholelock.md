# service-manager 整锁（方案 B，2026-10-02）

分支 `feat/sm-wholelock-20261002`（基于线上版本 `bdea7a72`）。用户已批准。

每个写操作从头到尾持有 SM 的一把全局写锁（Redis 写锁，先到先得），HTTP 的
`/sleep`、`/wake_up` 和物理确认都在锁内完成；同一个请求内部的多个 pod 仍然并行。
Redis 持久化、journal 恢复、物理确认、透明睡眠、replica floor、gpu-truth 唤醒门全部保留。

## 1. 为什么回到整锁

v1 的 SM 一直是整锁。v2 为了让不同请求的 sleep / wake 在时间上重叠，先在 09-30 把 wake
挪到锁外（S6 三段式），10-02 又把 sleep 挪到锁外。每挪一次，就多出一类"做到一半"的中间
状态，而处理这些状态又引入了新机制：sleep reservation 及其续约 / 过期 / 丢失处理、keepalive
线程、waking lease 及其孤儿回收、进程内在途集合、`/target` 与 `/power` 的 RetryLater、
锁外阶段到恢复流程的交接、transfer journal 及其续接。三轮评审查出的问题几乎都出在这些机制上。

收益与代价的对比：

| | 锁外方案（v2 到 10-02） | 整锁（方案 B） |
|---|---|---|
| 跨请求重叠的时间 | 至多一次 wake（1.5–3 s）或一次 sleep（1–3 s） | 无，请求在锁上排队 |
| sleep 在锁外的部分 | 不排空、一律 abort 之后只剩网关回执（7–15 ms） | 不适用 |
| 中间状态 | reservation、waking lease、在途集合、交接、transfer journal | 无（只剩崩溃 / 引擎读不到时的 journal 条目） |
| 调用方可见的"稍后重试" | `/target`、`/power` 的 RetryLater（409） | 无，锁上排队；等锁超时才 409 `writer_busy` |
| 代码（本分支核心提交，`tre_sm`） | — | +978 / −1753 行 |
| 测试（本分支核心提交） | — | +504 / −1601 行 |

SM 已经改成"不排空、一律 abort"，sleep 留在锁外的只是十几毫秒的网关回执，整锁损失的重叠
很小。所以选择整锁：删掉上面这些机制，不变量靠"锁内账面 + 物理确认 + journal 恢复"保证。

## 2. 形态

- **sleep**：`SleepPrimitive.sleep()` 是唯一的入口：floor 检查 → 写 journal 并隐藏（routable=false +
  route-gen）→ 网关回执 → 读一轮负载留档（`/metrics` 与 `/version` 同一轮并行探测）→ 一次
  `/sleep`（`mode=abort`；`sleep_mode_when_idle: wait` 可在无在途请求时改用 wait）→ 物理确认。
  多个 pod 的每一步都并行。不排空；在途请求由 sidecar 续发，结果里的 `aborted` 记录被打断的数量。
  vLLM 不支持 `mode` 参数且有在途请求（或负载读不到）时不发裸 `/sleep`，直接回滚
  （运维显式设 `vllm_sleep_mode_param: false` 才发裸 `/sleep`）。
- **wake**：prepare（账面、唤醒门、`awake` GPU 租约、wake journal、desired 意图）→ 本请求所有
  binding 并行 `/wake_up` + `/is_sleeping` → commit（annotation、store；失败的唤醒按物理状态结算，
  引擎其实醒了就补偿睡眠）。
- **接力** `POST /v2/transfers`：一次 `_writer("transfer")` 内完成选对、donor 并行睡、receiver 并行醒、
  提交（见 §6）。
- 启动准入、启动收敛、fleet repair、defrag、重启守卫、补偿睡眠都用同一个 sleep / wake 形态，
  各自在自己的一次持锁内完成。

## 3. 保留与删除

保留：Redis 写锁（fencing、先到先得队列）、sleep journal 与 wake journal（只用于崩溃恢复）、物理
确认、透明睡眠的"隐藏 → 网关回执"顺序、replica floor、gpu-truth 唤醒门与 power 标记、
`starting` / `awake` 租约与启动占位、fleet repair、defrag、结构化 409 的错误码。

删除：

- `state/sleep_reservations.py`（Redis key `tre:v2:sm:sleep_reservations`、三个 Lua 脚本）及其所有
  用法（`_assert_not_reserved`、`/v2/sleep` 的 `reservations`、admission 与 fleet repair 的预留检查）；
- 锁外排空与拆分 sleep：`_split_sleep`、`_finish_split_sleep`、`drain`、`resolve_lost`、`keepalive`、
  `abandon`、`release_unresolved`；
- S6 三段唤醒的锁外部分：`_finish_split_wakes`、`_hand_over_to_recovery`、`_wakes_in_flight(_of)`、
  `waking` 租约的相位语义（prepare 直接拿 `awake` 租约）、`reap_orphan_waking_leases`；
- 进程内 floor 锁 `_floor_lock`；
- `/target`、`/power` 的 RetryLater：并发请求在锁上排队，等锁超过 `writer_lock_wait_s` 才 409
  `writer_busy`；
- 沿用但不再生效的 registry 键：`sleep.budgets_s`、`sleep.no_drain_paths`、`sleep.hard_cap_s`、
  `sleep.reservation_ttl_s`、`commit_lock_wait_s`（仍可解析，SM 启动时打一行弃用日志）。

## 4. 不变量与保证方式

| 不变量 | 保证方式 |
|---|---|
| 1. 每张卡最多 1 个醒着的模型 | 唤醒的账面检查（store、GPU 租约、wake journal）、唤醒门（gpu-truth 或同卡居民探测）和拿租约，与 `/wake_up` 在同一次持锁内完成，没有别的写操作能插进来；租约由 Lua 脚本在 fence 下原子写入 |
| 2. 物理确认已睡后才释放 GPU | 租约只在 `/is_sleeping` 读到睡着之后释放（sleep 的 `slept` 结果、恢复读到睡着、失败唤醒结算读到睡着）；确认超时、`/sleep` 没有应答、状态读不到都保留租约 |
| 3. 物理状态未知时绝不重开路由 | 回滚前重新探测：读到睡着记为已睡，读不到记 `unconfirmed`（保持隐藏），读到醒着还要 `/is_paused` / `/resume` 确认没有暂停；`/sleep` 没有应答直接 `unconfirmed`；恢复流程要求两次相隔超过 `sleep_call_timeout_s` 的"醒着"读数并确认没有暂停，才恢复路由 |
| 4. floor 由 SM 在锁内保证 | floor 检查和随后的隐藏 / 睡眠在同一次持锁内；`/target` 的缩容按 floor 收紧（clamp），点名的 binding 与 SafeScale 隐藏拒绝（409 `floor_violation`）；routable 视图读不到时在任何隐藏之前 409 `routable_unknown`（repair / startup 路径豁免） |

崩溃时 journal 的写入顺序保证可恢复：sleep 的 journal 条目在 GPU 租约、store、desired 都写完之后
才结束；wake 的 journal 条目在 commit 或结算之后才结束。

## 5. 超时与最坏持锁

| 参数 | 默认值 | 理由 |
|---|---|---|
| `sleep.ack_timeout_s` | 5 s（原 10） | 实测回执 7–15 ms；超时回滚隐藏 |
| `sleep.sleep_call_timeout_s` | 10 s（原 45） | 实测 sleep 1–3 s；只有一次 `/sleep` |
| `sleep.physical_confirm_timeout_s` | 8 s（原 15） | 实测确认在 1 s 内 |
| `wake.call_timeout_s` | 10 s（新增，单次尝试） | 实测 wake 1.5–3 s；原来是探测超时 × 3 次重试 |
| `sleep.probe_timeout_s` | 2 s（原 5） | `/metrics`、`/version`、`/is_sleeping` 等探测；读不到按"未知"保守处理 |
| `sleep.io_margin_s` | 2 s（原 5） | Redis / Kubernetes 调用余量 |
| `writer_lock_wait_s` | 30 s（原 10） | 至少能排在一次最坏情况的 sleep 之后，或排在若干个 2–5 s 的普通操作之后 |

最坏持锁（`ServiceManagerConfig.worst_case_*`，多个 target 并行，与数量无关）：

- 一次 sleep：回执 5 + 一轮探测 2 + `/sleep` 10 + max(确认 8 + 最后一轮探测 2，失败回滚的 4 次探测 8)
  + 余量 2 = **29 s**（三项超时 23 s，探测与 IO 6 s）。典型 1–3 s。
- 一次 wake：居民探测 2 + `/wake_up` 10 + 收敛与结算探测 2 × 2 + 补偿睡眠 29 + 余量 2 = **47 s**；
  不需要补偿睡眠时 18 s。典型 1.5–3 s。
- 一次接力：选对时的一次探测 2 + donor 的 sleep 29 + receiver 的 wake 47 = **78 s**（两条失败路径
  同时到上限）；典型 3–6 s。
- 调用方看到的最坏时长：`writer_lock_wait_s` 30 + 最长持锁 78 + 余量 2 = 110 s，registry 校验它小于
  `api_call_timeout_s`（360 s，controller 的慢调用超时）。
- SIGTERM 等待：最长持锁 + 余量 = 80 s，小于 Deployment 的 `terminationGracePeriodSeconds`（300 s）。
- cold start、defrag、fleet repair 的持锁按设计以分钟计（等 pod 起来），不在 controller 的规划循环里。

**vLLM hang 时的行为**：所有 vLLM 调用都有 HTTP 超时，持锁时长因此有界。

- sleep：`/sleep` 10 s 内没有应答 → pod 保持隐藏，journal 标 `sleep_unconfirmed`，GPU 租约保留，
  立即返回并释放锁；之后每轮 supervisor 由 `recover_sleep_journal` 按物理状态结算（读到睡着记为
  已睡；仍读不到就继续隐藏并计数，审计报 `sleep_unconfirmed`）。
- wake：`/wake_up` 10 s 内没有应答（超时、连接错误；`VllmOps` 把它记为 `status_code` None，且
  `/wake_up` 不再重试）→ 结果是"不确定"而不是"失败"：wake journal（标 `uncertain`）与 `awake` 租约
  保留（卡继续被占），返回 409 `wake_failed` 并释放锁。`recover_wake_journal` 最早在
  `wake.transport_recheck_s` 之后复核（这是正确性下限，不是证据）：醒了补记账；同一轮里先读 sidecar
  的 `GET /tre-reissue/state` 得到 `waking == 0`、再读 `/is_sleeping == true` 才回滚并释放租约；其他组合
  （包括 sidecar 读不到）保留到下一轮（`wake_unsettled`）。有 HTTP 应答的失败（4xx/5xx）仍是确定的失败。
  引擎读不到时最多保留 `wake.recovery_unknown_attempts` 轮，之后（或 pod 不 Ready 时）恢复 desired、
  结束 journal，但**不释放租约**：binding 进入 suspect，由重启守卫的 suspect 收敛处理（同样要
  `waking == 0` 且读到睡着才释放；读到醒着按 desired 收敛），Pod 消失时由孤儿租约回收释放。
- 不变量 I1（10-04）：GPU 租约只凭新鲜的物理证据释放——动作结束后读到的 `/is_sleeping == true`、
  Pod 已从 Pod 列表消失、或 refresh_seq 新于该动作的 gpu-truth 样本；传输超时、HTTP 5xx、k8s 接受了
  delete、重试用尽、"放弃"都不算。
- 写锁 lease 由后台线程续约；进程活着但 Redis 连续一个 TTL 续约失败时 fence 视为丢失，操作在下一
  次检查（例如发 `/sleep` 之前）停下。持锁进程死掉时 lease 过期，下一个写操作拿到锁，
  `supersede_stale_operations` 把它留下的 running 记录标为 superseded，journal 恢复结算它做了一半的事。

## 6. 接口变化

- **RetryLater 消失**：`PUT /v2/models/{m}/target` 和 `PUT /v2/bindings/{b}/power` 不再返回"稍后重试"
  的 409；并发请求在写锁上排队，等锁超过 `writer_lock_wait_s` 返回 409 `writer_busy`（已有的结构化
  错误码）。`/power` 对 wake journal 里有条目的 binding（崩溃或引擎读不到留下的）返回 409
  `lease_conflict`（reason `wake_in_progress`）。
- **`/target` 的 floor**：所有缩容路径都按 floor 收紧，返回 200，带 `taken`（实际睡下的数量）和
  `clamped_by_floor`（floor 留下了副本时为 true）；只有 `/power` 点名的 sleep 和 SafeScale 隐藏仍返回
  409 `floor_violation`。Pod LIST 或 Redis 读取失败时，在任何隐藏之前返回结构化 409 `routable_unknown`。
- **相对量**（10-04 改回 main 的语义）：`/target` 只接受绝对量 `wake_replicas`，不再接受 `delta`。
  v1 的 `/scale_service`、`/wake_up` 在请求到达时（锁外）按"醒着且未隐藏"的数量把相对量换成绝对目标，
  再调 `/target`；返回形状不变。原因见 §7"重放与重试"。
- **`POST /v2/transfers`** `{donor_model, receiver_model, count, sleep_path（默认 urgent；safescale_commit
  → 400）, donor_bindings?, avoid_gpus?}`。响应 `{transfer_id, pairs[{donor, donors, donor_binding_ids,
  receiver, receiver_binding_id, node, gpu_ids, status: done|donor_sleep_failed|receiver_wake_failed,
  error?, compensating_sleep?}], done, taken, clamped_by_floor, donors_slept, receivers_woken, unfilled,
  refusals[], skipped{}, picked[], phases_ms}`。至少一对完成或整体被 floor 收紧时 200；一对都没完成时
  409 `partial`（结构化，带完整响应）；参数错误 400。选对规则：receiver 的每张卡都要被 donor 覆盖
  （否则 `uncovered_gpu`），第三居民用 `/is_sleeping` 探测，fault hook 否决时换对，donor floor 和
  receiver 的 `max_awake_replicas` 生效；TP=2 的 receiver 要所有 donor 都确认已睡才唤醒；donor 睡失败
  不动 receiver，receiver 醒失败做补偿睡眠、donor 不回滚。没有 transfer journal，也不续接：崩溃后由
  sleep / wake journal 按物理状态结算，donor 最多白睡一次，controller 下一个 tick 重新规划。
  `/v2/operations` 的记录带 `details.transfer` 和 `phases_ms`（select、donor_sleep、receiver_wake、total）。
- **`GET /v2/state`**：每个 binding 增加 `routable`（与 floor 检查同一个函数），每个模型增加
  `routable`、`floor`、`floor_headroom`，顶层增加 `fetched_ms`、`floor_enforced`。只在 `GET /v2/state`
  时计算（一次 Pod LIST）；v1 API 和内部调用走原来的轻量路径，不做 Pod LIST。`gpus[].reason` 的
  `draining` / `waking` 现在只表示 journal 里留下的条目。
- **`GET /v2/sleep`**：`policy` 去掉 `hard_cap_s`、`reservation_ttl_s`、`commit_lock_wait_s`、
  `budgets_s`、`no_drain_paths`，增加 `sleep_mode_when_idle`、`writer_lock_wait_s` 和各项
  `worst_case_*`；去掉 `reservations`。`GET /v2/wake` 去掉 `running_here`，增加 `wake_call_timeout_s`。
- registry：新增 `service_manager.wake.call_timeout_s`、`service_manager.sleep.sleep_mode_when_idle`；
  默认值见 §5；弃用键见 §3。
- 孤儿租约（同一分支的独立提交）：supervisor 每轮回收没有任何 Pod 的 binding 的 GPU 租约
  （`starting` 和 `awake`，事件 `orphan_awake_lease_released`），Pod 列表读不到时不回收；
  `POST /v2/reconcile {"drop_missing": true}` 删 binding 时在同一次持锁内释放它的租约
  （响应 `released_leases`）。

## 7. 与 APA 臂的等价性

AIBrix 的 APA 臂（`pkg/controller/podautoscaler/workload_scale.go`）先 `POST /models_replicas` 读当前数，
算出 `delta = desired - current`，再 `POST /scale_service` 发 up / down；任何非 200 都记 `FailedRescale`
事件、把 `AbleToScale` 置为 false 并返回错误，由 controller-runtime 退避后重新 reconcile（重新读数、
重新算 delta）。它的 HTTP 客户端超时是 10 s。

- 单次调用的结果与改动前相同：`/models_replicas` 仍是"醒着且未隐藏"的数量（轻量路径，不做 Pod
  LIST）；`/scale_service` 的 delta 在请求到达时以同一个数量为基数换算（与 main 相同）；缩容走 `apa`
  路径，floor 收紧与改动前相同（原来就只对 `apa` 收紧）；不排空、abort 的行为与改动前的 no-drain
  路径相同（无在途请求时由 `mode=wait` 改成 `mode=abort`，效果相同）；放不满的扩容仍按原来的方式报错；
  返回 `{requested, actual}` 不变。`/wake_up` 仍是"醒着且未隐藏的数量 + 1，不超过 binding 数，超过时
  `delayed`"。
- **重放与重试**（10-04）：外部请求按绝对目标执行（不变量 I2：重放无害）。10-02 版曾把换算移到锁内，
  结果是重试会叠加：APA 的调用超过 10 s 客户端超时后，AIBrix 重新 reconcile 时 SM 端的第一次调用可能
  还在锁上排队或执行，此时 `/models_replicas` 还没反映它，APA 再发一次同样的 "+1"，两次在锁内依次
  换算成 base+1 和 base+2，多醒一个。在到达时换算后，两次都是 base+1，第二次是空操作。
- **剩下的竞态**：同一次 reconcile 内，两个**不同的**调用方（或不同的 delta）从同一个基数各自换算，
  后执行的会覆盖先执行的（lost update）。完整的修法是 APA 直接发绝对的 `desiredReplicas`（改 AIBrix
  的 Go 代码和 SM 的 v1 接口），推迟到 E1 之后。gateway 的 `/wake_up`（+1）同样在到达时换算；
  当前部署 `HOT_SWITCH=0`，不会调用它。
- RetryLater 的 409 不再出现：原来一个唤醒还在锁外进行时，APA 的调用会收到 409，AIBrix 记一次
  `FailedRescale` 并退避重试；现在调用在锁上等待后执行，少了这类失败事件和退避。
- 需要注意的一点：调用现在可能在锁上等待。APA 臂里 TRE controller 处于 observe，只有 APA 自己和
  supervisor 的维护操作会写，排队通常只有几秒；但如果等待加执行超过 AIBrix 的 10 s 客户端超时，
  AIBrix 会把这次调用记为失败并重新 reconcile，而 SM 端的请求仍会执行（同步处理函数不会因为客户端
  断开而取消）。下一次 reconcile 会按新的数量重新算 delta；若第一次还没执行完，第二次按同一基数换算出同一个
  绝对目标，不会叠加（见上面"重放与重试"）。
  这一点在改动前也存在（原来的等锁上限是 10 s），只是现在的等锁上限变为 30 s。

## 8. 测试

- 删除了被移除机制的测试（reservation、keepalive、拆分 sleep / wake、在途计入、RetryLater、
  transfer 恢复 / 续接、排空与 hard cap）；保留并移植了选对纯函数、floor 收紧、`/v2/state` 口径、
  单次 abort、`/is_paused` 回滚等测试。
- 新增 `tests/test_wholelock_20261002.py`：每条风险一个测试，并发用事件控制，不依赖时序。
- `make check-redis` 同时跑真实 Redis 的 Lua 测试和 sleep primitive 的端到端测试（真实 Redis + HTTP）。
