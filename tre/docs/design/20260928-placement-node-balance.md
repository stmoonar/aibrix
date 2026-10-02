# GPU 放置：节点均衡 + 预留 + 阶数封顶（2026-09-28）

分支 `tre/pd-placement`（基于 `tre/post-deploy-fixes-20260928` = main 90e9fb25）。唯一实现：
`tre/common/tre_common/gpu_placement.py`；registry `placement:` 段（`tre/common/tre_common/registry.py`
`PlacementConfig`）。

## 1. 问题

06d5bfeb 把放置统一成 buddy best-fit，但 `_free_run_order` 一路爬到整节点阶数，等于替一个并不存在的
TP4 模型预留整节点；排序 `(split_cost, address, index)` 又按自然节点序，唤醒总是先填第一个节点，释放总是
先放最后一个节点，负载集中到一个节点（8b33780d / 45f7bb81 的节点均衡在 06d5bfeb 中被一并删除）。

## 2. 策略对象

`placement_policy_from_registry(registry, awake_counts=None)` 是唯一构造函数，controller（planner / tick）
和 service-manager 都用它：

- `max_order = log2(registry 中最大 tp_size)`（当前 14b tp=2 → 1）。放置的 split cost 与释放的 merge gain
  都在 `max_order` 封顶，不再为更大的块买单。
- `reserve_blocks = placement.reserve_tp_pairs`（默认 1）：尽量保留这么多个完全空闲、对齐的 max_order 块
  （当前即 TP2 对）。**软约束**：只参与排序，从不拒绝放置；`max_order == 0`（全是 tp1）时为 no-op。
  调用方知道各模型 awake 数时（planner、SM 都知道），有效预留 =
  `min(reserve, Σ(max_awake − awake))`，求和只含 tp 等于最大值的模型（`PlacementPolicy.for_awake`）。
- `policy=None` = `BEST_FIT`（纯 buddy best-fit，无均衡/封顶/预留），只作为参考实现和无 registry 的单测默认值；
  生产路径总是带 registry 策略：tick 把策略挂到 `ClusterView.placement`，SM 在构造时从 registry 生成。

## 3. 排序键

> 2026-09-30（S5）起的排序键如下；2026-09-28 版本的 `node_block_load` 键已删除，见文末第 7 节。

放置（每个空闲候选块，最小者胜）：

`(violation, split_cost_capped, node_eff_load_after, same_model_on_node, address, index)`

- `violation = max(0, reserve − 放置后剩余的空闲 max_order 块数)`：集群处于/低于预留线时，不消耗空闲对的
  候选优先；高于预留线时为 0。
- `split_cost_capped`：候选块所在最大空闲块的阶数（封顶 max_order）减去块阶数（碎卡优先少）。它排在负载
  之前，所以 TP1 副本总是先填半用对的另一半，不会一个对放一个把 TP2 能用的对都打散（原来靠按预留粒度
  计量的 `node_block_load` 做到这一点）。
- `node_eff_load_after`：放置后该节点的 GPU 占用比例 + registry `placement.placement_penalty[节点名]`
  （跑负载生成器 / 控制面的节点可以配一个惩罚，节点名只出现在 registry 里）。
- `same_model_on_node`：同模型已在该节点占用的 GPU 数（把一个模型的副本分散到不同节点）。
- `address`（自然序 node, base GPU）：稳定的最后兜底。

释放（该模型持有的块，最大者胜，与放置互为镜像）：

`(merge_gain_capped, node_eff_load, same_model_on_node, address, -index)`

2 节点 × 4 卡、A/B tp1、C tp2、reserve 1，从空集群依次唤醒 A,B,C,A,B：
A→n1:0，B→n1:1（与 A 共用一对），C→n2:0-1，A→n2:2（同模型分散），B→n2:3（不去碰最后一个空闲对 n1:2-3）。
缩容 A 先释放 n2:2（负载更重的节点），即最后放上去的那个。测试见
`tre/common/tests/test_gpu_placement_policy.py`（含 3 节点 × 8 卡、各节点占用块数始终相差 ≤ 1）。

所有调用方都走这一套：planner `_SlotOccupancy.plan_wakes / free_groups / donor_slot_pods`、
`_try_plan_same_slot_high_shrink` 的同分决胜（原来按 serve_id 字符串，现在按释放策略）、
SafeScale 探测选 pod（`tick._pods_to_probe` → `release_order`）、SM `_wake_pick` / `SlotAllocator.find_slot` /
`release_order`。

## 4. defrag 默认关闭（与 v1 一致）

registry `placement.defrag.enabled: false`（默认）：

- planner 不再生成 `critical_tp_defrag` 迁移（事件 `defrag_disabled:<model>` + `capacity_blocked:<model>`），
  代码保留，打开开关即恢复。SM 没有自动 defrag 路径。
- 手动 `POST /v2/defrag`：关闭时直接 409 `{"reason": "defrag_disabled", "message": ...}`（不拿写锁，
  SM 日志 WARNING）；请求体带 `"force": true` 才执行（同样记 WARNING）。console `/api/ops/defrag` 透传
  `force` 并写审计日志。开关打开时 controller 的 defrag 调用不需要 force。
- 理由：v1 没有自动迁移；迁移要睡一个副本再冷启/唤醒另一个，代价以分钟计，而节点均衡 + 预留已经尽量避免
  TP2 无对可用的碎片。

## 5. 基线唤醒集（campaign）

`deploy/scripts/campaign_queue.py` 的硬编码 node9 基线（`DEFAULT_BASELINE`）已删除，改为
`generate_baseline(registry)`：按 registry 模型顺序、每模型一个副本，在空集群上用同一策略在 registry 渲染出的
binding 中放置，返回 `model -> binding_id`，运行时经 SM 状态解析为 serve_id。显式覆盖：manifest 的
`baseline:`（serve_id 或 binding_id），或 CLI `--baseline MODEL=ID`（可重复），`--registry` 指定 registry 文件。
当前 registry 生成：7b→node9:0，8b→node9:1，14b→node10:0,1（2026-09-30 的新排序键下不变）。

`campaign_queue.py --manifest M --freeze-baseline`（2026-09-30）：manifest 还没有 `baseline:` 时，把生成的
基线连同来源（`baseline_frozen`）写进 manifest 后退出；之后两臂都用这份固定布局，不随排序键或
`placement_penalty` 的变化漂移。已有 `baseline:` 的 manifest 不会被覆盖。

**警告：历史 campaign 的基线全部在 node9（7b gpu0、8b gpu1、14b gpu2-3），且历史运行期间放置策略是
整节点打包。用新基线 / 新策略跑出的结果与历史实验在布局上不可比，不能直接并表对比；需要可比时用
`--baseline` 显式指定历史基线，并注意运行中的扩缩放置仍按新策略。**

## 6. registry 变更需重启 SM

SM 只在启动时读一次 registry（`tre_sm/server.py` `load_registry` → `ServiceManagerV2.__init__` 派生
placement policy 与 `SlotAllocator` 规则），没有热重载。placement 相关的 registry 修改（`placement.*`、
`models[].tp_size`、`max_awake_replicas` 等缩放上限、`cluster.nodes`）必须
`kubectl -n tre-v2 rollout restart deploy/tre-v2-service-manager` 才对 SM 生效；console 的
controller restart 只重启 controller，两者都要重启，否则 controller 与 SM 按不同策略放置。

`models[].tp_size` 的合法性只有一个来源：`tre_common.registry.tp_size_error`（2 的幂、>= 1、不超过最宽节点、
不超过 binding 布局支持的 `MAX_SUPPORTED_TP_SIZE = 2`）。registry 加载时即拒绝（SM、controller、UI 都经它加载），
`validate()`、SM `SlotAllocator`、`bindings.feasible_slots` 与 `placement_policy_from_registry`
（后者只用 2 的幂 + 最宽节点这条通用规则）同规则报错；SM 不再对不支持的 tp_size 静默退回 best-fit。

## 7. 2026-09-30 更新：选卡与并行唤醒（S1–S6）

分支 `feat/placement-parallel-wake-20260930`。

- **S1 唤醒门**：gpu-truth 样本在 Redis TTL 内即可用，不再等新采样；早于本 SM 在该卡上最后一次 power
  变更（sleep commit、wake、启动、删除；每次变更发一个 refresh 请求并记下样本必须回应的 `refresh_seq`）
  的样本不可信。可信且超过唤醒阈值 → 409 `gpu_busy`；缺失 / 不可信 → 探测同卡其它居民的
  `/is_sleeping`：全睡 → 放行，有醒着的 → `resident_awake`，读不到 → `resident_loading`
  （节点整体没有样本时 `truth_unavailable`，节点级）。冷启动门仍等新样本。
- **S2 启动占位**：启动准入的 `starting` lease 不再 120 s 过期，一直持有到 Pod 收敛或 Pod 消失
  （orphan reaper 释放）；唤醒账面把其它 binding 的任何有效 lease 视为占用。
- **S3 结构化 409**：`{detail, error, reason, binding_id, node, gpu_ids, scope, blocking_binding_id,
  retry_after_s}`；controller 解析后对该卡冷却 `placement.wake_cooldown.gpu_s`（默认 30 s），节点级
  拒绝冷却整个节点 `node_s`（默认 60 s），下一 tick 换卡（事件 `gpu_cooldown` / `wake_refused` /
  `placement_retry`）。旧 SM 的纯文本 409 仍按原样重试、不冷却。
- **S4 补偿睡眠**：唤醒失败但引擎实际醒了 → 经 sleep primitive（path repair）补发 sleep 并确认。
- **S5 选卡**：纯容量唤醒由 SM 选卡（planner 的 pod 只是 hint），不可行时 SM 自己换卡并在响应
  `picked` 里回传；同卡接力仍由 planner 指定。`/v2/state` 增加 `gpus[]`（wakeable、reason、占用者、
  truth_source、truth_age_s）与 `nodes{}`；planner 只把 wakeable 的卡算作容量。
- **S6 三段唤醒**：锁内（账面、lease、journal、desired 意图）→ 锁外并发 `/wake_up` + `/is_sleeping` →
  锁内提交；`tre:v2:sm:wake_ops` journal 供崩溃恢复（启动时与 supervisor 每轮）。观测：`GET /v2/wake`、
  JSON 日志事件 `wake_start/wake_done/wake_failed/gpu_truth_fallback/startup_placeholder`，
  ops 记录 `details` 带 binding、placement、truth_source、phases_ms、error_code、compensating_sleep。
  （S6 的三段式已被 [20261002-sm-wholelock](20261002-sm-wholelock.md) 取代：唤醒从检查到提交都在
  一次持锁内完成，请求内的 `/wake_up` 仍并行；journal 只用于崩溃恢复。）

### 7.1 评审后的加固（2026-09-30）

- （本条中锁外阶段、waking lease、commit 交接的部分已被 [20261002-sm-wholelock](20261002-sm-wholelock.md)
  取代。）等待中的唤醒：waking lease 不再过期（TTL 0），只由 commit 或 journal 恢复释放；账面把 journal 里的 binding
  视为占卡；prepare 阶段即打 power 标记。commit 拿不到锁会先重试一次；任何未完成的 commit（锁、Redis、
  fence）都交给恢复，不再卡在本进程。恢复等待写锁（commit_lock_wait_s）；pod UID 变了（pod 被重建）
  则回滚、不碰新 pod；物理状态读不到时最多保留 `wake.recovery_unknown_attempts` 轮（pod 非 Ready 则立即）
  然后回滚并告警；`/wake_up` 传输超时、补偿 sleep 失败都保留条目（前者延迟 `wake.transport_recheck_s` 复核）。
  SM 重启时 bootstrap 从 journal 重建 waking lease，并给这些卡打 power 标记。reconcile 不改 journal 中的
  binding（不打 routable、不改 store），defrag 在有 journal 条目时拒绝。
- 启动 / 重启占位：starting lease 在 Pod 非 Ready 且不能确认醒着时，超过
  `startup_admission.placeholder_max_s`（默认 900 s）或期间再次重启（CrashLoop）即释放并告警。
  vLLM 主容器原地重启（restartCount 增加、Pod 不重建）会带着醒着的引擎回来而不经过启动门：supervisor
  每轮比较 restartCount，增加即给该 binding 加 starting 占位（`container_restart_placeholder`）；active 下
  等 `/is_sleeping` 可读后按 desired 收敛（desired 睡则经 sleep primitive 补睡），observe 下只占位 + 告警。
  **根治方案（本轮不做）**：让主容器的 entrypoint 每次启动都先走一次准入（与 init 门相同的
  `/v2/startup/admit`，按容器实例而不是 Pod UID 去重），这样原地重启也拿到 starting 占位、由收敛负责；
  supervisor 的 restartCount 比较只能在它看到的两次采样之间发现重启，SM 停机期间的重启会漏掉。
- 部分扩容：精确目标（APA `/scale_service`）放不满返回 409 `partial`（已醒的保持醒）；`at_least`
  （controller）返回 `unfilled` 与 `refusals`，controller 视为未完成并按 refusals 冷却对应卡。hinted 唤醒带
  `avoid_gpus`（在途接力占用的卡），SM 不会把 wake 换到这些卡上；接力本身始终是精确 binding。

### 7.2 复审加固（2026-09-30）

- 占位只在引擎容器（`vllm-openai`）处于 Waiting/Terminated 且读不到醒着时释放；Running 但未 Ready 超过
  `placeholder_max_s` 只告警（`startup_placeholder_overdue`）。supervisor 先跑重启守卫再跑回收，CrashLoop 中
  每次重新启动都会先拿到新占位。被释放的 binding 记为可疑：它的卡不信任 gpu-truth（强制走居民探测，
  Running 未 Ready 的居民即拒绝），一旦 `/is_sleeping` 可读就按 desired 收敛（读到睡则解除可疑）。
- 重启计数持久化在 `tre:v2:sm:restart_seen`，SM 重启后首轮即可发现停机期间的原地重启（从未记录过的 pod
  只记基线）。SM observe 模式下重启占位也做记账收敛（醒着转 awake lease、睡着释放），只是不补 sleep。
- lease 释放失败（fence 丢失、Redis 出错）时保留 journal 交给恢复；另有 supervisor 回收没有 journal 对应的
  waking lease（先读物理状态：睡或 pod 不在则释放，醒着转 awake lease，读不到则保留）。（waking lease 及其
  回收已被 [20261002-sm-wholelock](20261002-sm-wholelock.md) 取代。）
- `avoid_gpus` 只是派发那一刻的快照：派发之后才入队的接力不在其中。SM 在 donor 排空期间本来就拒绝在该卡
  上唤醒（sleep reservation 覆盖该卡）；如果 donor 刚睡下、receiver 还没唤醒，SM 的换卡可能先占到这张卡，
  这时 receiver 会收到结构化 409，controller 冷却这张卡并重新规划（donor 那次 sleep 就浪费了）。这个窗口
  很短，本轮不加处理，验收时留意 `wake_refused` 与 `placement_retry` 事件。（已被
  [20261002-sm-wholelock](20261002-sm-wholelock.md) 取代：sleep reservation 已删除，同卡接力由
  `POST /v2/transfers` 在一次持锁内完成，这个窗口不再存在。）

