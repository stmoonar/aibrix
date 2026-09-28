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

放置（每个空闲候选块，最小者胜）：

`(violation, node_block_load_after, same_model_on_node, split_cost_capped, node_gpu_load_after, address, index)`

- `violation = max(0, reserve − 放置后剩余的空闲 max_order 块数)`：集群处于/低于预留线时，不消耗空闲对的
  候选优先；高于预留线时为 0，由节点均衡决定。
- `node_block_load_after`：放置后该节点 max_order 对齐组中"非全空"的比例。**按预留粒度计量是有意的**：
  填满半用对的另一半不增加节点负载，所以 TP1 副本会两两成对；若用朴素的 GPU 占用比例，TP1 副本会
  一个对放一个，把每个 TP2 能用的对都打散。`max_order == 0` 时退化为 GPU 比例。
- `same_model_on_node`：同模型已在该节点占用的 GPU 数（把一个模型的副本分散到不同节点）。
- `split_cost_capped`：候选块所在最大空闲块的阶数（封顶 max_order）减去块阶数。
- `node_gpu_load_after`、`address`（自然序 node, base GPU）：稳定的最后兜底。

释放（该模型持有的块，最大者胜，与放置对称）：

`(merge_gain_capped, node_block_load, same_model_on_node, node_gpu_load, address)`

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
当前 registry 生成：7b→node9:0，8b→node9:1，14b→node10:0,1。

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
