# 系统层 reissue sidecar + SM 统一排空（2026-09-24）

分支 `feat/reissue-sidecar-20260924`（基于 main 2caa0514）。**只写了代码和测试，没有部署。**
合入 main 后，在显式打开开关之前，线上行为**完全不变**：`make manifests` 的输出逐字节相同，
SM 的 hide / 排空 / 异步操作、controller 的逐次排空与异步派发全部默认关闭。

09-24 用户决定（本文 §3.3–§3.5）：
- **排空改为逐次调用决定**：hide 仍是 SM 全局开关；是否等在途请求结束由调用方在每次调用里给 `drain_s`。直接睡（planner 的各类 immediate、APA、defrag、admission/converge、fleet repair）一律 `drain_s=0`；只有 SafeScale commit 带预算。
- **SM 接口可异步**：target / power 支持 `?async=1`，返回 202 + operation id，controller 非阻塞派发、按 tick 轮询。

来源：`plan-20260923-icse-review-response.md` §10。不用 v1 的客户端 reissue，因为那是负载生成器的行为，会把失败“修”成成功。

## 1. 要解决的问题

缩容时 SM 直接调 `/sleep`。定制镜像 `vllm/vllm-openai:0.10.1-sleep` 的 `/sleep` 流程如下（已在运行中的 pod 里只读核对 `v1/engine/async_llm.py` 的 `pause_generation`）：

1. 先把引擎置为 `_paused=True`。
2. abort 所有在途请求。客户端收到的是 HTTP 200，最后一个 chunk 的 `finish_reason:"abort"`，后面跟 usage 和 `[DONE]`。
3. 等在途请求排空，再做 offload。
4. 返回被 abort 请求的快照，设计上只有 id 和长度，没有文本。

paused 期间新到的请求会挂在 `_pause_cond` 上，一直等到 wake。

**快照实际恒为 `[]`（评审 M1，09-24 在运行中的 7b pod 里只读核实源码）**：
- `AsyncLLM.pause_generation` 先调 `self.abort(request_ids)`：`output_processor.abort_requests` 立刻给每个请求的队列放入 `FinishReason.ABORT` 输出（客户端的 abort chunk 由此而来），再经 `engine_core.abort_requests_async` 到 `EngineCore.abort_requests` → `scheduler.finish_requests`。
- `finish_requests` 内部调 `get_unfinished_request_snapshot`，对每个请求调用 `_free_request`，后者经 `_free_blocks` 把请求从 `scheduler.requests` 删除。这份快照存进了 `EngineCore._last_abort_snapshots`，但**没有返回**。
- 随后 `pause_generation` 再调 `request_snapshot_async(request_ids)`，此时 `scheduler.requests.get(id)` 全是 None，于是返回 `[]`。`lxt-exm/model_switch2/sleep.txt` 里那份非空样例应当来自更早的构建。
- 结论：`/sleep` 的返回**不能**用来统计被中断的请求。sidecar 与 SM 都只把非空快照当排除依据或参考，并标注 `aborted_reliable=false`；权威计数是 sidecar 的 `tre_reissue_total`。

v2 replayer 不看 `finish_reason`，所以这类请求目前被记成成功（plan §11）。

## 2. 当前数据路径（从 main 重新推导；旧结论“主路径无 ext_proc”已过时）

- 两个臂的客户端都走 tre-v2 网关，NodePort 31094。
  - replayer 的 `run_trace` 和 campaign 默认发 `routing-strategy: least-gpu-cache` 与 `model` 两个头（commit 21a5139f）。
  - loadgen_v1 用 OpenAI SDK 的 `default_headers` 发 `routing-strategy`。
- 带 `routing-strategy` 头时，走 patch 进来的 `tre-original-route/<model>` 路由（`gateway-extproc.yaml`）：
  - ext_proc `tre-gateway-plugins:50052` 只在 `tre.aibrix.io/routable=true` 的 pod 中挑 `vllm:gpu_cache_usage_perc` 最低的，写入 `target-pod`。
  - 每个模型各有一个 ORIGINAL_DST cluster 负责转发到 `podIP:8000`；端口取 pod label `model.aibrix.ai/port`。
  - 请求体为 Buffered；响应体为 Streamed，SSE 行跨 chunk 时由插件重组（43aa0c31）。
  - 路由超时 150 s，连接超时 6 s。
- 不带该头时：per-model HTTPRoute → Service `default/<model>`（selector 含 routable=true）→ LEAST_REQUEST。
- **网关会拒绝 token-id prompt（已核实）**：`pkg/plugins/gateway/util.go:validateCompletionRequest` 把 `prompt` 反序列化进 Go 的 `string`，数组形式的 prompt 直接 400。chat 的 `messages` 以 raw JSON 解析，assistant 角色可以通过。所以续发只能用文本。
- 插件抓 pod 指标用的端口是 `model.aibrix.ai/metric-port`，没设时默认 8000。

## 3. 组件

### 3.1 sidecar（`tre/reissue/tre_reissue/sidecar.py`）

- 只依赖标准库和 aiohttp。vLLM 镜像里已有 python3.12、aiohttp 3.12，还有 uvloop；uvloop 可选，有就用。脚本通过 ConfigMap 挂载，不需要 build 镜像。
- vLLM 改监听 `127.0.0.1:8001`，sidecar 占 `:8000` 并透明代理**所有**路径，流式保持不变。
  - Service、网关 `target-pod`、SM（`/sleep` `/wake_up` `/is_sleeping` `/metrics`）、APA 与插件的指标抓取、readiness probe 都不用改。
  - pod label 仍是 `model.aibrix.ai/port=8000`。
- `POST /sleep`：
  - **fail-closed（评审 H3）**：请求缺少 `X-TRE-Hidden: 1` 时返回 409，根本不调用引擎。这个头只有在 SM 开了 `TRE_SM_HIDE_BEFORE_SLEEP`、先把 pod 摘掉路由之后才会带上；staggered 拉起脚本也带这个头，因为拉起阶段的 pod 本来就是 routable=false。开关为 `TRE_REISSUE_REQUIRE_HIDDEN`，manifest 里固定写 true。
  - **先**把本地标记为 sleeping，再转发。原因是 abort chunk 会先于 `/sleep` 的响应到达，否则会错过。
  - 非 2xx 或超时（300 s）时回滚标记。
  - 幂等（L4）：已经 sleeping 时再收到 `/sleep`，不改 epoch，也不重置计时。
- 2xx 的 `/wake_up` 清除标记。
- **与 vLLM 状态同步（H2）**：两个来源会纠正 sleeping 标记。
  - 每个经代理的 `/is_sleeping` 响应（SM 每次 sleep/wake 后都会调）。
  - 每 2 s 直连 `:8001/is_sleeping` 的探测。
  - 纠正受 epoch 保护，`/sleep` 或 `/wake_up` 进行中不纠正。典型场景：vLLM 重启后醒着，或 sidecar 重启时引擎已经在睡。
- 超时（L2）：`/health`、`/metrics` 等控制与元数据路径 60 s；`/sleep`、`/wake_up` 300 s；生成路径不设上限（由客户端和 Envoy 150 s 决定）。
- **快路径（M6）**：醒着时，如果一次读到的完整 SSE 事件里不含 `"abort"`、也不含 usage 对象，就原样转发，字节先存起来不解析；只有真要续发时才解析出已生成文本（每 64 KB 折叠一次，内存有界）。
- 自身接口：`GET /tre-reissue/metrics`、`GET /tre-reissue/state`；每次 reissue、sleep、状态纠正都在 stdout 打一行 JSON 日志。

### 3.2 manifests（`gen_model_manifests.py`，opt-in）

- registry 顶层 key `reissue_sidecar`，定义见 `tre_common.registry.ReissueSidecarSpec`。缺省或 `enabled: false` 时输出完全不变。
- 打开后：
  - 生成 ConfigMap `default/tre-reissue-sidecar`，内容为脚本。
  - 每个模型 Deployment 增加 `tre-reissue-sidecar` 容器：
    - 镜像同 `vllm_image`。
    - `:8000`，readiness 探测 `/health`，也就是经代理的 vLLM `/health`，参数和原来一样。
    - CPU request/limit 50m/500m（`cpu_limit` 可配），内存 64/256Mi。
    - env `TRE_REISSUE_REQUIRE_HIDDEN=true`（fail-closed，见 §3.1）。
    - `NVIDIA_VISIBLE_DEVICES=void`，不挂 GPU。
  - vLLM 容器去掉 readinessProbe 和 ports，改为 `--host 127.0.0.1 --port 8001`。
- SM 运行时创建 Deployment（defrag、create）走的是 `build_model_deployment`，读的是**同一个** registry key，所以迁移出来的 binding 和渲染出来的完全一致（有测试保证）。
- CLI 参数 `--reissue-sidecar` / `--reissue-gateway-url` 只用于 canary 渲染。正式上线必须改 registry，否则 SM 迁移出来的 pod 不会带 sidecar。

### 3.3 SM 统一 hide + 逐次排空（`TRE_SM_HIDE_BEFORE_SLEEP` 全局，排空按调用，默认都关）

**开关（评审 H3；09-24 改为逐次排空）**：
- `TRE_SM_HIDE_BEFORE_SLEEP=true`（**全局**）：每次 `/sleep` 前先 hide pod（routable=false 加 hidden 注解），等它退出可路由集合，`/sleep` 请求带 `X-TRE-Hidden: 1`。开 reissue sidecar 时必须开。
- **排空（等 vLLM running+waiting 归零）由调用方逐次决定**：`PUT /v2/models/{m}/target` 与 `PUT /v2/bindings/{id}/power` 的 body 可带 `drain_s`（秒）：
  - `0`：不排空。hide → 等不可路由 → 立即 `/sleep`，被 abort 的请求由 sidecar 续发。
  - `>0`：最多等这么久（再被 `TRE_SM_DRAIN_MAX_S` 与本次调用的截止时间截断），排空后再睡。
  - 不传：用默认值 `TRE_SM_DRAIN_BEFORE_SLEEP`（见下）。
  - `drain_s>0` 而没开 HIDE：该调用 400（给仍在接流量的 pod 排空永远不会收敛）；负数、NaN、inf、布尔值也是 400。
- **`TRE_SM_DRAIN_BEFORE_SLEEP` 迁移**：原来是“每次 sleep 都排空”的全局开关，现在**只是不传 `drain_s` 时的默认预算**：未设 / false / 0 = 不排空（默认）；true = 自动 `clamp(2·p95_e2e 或 TRE_SM_DRAIN_DEFAULT_S, TRE_SM_DRAIN_MIN_S, TRE_SM_DRAIN_MAX_S)`；正数 = 固定秒数。原来设成 true 的部署，行为只在“不传 `drain_s` 的调用”上保持；controller 开 `TRE_SM_CALL_DRAIN` 后会每次都传。
- **SM 内部的直接睡永不排空**（固定 `drain_s=0`）：defrag 迁移、startup admission / converge、fleet repair、过期标记恢复。APA 的 `/scale_service` 也显式传 0（见 §3.4）。
- **启动即失败（fail-closed）**：
  - 默认排空（`TRE_SM_DRAIN_BEFORE_SLEEP` 为 true 或正数）却没开 HIDE。
  - 默认排空非 0 却没设 `TRE_SM_ALLOW_DEFAULT_DRAIN=1`（评审 L4：不传 `drain_s` 的调用方会排空，而 APA 永远传 0，两臂不对称；必须显式确认）。
  - registry 里 `reissue_sidecar.enabled=true` 却没开 HIDE（`check_reissue_coupling`，在 `server.create_app` 连 Redis 和 k8s 之前检查，`ServiceManagerV2.__init__` 里再查一次）。
  - 开了 HIDE，但 vLLM 客户端的 `sleep()` 不支持 `hidden=`。
- 与 sidecar 的 409 一起，形成双保险：配置错了，要么 SM 起不来，要么 sidecar 拒绝 `/sleep`。
- 每个 sleep 结果（`sleep_outcomes[]`、sleep-audit）新增 `drained_s`、`drain_budget_s`、`drain_budget_source`（call / default）、`interrupted`（排空最后一次看到的 running+waiting，SM 侧估计）。

**三段式，排空期间不持全局写锁（评审 H1）**，适用于 `put_model_target` 和 `put_binding_power`：
1. **第一段，持锁**：
   - 先收尾过期标记。
   - 规划时，draining 的 binding 不算“在服务”的副本。扩容时优先**回收** draining 的 binding：删掉标记、取消 hide，不另外唤醒别的。
   - 本次调用里的 wake/create 都在这一段做完。实际上同一次调用不会既有 sleep 又有 wake，所以 wake 不会排在排空后面。
   - 对每个要睡的 binding：先写 draining 标记到 `tre:v2:sm:draining`（fencing token、instance、截止时间、`prior_hidden`；写入受 writer fence 校验），再写 hide 注解，并记 journal（phase `sleep_draining`）。
   - 这些 binding 在 legacy 状态里保持 `awake=True, hidden=True`，GPU lease 也不释放，因此 SlotAllocator、feasible-wake、create headroom、controller planner 都把这块 GPU 当作被占用（**draining 不算空闲容量**）。
2. **第二段，释放锁**：本次调用要睡的所有 binding **并行**（每个一个线程）等待不可路由，本次调用的排空预算 > 0 时再等排空。如果标记的 token 消失（被回收或被恢复），就提前结束。
3. **第三段，重新取锁**：`OperationBusy` 时重试，直到截止时间减去 reserve/2。
   - token 不见了：记为 `abandoned_reclaimed`，由新的所有者负责。
   - 期望状态已不是 sleeping：恢复可路由，记为 `abandoned_target_changed`。
   - 第二段出错：回滚为 awake 并可路由。
   - 其余情况：并行调 `/sleep`（带头），再用 `/is_sleeping` 核实物理状态。确认已睡 → 标为 sleeping 并释放 lease；确认仍醒着 → **回滚**为 awake、可路由，desired 也改回 awake；状态未知 → 保持 hidden（记为 `sleep_unverified`，fail-closed），交给 reconcile 处理。
   - 所有结果先落库（legacy 状态、标记、desired、journal phase `sleep_committed`），再对失败的情况抛出 `SleepCommitFailed`（HTTP 400）。评审 M5：不会再出现“hidden 又醒着”而无人处理的状态。

**单一截止时间（评审 M4）**：
- 每次调用只有一个截止时间 `TRE_SM_SLEEP_DEADLINE_S=240`，小于 controller 的 `TRE_SM_SLOW_TIMEOUT_SECONDS=300`。
- 第二段的窗口到截止时间减 `TRE_SM_SLEEP_COMMIT_RESERVE_S`（30 s）为止，所有 binding 的 unroutable 等待和排空都在这个窗口内。
- 到期仍没排空的，照常 sleep，并记下最后一次观测到的 running/waiting。

**controller 语义**：
- 同步接口（默认）：controller 不用改。action_queue 串行派发，一次排空仍会推迟它的其他动作，但不再阻塞其他 SM 操作（不再返回 409）。因为直接睡都是 `drain_s=0`，这段推迟只剩 SafeScale commit 那一次（≤ `TRE_SAFESCALE_COMMIT_DRAIN_MAX_S`=120 s）。
- 异步接口（`TRE_SM_ASYNC_OPS` + controller `TRE_SM_ASYNC`）：见 §3.4 / §3.5，排空不再推迟 controller 的任何动作。
- 只有开了 HIDE 时，`/v2/state` 才有以下变化：每个 binding 多一个 `draining` 字段，顶层列出 draining 标记；`models[m].awake` **不含** draining，另给 `draining` 计数。原因是 `sm_client.scale_model` 用 `awake + delta` 算下一个目标，这样算，+1 会回收正在 draining 的 binding，−1 也不会重复下发。

**同步调用等锁（评审 H2）**：
- 开了 HIDE 或 `TRE_SM_ASYNC_OPS` 后，排空阶段不持锁、异步 worker 在后台加锁，同步调用可能撞上正被短暂持有的 writer lock。
- 此时所有 `serialized_operation`（`put_model_routable`、defrag、reconcile、legacy target/power）以及三段式的第一段，锁忙时**有界等待** `TRE_SM_SYNC_LOCK_WAIT_S`（30 s，每 0.2 s 重试），不再立即 409；超时仍 409。
- 例外：异步 worker（有自己分阶段的重试）和 supervisor 发起的调用（startup converge、过期标记恢复）不等，下个 tick 再试。
- 开关全关时等待时长为 0，与 main 一样只试一次。

**其他路径**：
- `put_model_routable` 跳过 draining 的 binding，controller 的 UnhideAction 不会把它们放回路由。
- defrag 遇到 draining 的 binding 时拒绝，原因 `binding_draining`。
- fleet 漂移检测跳过 draining。
- **内联路径**（defrag 迁移、startup admission/converge、fleet repair）也做 hide → 等不可路由 → 带头 sleep，**不排空**（`drain_s=0`），在各自的锁里执行，受同一截止时间约束。这些路径很少触发，而且 admission/converge 睡的是刚启动、没有流量的 pod；失败时，只要确认 pod 仍醒着，就恢复 hide 之前的注解。

**过期标记恢复**：
- 由 supervisor 每个 tick、`reconcile()`、以及每次三段式调用开头触发。
- 满足任一条件即视为过期：超过截止时间 + `TRE_SM_DRAIN_STALE_GRACE_S`（30 s）——**本实例的标记也适用**，即使 token 仍登记为进行中（评审 H3：防任何泄漏路径）；或者是本实例写的标记，但已没有调用在处理它；或者标记的 `async_op_id` 所属异步操作已被判为孤儿（§3.4）。
- 评审 H3：异步操作的阶段通知放在 `_complete_staged` 的 try/finally 内，簿记（Redis 写）出错只记日志，不会让 token 泄漏。
- 处理原则：desired 仍为 sleeping，就补做 sleep；已是 awake，就取消 hide。

**审计（评审 M1）**：
- `/sleep` 响应的快照**始终**记入内存 ring（256 条，`GET /v2/sleep-audit`）。
- `[]` 记为 `aborted_count=None`；原始长度写在 `aborted_count_reported`，并标 `aborted_count_source=vllm_sleep_response_unreliable`。
- SM 侧的估计是 `interrupted_estimate`（排空最后一次观测到的 running+waiting）。
- 权威计数是 sidecar 的 `tre_reissue_*` 指标。
- 开关全关时，只做内存 append，不写 redis、不写 journal、不增加网络调用。

**开关全关时与 main 等价**：
- `put_model_target` 和 `put_binding_power` 只做 `drain_s` 校验（不传即 None），然后分派到 `_legacy` 方法；`_put_model_target_legacy`、`_put_binding_power_legacy`、`_put_binding_power_unlocked` 的装饰器 + 方法体与 main（merge base 2caa0514）AST 相等，由 `service-manager/tests/test_sm_legacy_ast.py` 固定（内嵌 main 源码，不依赖 git）。
- 不写标记，不带请求头，`/v2/state` 的结构不变（有测试）；`?async=1` 被忽略，按同步返回。

### 3.4 SM 异步操作（`TRE_SM_ASYNC_OPS`，默认关）

**接口**：
- `PUT /v2/models/{m}/target?async=1`、`PUT /v2/bindings/{id}/power?async=1`（或请求头 `Prefer: respond-async`）：
  - 同步部分只做校验和只读预览：模型 / binding 是否存在、`wake_replicas`、`drain_s`、扩容上限、预览规划（`_plan_model_target`，含 draining）。出错与同步接口一样返回 400 / 409（如 WakeConflict），**不创建操作**。
  - 通过后写一条操作记录（Redis hash `tre:v2:sm:async_ops`，SM 私有簿记，不走 writer fence），返回 `202 {async_operation: true, operation_id, status: "pending", plan, supersedes, status_url}`。
  - 后台 worker 执行与同步接口**同一段代码**（staged 或 legacy）。
- `GET /v2/operations/{id}`：先查异步记录，再查原来的 coordinator journal。字段：`status`（pending / running / succeeded / failed / superseded）、`phase`（queued / running / waiting_lock / draining / committing / 终态）、`bindings[]`（每个 binding 的 action、outcome、drained、drained_s、drain_budget_s、interrupted、error）、`summary`（woken / created / reclaimed / slept / abandoned / errors / drained_s / interrupted）、`error` + `error_code`、`latency_s`、`duration_s`、`supersedes` / `superseded_by`。
- `GET /v2/async-operations`：列出记录（最多保留 `TRE_SM_ASYNC_MAX_RECORDS` 条，超出时删最老的终态记录）；`?active=1` 只列 pending / running（controller 重启接管用）。
- 请求 body 可带 `meta`（JSON 对象，≤ 4 KB），原样存进异步记录；同步调用忽略。controller 用它存动作上下文和回滚意图（§3.5 M4）。
- 记录先落盘再入队：Redis 写失败时请求返回 500，不会跑一个没有记录的操作。终态记录写失败会重试 3 次，仍失败就留在内存、由心跳循环重写，不会一直显示 running（评审 L1）。
- 开关关着时 `?async=1` 被忽略，照常同步返回 200；controller 据响应里有没有 `async_operation` 区分。

**每模型规则（latest wins）**：
- 不同模型的操作并发执行，只在加锁的短阶段里被 writer lock 串行化。
- 同一模型同时最多一个操作处在**加锁阶段**（第一段的规划 + hide + wake/create，或第三段 commit）。
- 操作进入第二段（`draining`，不持锁）后不再占着模型：排队中的下一个操作立即开始。
  - 新的 target 若要扩容，它的第一段直接**回收**正在 draining 的 binding（复用 §3.3 的 reclaim：删标记、取消 hide）；旧操作的第三段发现 token 不见了，记 `abandoned_reclaimed`，旧操作终态为 `superseded`。
  - 因此一个立即可行的 wake 不会排在另一个 binding 的排空后面（有测试：排空中提交另一 binding 的 wake，排空结束前 wake 已完成）。
- 还没开始的排队操作会合并：同一模型的新 target 取代旧 target，同一 binding 的新 power 请求取代旧的；被取代的直接 `superseded`，从未执行。
- writer lock 忙（`OperationBusy`）：只在第一段落盘**之前**重试，最长 `TRE_SM_ASYNC_LOCK_WAIT_S`，超时记 `failed/writer_busy`。第一段之后若 commit 拿不到锁，按 §3.3 交给过期标记恢复，操作记 `failed`。

**SM 重启**：
- 活跃记录带 `instance` 和心跳（`TRE_SM_ASYNC_HEARTBEAT_S`）。
- supervisor 每个 tick（先于过期标记恢复）和 `POST /v2/reconcile` 会把**别的实例**留下、心跳超过 `TRE_SM_ASYNC_ORPHAN_AFTER_S` 的活跃记录标为 `failed`（`phase=orphaned`，记下 `orphaned_in_phase`）。评审 M1：本实例的记录一律跳过；每条候选在锁内重读一次再判定，期间被原主人更新过心跳或结束的不动：
  - 还在排队的：从未执行，什么都没改。
  - 已开始的：第一段已把期望状态落盘；它 staged 的 draining 标记带 `async_op_id`，所属操作被判孤儿后标记**立即**视为过期，由过期标记恢复按期望状态收尾（仍要睡就补睡，已改成醒就取消 hide）。wake / create 的物理状态由 reconcile 收敛。
- 心跳新鲜的别的实例（滚动更新期间新旧 pod 并存）不会被误判（有测试：两个实例共享 Redis，一个在排空中，另一个跑恢复不动它；它死后才被判孤儿）。
- **时钟（评审 L3）**：心跳和孤儿判定都用 wall clock，且由另一个实例比较。SM 应与 controller 同节点（pin node10，NTP 正常）。启动时比较本机时间与 Redis `TIME`，偏差 > `TRE_SM_ASYNC_MAX_CLOCK_SKEW_S`（5 s）或读不到 TIME，就**不启用** async ops（fail-closed，SM 照常同步服务，controller 自动退化为同步）。
- `DrainMarker.async_op_id` 为空时不写进文档，旧版 SM 回滚后仍能读标记。

**`/scale_service`（APA）保持同步**：Go controller-manager（TRE-PATCH P2-APA-001）调完 `/scale_service` 马上读 `/models_replicas`，只有同步返回才能看到结果；而它固定 `drain_s=0`，hide → 不可路由 → sleep 很快，同步不会拖住 APA。

### 3.5 controller 改动（`TRE_SM_CALL_DRAIN`、`TRE_SM_ASYNC`，默认都关）

**逐次排空（`TRE_SM_CALL_DRAIN`）**：
- 开了以后，controller 发出的每个 sleep 都带 `drain_s`：
  - SafeScale commit（`safescale_task` 把 commit 的 scale_down 加上预算）：`clamp(TRE_SAFESCALE_COMMIT_DRAIN_FACTOR × p95_e2e, _MIN_S, _MAX_S)` = 默认 `clamp(2·p95_e2e, 10 s, 120 s)`；窗口里没有 e2e p95 时用 `_DEFAULT_S`=30 s。理由：probe 期间 pod 已经隐藏了整个窗口，commit 时只剩残留在途请求。
  - 其余 sleep（critical / low-fairness donor immediate、idle proactive、model 级缩容）：`drain_s=0`。
  - 扩容请求不带 `drain_s`。
- 关着时什么都不发，SM 用自己的默认值，与 main 相同。
- 需要 SM 开 HIDE（否则 `drain_s>0` 会 400）。

**异步派发（`TRE_SM_ASYNC`）**：
- ScaleAction（model 级与 binding 级）带 `?async=1` 派发，拿到 202 就记下 operation id，模型**留在 inflight**，队列继续派发其他模型的动作；hide / unhide / defrag 仍同步。
- 每次 `drain_once` 先轮询已派发的操作：每个操作间隔 `TRE_SM_ASYNC_POLL_S`（1 s），每轮最多 `TRE_SM_ASYNC_MAX_POLLS_PER_TICK`（8）个，并发请求。404 连续 3 次记为 `operation_unknown`；网络错误下轮再试；超过 `TRE_SM_ASYNC_OP_TIMEOUT_S`（600 s）记 `op_timeout`。
- **完成时**才：写 F4 cooldown 的完成时间（冷却从操作完成算起，而不是派发）、放开 inflight、结算依赖。
- **按 binding 结算（评审 M2）**：
  - 扩容：`succeeded` / `superseded` 算成功（被 reclaim 的新操作就是要扩容）。
  - 缩容（"释放容量"：donor sleep、SafeScale commit）：只有目标 binding **全部** slept（或本来就已睡）才算成功。
  - 有 `abandoned_target_changed` / `abandoned_reclaimed`（含部分）→ 不算成功：不记 cooldown、不派 follow-up、**不回滚**（目标已被别人改了）。
  - `failed`、`rolled_back`、`sleep_unverified`、超时 → 失败，SafeScale commit 回滚 unhide。
- **依赖**：同一批里 donor-immediate 的接收方 wake 要等 donor 那次 sleep 成功才派发（它要用 donor 腾出的卡）；SafeScale commit 的 follow-up 扩容要等 commit 的 sleep 成功。被依赖的失败 / 超时 / 在 observe 下被跳过 → 依赖方 `dependency_failed`，不派发。评审 H1：被依赖的动作被 main 的 rescue 规则从队列里替换掉时也结算为失败；被依赖的 seq 既不在队列也不在跟踪中、又没有结果时立即判失败，接收方不会永远挂着。
- **防重复**：有活跃操作的模型在 `inflight_models()` 里，planner 不会给它规划；`submit` 按 main 的规则丢弃（`inflight`），main 原本放行的 rescue 动作在该模型有活跃操作时丢弃为 `active_op`；SafeScale 批次整批拒绝（`atomic_batch_conflict`），probe 不 resolve，下个 tick 重试（不丢一次性动作）。
- **supersede（评审 M3，已定）**：唯一例外是同模型的 **CRITICAL rescue 扩容**遇上该模型的活跃**缩容**操作（典型是 commit 排空中）：
  - planner 通过 `queue.supersedable_models()` 放行这个 receiver；
  - 动作改成 model 级 target 派发，SM 的 latest-wins 把正在排空、仍 awake 的 binding 回收，作为即时容量；
  - 旧操作不再跟踪，按"被 controller 取代"结算（不算成功、无 cooldown、无 follow-up、不回滚）。
  - 其他缩容类动作仍被 `active_op` / `inflight` 挡住。
- **一次性动作不丢（评审 H2）**：hide / unhide / SafeScale 来源的动作遇到 409、502/503/504 或超时，按 1→2→4→8→10 s 退避重试，最长 `TRE_SM_ONESHOT_RETRY_S`（120 s），observe 下保留；仍失败则打结构化告警 `sm_oneshot_action_failed`，交给 SM reconcile / controller HiddenOrphanDetector。routable 调用的超时 `TRE_SM_ROUTABLE_TIMEOUT_S`（45 s）大于 SM 等锁时间（30 s）。planner 动作不重试（下个 tick 重新规划）。
- **重启接管（评审 M4）**：
  - 异步派发时附带 `meta`（action_id、model、delta、pods、reason、source_loop；SafeScale commit 还有 `rollback_unhide`）。
  - 队列启动后第一次 drain 调 `GET /v2/async-operations?active=1`，把活跃操作按 action_id 分组，重新纳入跟踪直到终态（模型保持 inflight）；失败的 commit 照样回滚 unhide。没有 meta 的操作按 plan 方向重建为 ±1 动作，只保证 inflight 和跟踪。SM 不可达时每 5 s 重试。
  - planner 把 SM `/v2/state` 里 `draining=true` 的 binding 计为"正在离开"：不计入扩容上限的 awake 数，也不是可再缩容的副本（它是 hidden，本来就不在 routable 计数里）。
- **同步 + CALL_DRAIN 的多 pod commit（评审 L4）**：各 pod 的 power 调用并行发出，共用一个排空截止时间，而不是 N 倍预算。
- **planner 容量核算**（`tick.with_pending_ops`）：进行中的 wake = 即将到来的容量，其 binding 视为醒着（占卡）；model 级扩容先占该模型空闲卡上的 sleeping binding，再按 SM 分配器占空槽（占位 binding）；进行中的 sleep = 即将离开的容量，binding 在 SM 视图里本来就是 awake+hidden，卡保持占用，直到 SM 确认睡下。没有进行中的操作时视图原样不变。
- **失败 / 超时**：按同步失败处理（DispatchResult ok=False，不写 cooldown）；并限频（60 s）调一次只读的 `GET /v2/audit` 打日志。SafeScale commit 失败或超时时补发 unhide（等同 probe rollback；safescale 来源，observe 下保留不丢；SM 对已确认醒着的 pod 已经恢复可路由，unhide 幂等，且跳过仍在 draining 的 binding）。
- **observe / pause**：已派发的操作照常轮询到终态并记账，不会被遗忘；待派发的 planner 动作照旧丢弃，safescale 来源的一次性动作（含 commit follow-up 与 rollback unhide）保留到恢复 active 再派发。
- **SafeScale 状态机**：probe 在动作入队时 resolve（与 main 相同，保证一次性命令不重复）；commit 真正"完成"的判据是 SM 操作成功（此后才派发 follow-up、才开始 cooldown），失败则回滚 unhide。probe 记录里的 resolution 仍是入队时的决定，commit 的实际结果看 `sm_async_op_done` 日志。
- **指标 / 日志**：`ActionQueue.op_stats()` 按 `动作:方向:来源`（如 `scale:down:safescale`）累计 count / ok / failed / timeout / 延迟和与最大值 / drained_s 和 / interrupted 和；每个操作完成打一行 JSON `sm_async_op_done`（operation ids、状态、延迟、drained_s、interrupted、轮询次数）。SM 侧每个操作完成打 `sm_async_op`。
- SM 没开 `TRE_SM_ASYNC_OPS` 时会同步返回，controller 当作已完成处理，行为退化为同步派发，仍然正确。

**开关全关时与 main 等价**：`submit` / `drain_once` 分派到 `_submit_legacy` / `_drain_once_legacy`；这两个与 `_dispatch`、`_dispatch_binding_power`、`_record_done`、SM client 的 `scale_model` / `set_binding_power` 与 main AST 相等（`controller/tests/test_controller_legacy_ast.py`）。新加的 `ScaleAction.drain_s` 默认 None，不进 decision snapshot；`with_pending_ops` 在无进行中操作时返回原视图。

## 4. 语义

### 4.1 触发续发

同时满足以下三条才续发：
- **本地 vLLM 上游**的 chunk 出现 `finish_reason=="abort"`；
- 本地处于 sleeping，包括 `/sleep` 调用进行中；
- 客户端连接仍在（transport 未关闭）。

以下情况不续发：
- 客户端断开：写失败后 sidecar 会关闭上游，vLLM 因此 abort，但 sidecar 已不再读，这种 abort 永远不会触发续发。
- 本地未 sleeping 时的 abort：计为 `abort_not_sleeping` 并透传。
- 续发请求**从网关返回**的 abort：由下游 sidecar 决定，本层只透传，计 `failed`。
- 不支持的请求形态：`n>1`、`best_of>1`、`echo`、token-id prompt，一律纯透传。

### 4.2 构造续发请求（文本续写）

- **预算必须显式（M3）**：续发请求一定带剩余的 `max_tokens`。
  - completions 没带 `max_tokens` 时，按 vLLM 默认 16 计算。
  - chat 没带上限时，取 `max_model_len - prompt_len - 已生成`，两个数都来自本地 `/tokenize`。
  - 剩余 ≤0 时直接以 `length` 收尾，不再续发。
- **参数（L3）**：
  - 去掉 `logprobs`、`top_logprobs`、`prompt_logprobs`（跨段无法一致拼接），并在 `tre_reissue.dropped` 中标出。
  - 保留 `response_format` 和 `guided_*`。注意 guided 语法会从续写点重新开始，这点不保证。
  - `seed` 改为 `seed + depth`，避免续写重放第一段的随机流。
- `/v1/completions`：
  - `prompt = 原 prompt + 已生成文本`。
  - `max_tokens` 与 `min_tokens` 都减去已生成 token 数。
  - `echo=false`。
  - `ignore_eos`、`temperature` 等其余参数原样保留。
  - 已生成 token 数取第一段 abort 之后 usage 里的 `completion_tokens`。sidecar 对流式请求会**强制打开** `include_usage`；客户端没要 usage 时，这个 usage chunk 会被吞掉。
- `/v1/chat/completions`，默认 `render` 模式：
  - 用本地 API server 的 `/tokenize`（原 messages 和原 `add_generation_prompt`）加 `/detokenize` 拿到 vLLM **自己渲染**的 prompt 文本。这两个接口只用 tokenizer，引擎 paused 或 sleep 时照样能用。
  - 续发请求改为 `/v1/completions`：`prompt = 渲染文本 + 已生成文本`，`add_special_tokens=false`；采样参数按白名单拷贝；`max_tokens` 取 `max_completion_tokens`、`max_tokens` 或 `max_model_len - prompt_len` 之一，再减去已生成数。
  - 返回的 completion chunk 转成 chat chunk 形态再拼接。
  - **往返校验（L5）**：渲染出的文本会再用 `/tokenize`（`add_special_tokens=false`）重新分词，token 数必须与原来一致，否则回退到 `continue_final_message` 并计数 `render_fallback_roundtrip`。三个在线模型的 `clean_up_tokenization_spaces` 都是 False（已核对 tokenizer_config），vLLM 的 `/detokenize` 不能传这个参数，所以用校验来兜底。
  - **为什么不默认用 `continue_final_message`**：DeepSeek-R1-Distill 的 chat template 在渲染 assistant 历史时，会删掉 `</think>` 之前的全部内容，而且 generation prompt 是 `<｜Assistant｜><think>\n`，历史里的 assistant 却渲染成 `<｜Assistant｜>`。这两点都会让续写上下文和原生成对不上；transformers 甚至会直接报错“final message does not appear”。
  - `TRE_REISSUE_CHAT_MODE=continue_final_message` 保留为可选模式：追加 assistant 消息，设 `continue_final_message=true`、`add_generation_prompt=false`；render 失败时也会回退到这个模式。
- 已生成文本为空时（排队中被 abort 的请求，或 stuck 请求）：原样重发原请求。
- 续发**经本臂网关**：`TRE_REISSUE_GATEWAY_URL`，默认 `http://envoy-tre-v2-tre-aibrix-eg-161007f9.envoy-gateway-system.svc.cluster.local:80`，也就是 31094 背后那个 Envoy 的 Service。**绝不**直接发给 pod，也不发给模型 Service。
  - 原请求头：去掉 hop-by-hop 头、`x-request-id`、`target-pod`、`x-forwarded-*`、`x-envoy-*`，其余全部拷贝（`model`、`routing-strategy`、自定义头都会带上）。缺 `model` 头时从 body 里补。
  - 另加 `X-TRE-Reissue-Depth: n+1` 和 `X-TRE-Reissue-Origin: <pod>`。
- 深度上限 `TRE_REISSUE_MAX_DEPTH=3`。超限时如实透传 abort（abort chunk + usage + `[DONE]`），计 `depth_limit`。

### 4.3 拼接

- 吞掉第一段的 abort chunk、usage 和 `[DONE]`。abort chunk 如果带文本（detokenizer 在 finish 时会冲出残留文本），先把这段文本作为普通 chunk 发出去。
- 续流 chunk 的 `id`、`created`、`object` 改写成原值。chat 的 `role` 只保留最初那一个，空的 delta 丢弃。
- 结尾依次发出：
  - 合并后的 usage chunk（仅当客户端要了 usage）：`prompt_tokens` 取原值，`completion_tokens` 为各段之和，另附扩展字段 `tre_reissue{n, gap_ms, target, depth, kind, outcome}`；
  - SSE 注释行 `: tre-reissue {...}`；
  - `[DONE]`。
- 多级续发时 `n` 与 `gap_ms` 逐级累加。
- **TTFT 保持第一段的值**。续发间隙会计入 TPOT 和 E2E。这就是切换代价，本来就应该被看到。
- 兼容性（有测试）：OpenAI Python SDK 1.77 能解析拼接后的 chat 和 completion 流（扩展字段出现在 `model_extra`）；replayer 的 `_default_stream_call` 读到 `completion_tokens == max_tokens`；网关插件只解析 `data:` 行，会忽略注释行。
- 非流式：先缓冲。遇到 abort 时用非流式续发，合并文本、`finish_reason`、usage，并在响应顶层加 `tre_reissue`。
- **stop 字符串跨拼接点（M2）**：
  - vLLM 为了匹配 stop，会扣住最后 `max_stop_len-1` 个字符，直到结束时才吐出；abort chunk 会把这段文本冲出来。
  - 请求带 stop 时，sidecar 不立即转发这段文本（held），而是在续流的前 `max_stop_len-1` 个字符里一起匹配。
  - 若命中一个**起点落在 held 内**的 stop，就在该处截断（`include_stop_str_in_output` 时截到 stop 之后），`finish_reason=stop`，并关闭续流连接，让下游引擎 abort 剩余部分。此时续流的 token 数按收到的 chunk 数计，`tre_reissue.stop_at_seam=true`。
  - 完全落在续流内的 stop 仍交给下游引擎处理，续发请求保留原 stop 列表。
  - 非流式同理：在合并文本上检查跨越拼接点的 stop。

### 4.4 卡在 pause 后面的请求（stuck）

- 竞态：请求在 pause **之后**才到达 vLLM，所以不在 abort 列表里，会一直挂到 wake。
- 判定（M1）：
  - 本地在途表里的请求，在 sleep 开始前就已发给引擎；
  - 在 `/sleep` 返回后 `stuck_grace_s`（默认 0.5 s）内，一个响应字节都没收到。
  - 满足以上两条即判为 stuck：切断上游，原样重发，计 `kind="stuck"`。
- 为什么不会把“已 abort、但响应还没到”的请求误判：引擎在开始 offload 之前就已产出 abort 输出，而 `/sleep` 要等 offload 完才返回。所以到宽限期结束时，被 abort 的请求（包括非流式）早就收到字节了。测试里专门覆盖了非流式 abort 响应比 `/sleep` 晚到的情况。
- 快照只在**非空**时用来排除请求；`[]` 视为“未知”。
- 周期化（L1）：扫描在 `/sleep` 返回后执行一次，之后随监控循环每 ≤0.5 s 执行一次，并重复切断直到 handler 释放。因此由探测纠正出来的 sleeping 状态（没有经过 `/sleep`）也能释放卡住的请求。

### 4.5 sleeping 期间的新请求

- 直接转发到网关（`X-TRE-Forward-Hops+1`），不会挂在本地 vLLM 上。
- 请求可能被路由回正在睡的 pod：没有先 hide 时，routable=false 要等 `/sleep` 返回后才写，least-gpu-cache 还特别偏好刚释放 KV 的 pod。这种情况下会从第 2 跳起指数退避（0.25 s 起），最多 5 跳，然后返回 503。
- 现在这一点由代码强制：sidecar 拒绝没有 `X-TRE-Hidden` 的 `/sleep`；SM 在 registry 打开 `reissue_sidecar` 而 `TRE_SM_HIDE_BEFORE_SLEEP` 未开时拒绝启动（见 §3.3）。

## 5. 各环节的计数口径

| 位置 | 看到什么 |
|---|---|
| 客户端（replayer / loadgen_v1） | 1 个请求。TTFT 取第一段；E2E/TPOT 含续发间隙；`completion_tokens` 为各段之和；`finish_reason` 取最后一段；另有 `tre_reissue` 扩展字段。 |
| Envoy per-model cluster（`upstream_rq_total` 等，controller `gateway_health` 与 openloop sentinel 读这些） | 原请求 1 次 + **每次续发 1 次**。续发也吃准入与熔断配额，这是有意为之（Fable 判定：绕开网关会污染控制信号）。 |
| 网关插件 usage 计数（request trace） | 原请求上报的是**合并后**的 usage，续发请求再上报一次它自己的 usage，所以续发那段 token 被计了两次。这只影响插件自身的 token 统计，不影响路由（least-gpu-cache 看的是 pod 的 KV 使用率）。 |
| vLLM `/metrics`（TSS、APA、KV-Auto 的输入） | 每个 pod 只计自己实际算过的 token，没有重复。续发会触发 **re-prefill**，在 B 上多出 prefill 负载，这是真实代价。 |
| sidecar `/tre-reissue/metrics` | `tre_reissue_total{model,kind,outcome}`：kind 取 abort/stuck；outcome 取 ok/failed/depth_limit/client_disconnected/abort_not_sleeping/no_gateway。另有 `tre_reissue_gap_seconds` 直方图、`tre_reissue_sleep_forward_total{outcome}`、`tre_reissue_events_total{event}`（包括 state_corrected_*、sleep_rejected_not_hidden、sleep_failed、sleep_repeated、stuck_detected、stop_at_seam、render_fallback_*）、`tre_reissue_sleeping`、`tre_reissue_local_inflight`。**被中断请求的权威计数在这里。** |
| SM `/v2/sleep-audit` 与 journal | 每次 `/sleep` 的引擎快照（真实镜像恒为 `[]`，标注为不可靠，见 §1）；开 HIDE 时还有排空预算与来源、排空时长、是否排空，以及最后一次观测到的 running/waiting（作为 SM 侧的被中断估计）。 |
| SM 异步操作记录 / controller `sm_async_op_done` | 每个操作的 drained_s、interrupted（SM 侧估计）、延迟；controller `op_stats()` 按动作类型累计。 |

**在途占用翻倍（L6）**：续发期间，原请求的流（客户端 → Envoy → pod A 的 sidecar）仍然开着，续发请求（sidecar A → Envoy → pod B）又是同一模型 cluster 上的另一个活动请求。影响如下：
- 每个被续发的请求在续发期间会占用 per-model 熔断配额 `max_requests=4096` / `max_pending_requests=1024` 的 **2 个**名额。大批量同时 abort（N 个）会在短时间内额外吃掉 N 个，接近上限时可能触发 `upstream_rq_pending_overflow`。
- controller 的 `gateway_health`（SafeScale 供体健康守卫）计数：续发成功时 `requests` 加 1、`errors` 不变，所以错误率的分母被抬高，结果略偏乐观；续发失败时原请求仍是 200（abort 透传），不计入 errors。
- 插件的 per-pod 在途与 least-request 类路由会看到 B 多了一个请求，A 上的原连接只是在转发，不占 A 的引擎。
- openloop 的 pending-overflow 哨兵同样会看到这部分额外占用。
- 对比实验时，应同时报告 reissue 次数与峰值在途数。

## 6. 公平性

- 所有臂共用同一批 20 个 pod 和同一个 tre-v2 网关（X5），sleep 全部经过 SM。sidecar 与 SM 排空都是**运行时属性**，一旦打开对所有臂同时生效。
- 实验口径（plan §10）：默认所有臂开启；另外只在 TRE 臂的 t1/t7/t8 上做 on/off 消融。
- APA 缩容也经过 SM，同样先 hide、被续发，但**不排空**（`/scale_service` 固定 `drain_s=0`）；TRE 除 SafeScale commit 外的直接睡也是 0。两臂的差别只剩 SafeScale commit 的残留排空，这本来就是 SafeScale 机制的一部分。拿 v2 历史 APA 数字对比时必须注明：那时没有 reissue。

## 7. 已知限制

- **采样不确定性**：续发在另一张卡、另一个 batch 上继续生成。temperature>0 时，续写和“没被打断”的生成在分布上等价，但逐 token 不同；temperature=0 时也可能因 batch 不同出现微小数值差异。
- **文本边界重分词**：completions 续发时，`prompt + 文本` 重新分词可能在拼接点合并成与原 token 不同的切分，token 总数可能差 ±1。render 模式同理。
- **re-prefill 成本**：部署时关闭了 prefix caching（`--no-enable-prefix-caching`），续发要在 B 上重新 prefill 原 prompt 加已生成部分，占 B 的算力，也拉长 gap。
- **TPOT/E2E 含 gap**：gap 包括 abort 到续发首 token 的全部时间（网关路由、B 的排队、prefill），这是故意的。
- **Envoy 150 s 路由超时**作用于原请求的**全程**，续发的时间也算在里面。
- **不支持**：`n>1`、`best_of`、`echo`、token-id prompt、logprobs 的跨段拼接、reasoning parser（当前部署未开启）、tool call 流。
- **CPU（M6）**：在 76 主机上做的本地微基准（CPython 3.10、无 uvloop，fake 引擎每 20 ms 出一个 token，64/256 并发）：
  - 快路径之后，生成路径约 **70–78 µs CPU/chunk**，纯代理（`TRE_REISSUE_ENABLED=false`）约 64–68 µs，快路径之前约 88–98 µs。
  - 吞吐与直连相同；剩下的开销主要是 aiohttp 每个 chunk 的读写。
  - 默认 limit 提到 **0.5 核**（registry `reissue_sidecar.cpu_limit` 可配），约支撑每 pod 6–7k chunk/s。
  - **pod 内复测方案**：canary pod 上分别设 `TRE_REISSUE_ENABLED=true` 和 `false`，用 replayer 以 32/128/256 并发发 ignore_eos 长流。记录 sidecar 容器的 `container_cpu_usage_seconds_total` 增量除以 chunk 数（chunk 数由客户端统计 `completion_tokens` 之和），以及 `container_cpu_cfs_throttled_periods_total`，同时对比三种配置的 TPOT p99（无 sidecar / 纯代理 / 开启）。验收：没有 throttled period，TPOT p99 增加不超过 1 ms；否则提高 `cpu_limit`。
- 旧版关于“排空拉长 SM 调用、失败后保持 hidden”的限制已由 H1/M4/M5 解决，见 §3.3。

## 8. 开关

| 开关 | 位置 | 默认 |
|---|---|---|
| `reissue_sidecar.enabled` | `deploy/registry.yaml`（顶层）+ `tre-v2-registry` ConfigMap（供 SM 使用） | 缺省 = 关 |
| `reissue_sidecar.{gateway_url,max_depth,vllm_port,chat_mode,image,cpu_limit,...}` | 同上 | 见 `ReissueSidecarSpec` |
| `--reissue-sidecar` / `--reissue-gateway-url` | `gen_model_manifests.py` 命令行（仅 canary） | 关 |
| `TRE_REISSUE_ENABLED=false` | sidecar 容器 env | true；设为 false 时是纯代理，用于测开销基线 |
| `TRE_REISSUE_MAX_FORWARD_HOPS` / `_FORWARD_BACKOFF_S` / `_STUCK_GRACE_S` | sidecar env | 5 / 0.25 / 0.5 |
| `TRE_REISSUE_REQUIRE_HIDDEN` | sidecar env（manifest 固定 true） | true |
| `TRE_REISSUE_PROBE_INTERVAL_S` / `_PROXY_TIMEOUT_S` / `_CONTROL_TIMEOUT_S` | sidecar env | 2 / 60 / 300 |
| `TRE_SM_HIDE_BEFORE_SLEEP` | SM Deployment env，**全局**（开 reissue sidecar 时必须开，否则 SM 启动失败；任何 `drain_s>0` 也需要它） | 关 |
| `drain_s`（请求 body） | `PUT /v2/models/{m}/target`、`/v2/bindings/{id}/power`，逐次调用 | 不传 = 用下一行的默认值 |
| `TRE_SM_DRAIN_BEFORE_SLEEP` | SM env：**仅是不传 `drain_s` 的调用的默认预算**（false/0 = 不排空，true = 自动 clamp(2·p95)，数字 = 秒）；非零需要 HIDE **且** `TRE_SM_ALLOW_DEFAULT_DRAIN=1`，否则启动失败 | 关（0） |
| `TRE_SM_ALLOW_DEFAULT_DRAIN` | SM env：显式允许非零默认排空（评审 L4） | 关 |
| `TRE_SM_SYNC_LOCK_WAIT_S` | SM env：同步调用等 writer lock 的上限（只在 HIDE 或 ASYNC_OPS 开时生效） | 30 |
| `TRE_SM_DRAIN_{DEFAULT,MIN,MAX,POLL}_S` | SM env（MAX 也截断逐次预算） | 60 / 30 / 300 / 1 |
| `TRE_SM_SLEEP_DEADLINE_S` / `TRE_SM_SLEEP_COMMIT_RESERVE_S` | SM env（截止时间必须小于 controller 的 `TRE_SM_SLOW_TIMEOUT_SECONDS`） | 240 / 30 |
| `TRE_SM_UNROUTABLE_TIMEOUT_S` / `TRE_SM_DRAIN_STALE_GRACE_S` | SM env | 30 / 30 |
| `TRE_SM_ASYNC_OPS` | SM env：target / power 接受 `?async=1`，返回 202 + operation id（§3.4）；关时 `?async=1` 被忽略 | 关 |
| `TRE_SM_ASYNC_LOCK_WAIT_S` / `_HEARTBEAT_S` / `_ORPHAN_AFTER_S` / `_MAX_RECORDS` / `_MAX_CLOCK_SKEW_S` | SM env | 120 / 5 / 60 / 500 / 5 |
| `TRE_SM_CALL_DRAIN` | controller env：每个 sleep 带 `drain_s`（commit 带预算，其余 0）；需要 SM 开 HIDE | 关 |
| `TRE_SAFESCALE_COMMIT_DRAIN_{FACTOR,MIN_S,MAX_S,DEFAULT_S}` | controller env（commit 预算 = clamp(FACTOR·p95_e2e, MIN, MAX)，无 p95 用 DEFAULT） | 2 / 10 / 120 / 30 |
| `TRE_SM_ASYNC` | controller env：异步派发 + 轮询（§3.5）；SM 没开 `TRE_SM_ASYNC_OPS` 时自动退化为同步 | 关 |
| `TRE_SM_ASYNC_POLL_S` / `_OP_TIMEOUT_S` / `_MAX_POLLS_PER_TICK` / `_AUDIT_ON_FAILURE` | controller env | 1 / 600 / 8 / true |
| `TRE_SM_ONESHOT_RETRY_S` / `TRE_SM_ROUTABLE_TIMEOUT_S` | controller env（只在 ASYNC 或 CALL_DRAIN 开时的派发路径上用） | 120 / 45 |

**组合约束**：非零默认排空 ⇒ HIDE + `TRE_SM_ALLOW_DEFAULT_DRAIN=1`；`TRE_SM_ASYNC_OPS` ⇒ SM 与 Redis 时钟偏差 ≤ 5 s（否则自动不启用），建议 SM pin 到 node10 与 controller 同节点；sidecar ⇒ HIDE；`drain_s>0`（含 controller `TRE_SM_CALL_DRAIN` 的 commit 预算、SM 默认排空）⇒ HIDE；controller `TRE_SM_ASYNC` 只有配 SM `TRE_SM_ASYNC_OPS` 才真正异步；`TRE_SM_ASYNC_OPS` 不依赖 HIDE（HIDE 关时 worker 跑 legacy 路径，没有 draining 阶段，同模型操作纯排队）。

## 9. 上线步骤（需用户确认，且须在标定 M 收口之后，plan §13）

1. **合并**：本分支与 scaling 分支合成一个变更集。`make check` 必须通过；`make manifests` 在未打开开关时应无 diff。
2. **SM**：按 CLAUDE.md 流程 build 并滚动 SM 镜像（两处 tag 都要改）。env 先只加 `TRE_SM_HIDE_BEFORE_SLEEP=true`（`TRE_SM_DRAIN_BEFORE_SLEEP` 保持不设 = 默认不排空），观察 `/v2/sleep-audit` 与 `/v2/state` 的 draining 字段。先不开 sidecar。
   - **controller 逐次排空**：滚 controller 镜像，加 `TRE_SM_CALL_DRAIN=true`。观察 SafeScale commit 的 `drain_budget_s` / `drained_s` / `interrupted`，以及其余 sleep 的 `drain_budget_source=call`、`drained=false`。
   - **异步**：SM 加 `TRE_SM_ASYNC_OPS=true`（此时 controller 不开也无影响），手工 `curl -X PUT ...?async=1` 验一次 202 + `GET /v2/operations/{id}`；再给 controller 加 `TRE_SM_ASYNC=true`，观察 `sm_async_op_done` 日志的延迟分布、`op_stats()`、有没有 `op_timeout` / `dependency_failed`，并确认 commit 排空期间其他模型的动作照常派发。
   - 回滚顺序反过来：先关 controller 的 `TRE_SM_ASYNC`，再关 SM 的 `TRE_SM_ASYNC_OPS`（SM 先关的话 controller 只是退化为同步，也安全）。
   - 开 `TRE_SM_ASYNC_OPS` 前：确认 SM 调度在 node10（与 controller 同节点）、NTP 正常；SM 启动日志没有 "TRE_SM_ASYNC_OPS disabled"。
   - **supersede（M3）验收**：制造一次 commit 排空（commit 预算调大到 60 s）期间让同模型进入 CRITICAL，确认日志出现 `sm_async_op_superseded`、SM 记录里旧操作 `abandoned_reclaimed`、新操作 `reclaimed=[该 binding]`，且没有多余的 wake / create。
   - **重启接管（M4）验收**：commit 排空中重启 controller pod，确认日志 `sm_async_op_adopted`，该模型在操作结束前不被重复规划。
3. **canary（单 pod）**：
   - 用 `--reissue-sidecar` 渲染到 /tmp。只对 1 个空闲的 7b binding 应用它的 Deployment，同时 apply ConfigMap，并确认 SM 不会在 canary 期间迁移这个 binding。
   - 在该 pod 上确认 `/health`、`/metrics`、`/is_sleeping` 经代理正常，`/tre-reissue/state` 正常。
   - 对它发 30 个长流式请求（ignore_eos，max_tokens≈2000，走 31094 网关）。生成中途先把该 pod 设为 routable=false（或经 SM `PUT /v2/models/<m>/routable` 隐藏它），再**手动** `POST /sleep`，并带上 `X-TRE-Hidden: 1`；不带这个头会得到 409，这本身也是一项验收。
   - 验收：30/30 客户端 `finish_reason != abort`；每个请求 `completion_tokens == max_tokens`；`tre_reissue_total{outcome="ok"} == 30`；gap 的 p50/p99 有记录；网关 per-model `upstream_rq_total` 增加 60（30 + 30）。
   - 再测一轮不开 sidecar 的对照，abort 应为 30/30。
4. **开销微基准**：同一个 pod，分别测 sidecar 开（`TRE_REISSUE_ENABLED=true`）、纯代理（`=false`）、无 sidecar（原 manifest）三种情况，在 1/32/128 并发下比较 TTFT、TPOT 的 p50/p99，以及 sidecar 容器的 CPU 与 `container_cpu_cfs_throttled_periods_total`。结果写进论文披露（预计 TTFT +0.5–1 ms）。
5. **全量**：
   - 在 registry 设 `reissue_sidecar.enabled: true`，同步 `tre-v2-registry` ConfigMap，让 SM 运行时创建的 Deployment 也带 sidecar。
   - `make manifests` 并提交。
   - 用 `deploy_models.sh --staggered` 重滚 20 个 Deployment。它会一并 apply ConfigMap；每张卡同一时间最多 1 个在加载。
6. **回滚**：registry 改回 `enabled: false`，`make manifests`，同步 ConfigMap，`--staggered` 重滚。SM 排空通过 env 关闭即可。

## 10. 未决 / 未核实

- draining 标记的 Lua 写脚本只在 fake redis 上测过，没在真实 Redis 上跑过。
- controller 的 HiddenOrphanDetector 直接读 SM 状态哈希：draining 的 binding（awake 且 hidden）要满 600 s 才会告警，而排空截止时间是 240 s，按设计不会触发，但没实测。
- ~~同步 target 调用最长约 240 s，会推迟 controller action_queue 里的其他动作~~：已由逐次排空（直接睡 `drain_s=0`）+ 异步接口（§3.4 / §3.5）解决，但只在开关打开后生效；同步模式下 SafeScale commit 仍会推迟队列最多约 120 s + sleep。
- 异步操作与逐次排空只有单元 / 集成测试（fake vLLM、fake Redis、FastAPI TestClient），没在集群跑过：
  - `tre:v2:sm:async_ops` 的 HSET / HGETALL / HDEL 没在真实 Redis 上跑过；记录写入不走 writer fence（SM 单副本；滚动更新时新旧实例各写各的记录，靠心跳区分）。
  - SM 重启恢复只用构造的孤儿记录测过，没真杀过 SM pod；孤儿判定依赖 wall clock（心跳 60 s）。
  - controller 的 planner 容量核算（`with_pending_ops`）对 model 级扩容的占位槽是近似（SM 可能选别的槽），最多让一个 poll 周期内的规划偏保守。
  - 同步模式开 `TRE_SM_CALL_DRAIN` 时 profiler 不记 `dispatch` 事件（只影响 profiling 数据）。
  - 只有 CRITICAL rescue 扩容能 supersede 同模型的活跃缩容（M3）；其他情况模型在操作结束前一直 inflight。
  - 评审修复（H1–H3、M1–M4、L1–L4）都有先红后绿的单元测试，但同样只在 fake 上跑过：同步等锁对真实 Redis 锁租约的影响、重启接管时 SM 记录的 meta 能否完整还原、时钟偏差检查在真实 Redis 上的读数，都没验证。
  - M4 接管只能还原 meta 里有的上下文：旧版 controller 派发的、或非 controller 调用方的操作只按 ±1 跟踪，不会派 follow-up。
  - 同步等锁（H2）让 routable / defrag 等请求最长多挂 30 s。**组合风险**：SM 开了 HIDE 而 controller 两个开关都关时，controller 走 main 的旧路径，routable 超时仍是 5 s，SM 却可能等锁 30 s，controller 会先报超时、SM 随后照样执行（操作幂等，但 controller 记成失败）。这种组合下应把 `TRE_SM_SYNC_LOCK_WAIT_S` 设到 4 s 以下，或同时打开 controller 的 `TRE_SM_CALL_DRAIN`。
  - `op_timeout`（600 s）后 controller 放开该模型，但 SM 那边的操作可能还在跑；此时新的动作由 SM 的每模型规则排队 / 取代。

- 集群内调 `/tokenize` + `/detokenize` 渲染 DeepSeek-R1 prompt 的往返一致性（特殊 token 文本回解析成单个 token、没有重复 BOS）只在 fake 上测过，canary 时需在真 pod 上核对一次：比较 `/tokenize` 前后的 token 数。
- 续发请求不带原 `X-Request-Id`（sidecar 已丢弃），由 Envoy 或插件重新生成（未在集群核实）。续发 chunk 的 id 由 sidecar 改写回原值，所以客户端看不到差异。
- sidecar 在 0.5 核 limit 下、pod 内（有 uvloop）的真实 CPU 余量，复测方案见 §7。
- Envoy 默认的 `x-envoy-*` 头在 ORIGINAL_DST 路径上是否全部被剥掉，影响不大，sidecar 已统一丢弃。
