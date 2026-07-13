# TRE v2 安全与 Kubernetes 设计审查（排除磁盘告警）

日期：2026-07-13  
审查基线：`main@f87a5e67ddad4e3d7dea9550719f8ee70aee7cfa`  
验证：`cd tre && make check`，522 passed  
范围：`tre/common`、`controller`、`service-manager`、`ui`、`replayer`、`deploy`

## 1. 范围声明

本报告按当前决定，**不讨论本次 node9 DiskPressure、Evicted Pod 堆积、gpu-truth DaemonSet 被驱逐以及它们直接导致的在线异常**，也不提出本次磁盘事件的处置步骤。

但是，以下与磁盘事件无关、能够从代码和部署配置独立证明的问题仍在范围内：

- API/RBAC/网络安全边界；
- service-manager 并发、原子性、幂等与故障恢复；
- SafeScale 和 observe 模式的状态机正确性；
- Redis 持久化与单例滚动升级语义；
- reconcile 的 source-of-truth 和物理状态收敛；
- Kubernetes 调度、资源、探针、安全上下文和高可用设计；
- 配置一致性、命名空间隔离和实验入口漂移。

`nodeName` 作为长期 Kubernetes 设计问题保留在报告中，但不关联本次磁盘告警的现象或数据。

## 2. 总体结论

当前实现适合作为受控研究集群中的实验原型，但尚不满足可安全进入无人值守 active 模式或多租户 Kubernetes 集群的要求。最优先的问题不是算法，而是控制面安全边界、状态事务和故障恢复：

1. Console、service-manager、Redis 和 vLLM 管理接口缺少可信认证边界；
2. gateway-plugins 拥有集群级 Pod 写权限，service-manager 可跨命名空间修改共享资源；
3. service-manager 在持久化状态之前执行物理副作用，且 Redis 版本检查不是原子 CAS；
4. SafeScale 在 observe 模式下可能跳过 hide，却仍推进 probe 并保留后续缩容命令；
5. 多个安全门在依赖不可用时 fail-open；
6. 关键状态放在无持久卷、默认 RollingUpdate 的单实例 Redis 中；
7. reconcile 能保留 ghost binding，并把“只改内存状态”描述为 auto-sleep；
8. 模型 Pod 绕过调度器和资源记账，缺少探针、资源约束和容器硬化。

## 3. Critical / P0

### SEC-01：gateway-plugins 和 service-manager 权限过大

证据：

- `deploy/overlays/tre-v2/gateway-plugins.yaml:17-43`：ClusterRole 对所有命名空间的 Pod、ModelAdapter、HTTPRoute 授予 create/delete/get/list/patch/update/watch。
- `deploy/overlays/tre-v2/rbac.yaml:1-29`：controller 与 service-manager 共用高权限 Role。
- `deploy/overlays/tre-v2/rbac.yaml:56-105`：service-manager 可管理 `default` Deployment/Pod，并写 `aibrix-system` HTTPRoute。
- 线上只读授权检查确认 gateway-plugins ServiceAccount 可以删除 `kube-system` Pod，service-manager 可以修改 `aibrix-system` HTTPRoute。

影响：

- gateway-plugins 镜像、pprof/metrics 端口或依赖一旦被利用，可破坏任意命名空间工作负载；
- controller 获得了代码路径并不需要的 Kubernetes 写权限；
- TRE overlay 主动创建 `aibrix-system` 资源，与 ADR-0008 和“绝不改动 aibrix-system”约束冲突。

修复：

1. gateway-plugins 仅保留模型命名空间内 `pods get/list/watch`；删除全部写权限。
2. 若现有 gateway-plugins 二进制强制 watch ModelAdapter/HTTPRoute，增加 namespace/watch 配置或构建只读 scraper，不能用集群级写权限绕过启动报错。
3. controller 移除 manager RoleBinding，并设置 `automountServiceAccountToken: false`。
4. service-manager 使用独立、命名空间级 Role；长期把模型资源迁入 TRE 专用命名空间。
5. 删除 TRE 在 `aibrix-system` 中的 Role/RoleBinding。
6. 增加 ValidatingAdmissionPolicy，限制 service-manager 只能修改带 `tre.aibrix.io/managed=true` 的允许资源。

验收：

- `kubectl auth can-i delete pods -n kube-system --as=...tre-gateway-plugins` 必须为 `no`；
- controller ServiceAccount 对 Kubernetes workload API 不应有写权限；
- overlay 不再生成任何 `aibrix-system` 命名空间对象。

参考：[Kubernetes RBAC good practices](https://kubernetes.io/docs/concepts/security/rbac-good-practices/)。

### SEC-02：控制 API、Redis 和 vLLM 管理接口无认证/隔离

证据：

- `deploy/overlays/tre-v2/ui.yaml:43-60`：Console 通过明文 NodePort 30812 暴露。
- `ui/tre_ui/app.py:157-326`：无需认证即可切换 active/observe、修改参数、重启 controller、改副本、隐藏路由和执行 defrag。
- `service-manager/tre_sm/api/v2.py:525-583`：SM mutation API 无身份认证。
- `deploy/overlays/tre-v2/redis.yaml`：Redis 无 Secret、ACL/TLS 配置。
- 当前集群无 NetworkPolicy。
- vLLM `/sleep`、`/wake_up` 通过 Pod IP 明文调用。

影响：

- 能访问 NodePort 的主机可直接取得 TRE 操作员权限；
- 集群内任意 Pod 可伪造 controller/UI 调用 SM，或篡改 Redis 中的 mode、binding、metrics 和 probe；
- 任意可达 vLLM Pod IP 的工作负载可能直接 sleep/wake 模型。

修复：

1. UI 改为 ClusterIP，经 TLS Gateway 暴露，并接入 OIDC 或 mTLS。
2. 定义 viewer/operator/admin 三类权限；所有写操作记录经过认证的 actor。
3. Cookie 模式增加 CSRF token/Origin 校验；所有 mutation 增加速率限制和 idempotency key。
4. Redis 使用 Secret 管理 ACL/password；条件允许时启用 TLS。
5. 建立 default-deny ingress/egress NetworkPolicy，只开放以下边：
   - controller/UI -> service-manager；
   - controller/SM/UI/gpu-truth/scraper -> Redis；
   - service-manager -> vLLM 管理端口；
   - Gateway -> vLLM 推理端口。
6. vLLM 管理接口与推理接口最好拆分监听地址/端口；至少用 NetworkPolicy 保证只有 SM 可访问 sleep/wake。

验收：

- 未认证访问所有 `/api/ops/*` 和 `/v2/*` mutation 均返回 401/403；
- 非允许 Pod 无法连接 Redis、SM 和 vLLM 管理端口；
- SSE 连接和 mutation 均有并发/速率上限。

参考：[Kubernetes NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)。

### COR-01：service-manager 状态 CAS 非原子，物理副作用先于持久化

证据：

- `service-manager/tre_sm/state/store.py:48-62`：save 顺序为读取 version、DELETE hash、HSET hash、SET version，没有 Redis transaction/Lua。
- `service-manager/tre_sm/api/v2.py:94-184`：wake/sleep/create 在 `StateStore.save()` 前执行。
- `service-manager/tre_sm/api/v2.py:192-255`：binding power 和 routable 修改同样先写外部状态再提交 Redis。
- FastAPI 同步 handler 可在线程池并发执行，ServiceManagerV2 没有 mutation lock。
- `deploy/overlays/tre-v2/service-manager.yaml` 未设置 Recreate，默认 RollingUpdate 可短暂运行两个可写 SM Pod。

失败模式：

- 两个请求从同一 version 规划并执行不同物理动作；
- 两个请求选择同一 GPU；
- StateConflict 发生时，已经执行的 wake/sleep 不会回滚；
- 进程在 DELETE 与 HSET/SET 之间退出，状态部分丢失；
- SM rollout 期间两个实例并发操作同一 Redis/Kubernetes/vLLM 状态。

修复：

短期：

1. SM Deployment 设置 `strategy: Recreate`。
2. 所有 mutation 使用同一个进程内锁串行化。
3. StateStore 使用 Lua 或 WATCH/MULTI/EXEC，把 version 检查、binding 替换和 version 递增做成一个原子操作。
4. HTTP 超时后先重新探测物理状态，不得直接重复副作用。

长期：

1. API 只提交 desired state 和 operation ID；
2. 单一 leader worker 执行副作用；
3. 持久化 operation journal/outbox，包括 planned、running、applied、compensating、failed；
4. 所有动作支持幂等重放和补偿。

验收测试：

- 并发 target/power/defrag 请求只能有一个 operation 获得执行权；
- 在每个外部副作用之后注入进程崩溃，重启后都能确定性收敛；
- Redis 中永远看不到 bindings/version 半更新状态。

### COR-02：SafeScale observe 生命周期可能绕过 hide

证据链：

1. `controller/tre_controller/planning/safescale.py:86-114`：start_probe 先持久化 probe，再返回 hide command。
2. `controller/tre_controller/loops/tick.py:156-162`：调用 `queue.submit(actions)`，但忽略 accepted/rejected 结果，并把 `submitted` 记录为 action 数量。
3. `tick.py:263-272`：初始 hide 沿用原 action 的 fairness/rescue source_loop。
4. `controller/tre_controller/loops/action_queue.py:123-152`：observe 模式只保留 source_loop 为 safescale 的 resolution command；普通 fairness/rescue hide 会被 `observe_skipped` 丢弃。
5. probe 仍处于 active，可继续观察并产生 commit/rollback；resolution command 可能在恢复 active 后执行。

影响：

- 副本从未真正隐藏，probe 却可能判定成功；
- 恢复 active 后直接执行 scale-down，相当于绕过 SafeScale 的核心安全假设；
- queue 只在内存中，controller 重启会丢失已 resolved 但尚未 dispatch 的命令；
- 所谓 atomic safescale batch 只保证入队时无冲突，不保证多个 SM 动作全部成功或补偿。

修复：

将状态机改成：

`Planned -> HideSubmitted -> HiddenConfirmed -> Probing -> ResolutionPending -> Applied`

要求：

- observe 模式不得创建具有执行语义的 probe；若要保留 counterfactual，应使用独立的 simulated probe 类型；
- probe deadline 只能从 SM 确认 Pod 已从路由摘除后开始；
- submit rejected 时取消 probe，不能继续观察；
- resolution 只有在 command 被 SM ACK 且 observed state 收敛后才能标记完成；
- command 写入 durable outbox，重启后恢复；
- receiver 扩容和 donor 缩容采用可补偿顺序，优先保证 receiver 成功。

验收测试：

- observe 全生命周期内 Kubernetes、SM、vLLM 不发生任何副作用；
- controller 在 hide、commit、rollback 任意阶段重启后均能恢复；
- rejected hide 不会留下 active probe；
- batch 第二步失败时第一步会回滚或进入明确的 Degraded 状态。

### SAFE-01：关键控制门在依赖异常时 fail-open

证据：

- `controller/tre_controller/mode.py:39-47`：Redis 读取失败或 key 不存在时返回 active。
- `service-manager/tre_sm/api/v2.py:454-465`：GPU truth 为 None 时跳过 headroom 拒绝，允许 create。
- reconcile 的物理 probe 不可达时回退到 annotation/persisted 状态。

影响：

- Redis 重启、网络分区或数据丢失可能意外解除 observe；
- GPU truth 缺失时最需要保守处理，当前却允许复用 GPU；
- 不确定的物理状态可能被当作可执行状态继续规划。

修复：

- controller 缺省为 observe；Redis 异常时保持 last-known-safe mode 或 pause；
- active 使用带 generation/TTL 的显式 operator grant，不以 key 缺失表示 active；
- wake/create 必须要求目标 GPU truth 完整、新鲜，节点健康且无冲突；
- 缩容、迁移、GPU 复用一律 fail-closed；紧急扩容若要 fail-open，必须建立独立且明确的策略。

## 4. High / P1

### COR-03：直接 scale-down 的顺序不安全

`service-manager/tre_sm/api/v2.py:335-354` 对 sleep 的顺序是先调用 vLLM `/sleep`，成功后才写 annotation/routable label。UI、v1 compatibility API 和直接 SM 调用可以绕过 controller SafeScale。

统一的缩容协议应为：

1. `routable=false`；
2. 确认 Service/EndpointSlice 已摘除；
3. 等待 in-flight 请求清空或 drain timeout；
4. 调用 `/sleep`；
5. 验证 `/is_sleeping=true`；
6. 提交 observed state。

所有 sleep 入口必须复用同一协议，不能只依赖上层调用者先 hide。

### COR-04：defrag/create 缺少 saga 和补偿

证据：

- `service-manager/tre_sm/api/v2.py:367-387`：Deployment 创建后 readiness/wake/annotation 任一步失败会留下 orphan。
- `api/v2.py:389-433`：defrag 已 hide、sleep、delete 旧 Deployment 后，新 Deployment 创建或 wake 失败没有回滚。
- 最终 StateStore save 仍在全部物理副作用之后。

修复：

- 每次迁移持久化 from/to slot、旧/new serve ID 和阶段；
- 创建新副本并验证后，再删除旧副本；如果容量不允许重叠，必须至少提供重建旧位置的补偿；
- 失败后进入可观测的 Degraded/Compensating 状态，不得让 Redis 继续声称旧副本 awake；
- 定期 reconciler 接管 orphan cleanup。

### COR-05：reconcile 保留 ghost binding

证据：

- `service-manager/tre_sm/state/reconcile.py:112-121`：persisted binding 没有 Pod observation 时仍被保留，只产生 warning。
- `service-manager/tre_sm/ops/k8s_ops.py:97-130`：只观察 Running Pod，不观察 Deployment desired/available、Pod UID、Ready condition 和失败原因。

修复：

- 同时观察 Deployment、ReplicaSet、Pod UID、Pod Ready 和物理 sleep 状态；
- binding 增加 Pending/Starting/Awake/Sleeping/Lost/Failed 状态；
- Pod 不存在且 Deployment 不存在时，经过短 grace 后清理 binding；
- Deployment 存在但 Pod 不健康时标记 Lost，不计入 awake/available；
- `get_state()` 同时返回 desired、observed、available，并暴露 condition/reason。

### COR-06：auto-sleep 只改逻辑状态，没有物理 sleep

`service-manager/tre_sm/state/reconcile.py:174-195` 在发现同 GPU 多个 awake binding 时，只把后一个 binding 改成 `awake=False`；没有调用 vLLM sleep。路由可被隐藏，但显存占用和物理冲突仍存在。

修复：

- 发现冲突先 quarantine 全部冲突 Pod；
- 用确定性规则选 winner；
- loser 执行 hide/drain/sleep/verify；
- 物理 remediation 失败时保持 conflict condition，禁止把结果描述为 auto-slept。

### STATE-01：Redis 承载关键状态，却按无状态 Deployment 部署

`deploy/overlays/tre-v2/redis.yaml` 没有 PVC、AOF、认证、探针或备份，Deployment 默认 RollingUpdate。rollout 时新旧 Redis Pod 可能同时被 Service 选中，而新 Pod拥有空状态。

修复：

- 短期：Recreate、PVC、AOF、readiness、认证；
- 长期：外部 HA Redis，或 StatefulSet + PVC；
- 将可丢弃 metrics 与不可丢失的 controller mode、SM state、SafeScale journal 分离；
- 对关键状态建立备份、恢复和 schema migration 测试。

参考：[Kubernetes StatefulSet](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)。

### CFG-01：动态 registry 只对 controller 生效

证据：

- controller：`TRE_REGISTRY_PATH=/etc/tre/registry.yaml`，挂载 ConfigMap；
- service-manager：`TRE_REGISTRY_PATH=/app/tre/deploy/registry.yaml`；
- UI：`TRE_REGISTRY_PATH=/app/tre/deploy/registry.yaml`；
- Console 允许修改 `min_replicas/max_replicas`，但 restart endpoint 只重启 controller。

影响：

- controller、SM、UI 对 min/max、拓扑和模型元数据可能使用不同版本；
- controller 可规划出 SM baked registry 拒绝的目标；
- UI 的 applied hash 只表示 controller 被重启，不代表整个控制面配置一致。

修复：

- 将动态控制参数与不可变 topology/model identity 分离；
- 如果 min/max 保持动态，则 controller、SM、UI 必须读取同一版本化 ConfigMap，并进行协调 rollout；
- applied status 应记录每个组件确认的 config generation，而不是单一 Redis hash。

## 5. Medium / P2：Kubernetes 规范与工程质量

### K8S-01：`nodeName` 绕过调度器

`deploy/gen_model_manifests.py:251-275` 生成固定 `nodeName`，绕过 scheduler 对 taint、cordon、压力和资源容量的判断。`NVIDIA_VISIBLE_DEVICES` 配合 `resources: {}` 也绕过 Kubernetes GPU 资源记账。

修复方向：

- 使用 required node affinity，而不是 nodeName；
- CPU/RAM 继续由默认 scheduler 负责；
- awake GPU 独占用 Lease/CRD + admission 或 scheduler plugin 管理；
- sleeping Pod 的 RAM、共享 GPU 槽位也应成为显式容量模型。

参考：[Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)。

### K8S-02：模型 Pod 无资源 requests/limits

所有模型 manifest 都是 `resources: {}`，同时使用 20Gi memory-backed `/dev/shm`，sleeping weights 还会占用大量主机 RAM。调度器无法预留真实容量，Pod QoS 低，节点级资源竞争不可预测。

修复：

- 按模型校准 CPU/RAM request；
- memory limit 要覆盖权重 sleep 到 RAM 和 `/dev/shm`，避免设置过小导致 OOM；
- 设置 ephemeral-storage request/limit；
- 把 RAM 中可同时保留的 sleeping 模型数纳入 planner 容量约束。

参考：[Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)。

### K8S-03：缺少有效探针

controller、SM、UI、Redis、gateway-plugins 都没有 liveness/readiness/startup probe。UI 和 SM 的 `/healthz` 无条件返回 true，无法反映 Redis、Kubernetes API、配置、物理状态或路由可用性。

建议：

- liveness 只判断事件循环/进程是否卡死；
- readiness 判断关键依赖和本组件能否安全服务；
- startup probe 覆盖 registry 加载、Redis/Kubernetes 初始化和 vLLM 长启动；
- controller readiness 应暴露 mode、metrics freshness、SM state freshness；
- SM readiness 应验证 Redis CAS、Kubernetes API 权限及 reconcile 最近成功时间。

参考：[Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/)。

### K8S-04：容器和 Pod 缺少安全上下文

Dockerfile 没有 USER，manifest 没有 runAsNonRoot、seccomp、capability drop、readOnlyRootFilesystem 或 allowPrivilegeEscalation 限制。Redis、gpu-truth、controller 等不需要 API token 的 Pod 仍默认挂载 ServiceAccount token。

修复：

- 建立非 root UID/GID；
- `allowPrivilegeEscalation: false`；
- `capabilities.drop: [ALL]`；
- `seccompProfile.type: RuntimeDefault`；
- 适用组件启用 readOnlyRootFilesystem，并显式提供 `/tmp` emptyDir；
- 不调用 Kubernetes API 的组件设置 `automountServiceAccountToken: false`；
- 为 tre-v2 namespace 逐步启用 Pod Security `warn/audit`，稳定后 enforce baseline/restricted。

参考：[Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)。

### K8S-05：整个 `/data` 以读写 hostPath 暴露

模型 Pod 将主机 `/data` 整体读写挂载。容器或依赖被利用后可以修改共享权重和其他数据；hostPath 也绕过常规存储隔离和配额。

修复：

- 优先使用 ReadOnlyMany PVC；
- 若必须 hostPath，仅挂载模型精确目录，设置 `readOnly: true` 和 `type: Directory`；
- 每个模型使用独立 volumeMount，不暴露整个 `/data`。

参考：[Kubernetes hostPath warning](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath)。

### K8S-06：单例和单故障域

controller、SM、Redis、UI、gateway-plugins 均为单副本，控制组件又固定在同一节点。controller 用 Recreate 避免双写是合理的短期措施，但带来升级停机且没有自动 leader failover。

修复：

- controller/SM 引入 `coordination.k8s.io/v1 Lease` leader election；
- 两副本 standby，仅 leader 执行 mutation；
- UI 可无状态多副本；
- Redis 使用独立 HA/StatefulSet；
- 增加 topologySpread/anti-affinity 和 PDB。

参考：[Kubernetes Leases](https://kubernetes.io/docs/concepts/architecture/leases/)。

### K8S-07：镜像供应链与运行时镜像过大

三个 Python Dockerfile 安装 `requirements-test.txt`，把 pytest/httpx 带入生产；基础镜像和 Python 依赖只有范围版本，无 digest、lock/hash、SBOM 或签名，容器默认 root。

修复：

- 生产镜像只安装 runtime requirements；
- 生成带 hash 的 lockfile；
- base/runtime 镜像固定 digest；
- 多阶段构建、非 root、最小 COPY 范围；
- CI 生成 SBOM、扫描 CVE，并对镜像签名/验证。

### ISO-01：命名空间隔离与代码默认值漂移

问题：

- TRE 模型资源实际位于 `default`；
- overlay 在 `aibrix-system` 创建 Role/RoleBinding；
- `service-manager/tre_sm/server.py:75-76` 的缺省 route namespace/gateway 仍为 `aibrix-system/aibrix-eg`；
- `replayer/run_trace.py` 和 `run_comparison.py` 缺省 gateway 仍是共享 31592，而 TRE 专用 gateway 是 31094；
- `deploy/gateway-hardening` 包含修改共享 aibrix-system EnvoyProxy 的操作。

修复：

- 将 TRE 模型、Service、Route、控制面统一放入 TRE 专用命名空间；
- 删除 production code 中所有 aibrix-system fallback；
- replayer 强制显式 `--arm` 或 `--gateway-url`，禁止静默默认到共享平面；
- CI 扫描渲染产物，拒绝除白名单 baseline 文件外的 `aibrix-system` 对象；
- 共享 GatewayClass/EnvoyProxy 如确需修改，应由独立集群平台变更管理，不属于 TRE overlay。

### OBS-01：审计和可观测性不足

- UI 参数审计使用无界 Redis list，未记录可信 actor，写失败被静默忽略；
- 多处 `except Exception` 只降级或吞掉错误，缺少统一 error condition/metric；
- warning 与 observed condition 没有结构化状态 API；
- action queue 的 dispatch 失败不持久化，重启后不可追溯。

修复：

- 审计写入有保留策略的持久化流/日志系统；
- 记录 actor、request ID、old/new generation、result；
- 为 controller/SM 暴露 Prometheus 指标：reconcile error、state conflict、ghost/lost binding、operation age、queue depth、probe phase、mode-read failure；
- 关键异常使用 Kubernetes-style Conditions：type/status/reason/message/observedGeneration/lastTransitionTime。

## 6. 测试缺口

现有 522 个测试覆盖面较广，但部分测试固定了需要改变的危险行为：

- `controller/tests/test_mode.py` 期望 Redis 错误和 key 缺失时 active；
- `deploy/tests/test_gen_model_manifests.py` 期望生成 nodeName；
- `service-manager/tests/test_reconcile.py` 期望保留无 Pod 的 persisted binding；
- reconcile auto-sleep 测试只验证逻辑 binding，没有验证物理 `/is_sleeping`；
- ActionQueue observe 测试覆盖 resolution command hold，但没有覆盖“初始 hide 被跳过、probe 仍推进”的完整生命周期。

必须新增：

1. SM 并发 mutation 和 Redis 原子 CAS 测试；
2. wake/sleep/create/defrag 每一步后的 crash fault injection；
3. observe 模式 SafeScale 全生命周期零副作用测试；
4. hide submit rejected、controller restart、outbox replay 测试；
5. Pod 删除/重建/UID 改变/物理 probe 不可达时的 reconcile 收敛测试；
6. RBAC negative `kubectl auth can-i` 守卫；
7. NetworkPolicy 未授权连通性测试；
8. kubeconform/server-side dry-run、Pod Security 和容器安全上下文守卫；
9. Redis/SM rollout 与状态恢复测试；
10. controller、SM、UI 读取同一 registry generation 的一致性测试。

## 7. 推荐实施顺序

### Phase A：安全边界与阻断高风险路径

1. RBAC 最小化，移除 aibrix-system 权限和无用 SA token；
2. UI/SM/Redis/vLLM 管理接口认证与 NetworkPolicy；
3. mode、GPU truth、物理状态全部改 fail-closed；
4. SafeScale observe 模式禁止产生 operative probe。

### Phase B：状态正确性

1. SM Recreate + mutation lock；
2. Redis 原子 CAS；
3. operation journal/outbox；
4. ghost/lost binding 状态和 reconcile 收敛；
5. 统一 hide/drain/sleep/verify 协议；
6. defrag saga 与补偿。

### Phase C：Kubernetes 化

1. nodeName 改 affinity，建立显式 GPU Lease/调度扩展；
2. 补资源、探针、安全上下文、只读存储；
3. Redis 持久化/HA；
4. controller/SM Lease leader election；
5. 统一命名空间和配置 generation。

## 8. 完成标准

在以下条件全部满足前，不应把系统定义为 production-ready：

- 非授权主体无法调用任何 mutation 或写 Redis/vLLM 管理接口；
- gateway-plugins 和 controller 无集群级写权限；
- 任意单点进程崩溃不会留下无法恢复的物理/Redis 分裂状态；
- observe 模式在所有情况下保持零副作用；
- SafeScale 只有在 hide 已确认并完成完整观察窗口后才能 commit；
- Redis/SM/controller rollout 不产生双 writer 或状态丢失；
- desired、observed、physical、routable 四层状态能够自动收敛；
- 所有 workload 具备有效 probes、资源声明和最小安全上下文；
- 渲染产物不写 aibrix-system，不默认使用共享 gateway；
- 新增故障注入、并发、安全和集成测试全部通过。
