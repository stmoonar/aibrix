# TRE 模型驻留池恢复与状态一致性设计（2026-07-14）

## 1. 事故与恢复结论

2026-07-13 的 node9 磁盘压力造成模型 Pod 大量 Evicted。磁盘压力解除后，Deployment 同时重建同卡上的多个 vLLM：7B 先占满 GPU1/2/3，Llama 和 14B 因冷启动需要约 33.5 GiB 空闲显存而持续失败，单 Pod 重启次数超过 300。同时 Redis `tre:v2:sm:state` 仍保存被驱逐前的 Pod 名称和 awake 状态，形成三类分裂：

- Kubernetes 中是新 Pod，SM 中是旧 `serve_id`；
- Pod annotation/label 表示 awake，但 vLLM 进程可能不可达或处于 CrashLoop；
- SM 期望的基线是 node9 的 7B/GPU0、Llama/GPU1、14B/GPU2-3 awake，实际显存却被 node9 的三个 7B 占用。

本次恢复按最后一次健康 `reset_state.json` 重建完整驻留池：7B 8 个、Llama 8 个、14B 4 个，共 20 个 Pod。恢复过程遵守以下顺序：

1. controller 保持 `observe`，确认 APA、safescale、orphan guard 均未运行。
2. 物理探测 node10 现存 Pod 的 `/is_sleeping`，确认全部 asleep。
3. 删除 node9 全部模型 Deployment 和旧 Pod。
4. 所有待建 Deployment 先以 `replicas=0`、`routable=false`、`state=sleeping` 创建。
5. 每次只放大一个 Deployment；冷启动前逐个探测其 GPU 上其他驻留 Pod，必须全部返回 `is_sleeping=true`。
6. 等待 vLLM HTTP 真正可用，立即调用 `/sleep`，再次物理探测成功后才启动同卡下一个模型。
7. 20 个 Pod 全部 physically sleeping 后调用 SM reconcile，清除旧 Pod 名称和 ghost binding。
8. 只通过 SM wake 标准基线：node9 7B/GPU0、Llama/GPU1、14B/GPU2-3。

最终独立验收：

- 20/20 Deployment 为 desired=ready=available=1；
- 20/20 Pod Running、Ready、restart=0；Failed/Pending=0；
- SM 为 7B `1 awake / 8 bound`、Llama `1/8`、14B `1/4`；
- 20 个 `/is_sleeping` 结果逐个与 SM awake、Pod annotation、`routable` label 一致；
- 三个 Service 各只有一个 endpoint，均指向标准基线 Pod；
- node9 GPU0/1/2-3 分别由标准基线模型占用；node10 四张 GPU 均为 sleeping 驻留占用；
- `DiskPressure=False`，controller 仍为 `observe`，三个 guard hash 均为 0。

恢复证据保存在 76 的 `/tmp/tre-model-recovery-20260714-200021/`。

## 2. 架构决策：冷启动必须归 service-manager 管

结论是 **必须归 service-manager 管**，但不是让 SM 取代 Kubernetes 的自愈能力，而是把职责分成两层：

- Kubernetes 负责 Pod/容器存活、节点故障检测和对象生命周期；
- service-manager 独占模型冷启动、GPU admission、sleep/wake、路由和 desired power state 的控制权。

普通 Deployment 的问题是：Pod 被驱逐后 ReplicaSet 会并发补 Pod，vLLM 容器一启动就加载模型。这个过程绕过 SM 的 GPU 冲突检查，因此只把“手工创建 Deployment”移入 SM 仍不够；必须保证 **新 Pod 在获得 SM 启动许可前不会加载模型**。

推荐最终形态：

```text
Kubernetes 创建空壳 Pod
        ↓
launcher 等待 SM 的 binding/gpu lease（默认不可路由）
        ↓
SM 原子获取该 binding 覆盖的全部 GPU lease
        ↓
确认同卡其他模型 physically sleeping + GPU headroom + Node 无压力
        ↓
launcher 启动 vLLM，SM 等待真实 HTTP readiness
        ↓
desired=sleeping: 立即 /sleep
desired=awake: 保持 awake
        ↓
SM 写 observed state，最后设置 routable
```

TP2 的两个 GPU 必须用一个 lease 原子获取，不能分卡获取。

## 3. 状态模型：分离 desired、observed 和 operation

当前一个 Redis hash 同时承担“期望状态”和“观测状态”，Pod 名称又被当作 binding 主键，Pod 重建后必然漂移。应拆为三层：

### 3.1 Stable Binding ID

主键改为稳定的 slot 身份，而不是 Pod 名：

```text
binding_id = <model>/<node>/<gpu_ids>
例如 dsqwen-14b/nscc-ds-4a100-node9/2,3
```

Pod name、UID、IP 只是 observed instance 字段。Pod 换代不应改变 desired binding，也不应要求 campaign manifest 固定旧 Pod 名。

### 3.2 DesiredState

由 controller/运维写入，SM 是唯一执行者：

- lifecycle：`Absent | Resident`
- power：`Sleeping | Awake`
- hidden/routable intent
- generation、写入者和原因

### 3.3 ObservedState

只能由 SM 通过 Kubernetes、vLLM 和 GPU truth 写入：

- Pod UID/name/IP、phase、Ready、restart count；
- `/is_sleeping` 物理结果；
- GPU used memory；
- annotation/label/EndpointSlice 状态；
- last_seen、error、observed_generation。

探测不可达时必须标为 `Unknown`，不能从 annotation 推断 awake，也不能用内存中的 auto-sleep 覆盖物理事实。

### 3.4 Operation Journal

冷启动、sleep、wake、迁移和修复都使用可恢复状态机：

```text
Pending → Quarantined → LeaseAcquired → Starting → Ready
        → Sleeping/Awake → Routable → Committed
```

每步记录 operation_id、fencing token、前置条件和结果。SM 重启后从 journal 继续，而不是重新猜状态。

## 4. Reconcile 的规则

建议把 `/v2/reconcile` 拆成 `audit` 与 `repair`：

- `audit` 只报告 desired/observed mismatch，不产生物理副作用；
- `repair` 在 leader、全局 operation lock 和每 GPU lease 下执行收敛。

核心不变量：

1. 每个 GPU 最多一个 `Starting | Waking | Awake` binding；TP2 同时占两个 key。
2. `routable = desired awake AND physically awake AND Pod Ready AND not hidden`。
3. physically sleeping 才允许同卡另一个模型进入 Starting。
4. Redis 中每个 observed Pod UID 必须在 Kubernetes 存在；缺失超过 grace period 必须转 Missing，不能永久保留 ghost。
5. Kubernetes 中每个 managed Pod 必须能映射到唯一 stable binding；未知 Pod 立即 quarantine。
6. annotation/label 只是缓存，不是真相源；`/is_sleeping` 和 GPU truth 才是物理证据。
7. `DiskPressure/MemoryPressure` 存在时暂停 repair 和冷启动，保持路由 fail-closed；压力解除并经过 hysteresis 后才串行恢复。

状态写入必须用 Redis Lua 或事务实现 generation CAS；当前 `GET version → DEL hash → HSET → SET version` 不是原子的，多请求并发会丢更新。SM 还需要 leader election 或单写者 fencing。

## 5. 当前代码中与本事故直接相关的缺口

1. `deploy/scripts/deploy_models.sh` 已注明 fresh bring-up 不能并发 apply，但 staggered 模式仍是 TODO。
2. 生成的 Deployment 默认 `replicas=1`、`routable=true`，新 Pod 在模型尚未准备好时可能进入 Service selector。
3. `ServiceManagerV2._create_and_wake_runtime_binding` 只支持“创建后成为 awake”，没有“创建 resident 并最终 sleeping”的一等操作。
4. Deployment 自愈冷启动绕过 SM；仅在 API 创建前检查 GPU headroom 无法约束 ReplicaSet 自动重建。
5. reconcile 对“Redis 有 binding、Kubernetes 无 Pod且 slot 未被占用”的情况保留 ghost，旧 Pod 名可永久存在。
6. `_auto_sleep_awake_conflicts` 只把内存中的 Binding 改为 sleeping，没有调用 vLLM `/sleep`，会制造新的逻辑/物理分裂。
7. Pod 物理探测不可达时会回退到 annotation；CrashLoop Pod 因此可能被当作 awake。
8. reconcile 的 observed 列表使用普通字符串排序，StateStore load 使用 natural sort；即使 `warnings=[]`、值完全相同也会误判变化并持续增加 version。本次延迟复核即出现 version 492→493。
9. `serve_id` 使用 Pod name，Pod 重建会使外部 baseline、action journal 和 Redis binding 全部失效。

## 6. 实施优先级

### P0：先封住再次并发冷启动

- 实现 `deploy_models.sh --staggered`，复用本次已验证的“scale 0 → 单个 scale 1 → HTTP ready → sleep”流程。
- 所有模型模板默认 `routable=false`，增加明确的 `provisioning` 状态。
- 增加 `POST /v2/fleet/repair` 异步操作：controller 自动进入 observe，按 GPU 串行修复，最后恢复目标 awake 集。
- Pod/Node watcher 遇到 DiskPressure 或批量 Pod UID 变化时自动 quarantine，禁止并发 cold start。
- 修复 reconcile ghost、物理 auto-sleep、排序 version churn；StateStore 改原子 CAS。

### P1：稳定身份与 desired/observed 分层

- 引入 stable binding ID；所有 controller action 改为 binding ID，Pod name 只作为 observed instance。
- Redis 拆成 desired、observed、operation journal；UI 同时展示三者和 mismatch。
- 加入每 GPU/TP slot lease、fencing token、leader election。

### P2：启动门控

- 增加受 SM 控制的 launcher/init gate，使 ReplicaSet 重建 Pod 时不会自动加载模型。
- 可进一步引入 `ModelBinding` CRD，由 SM controller 根据 CR 生成 Pod/Deployment 并持续收敛。

## 7. 验收测试

必须新增以下故障注入测试：

- 同一 GPU 上三个 sleeping 模型，删除全部 Pod，验证始终只有一个处于 Starting；
- TP2 与单卡模型同时重建，验证两个 GPU lease 原子性；
- SM 在 Starting、sleep 完成前、Redis commit 前分别重启，验证 journal 可恢复；
- Redis desired 与 annotation、`/is_sleeping`、GPU truth 分别制造冲突，验证 physical truth 与 fail-closed 路由；
- Node DiskPressure 下删除 Pod，验证不产生重建风暴；解除后按 GPU 串行恢复；
- 重复 reconcile 不改变 version；Pod 换 UID 后 ghost 在 grace period 后消失；
- 全过程 Service endpoint 不包含 provisioning、sleeping、Unknown 或未 Ready Pod。

## 8. 运维原则

在上述 P0/P1 落地前，禁止对 fresh fleet 直接执行 `kubectl apply -k deploy/models`。批量驱逐后的标准恢复步骤是：

1. controller 置 observe，清查 APA/guard；
2. 所有新 Deployment 先 scale 0、routable false；
3. 逐 GPU 串行冷启动并立即 sleep；
4. 全量物理探测通过后 reconcile；
5. 仅通过 SM wake 目标基线；
6. 同时核验 SM、Pod UID、`/is_sleeping`、GPU memory、routable label 和 Service endpoints。

## 9. 2026-07-15 P0 第一阶段实现记录

本阶段已在 76 的权威仓库实现以下最小闭环，目标是先消除本次事故中已经确认的“假一致”和并发冷启动入口：

1. `Binding.binding_id` 使用 `<model>/<node>/<gpu_ids>` 计算稳定槽位身份；`/v2/state` 和 reconcile 响应同时保留临时 `serve_id` 并新增 `binding_id`，兼容现有调用方。
2. 新增只读 `GET /v2/audit`，按 stable binding 对比 Redis、Kubernetes Pod、物理 `/is_sleeping` 和 routable label，报告 `ghost_binding`、`untracked_pod`、`instance_replaced`、`power_mismatch`、`physical_state_unknown`、`routable_mismatch` 等结构化问题，不写 Redis、不改 Pod。
3. `POST /v2/reconcile` 新增可选请求体 `{"drop_missing": true}`。默认仍保留暂时不可见的 binding，避免把短暂 Pending/重启误判为永久删除；fleet 恢复在 Pod 全部明确 scale 0 后使用严格模式清 ghost。
4. reconcile 与 StateStore 统一 natural sort，重复 reconcile 不再因 `pod-2`/`pod-10` 排序差异无意义增加 version。
5. 物理探测不可达时不再根据 annotation 放行路由，而是保持当前物理事实为 Unknown、将 binding hidden 并强制 unroutable。
6. 同卡出现两个 physically awake binding 时不再伪造其中一个 `awake=false`。后出现的 binding 保持真实 `awake=true`、标记 hidden 并摘路由，等待显式 repair；allocator 只在 reconcile 结果中允许表示这种非法现场，普通 wake/allocate 路径仍拒绝冲突。
7. 新增 `deploy/scripts/staggered_model_fleet.py`，并由 `deploy_models.sh --staggered` 调用。工具默认 dry-run；真实执行必须同时给 `--execute --confirm-reset-fleet`，并执行以下门控：controller 必须为 observe、所有 Node 必须 `DiskPressure=False`、先把模型 Deployment 以 replicas=0/hidden/routable=false 应用、严格清 ghost、逐 binding 启动、启动前确认重叠 GPU 上所有居民 physically sleeping、HTTP ready 后立即 `/sleep` 并复核，最后仅通过 SM wake 指定 stable binding，且以 `/v2/audit healthy=true` 收尾。
8. 恢复工具只 apply `default` 模型命名空间内的 Service/ReferenceGrant/Deployment，不修改共享 `aibrix-system` 中的 HTTPRoute。

标准基线的 dry-run 命令：

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre
./deploy/scripts/deploy_models.sh --staggered \
  --wake dsqwen-7b/nscc-ds-4a100-node9/0 \
  --wake dsllama-8b/nscc-ds-4a100-node9/1 \
  --wake dsqwen-14b/nscc-ds-4a100-node9/2,3
```

确认确实需要销毁并重建整个 resident pool 后，才追加：

```text
--execute --confirm-reset-fleet
```

第一阶段明确没有把以下事项伪装成已完成：StateStore 原子 CAS、desired/observed/journal 拆分、GPU lease/fencing、DiskPressure watcher、受 SM 控制的 launcher/startup gate、异步 `/v2/fleet/repair`。这些仍是 P0 后半与 P1/P2；在 startup gate 落地前，普通 Deployment 自愈仍可能绕过 SM，因此 fresh fleet 禁止直接 `kubectl apply -k deploy/models`。
