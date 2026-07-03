# TRE 系统重构总计划（交付执行模型，无人值守执行）

> 本文档是 TRE（基于 AIBrix 的 MaaS 多模型热切换扩缩容系统）的完整重构方案。
> 读者是**自主执行重构的 AI 模型**。执行模式为无人值守连续运行：
> **任何情况下不要停下来等待人类确认**，遇到无法裁决的问题按第 9 章守则记录后跳过，
> 继续做不受影响的部分。
>
> 本文档已经由架构师做完关键技术决策（第 4 章）并给出目标代码框架（第 5 章）。
> 执行时**不要重新发明架构**：接口、目录、schema 按第 5 章实现；实现中发现框架与
> 事实冲突（如新版 AIBrix 接口对不上），在 `DECISIONS.md` 记录偏离原因后做最小偏离。
>
> 配套阅读：`START.md`（背景）、`RUN_SYSTEM.md`（启动流程）、`MACHINE_CONFIG.md`
> （机器配置）、`TRE_MODEL_PARAMETERS.md`（参数）、`paper/`（论文，与实现有差异，
> 以代码为准，差异要记录）。

---

## 0. 术语对照

| 术语 | 含义 |
| --- | --- |
| TSS / TRS | 同一个指标。代码中叫 TRS，论文/新命名为 TSS。计算：加权吞吐 Y_m（prefill×w_p×(1-kv_hit) + decode×w_d）÷ 控制队列 Q_ctl（running + swapping + waiting×lambda_wait，下限 qmin），再做 EMA 平滑（alpha）。实现：`python/tre/controller/trs.py` |
| theta_m | 每模型拟合出的"临近违规"TRS 阈值，用于归一化 |
| Z_m | Z_m = TRS / theta_m，跨模型可比的容量健康度 |
| tau_crit / tau_low / tau_high | Z_m 分档阈值 → CRITICAL / LOW / HEALTHY / HIGH 四态（`paper_state.py::ModelState`），START.md 里的 delta_crit/delta_high 即此 |
| 热切换 | vLLM sleep mode（/sleep、/wake_up），1–2s 内权重换入/换出显存，pod 进程保留 |
| service manager | 自研 pod 睡眠/唤醒/GPU 占位管理服务，`python/service_manage_aibrix/` |
| 快环/慢环 | rescue loop（默认 5s，CRITICAL 紧急扩容）/ fairness loop（默认 10s，HIGH→LOW 容量再平衡），`dual_cadence.py` |
| SafeScale | 缩容前先隐藏路由观察一个窗口，无 SLO 违规才真正缩容，否则回滚 |
| APA | AIBrix 基线 autoscaler（KVCache 指标），已改造为走 service manager sleep/wake，`pkg/controller/podautoscaler/podautoscaler_controller.go`（`APA_SCALE_SLEEP_MODE`） |

## 1. 系统现状地图

```
本仓库（基于 AIBrix v0.4.0）
├── pkg/plugins/gateway/           # 网关插件（路由 + wakeup.go 唤醒逻辑 + Redis 指标写入）— Go
├── pkg/controller/podautoscaler/  # APA 基线（已改造为 sleep/wake 路径）— Go
├── python/tre/                    # TRE 控制器（Python）
│   ├── controller/                #   main.py(1096行) planner(1160行) safescale(1369行)
│   │                              #   trs.py dual_cadence.py paper_state.py state_store.py ...
│   ├── monitor/collector.py       #   Redis 指标采集（1168行，SCAN 全量扫键）
│   ├── executor/scaler.py         #   K8s 执行残留（兼容用）
│   ├── calibration/               #   拟合流水线（fit 脚本 1939 行）
│   └── configs/                   #   model_slo_profiles.json、seed_calibration.json
├── python/service_manage_aibrix/  # service manager（FastAPI，slot-aware 选卡，内存态）
├── CustomTraceGenerator/          # trace 重放器 + 实验编排 + 画图
├── config_tre/                    # 网关/TRE controller/APA 部署清单
├── config_model_replicas/         # 模型 deployment 生成器 gen-deployment.sh + run_all.sh
└── paper/                         # 论文 LaTeX
```

数据流：vLLM pod `/metrics` → 网关插件写 Redis（key 形如
`aibrix:pod_histogram_metrics_<pod>_<ts>`，时间戳做 key 后缀）→ TRE collector 每轮
SCAN 按窗口取 → 计算 TRS/Z_m → planner 出计划 → HTTP 调 service manager →
service manager 调 vLLM `/sleep` `/wake_up` 并维护 GPU 占位 → 网关按 routable pods 路由。

## 2. 环境说明与硬性红线

### 2.1 环境

- 两台 4 卡 A100(40G)：`192.168.223.75`、`192.168.223.76`，其上一套 k8s。
- 旧系统（可运行基线）：76 的 `/root/aibrix-main`（AIBrix 0.4.0，与本仓库对应）。
- **新版工作目录：76 的 `/data/nfs_shared_data/xxy/aibrix`**（最新版 AIBrix 检出），
  所有代码修改与测试在此进行。
- 模型权重：`/data/nfs_shared_data/` 下（NFS 共享）。
- 端口约定：网关 port-forward 8888；service manager 8001；Redis `aibrix-redis-master:6379`。
- 旧可视化工具：76 的 `/root/aibrix-main/python/gateway_visualizer`（做 UI 前先读）。

### 2.2 红线（禁止事项）

1. **禁止改动 k8s 集群本身**：不重装/升级 k8s、不改 kubelet/containerd/网络插件、
   不 drain/cordon 节点、不动 GPU 驱动和 nvidia runtime。
2. **禁止删除或移动模型权重**；禁止向 NFS 写大量临时文件，实验输出写自己的工作目录。
3. **禁止修改 `/root/aibrix-main`（旧系统）**：整个重构期间冻结，只读参考。
4. **禁止 `docker system prune -a`、`kubectl delete ns` 等大范围清理**；删除任何 k8s
   资源前先 `kubectl get -o yaml` 备份到文件。
5. **新旧系统不能同时抢 GPU**：任何要起模型 pod 的操作前，用 `nvidia-smi` +
   `kubectl get pods -A -o wide` 确认无冲突；检测到未知进程占卡就跳过该步并记录。
6. 新系统 k8s 资源一律用**新 namespace（`tre-v2`）和新镜像 tag**，与旧系统区分。
7. 长时间操作前检查 `df -h` 和 `redis-cli info memory`。
8. 不确定某操作是否影响机器环境时，**在工作日志中记录问题并继续做其它不受影响的
   部分**，不要猜、不要冒险执行。

## 3. 执行模式与验证原则（本次运行的边界）

本次是**无人值守的一次性长时间运行**，目标是完成代码重构并保证代码正确性，
**不是**复现实验数据。因此：

1. **不跑长时间实验**。以下全部列入"遗留运行清单"（第 10 章），本次只保证相应代码
   正确、能启动、有单测：拟合训练网格（约 10 小时）、7 条 30 分钟正式 trace、
   TRE vs APA 回归矩阵、消融矩阵。
2. **验证金字塔**（每个阶段的"验证"步骤只允许用这三层，全部可离线/短时完成）：
   - L1 静态：`make check` = ruff lint + mypy（宽松模式）+ 单元测试 + `docker build`
     （若环境无 docker 则降级为语法/导入检查并记录）+ `kustomize build` 全清单；
   - L2 回放：用夹具数据（录制的 Redis dump 或合成数据）离线驱动模块，断言输出
     （见 8.2 夹具制作方法）；
   - L3 冒烟（条件执行）：仅当能访问集群且红线 5 检查通过时，跑 ≤5 分钟的部署冒烟；
     不满足条件就跳过 L3，在 WORKLOG 标注"L3 未执行，原因"，**不阻塞后续阶段**。
3. **拍板已完成**：第 4 章的决策不需要再写"方案对比文档"，直接按决策实现；
   每个决策附带了"若事实推翻前提怎么办"的降级路径。
4. **兼容性开关原则**：所有改变数值口径的改动（分位数插值、窗口语义、TRS 输入）
   必须带开关且默认保持旧口径（原因：theta_m 等参数是旧口径拟合的，本次不重跑拟合，
   新口径参数等人类触发重拟合后再切换）。

## 4. 关键架构决策（已拍板，直接执行）

### D1 起 pod 与扩缩容方式：保留"预置 pod 池 + service manager"，重写内核

**问题**：现在用 deployment 起满 pod、发 /sleep 压成池子，扩缩容不走 k8s scale 而走
service manager 调 /sleep /wake_up，并用进程内锁管理 GPU 占位。这是不是最佳实践？

**结论**：**架构方向是对的，本轮保留这个形态，但内核按 5.3 节重写。** 论证：

- 热切换的本质约束是"vllm 进程必须常驻、权重驻留 RAM、唤醒只能回到原来绑定的 GPU"。
  k8s 原生 scale/HPA 的语义是创建/销毁 pod，天然与此矛盾——scale down 杀掉的进程
  再也无法 1-2s 唤醒。所以"不用 k8s scale 接口"不是 hack，而是必然。
- 业界对这类"pod 活着但逻辑上缩容"的标准做法就是**路由层摘除 + 池化管理**
  （serverless 推理平台的 warm pool 同理）。我们缺的不是换架构，而是三点工程质量：
  分配器有碎片化缺陷、状态不持久、k8s 侧不可观测。这三点在 P4 修。
- CRD + operator（把 sleep/wake 做成自定义资源的 reconcile）是更 k8s native 的形态，
  但本轮**不做**：工作量大、调试成本高、且不解决任何算法问题。作为妥协，P4 会把
  serve 状态和 GPU 绑定写到 pod annotation（`tre.aibrix.io/state`、
  `tre.aibrix.io/gpu-ids`），让 kubectl 能看到全部状态——拿到 operator 的可观测性
  好处而不付 operator 的成本。
- **降级路径**：若 P2 调研发现新版 AIBrix 已内置 sleep-mode 编排（社区在跟进 vLLM
  sleep mode），评估其成熟度：能覆盖"多模型抢占 + GPU 绑定"才考虑替换，否则仍用
  自研 service manager，只在 DECISIONS.md 记录评估结论。

**起 pod 方式**同样保留"生成式 deployment（gen-deployment.sh 思路）"，但改进为：
每个 (模型, 槽位) 生成一个单副本 Deployment（而不是多副本共享一个 deployment），
名字含槽位编号（如 `dsqwen-7b-n76g01`），`CUDA_VISIBLE_DEVICES` 与槽位一一对应。
理由：多副本 deployment 下 k8s 随机杀 pod 会破坏 GPU 绑定，单副本 deployment 使
"哪个 pod 绑哪张卡"完全确定，service manager 的对账逻辑大幅简化。生成脚本改为
Python（读模型注册表，见 5.2），输出 kustomize 资源。

### D2 TRE 控制循环开销：三处削减，目标单轮 <100ms

**问题**：每轮循环从 Redis SCAN 全量键、重新拉取重新计算，耗时大；快环 5s 一次
放大成本。

**结论**（三层配合，P3 + P5 实现）：

1. **存储层**（治本）：Redis 键格式从"key 后缀带时间戳 + SCAN 过滤"改为
   **每 pod 每类指标一个 Sorted Set**（score = 毫秒时间戳，member = JSON 样本），
   读窗口 = `ZRANGEBYSCORE`，复杂度从 O(全库键数) 降到 O(log n + 窗口样本数)；
   写入侧（网关 Go 代码）同步修改，并用 `ZREMRANGEBYSCORE` 滚动裁剪（保留 30 分钟）
   + 键级 TTL 兜底，Redis 不再无限膨胀。键 schema 见 5.2。
2. **采集层**：collector 拆为 `MetricsStore`（唯一碰 Redis 的类，进程内缓存
   "已读窗口"的原始样本，快环两次 tick 落在同一监控窗口时第二次零 Redis 读）+
   纯函数聚合。所有模型的窗口读取用 pipeline 一次往返批量发出。
3. **循环层**：控制器改为 asyncio 三任务——`metrics_task`（唯一的取数方，按窗口
   刷新共享快照 `MetricsSnapshot`）、`rescue_task`（5s）、`fairness_task`（10s）。
   快慢环只读快照做决策，永不直接碰 Redis；动作经带去重仲裁的 `ActionQueue` 下发
   （规则：rescue 优先，同模型有在途动作时丢弃后来的 fairness 动作）。慢环执行慢
   不再能阻塞快环。结构见 5.4。

**分位数**：`_histogram_percentile` 取桶上界改为桶内线性插值（Prometheus
`histogram_quantile` 标准公式），实现为 `percentile_mode: bucket_upper | interpolated`
配置，**默认 bucket_upper**（见第 3 章开关原则），单测两种模式都覆盖。

### D3 迁移策略：先提差异清单，再在新版 AIBrix 上按第 5 章框架重建

不在旧代码上原地改。以 AIBrix v0.4.0 官方 tag 为基准 diff 出全部 Go 侧自研改动 →
在新工作目录以"upstream 分支 = 上游原样，main 分支 = 自研"的双分支结构重建，
`git diff upstream..main` 永远是干净的自研差异。Python 侧三个项目是纯自研，
按第 5 章新结构迁移重组。

### D4 消融机制：env 开关 + kustomize overlay

四个开关（`TRE_ABLATION_DISABLE_FAST_LOOP`、`TRE_ABLATION_DISABLE_SAFESCALE`、
`TRE_SIGNAL_SOURCE=zm|latency_p95|queue_len|kv_cache`、`ENABLE_TRE_SCALING`），
在 planner 的信号选择与任务启动处生效，`deploy/overlays/ablation-*` 各放一个组合。
本次只保证开关逻辑有单测、清单能 build，不跑消融实验。

### D5 拟合与排序度评估：只重构代码 + 离线验证，不重跑

calibration 拆成 collect / dataset / fit / evaluate 四个独立 CLI 阶段。正确性验证
全部离线：用旧的训练输出目录（若 76 上还在）或合成数据驱动 fit 与 evaluate，
断言"重构后 fit 在旧数据上产出与 `TRE_MODEL_PARAMETERS.md` 记录一致的 theta_m
（容差 1e-6，若算法本身修了 bug 则记录差异原因）"。真实重拟合列入遗留运行清单。

### D6 trace 重放器：审查 + 修复，用本地 vLLM 对拍替代大实验

按 P7 清单静态审查 `client_dispatcher.py` 等；发现的问题（闭环退化、时间漂移等）
直接修。对拍降级为：若集群可用则单模型 2 分钟对拍 vllm bench serve（L3），否则
写一个"离线发射精度测试"——用本地 mock server（立即返回的 aiohttp stub）跑 60s，
断言实际发射时间戳与计划时间表的偏差 P99 < 10ms。

## 5. 目标代码框架（详细，按此实现）

### 5.1 目录树（新工作目录 `/data/nfs_shared_data/xxy/aibrix` 内）

```
（上游 AIBrix 目录结构保持原样；侵入改动处加 // TRE-PATCH(<编号>) 注释）
pkg/plugins/gateway/...            # 移植的网关改动
pkg/controller/podautoscaler/...   # 移植的 APA 改动

tre/                               # 自研代码全部收拢于此
  common/                          # ← 先建这个，所有组件依赖它
    pyproject.toml
    tre_common/
      registry.py                  # 模型注册表加载与校验（唯一配置源）
      rediskeys.py                 # Redis 键 schema 常量 + 构造函数
      metrics_schema.py            # MetricsSample / MetricsSnapshot dataclass
      percentile.py                # 直方图分位数（两种模式）
      logging.py                   # 统一 JSON 日志 setup
    tests/
  controller/                      # TRE 控制器（原 python/tre 重组）
    tre_controller/
      app.py                       # asyncio 入口：装配三任务 + ActionQueue
      loops/metrics_task.py        # 快照刷新
      loops/rescue_task.py         # 快环
      loops/fairness_task.py       # 慢环
      loops/action_queue.py        # 动作仲裁与下发
      signals/trs.py               # TRSComputer/SaturationGuard（从旧 trs.py 迁移）
      signals/sources.py           # TRE_SIGNAL_SOURCE 的信号抽象（消融用）
      planning/classify.py         # 原 paper_state.py：四态分类
      planning/planner.py          # 纯函数 build_plan()
      planning/safescale.py        # SafeScale 状态机（从 1369 行拆薄）
      store/metrics_store.py       # 唯一 Redis 读方 + 窗口缓存
      store/state_store.py         # 控制器自身状态持久化
      sm_client.py                 # service manager v2 API 客户端
      config.py                    # 全部 env 的集中解析（含消融开关）
    tests/
  service-manager/                 # 原 python/service_manage_aibrix 重组
    tre_sm/
      app.py                       # FastAPI 入口
      allocator/slots.py           # 槽位模型（纯函数，无 IO）★核心
      allocator/topology.py        # 节点/GPU 拓扑（从模型注册表 + 发现构建）
      state/store.py               # 占位表持久化（Redis hash + 版本号）
      state/reconcile.py           # 启动对账：k8s 实况 + 持久化记录 → 内存态
      ops/vllm_ops.py              # /sleep /wake_up 调用（幂等 + 重试 + 超时）
      ops/k8s_ops.py               # pod annotation 读写、pod 发现
      api/v2.py                    # 新 API（声明式）
      api/v1_compat.py             # 旧接口薄适配层（过渡期）
    tests/
  replayer/                        # 原 CustomTraceGenerator 重组
    tre_replayer/
      engine/schedule.py           # 预生成到达时间表（开环）
      engine/dispatcher.py         # 异步发射器（修复审查问题后）
      traces/...                   # 原 config/traces 原样迁入
      orchestrate.py               # 原 run_experiment.sh 的 Python 化
      design.py                    # ρ 空间相位 DSL → trace.json 生成器（第 12 章）
      lint.py                      # trace 三硬约束检查器（第 12 章）★每条 trace 必过
      oracle.py                    # 事后最优分配器：违规下界 + 参考分配序列（第 12 章）
    tests/                         # 含离线发射精度测试（D6）
  calibration/
    tre_calibration/
      collect.py  dataset.py  fit.py  evaluate.py   # 四阶段 CLI
      signals.py                   # 可拟合信号列定义（trs/latency/queue/kv）
      capacity.py                  # 从训练网格拟合单 pod 容量面 C_m(i,o)（第 12 章用）
    tests/  fixtures/
  ui/                              # P8
  deploy/
    base/                          # gateway、tre-controller、service-manager、redis 引用
    models/                        # gen_model_manifests.py 生成的每槽位 Deployment
    overlays/cluster-75-76/
    overlays/ablation-{noflash,nosafescale,sig-latency,sig-queue}/
  Makefile                         # check / smoke / build-images / manifests
docs/refactor/                     # 过程文档（第 7 章）
experiments/                       # 夹具与冒烟输出（大文件不进 git）
```

### 5.2 tre/common —— 单一配置源与共享 schema

**模型注册表** `tre/deploy/registry.yaml`（消灭"改一个模型要同步 6 处"；
controller、service manager、manifest 生成器、calibration、replayer 全部只读它）：

```yaml
cluster:
  nodes:
    - {name: node-75, gpus: 4, two_gpu_slots: [[0,1],[2,3]]}
    - {name: node-76, gpus: 4, two_gpu_slots: [[0,1],[2,3]]}
models:
  - name: dsqwen-7b
    weights_path: /data/nfs_shared_data/Qwen1.5-7B-Chat
    tp_size: 1
    min_replicas: 1
    max_replicas: 4
    vllm_image: vllm/vllm-openai:0.10.1-sleep
    slo: {ttft_p95_ms: 500, tpot_p95_ms: 75, e2e_p95_ms: 10000}
    trs:                      # 全部来自旧拟合值（本次不重拟合）
      w_p: 0.04
      w_d: 1.0
      lambda_wait: 2.625
      qmin: 1.0
      ema_alpha: 0.5
      theta_m: <旧值>
      tau_crit: <旧值>
      tau_low: <旧值>
      tau_high: <旧值>
      qsat: 4.0
      epsat: 0.05
      hsat: 3
  # dsllama-8b (tp=1)、dsqwen-14b (tp=2) 同结构
```

`registry.py` 接口：

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str; weights_path: str; tp_size: int
    min_replicas: int; max_replicas: int; vllm_image: str
    slo: SloSpec; trs: TrsParams

def load_registry(path: str | None = None) -> Registry: ...
class Registry:
    def model(self, name: str) -> ModelSpec
    def models(self) -> list[ModelSpec]
    def topology(self) -> ClusterTopology
    def validate(self) -> list[str]      # 启动时调用，返回问题清单
```

**Redis 键 schema v2**（`rediskeys.py`，Go 写入侧用同名常量并在文件头互相引用注释）：

```
tre:v2:hist:{pod}          Sorted Set  score=ts_ms  member=JSON(直方图快照: 各 metric 的 sum/count/buckets)
tre:v2:inst:{pod}          Sorted Set  score=ts_ms  member=JSON(即时指标: waiting/running/swapping/kv_hit...)
tre:v2:pods:{model}        Set         该模型当前上报中的 pod 名（写入侧维护，TTL 刷新）
tre:v2:decision:latest     Hash        controller 每轮决策快照（UI/调试读）
tre:v2:sm:state            Hash        service manager 占位表（field=serve_id, value=JSON）
tre:v2:sm:version          String      占位表乐观版本号
保留策略：写入侧每次写后 ZREMRANGEBYSCORE (-inf, now-30min)；所有键 TTL 2h 兜底。
```

**指标快照**（`metrics_schema.py`）：

```python
@dataclass(frozen=True)
class ModelWindowMetrics:
    model: str; window_start_ms: int; window_end_ms: int
    prompt_tokens: float; generation_tokens: float
    avg_waiting: float; avg_running: float; avg_swapping: float
    kv_cache_hit_rate: float
    ttft_p95_ms: float | None; tpot_p95_ms: float | None; e2e_p95_ms: float | None
    routable_pods: int; assigned_replicas: int
    per_pod: dict[str, PodWindowMetrics]

@dataclass(frozen=True)
class MetricsSnapshot:
    ts_ms: int
    models: dict[str, ModelWindowMetrics]
    stale: bool                     # metrics_task 刷新失败时置 True，环据此保守决策
```

### 5.3 service manager（P4 实现）

**槽位分配器** `allocator/slots.py` —— 纯函数、无 IO、强制单测：

```python
@dataclass(frozen=True)
class Slot:
    node: str
    gpu_ids: tuple[int, ...]        # 长度 = tp_size，必须落在同一个 two_gpu_slot 内或恰为其一半

@dataclass(frozen=True)
class Binding:                       # "绑定"与"唤醒"分离：睡着的 serve 仍占用 Slot
    serve_id: str; model: str; slot: Slot; awake: bool

class SlotAllocator:
    """buddy 风格：2 卡槽可拆两半给 1 卡模型；1 卡分配优先填已拆开槽的另一半，
    绝不无谓拆开完整 2 卡槽；释放自动合并。"""
    def __init__(self, topology: ClusterTopology, bindings: list[Binding]): ...
    def find_slot(self, tp_size: int) -> Slot | None          # best-fit，不修改状态
    def bind(self, serve_id: str, model: str, slot: Slot) -> None
    def release(self, serve_id: str) -> None
    def feasible_wake(self, serve_id: str) -> bool            # 该 serve 的槽当前是否可唤醒
    def plan_defrag(self, tp_size: int) -> list[Migration] | None
        # 总空闲卡数够但无完整槽时，给出最小迁移计划（sleep→rebind→wake 序列）；
        # 无解返回 None。Migration = (serve_id, from_slot, to_slot)
    def snapshot(self) -> dict      # 序列化到 Redis
```

**必测反例**（写进 `tests/test_slots.py`，这是 START.md 点名的缺陷）：
node 有 4 卡 {0,1,2,3}，两个 1 卡 serve 分别绑 0 号和 2 号（各拆开一个 2 卡槽）→
`find_slot(2)` 返回 None 但 `plan_defrag(2)` 必须给出"把 2 号迁到 1 号"的计划；
以及正向断言：1 卡分配序列永远先填半空槽（放 0 后再放 1，而不是 2）。

**状态持久化与对账**：占位表每次变更写 `tre:v2:sm:state`（乐观版本号防并发写坏）；
GPU 绑定同时写 pod annotation `tre.aibrix.io/gpu-ids`、状态写 `tre.aibrix.io/state`
（sleeping/awake/hidden）。启动时 `reconcile.py`：k8s pod 实况 ∪ Redis 记录 →
不一致时以"pod 实际存在 + 其 CUDA_VISIBLE_DEVICES env"为准并记 warning。
kill -9 重启后状态一致是 P4 的验收单测（用 fake k8s client + fakeredis 模拟）。

**API v2**（声明式，替代 scale_service/sleep_all 等命令式散接口；v1 保留薄适配）：

```
GET  /v2/state                     → 全集群视图（节点/槽位/绑定/唤醒状态/版本号）
PUT  /v2/models/{model}/target     body {"wake_replicas": n}     幂等，返回 diff 动作列表
PUT  /v2/models/{model}/routable   body {"hidden_pods": [...]}   SafeScale 用
POST /v2/reconcile                 → 手动触发对账
GET  /healthz
```

所有变更接口幂等（同一目标状态重复调用无副作用），返回体带执行的动作列表与新版本号。

### 5.4 TRE 控制器（P5 实现）

`app.py` 装配骨架（按此写，不要回到单循环）：

```python
async def main() -> None:
    cfg = ControllerConfig.from_env()          # 集中解析全部 env，启动时整体打印
    registry = load_registry()
    store = MetricsStore(redis, registry, cfg) # 唯一 Redis 读方，带窗口缓存
    snapshot_box = SnapshotBox()               # 持有最新 MetricsSnapshot（原子替换）
    queue = ActionQueue(sm_client, cfg)        # 仲裁 + 下发 + 在途追踪

    tasks = [metrics_task(store, snapshot_box, cfg)]
    if not cfg.ablation_disable_fast_loop:
        tasks.append(rescue_task(snapshot_box, queue, registry, cfg))
    tasks.append(fairness_task(snapshot_box, queue, registry, cfg))
    tasks.append(queue.run())
    await asyncio.gather(*tasks)
```

关键契约：

- `metrics_task`：每 `MONITOR_INTERVAL_SECONDS` 取"最后一个完整窗口"（窗口边界对齐
  写入周期整数倍，向 P3 的 store 要数据），替换 snapshot_box；失败时保留旧快照并置
  `stale=True`。
- `rescue_task` / `fairness_task`：`while True: 读快照 → plan → queue.submit(...) →
  asyncio.sleep(interval)`。环内**没有任何 Redis/HTTP 调用**（HTTP 全在 queue）。
- `planner.build_plan(snapshot, classifications, cluster_view, inflight, cfg) -> list[Action]`
  纯函数。`Action = Scale(model, delta, reason) | Hide(pod) | Unhide(pod) | Defrag(migrations)`。
- **TP-aware**：planner 找容量时调用 `GET /v2/state` 的缓存视图 + 分配器语义——
  2 卡模型 CRITICAL 时：完整空闲 2 卡槽 → 可由 HIGH 模型缩容腾出的同槽两半 →
  `plan_defrag` 迁移 → 都不行则记 `capacity_blocked` 事件。1 卡模型优先吃碎片。
- `ActionQueue`：同模型在途动作未确认前丢弃后续动作（rescue 来源除外，rescue 可
  抢占取消 fairness 在途意图）；每个动作落 JSON 日志（含 reason 和来源环）。
- 信号抽象（消融）：`signals/sources.py` 提供
  `get_signal(model_metrics, spec, cfg) -> SignalValue`，`TRE_SIGNAL_SOURCE=zm` 时为
  Z_m，其它取延迟/队列/kv 并用各自阈值（阈值字段在 registry 的 trs 段预留
  `alt_thresholds`，本次可为空，fit 重构后由 calibration 产出）。
- 每轮决策快照写 `tre:v2:decision:latest`（UI 与调试用）。
- SafeScale 状态机显式化：`PROBING(pod, deadline) → COMMIT | ROLLBACK`，状态入
  state_store，重启可恢复（迁移旧 safescale.py 逻辑时先画状态图进
  `docs/refactor/05_controller_design.md` 再写代码）。

### 5.5 calibration（P6 实现）

```
tre-calib collect  --config <grid.json> --out runs/     # 跑负载网格（本次不执行）
tre-calib dataset  --runs runs/ --out dataset.parquet   # 窗口化+SLO标签
tre-calib fit      --dataset dataset.parquet --signal trs --out fitted.json
tre-calib evaluate --dataset test.parquet --params fitted.json --out report/
```

- `dataset`：SLO 标签用 `percentile_mode` 可配（默认旧口径，见 D2/第 3 章）。
- `fit`：`--signal trs|latency_p95|queue_len|kv_cache`，同一套拟合代码多信号复用，
  输出直接是 registry.yaml 可合并的片段（消灭手工抄数）。
- `evaluate`：Spearman/Kendall τ、AUROC（区分违规窗口）、跨模型一致性（同 Z_m 分档
  各模型违规率）；训练/测试按 **scenario 整段**划分，禁止同 scenario 窗口跨集合
  （修复时间自相关泄漏）。
- 离线验证（本次执行）：`fixtures/` 放一份小型合成数据集（生成脚本
  `tests/make_fixture.py`：构造已知 theta 的合成 TRS-健康度关系，fit 应恢复出该
  theta）+ 若 76 上找得到旧训练输出目录，用它跑 dataset+fit，比对旧记录值。
- `capacity.py`：从训练网格数据（每个 (i,o,rps) 组合的 SLO 健康标签）拟合
  **单 pod 容量面 C_m(i,o)** = 该 (i,o) 下 SLO 内最大可持续 RPS（对网格内插值，
  网格外外推要标记 low-confidence）。输出 `capacity_<model>.json`，是第 12 章
  trace 设计与 lint 的输入。纯离线，旧数据在则本次可产出。

### 5.6 replayer（P7 实现）

审查并修复五点（结论写 `docs/refactor/07_replayer_audit.md`）：
① 开环性：请求发射不得依赖上一响应完成（检查并发上限、连接池、信号量）；
② 到达过程：改为**预生成到达时间表**（`engine/schedule.py`：确定性或 Poisson，
   seed 可配），dispatcher 按绝对时间戳发射并记录 `scheduled_ts` vs `actual_ts`；
③ token 控制：确认 tokenizer 一致、`ignore_eos`/`max_tokens` 固定 decode 长度；
④ 指标口径：TTFT= 首个 SSE content 字节到达，与 vllm bench 定义对齐并写注释；
⑤ 精度测试（离线，必做）：aiohttp stub server + 60s 发射，断言
   `P99(actual-scheduled) < 10ms`、实际总 RPS 与目标偏差 < 1%。

多模型编排、trace 配置格式、TRE/APA 切换流程保留（`orchestrate.py` Python 化时
行为不变优先）。

trace 的**设计与合格性**遵循第 12 章方法论；`design.py / lint.py / oracle.py`
三个工具在 P7 一并实现（都是纯离线计算）。

### 5.7 UI（P8 实现，从简）

FastAPI 后端聚合三源：Redis（`tre:v2:*` 时序 + decision:latest）、
service manager `/v2/state`、registry.yaml。前端单页（原生 JS + 轻量图表库打包进
镜像，禁止运行时外网 CDN）。页面优先级：
① 集群网格图（2 节点 × 4 GPU：占用模型/睡眠/隐藏，轮询 /v2/state）；
② 模型时间线（Z_m、TRS、队列、P95 + 扩缩容事件标记）；
③ 参数页（registry 当前生效值）；
④ 实验面板（对 orchestrate.py 的薄封装）——④ 若时间不够可以只留 stub。
先读旧 `gateway_visualizer` 可复用的取数逻辑。

## 6. 阶段计划（每阶段以"可离线验证"收口）

> 依赖：P1→P2；P3/P4 可并行（都依赖 P1 的 common）；P5 依赖 P3+P4 接口；
> P6/P7 独立可穿插；P8 最后。每阶段完成即 commit + tag，再进下一阶段。

### P0 差异清单与冻结（不改代码、不跑实验）
1. 本仓库打 tag `baseline-v0`（或记录 commit 号于 WORKLOG）。
2. 以 AIBrix v0.4.0 官方 tag 为基准 diff `pkg/ api/ cmd/ config/`，逐文件写
   `docs/refactor/00_custom_diff_inventory.md`：改动目的 / 是否保留 / 迁移去向。
   已知重点：`pkg/plugins/gateway/`（含 wakeup.go、Redis 写入）、
   `pkg/controller/podautoscaler/podautoscaler_controller.go`。
3. 登记 Python 侧模块边界与相互 HTTP/Redis 接口（现状 v1 接口清单，供 v1_compat 用）。
4. 若 76 可访问：`kubectl get all -A -o yaml` 与 `nvidia-smi` 快照存档；不可访问则跳过。

**验证**：diff 清单覆盖率 = 全部非上游文件都有条目（脚本核对文件列表差集）。

### P1 新仓库骨架 + common
1. 新工作目录建 git 双分支（upstream=上游原样 tag，main=工作分支）。
2. 按 5.1 建目录骨架 + Makefile（check/smoke/manifests 三个目标先可运行）。
3. 实现 `tre/common` 全部模块（registry/rediskeys/metrics_schema/percentile/logging）
   + 单测；registry.yaml 按 5.2 填入三个模型的现有参数（从旧 configs json 抄）。
4. `gen_model_manifests.py`：读 registry 生成每槽位单副本 Deployment（D1），
   `kustomize build` 通过。

**验证**：`make check` 全绿；percentile 两种模式单测（构造已知分布的桶，断言
bucket_upper 偏大、interpolated 接近真值）；manifests 生成结果与槽位一一对应
（单测断言 8 卡 → 正确的 deployment 集合）。

### P2 AIBrix 新版迁移（Go 侧）
1. 读新版对应目录，逐条判断 P0 清单条目的移植方式；新版已内置的能力记
   DECISIONS.md 并按其降级路径处理（见 D1）。
2. 逐 patch 移植，一条目一 commit，侵入处 `// TRE-PATCH(N)` 注释；同时把网关
   Redis 写入改到 v2 键 schema（保留 `TRE_REDIS_SCHEMA=v1|v2|dual` 环境开关，
   默认 dual 双写，供对拍）。
3. 硬编码 URL（wakeup.go 的 servementURL 等）改纯 env + 启动校验。
4. `go build ./...` + `go test ./pkg/plugins/gateway/... ./pkg/controller/podautoscaler/...`；
   为 Redis 写入路径补一个用 miniredis 的单测（写入后 ZRANGEBYSCORE 可读回）。

**验证**：go build/test 通过；patch 清单 `docs/refactor/02_upstream_patches.md` 与
commit 一一对应。（L3 冒烟：条件满足才部署到 tre-v2 namespace 发 curl。）

### P3 指标管道（monitor → store）
1. 先读网关写入侧代码，把每个指标的写入周期/单位/时间戳来源写进
   `docs/refactor/03_metrics_pipeline.md`（字段级）；据此定死窗口语义：
   窗口长度 = 写入周期整数倍，取数取"最后一个完整窗口"。
2. 实现 `MetricsStore`（v2 键读取 + pipeline 批量 + 窗口缓存 + v1 兼容读模式）
   与聚合纯函数（差分、加权、窗口均值——从旧 collector.py 迁移公式）。
3. 分位数接入 common.percentile（默认 bucket_upper）。
4. 夹具：写 `tests/make_redis_fixture.py` 合成夹具（多 pod、多窗口、含乱序/缺样本
   /重启计数器归零等边界）；若 76 Redis 可访问，另 dump 一份真实数据做第二夹具。

**验证**：fakeredis 回放夹具，断言聚合值与手算一致；新旧 collector 在同一夹具上
输出对比（差异逐项解释，写进 03 文档）；基准测试：夹具 3 模型 × 8 pod × 30min
数据，单轮全模型取数 + 计算 < 100ms（fakeredis 下近似衡量，记录数字）。

### P4 service manager
按 5.3 实现。顺序：slots.py（纯函数+单测，含必测反例）→ topology/state →
reconcile → ops → api v2 → v1_compat。单测用 fakeredis + fake k8s client；
"kill -9 后重启状态一致"做成单测（构造持久化态 + 模拟 pod 实况 → reconcile →
断言）。幂等性单测：同一 target 连续 PUT 两次，第二次动作列表为空。

**验证**：`make check`；分配器性质测试（随机分配/释放序列 1000 轮，断言不出现
"总卡数够但 find+defrag 都失败"、断言无重叠绑定）。

### P5 TRE 控制器
按 5.4 实现。顺序：config.py（全 env 集中）→ signals（迁移 trs.py，公式不改，
对拍旧实现）→ classify/planner（纯函数化迁移，旧 planner 的 paper 路径为准，
legacy 路径丢弃并记录）→ safescale 状态机 → loops + queue → app.py。
迁移 TRS/分类时**逐函数带上旧实现做黄金对拍单测**（同输入断言同输出），
之后删除旧文件。消融开关四个全部接通并各有单测（如 DISABLE_FAST_LOOP=1 时
rescue_task 不启动）。论文 vs 实现差异随迁移随手记录到
`docs/refactor/05_paper_vs_impl.md`（只记录，不擅自改公式；确定是 bug 的改动
单独 commit 并在 DECISIONS.md 标注）。

**验证**：黄金对拍单测全绿；夹具驱动的端到端回放测试——用 P3 夹具 + mock
service manager，跑 60 个模拟 tick，断言决策序列符合预期场景（构造一个
CRITICAL 场景、一个 HIGH→SafeScale 场景、一个 2 卡模型需 defrag 场景）；
快环 tick 抖动测试（asyncio 下慢环注入 2s 延迟，断言快环周期仍 5±0.5s）。

### P6 calibration
按 5.5 实现。fit 的数学逻辑从旧 1939 行脚本迁移（先读懂并在
`docs/refactor/06_calibration_design.md` 写下拟合目标函数与流程，再拆），
evaluate 按新方法实现。**不跑任何真实负载**。

**验证**：合成夹具恢复已知 theta；若找到旧训练输出则复算比对旧值；evaluate 在
合成数据上的 τ/AUROC 方向正确（构造正相关信号断言 τ>0.9）。

### P7 replayer
按 5.6 审查与修复；实现第 12 章的 `capacity.py`（在 calibration 侧）、
`design.py`、`lint.py`、`oracle.py`；对现有 7 条 trace 全部跑 lint + oracle，
结果写入 `docs/refactor/07_replayer_audit.md`（不合格的 trace **只标注不删除**，
重新生成列入遗留清单 R7）。

**验证**：离线发射精度测试达标；orchestrate.py 对旧 shell 流程的行为对照表写入
07 文档；trace 配置加载单测（7 条现有 trace 全部能解析）；lint 自身的单测
（构造一条明显超容量 trace 和一条不触发扩缩容的 trace，断言分别被 C1/C2 拒绝）；
oracle 单测（构造手算可验的小场景，断言 oracle 违规下界正确）。

### P8 UI
按 5.7 实现。**验证**：后端接口单测（mock 三数据源）；前端构建通过；
用 P3 夹具 + mock sm 起本地 UI，截图存 docs（若无浏览器环境则跳过截图，记录）。

### P9 集成收口
1. `make check` 全仓通过；`make manifests` 产物齐全。
2. 端到端离线联调：fakeredis + 真 service manager 进程（fake k8s）+ 真 controller
   进程 + 夹具数据泵，跑 5 分钟，断言链路日志完整（指标→决策→v2 API 调用）。
3. （L3，条件满足才做）tre-v2 namespace 部署 + 1 模型 + replayer 2 分钟冒烟。
4. 写 `docs/refactor/09_final_report.md` + 更新遗留运行清单（第 10 章模板）。

## 7. 可回溯性与文档规范（强制）

- **Git**：upstream/main 双分支；一阶段一 tag（`p<N>-done`）；commit 消息
  `[P<N>] <module>: <what>`，结构移动与逻辑修改分开提交；每阶段合并前 `make check`。
- **docs/refactor/**：`00_custom_diff_inventory.md`、各阶段设计文档（动手前写方案
  骨架，收口时补验收结果）、`WORKLOG.md`（**每个工作会话追加**：做了什么/问题/
  系统当前状态/下一步）、`DECISIONS.md`（ADR 风格，含所有"框架偏离"与"假设待
  确认"条目）、`05_paper_vs_impl.md`、`09_final_report.md`。
- **experiments/**：本次只有夹具与冒烟输出，同样带 `meta.json`（commit、配置快照）。
- **回滚**：按阶段 tag 回退；集群侧新资源全在 tre-v2 namespace，可整体删除；
  参数文件全部进 git。

## 8. 测试与验证汇总

### 8.1 质量门
`make check` = ruff + mypy(宽松) + pytest + docker build(条件) + kustomize build。
做成 pre-push hook。任何阶段收口必须全绿。

### 8.2 夹具（离线验证的基石，P3 优先产出，后续阶段复用）
- `tests/make_redis_fixture.py`：合成多模型多 pod 时序（参数可控：RPS 形状、
  直方图桶、缺样本、计数器重置），供 P3/P5/P8 回放。
- 真实 dump（条件产出）：76 Redis 可访问时 `redis-cli --rdb` 或按键导出一段窗口。
- calibration 合成夹具：已知 ground-truth theta 的数据生成器。

### 8.3 黄金对拍
凡"迁移旧逻辑"（TRS、分类、planner、聚合公式），迁移期间保留旧函数副本于
`tests/golden/legacy_*.py`，写"同输入同输出"单测，通过后旧副本保留在测试目录
（不删，作为行为契约），生产代码删旧。

## 9. 执行守则（无人值守）

1. **永不停下等人**。需要裁决时：选择最保守/可逆的默认方案 → DECISIONS.md 记录
   （背景/选项/所选默认/如何撤销）→ 继续。整晚运行结束时人类只看
   WORKLOG + DECISIONS 即可复盘。
2. **先读后写**：动一个模块前读完其现有代码；本文档的问题描述基于抽查，以代码
   事实为准，事实冲突时按最小偏离实现并记录。
3. **小步提交**：每个可独立回滚的改动一个 commit；结构移动与逻辑修改分开。
4. **不越阶段**：发现其它模块问题记 WORKLOG 的 TODO 区，不顺手修。
5. **阶段收口硬条件**：`make check` 绿 + 该阶段"验证"项完成（或标注跳过原因）+
   tag。收不了口就缩小该阶段范围（在 WORKLOG 声明缩了什么），也要收口再前进。
6. **红线自查**：任何 kubectl 写操作、docker 清理、NFS 写入前对照 2.2 节。
7. **可运行优先**：时间不够时砍功能不砍质量门；宁可 P8 只出后端 stub，不可留下
   `make check` 红的仓库。
8. **会话结束动作**（每次上下文接近极限或运行结束前）：commit 当前工作 →
   更新 WORKLOG（含"下一步从哪继续"）→ 确认无残留端口转发/后台进程/临时集群资源。

## 10. 遗留运行清单（本次不执行，留给人类触发）

完成重构后，`docs/refactor/09_final_report.md` 末尾按此模板列出，附现成命令：

| # | 事项 | 前置条件 | 预计耗时 | 命令入口 |
| --- | --- | --- | --- | --- |
| R1 | 基线对照实验（旧系统 TRE/APA 各一条 trace） | 集群空闲 | ~2h | 旧系统 run_experiment.sh |
| R2 | 新系统冒烟 + 7 条 trace 正式回归（TRE/APA） | R1、镜像构建 | ~8h | `tre/replayer orchestrate` |
| R3 | 重拟合（新分位数口径）：训练网格 + fit + 阈值写回 registry | 集群空闲 | ~10h/模型 | `tre-calib collect/fit` |
| R4 | 口径切换：percentile_mode→interpolated + R3 新参数，重跑 R2 | R3 | ~8h | 同上 |
| R5 | 消融矩阵（4 开关 × 3 trace） | R4 | ~6h | overlays/ablation-* |
| R6 | replayer 与 vllm bench serve 真机对拍 | 单模型 pod | ~0.5h | 07 文档附命令 |
| R7 | 按第 12 章方法论重新生成/修复未过 lint 的 trace，冻结正式 trace 集 | capacity 表（P7 已产出） | ~1h（纯生成）| `tre_replayer design + lint` |

## 11. 风险与降级

| 风险 | 缓解/降级 |
| --- | --- |
| 新版 AIBrix 网关/autoscaler 架构大改，patch 无法平移 | P2 先调研；最坏情况：网关插件作为独立进程部署（不进上游进程），APA 基线降级为直接沿用旧版镜像跑基线 |
| 新版内置 sleep 编排与自研冲突 | 按 D1 降级路径评估，默认仍用自研 |
| Redis v2 schema 切换期新旧组件不兼容 | 写入侧 dual 双写开关；controller 的 store 支持 v1 兼容读 |
| 76/集群在夜间不可访问 | 所有 L3 验证自动跳过并记录，纯代码工作不受影响 |
| 执行中发现旧参数与新代码口径不匹配 | 一律保持旧口径默认值（第 3 章开关原则），差异记入 R3/R4 |
| 上下文/会话中断 | 守则 9.8 的会话收尾保证任意断点可续；WORKLOG 是唯一恢复入口 |

## 12. 造 trace 方法论（评估负载设计规范）

> 本章回答"什么样的 trace 是**正确且有说服力**的"。所有正式评估 trace（人造与
> 真实切片）必须按本章设计并通过 `lint.py` 检查。工具在 P7 实现，全部离线可算。

### 12.1 三类作废 trace（先明确要避免什么）

1. **超容量**：需求峰值超出集群总能力 → 任何调度器都违规，比的是噪声，结论无效。
2. **不触发扩缩容**：每模型初始 1 副本始终够用 → 所有系统打平，trace 白跑。
3. **为自己定制**：负载周期与 TRE 控制环共振、或只覆盖 TRE 擅长的场景 →
   评审可以合理质疑基线被调差。必须有对照场景 + 事前冻结纪律（12.6）。

### 12.2 设计单位：pod 需求空间（ρ 空间）

不要直接在 RPS 上设计 trace——RPS 的"高低"因模型和 i/o 长度而异，无法判断是否
在容量范围内。改为：

- **容量面** `C_m(i, o)`：模型 m 单 pod 在输入长 i、输出长 o 下 SLO 内最大可持续
  RPS。由 `tre_calibration/capacity.py` 从**已有**训练网格数据插值得到（网格是
  i×o×rps 笛卡尔积扫描 + SLO 健康标签，正好是这个函数的采样）。
- **归一化需求** `ρ_m(t) = RPS_m(t) / C_m(i(t), o(t))`，物理意义 = 该时刻模型 m
  需要的 pod 数（连续值）。
- **设计在 ρ 空间进行**（"让 7b 模型从 0.5 个 pod 涨到 2.5 个 pod"），生成
  trace.json 时由 `design.py` 乘 C_m 映射回 RPS。好处：负载是否可行一目了然、
  换硬件/换模型只需换容量面即可复用全部 trace 形状。

集群总容量以**槽位**计：8 卡 = 4 个 2 卡槽；1 卡模型 1 pod 占半槽，2 卡模型
1 pod 占整槽。定义**占用需求** `S(t) = Σ_m ceil_frac(ρ_m(t)) × slot_width_m`
（lint 中用连续松弛 + 整数两种口径都算）。

### 12.3 三条硬约束（lint.py 逐条检查，全过才算合格 trace）

- **C1 可行性（对 oracle 公平）**：存在一个"事后诸葛亮"分配方案使违规≈0。
  `oracle.py` 用带滞后的贪心算：已知全程 ρ_m(t)，在槽位形状约束下分配整数副本，
  切换计入热切换耗时（1–2s）与 SafeScale 类观察窗为 0（oracle 不需要）。
  要求 oracle 违规窗口占比 < 1%。同时检查瞬时上界：
  `max_t Σ_m ρ_m(t)·slot_width_m ≤ 0.95 × 总槽数`（连续松弛都放不下的直接拒绝）。
- **C2 非平凡性（对静态不公平）**：静态基线（每模型固定初始 1 副本、永不扩缩）
  必须出现明显违规——存在总时长 ≥ 3 个慢环周期的时段使某模型 ρ_m(t) > 1.2。
  否则 trace 测不出任何扩缩容能力。
- **C3 头部空间分档**：`H = max_t Σ_m ρ_m(t)·slot_width_m / 总槽数` 必须落在
  声明档位上：**宽松 H≈0.6 / 中等 H≈0.75 / 紧张 H≈0.9** 三档（±0.05）。
  正式 trace 集三档都要有——系统间差距通常在中、紧档才拉开，宽松档用来证明
  "不折腾"（无谓切换次数也是要报告的指标）。

lint 输出 `lint_report.json` 存进 trace 目录（含 C1/C2/C3 数值、oracle 违规率、
使用的容量面版本），进 git。

### 12.4 机制覆盖矩阵（每条 trace 声明它测什么）

正式 trace 集合起来必须覆盖以下六个轴，每条 trace 在其 README 声明覆盖哪几个
（对照现有 7 条 trace 填矩阵，缺的轴由 R7 补造）：

| 轴 | 设计手法 | 区分出什么 |
| --- | --- | --- |
| A1 需求转移速度 | 相位切换斜率三档：慢坡（冷启动也跟得上）/ 快坡（只有热切换跟得上）/ 阶跃 | 热切换价值本身；快环响应速度 |
| A2 模型间反相关 | 一个模型升、另一个降，总量 Σρ 基本恒定（如现有 Alternating_hot） | 容量再平衡（慢环/donor-receiver），APA 这类各管各的指标没有全局视角 |
| A3 i/o 混合漂移 | **RPS 恒定**，只把输出长度从短改长（decode 变重）或输入变长 | **指标优越性的关键场景**：队列长度、KVCache 占用等信号变化滞后或不敏感，而 TSS 的加权吞吐直接反映有效负载。设计一条"RPS 不变、o 从 100→600"的 trace，预期 TRE 提前扩容而 APA 迟钝 |
| A4 突发与毛刺 | 窄脉冲（< EMA 时间常数）vs 宽突发（> 3 个慢环周期），紧邻回落 | 平滑参数 alpha 的价值：不该被毛刺骗去扩容（无谓切换数），该对真突发快响应；也测 SafeScale 回滚 |
| A5 TP 异构压力 | 先让两个 1 卡模型把 2 卡槽拆散，再抬升 2 卡模型（14b）需求 | 分配器防碎片化 + defrag（D1/P4 的核心卖点） |
| A6 对照场景 | 单模型平缓正弦、低 H，所有系统都应该轻松应对 | **公平性证据**：TRE 与基线打平 → 证明其它 trace 的差距不是基线被调差 |

### 12.5 相位结构与时长规则（design.py 内置校验）

- 结构：`warmup（≥2 个监控窗口，恒定低载）→ 相位 1..N → cooldown`。
- 每个相位时长 ≥ 5 × 慢环周期（当前 10s → ≥50s；建议取 2–5 min 便于图上可读）。
- 周期性负载的周期**不得是任何控制周期（5s/10s/20s 监控窗口）的整数倍**，
  用互素值（如 170s、230s）避免共振假象——对 TRE 有利或不利的共振都要避免。
- 随机性（Poisson 到达、长度分布抽样）必须带显式 seed，写入 trace 配置。
- 总时长：正式 trace 25–35 min（与现有一致）；每条正式 trace 配一个 4 min 缩短版
  （相位等比例压缩，仅用于冒烟，不用于结论）。

### 12.6 真实 trace 切片规范（防 cherry-pick）

1. **先写选片规则再看数据**，规则进 `retrace_source.yaml`：例如
   "在源数据集中选跨模型需求转移量（Σ_m |Δρ_m| 的窗口积分）最大的连续 30 分钟"。
2. 映射：源 trace 的到达时间戳保形；模型映射规则写明（源的哪类流量映到哪个模型）；
   i/o 长度用源数据经目标 tokenizer 重新计数后的真实分布。
3. **缩放只允许乘一个全局系数**把 H 对齐到目标档位（12.3 C3），不允许对单模型
   或单时段做形状修改；系数记录在案。
4. 切片同样必须过 lint 三约束——真实 trace 不豁免。

### 12.7 报告口径与冻结纪律

- **oracle 归一化得分**（主报告指标，比裸违规率有说服力得多）：
  `score_sys = (V_static − V_sys) / (V_static − V_oracle) ∈ (−∞, 1]`，
  V = SLO 违规时间占比（或违规请求占比，两种都报）。score=1 即达到事后最优，
  score=0 即不比静态好。TRE、APA、消融组都报这个分数 + 无谓切换次数 + 平均
  在显存副本数（资源代价）。
- **事前冻结**：正式 trace 集在跑任何对比实验之前 git tag 冻结
  （`traceset-v1`）。此后任何修改 trace 的行为都要记 DECISIONS.md，且该 trace
  之前的对比数据全部作废重跑。禁止"跑完看结果不理想再改 trace"的迭代——如果
  确要迭代设计，在探索用的 scratch trace 上做，正式集只换版本号整体更替。

### 12.8 本次执行范围

本次运行内完成：`capacity.py`（有旧网格数据则产出容量面，否则输出占位并记录）、
`design.py / lint.py / oracle.py` 及其单测、对现有 7 条 trace 的 lint 报告与
机制覆盖矩阵（写入 07 文档）。**不生成新的正式 trace 集、不跑对比实验**——
那是 R7/R2 的事，因为容量面若来自旧口径数据，正式定稿应等 R3 重拟合后复核。
