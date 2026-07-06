# TRE v2 收尾总方案（Endgame Plan v1）

> 文档编号：`docs/refactor/14_endgame_plan.md`
> 写作时间：2026-07-05 晚（架构师第三轮审计后）
> 性质：**这是一份从当前状态直通"TRE 工作全部做完"的完整方案**，不是阶段方案。
> 覆盖：N4b 两个阻塞点的架构决策与修复 → N4b 收尾 → **F4 清场 + AIBrix 0.7.0
> 声明式重部署（D11，N5 前强制关口）→ 干净集群上 n4b-done** → N5 全部实验
> （R1–R7）→ 论文数据打包与最终封版。执行模型从头到尾照本文档执行即可。
> 版本口径：本系统底座 = **AIBrix 0.7.0**（HEAD 在 `v0.7.0` 之后）；`upstream-v0.4.0`
> 只是**旧系统**的对拍基线，不是本系统版本，勿混淆。
> 与 `10_next_steps.md` 的关系：第 10 章的 10.1–10.4 已完成不再重复；10.5/10.6/10.7
> 被本文档第 4 章**取代**（因为两个阻塞点改变了执行前提）。其余纪律条款继续有效。

---

## 0. 架构师审计结论（2026-07-05 晚，全部为本人亲自实查，非转述）

### 0.1 N4b.1–N4b.4 复核：通过，质量好

- git 链完整：`943ab486`（写入计划）→ `9d9a31fb`（UUID 绑卡）→ `d7263b6c`（defrag 重建）
  → `a3bcf79b`（canary PASS）→ … → `4ebab326`（全拓扑验证）。`make check` 一路从 231 涨到 233。
- D7 canary 一次通过，**未动用任何 fallback**（无 runtimeClassName、无 privileged、无 hostPath），
  no-GPU-request pod 容器内 `nvidia-smi` 只见注入的那一张 UUID 卡。D7 路线彻底坐实。
- 共卡双模型 20 轮切换：gateway `errors=0`，wake P95 `dsllama-8b 1.34s / dsqwen-7b 1.23s`。
- 全拓扑：三模型 12 个 binding 全部 D7 化，reconcile 无告警，三模型 gateway 各 20/20。
- 执行模型自主修了三个真 bug 并都有回归测试：双 awake 观测被 reconcile 前拒收、
  目标增长时跳过不可行 sleeping binding（409 改为尝试下一个）、create 后等 vLLM HTTP 就绪
  （k8s Ready ≠ `/wake_up` 可用）。判断和写法都对。

### 0.2 阻塞点 B1 确证：**vLLM sleep 显存泄漏**（比执行模型的归因更深一层）

执行模型把 10.5/10.6 的失败归因为"D7 共驻导致启动 headroom 不足，需要调启动参数"。
我实查后发现根因更严重——**是 sleep 之后显存没有真正释放**，不是参数问题：

```
node9 实测（nvidia-smi compute-apps + /proc cgroup → pod 映射 + /is_sleeping 探针）：
  健康 sleeping pod 驻留 ≈ 1.0–1.8 GiB（与 N4 时代一致）
  泄漏 sleeping pod：
    dsllama-8b-gpu-2   is_sleeping=true 却驻留 22,856 MiB（GPU2）
    dsqwen-14b-gpu-2-3 is_sleeping=true 却驻留 16,946 MiB（GPU2）+ 37,838 MiB（GPU3）
  → node9 GPU2/GPU3 被"睡着的"pod 占满（39.9/40.1 GiB），任何冷启动必然 OOM。
```

特征：泄漏 pod 都经历过 **wake → 承载流量 → sleep** 循环（10.6 的 queue_len canary 期间被
控制器唤醒过）；TP=2 的 14b 两个 shard 泄漏量不对称（16.9 vs 37.8 GiB），说明不是"权重
没卸载"这么简单，更像 sleep 的 CuMem 池外分配（NCCL buffer / CUDA graph / 碎片）未回收。
而 N4 的 20 轮热切换（wake→少量请求→sleep）从未出现泄漏，说明泄漏有触发条件，需要 E1
实验刻画（见 2.1）。

**推论**：调 `gpu_memory_utilization` 治不了这个病（泄漏 22–38 GiB，调参最多省 2 GiB）。
正确的治理是"检测 + 重建"（决策 D8，见 1.1）。

### 0.3 阻塞点 B2 确证：zm 信号断链的精确代码位置

- `tre/controller/tre_controller/store/metrics_store.py`：
  - 第 69/98 行：hist 文档只读 `[window_start, window_end]` **窗口内**的 zset 成员；
  - 第 215 行 `_hist_sum_delta`：算 delta 需要窗口内**至少 2 个** hist 文档。
  - hist 文档由 AIBrix 底座写入（`aibrix:pod_histogram_metrics_*`，写侧不归我们管），
    写入节奏与窗口长度同量级 → 窗口里经常只有 0 或 1 个文档 → prompt/generation tokens
    无法成 delta → 分类 UNKNOWN / Z_m=None。
- `tre/controller/tre_controller/planning/planner.py:577` `_paper_state_incomplete`：
  **任一**非 IDLE 模型不完整 → 第 109–115 行丢弃**整轮所有动作**。一颗老鼠屎坏一锅粥。
- registry 的 `theta_m` 已是真值（738.67/738/534，N1.3 对齐过），不是 Z_m=None 的原因。

**推论**：这是标准的"counter delta 需要窗口外基线"问题 + 降级粒度过粗问题，读侧可修，
不需要动 AIBrix 底座（决策 D9，见 1.2）。

### 0.4 当前集群状态（截至审计时）

| 模型 | awake | bound | 备注 |
|---|---|---|---|
| dsqwen-7b | 1 (node9 GPU0) | 2 | gpu-1/gpu-2 的 Deployment 在 10.5 失败后已删，需恢复到 4 |
| dsllama-8b | 1 (node9 GPU1) | 4 | gpu-2 的 binding 是泄漏 pod |
| dsqwen-14b | 2 (node10 GPU0-1, GPU2-3) | 4 | node9 gpu2-3 的 binding 是泄漏 pod |

- 控制器已恢复默认 zm 信号 env，因 B2 实际处于"只保活不决策"状态。
- 一个未解之谜：10.6 期间 `dsqwen-7b-router` HTTPRoute 凭空消失（执行模型已按同模式
  重建）。谁删的没有结论，列入 4.4 排查项。
- 其余组件健康：SM/controller/redis/ui 全部 Running，`POST /v2/reconcile` 无告警。

### 0.5 对执行模型两个 Blocked 记录的裁决

- 记录质量好：都停在了该停的地方，没有瞎猜；queue_len canary 的对照实验设计得当，
  正好把 B1、B2 同时暴露出来了。
- 但 10.5 的 Blocked 归因（"启动参数问题"）**不采纳**，以 0.2 的泄漏结论为准。
- "12h soak 不要在决策前跑"的判断**采纳**——本文档就是那个决策。

---

## 1. 架构决策 D8 / D9 / D10（本文档拍板，执行时各记一条 ADR 进 DECISIONS.md）

### 1.1 D8：sleep 显存泄漏治理 = "检测 + 重建"，不追求根治

- **立场**：泄漏发生在 vLLM 内部（0.10.1-sleep 镜像），我们不改 vLLM、不升镜像
  （升镜像 = 重新验证全部 N4/N4b 结论，代价不成比例）。把泄漏当作环境属性来治理。
- **治理三件套**：
  1. **GPU 真值源**：优先用 prometheus 里 gpu-operator 的 DCGM 指标
     （`DCGM_FI_DEV_FB_USED`，按 GPU UUID 标签查询）；若 DCGM 指标不可用，降级方案是
     一个独立的采集脚本（见 2.3 Plan B）。SM 自己不 ssh、不跑 nvidia-smi。
  2. **泄漏检测**：reconcile 扩展——binding 记录为 sleeping 但该 pod 在其 GPU 上的
     真值驻留 > 阈值（env `TRE_SLEEP_LEAK_MIB`，默认 4096）→ reconcile 响应带
     `sleep_leak:<serve_id>` 告警。只告警，不自动动手。
  3. **重建即治愈（hygiene recreate）**：对泄漏 binding 执行"删 Deployment → 原
     GPU/UUID 重建 → 等 vLLM 就绪 → sleep → 验证驻留 ≤ 2 GiB"。复用 defrag 的
     delete+create 代码路径，不新写一套。**触发是人工/脚本运维动作**，控制环不自动
     重建（避免控制环在泄漏风暴里自激振荡）；12h soak 中若出现泄漏导致扩容失败，
     如实记录为泄漏事件，soak 判定规则见 4.3。
- **启动参数**：`gpu_memory_utilization` 暂**维持 0.9**。只有当 E1 实验（2.1）证明
  健康路径下冷启动 headroom 仍不足时，才在 registry 的 `vllm_extra_args` 里显式加
  `--gpu-memory-utilization 0.85`（这个改动要重建全部模型 pod，代价大，不轻动）。

### 1.2 D9：zm 信号读侧修复，zm 保持为唯一生产信号

- **立场**：论文的核心贡献就是 TRS/Z_m 控制，长实验（N5）绝不允许跑在 queue_len 上。
  queue_len 只保留两个用途：R5 消融实验的一个 arm；文档化的紧急逃生舱。
- **修复三点**（全在读侧，细节见第 3 章）：
  1. **基线回看**：hist 文档读取范围扩为 `[window_start − lookback, window_end]`，
     delta = 窗口内最后一个文档 − 窗口起点前最后一个文档。窗口内**有 1 个文档即可**成 delta。
     lookback 默认 90s，由 env `TRE_HIST_BASELINE_LOOKBACK_MS` 控制，执行前先实测
     AIBrix 实际写入节奏来定值（3.1 第 0 步）。
  2. **按模型降级**：`_paper_state_incomplete` 从"任一不完整→全丢"改为"谁不完整丢谁的
     动作"，其它模型正常决策。丢弃事件带模型名。旧行为保留在
     `TRE_INCOMPLETE_POLICY=drop_all` 之后（golden 对拍如依赖旧语义则 golden 路径固定
     用 drop_all，live 默认 drop_model）。
  3. **EMA 保持**：某模型某窗口无 hist 文档时，TRS EMA 与分类**保持上一窗口值**并计
     staleness，连续超过 3 个窗口（env `TRE_PAPER_STALE_MAX_WINDOWS`）才降为 UNKNOWN。

### 1.3 D10：显存预算与创建前置检查（SlotAllocator/SM 的新硬规则）

- 每 GPU bound 预算 ≤ 3（1 awake + 2 sleeping）——维持 D7 已定规则。
- **冷启动（create）前置检查**，在 SM 的 target-growth 与 defrag create 路径统一生效：
  1. 目标 GPU 上**不得有 awake binding**（awake ≈ 36–37 GiB，物理上不可能共存）；
  2. 目标 GPU 真值空闲 ≥ 该模型冷启动需求（registry 每模型新增字段
     `startup_free_mib`，取值来自 E1 实测，先按 37500 占位）；真值源不可用时降级为
     账面检查（假设每个健康 sleeping 占 2 GiB）并在响应中注明 `headroom_check:bookkeeping`。
  3. 检查不过 → 该 GPU 跳过，尝试下一个可行 binding；全部不可行才 409（沿用
     `7278875d` 已建立的"跳过不可行"语义）。

---

## 2. 阶段 F1：泄漏刻画与治理落地（预计 1–1.5 天）

### 2.1 E1：泄漏复现实验矩阵（真机，先做，结论决定后面参数）

在 node9 上腾一张干净卡（先按 2.2 清掉泄漏 pod 即得 GPU2/GPU3）。每步用
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 在 node 上实测，
配合 pod 的 `/is_sleeping`。矩阵（模型用 dsllama-8b 单卡 + dsqwen-14b TP=2 各跑一遍）：

| # | 场景 | 记录 |
|---|---|---|
| a | create → ready → sleep（零流量） | sleep 后驻留 MiB |
| b | wake → 20 条短请求 → sleep | 同上 |
| c | wake → 200 条混合长短请求（并发 8）→ sleep | 同上 |
| d | 场景 c 连续 10 轮 | 每轮 sleep 后驻留，看趋势（一次性台阶 or 递增） |
| e | 场景 c 但 sleep 用 level 2（`POST /sleep?level=2`） | 驻留 + 之后 wake 是否可用、wake 耗时 |

判定规则：
- 若 c/d 复现泄漏而 b 不复现 → 坐实"承载流量后 sleep 才泄漏"，与 0.2 推断一致；
- 若 e（level 2）驻留 ≤ 2 GiB 且 wake 功能正常 → **追加决策**：常备 sleeping binding
  改用 level 2 睡眠（wake 会变慢，因为权重要重新从盘/内存加载——把实测 wake 耗时记下来，
  若 P95 > 5s 则放弃 level 2，维持 D8 的重建治理）；
- 所有结果写进 `docs/refactor/12_realenv_tests.md` 新增小节"N4b.E1 sleep 泄漏刻画"，
  并回填 `startup_free_mib` 真实值（= 冷启动峰值需求 + 512 MiB 余量）。

### 2.2 泄漏 pod 清理与重建演练（顺手完成 hygiene 流程首验）

1. 备份当前三模型 Deployment yaml 到 `docs/refactor/p11_evidence/`（惯例纪律）。
2. 删除两个泄漏 Deployment：`dsllama-8b-...-gpu-2`、`dsqwen-14b-...-node9-gpu-2-3`；
   删除后在 node9 实测确认对应进程消失、GPU2/GPU3 驻留回落到只剩健康 residue。
3. 用 manifests 生成器**原位重建**（同 GPU UUID）→ 等 vLLM 就绪 → sleep →
   实测驻留 ≤ 2 GiB → `POST /v2/reconcile` 无告警。
4. 把 dsqwen-7b 的 gpu-1/gpu-2 Deployment 也补回来（恢复 bound=4），同样走
   就绪→sleep→实测流程。**注意创建顺序**：同一张卡上先建的 pod 必须 sleep 完成、
   驻留回落后再建下一个（否则冷启动 headroom 不够，这就是 D10 规则的人肉版）。
5. 结束状态 = 12 binding 全拓扑 + 每卡驻留可解释（awake 1 个 36–37 GiB，sleeping
   各 ≤ 2 GiB）。记 WORKLOG。

### 2.3 GPU 真值源接入（TDD）

1. **先探测**（只读）：`kubectl -n <prometheus-ns> port-forward` 或 ClusterIP 直接查
   `DCGM_FI_DEV_FB_USED`，确认指标存在且带 GPU UUID 维度标签（gpu-operator 默认有
   dcgm-exporter）。把一次真实查询的返回样例存进 WORKLOG。
2. **Plan A（DCGM 可用）**：`service-manager` 新增 `GpuTruthProvider` 接口 +
   `PrometheusGpuTruth` 实现（HTTP 查询 prometheus，env `TRE_PROM_URL`；查不到时返回
   None，调用方降级账面检查）+ `NullGpuTruth`（测试/降级用）。单测用假 HTTP 响应。
3. **Plan B（DCGM 不可用才做）**：写 `tre/deploy/scripts/gpu_truth_agent.py`——在
   两台 node 上各跑一个 hostPID=false 的普通脚本（systemd timer 或 nohup），每 30s
   把 `nvidia-smi --query-gpu=uuid,memory.used` 结果 `SETEX` 进 tre-v2 Redis
   （key `tre:gpu_truth:<node>`，TTL 120s）；SM 的 provider 改读 Redis。
4. reconcile 泄漏检测 + create 前置检查按 D8/D10 实装（RED→GREEN：
   `test_reconcile_sleep_leak.py`、`test_create_headroom.py`，fake provider 注入
   泄漏/健康两种真值）。`make check` 全绿后 build/roll SM 镜像，tag 规范不变。
5. 真机验收：人为不清理一个泄漏 pod 之前（或用 E1 的 d 场景造一个），
   `POST /v2/reconcile` 响应里出现 `sleep_leak:<serve_id>`；hygiene 重建后告警消失。

---

## 3. 阶段 F2：zm 信号修复（预计 1 天）

### 3.1 离线部分（TDD，全部可在本地跑）

0. **先测量再定参**：读生产 Redis（`TRE_METRICS_REDIS_URL` 指向的 aibrix-redis），对
   一个活跃 pod 的 `aibrix:pod_histogram_metrics_*` zset 取最近 50 个成员的 score，
   算相邻间隔分布（min/p50/p95）。lookback 取 `max(90s, 3×p95间隔)`，把测量记进
   `docs/refactor/03_metrics_pipeline.md` 新小节。
1. `metrics_store.py` 基线回看改造：
   - `_read_zset_docs` 增加 lookback 参数（对 v1 legacy 路径 `_read_legacy_docs` 同改）；
   - `_hist_sum_delta` / `_hist_avg_delta` / `_hist_percentile` 改为
     "基线文档（ts ≤ window_start 的最后一个）+ 窗口内文档"计算 delta；
     窗口内 ≥1 个文档即可用；基线不存在时退回旧行为（窗口内 ≥2）。
   - RED 用例：窗口内 1 个文档 + 基线在窗口前 → tokens delta 正确；窗口内 0 个文档 →
     该窗口 tokens=None（进入 EMA 保持路径，不是 0）。
2. TRS/分类 EMA 保持：`signals/trs.py` 与 `planning/classify.py`——窗口 tokens=None
   时保持上一窗口 TRS EMA 与分类结果，staleness 计数；超过
   `TRE_PAPER_STALE_MAX_WINDOWS`（默认 3）→ UNKNOWN。事件
   `paper_state_stale_hold:<model>` 进 plan events。
3. planner 按模型降级：`_paper_state_incomplete` 拆分——返回不完整模型集合；
   `plan()` 只剔除这些模型的动作，事件 `paper_state_incomplete_drop:<model>`；
   全部非 IDLE 模型都不完整时行为等价旧版全丢。`TRE_INCOMPLETE_POLICY`
   （`drop_model` 默认 / `drop_all` 兼容）。
4. **golden 对拍保护**：先跑一遍现有 golden 测试确认新默认值是否破坏对拍；若破坏，
   golden fixture 加载路径强制 `drop_all` + 旧 delta 语义（构造函数参数，不读 env），
   保证"对拍冻结副本行为永不漂移"。这一步做完 `make check` 必须全绿。

### 3.2 真机验收

1. build/roll controller 镜像（tag 规范不变），env 保持 zm 默认信号。
2. 复跑 15 分钟三模型交替 precheck（复用 `/tmp/n4b_three_model_precheck_*.py` 的脚本，
   脚本本体请 commit 到 `tre/deploy/scripts/` 免得再丢在 /tmp）。验收：
   - controller 日志中 `paper_state_incomplete_drop` 出现次数为 0（模型活跃期间）；
   - Redis decision 记录里三个模型的 `Z_m` 全程非空；
   - `stale_hold` 事件允许零星出现（< 5% 窗口）。
3. 负载压高验证决策产出：用 precheck 脚本把 dsqwen-7b 打到排队（并发拉高 4×，5 分钟），
   验收：controller 基于 **zm** 发出该模型 scale 动作（不再需要 queue_len canary），
   且 F1 的 headroom 检查放行/拒绝行为符合当时的真值。证据 JSON 收进
   `docs/refactor/p11_evidence/`。

---

## 4. 阶段 F3：N4b 收尾（取代 10_next_steps.md 的 10.5–10.7；预计 1.5 天 + 一夜）

### 4.1 live defrag 真机 PASS（原 10.5）

前置：F1/F2 完成，全拓扑 12 binding、无泄漏告警。

1. **构造碎片**（只用 sleep/wake 与删建，不在有 awake 的卡上冷启动）：
   把 dsqwen-14b 压到 node9 只剩零散半卡 sleeping（具体拓扑按当时 SlotAllocator 视图，
   原则：让 14b 的可行 wake 目标为 0，但存在"搬走一个单卡 sleeping binding 即可空出
   完整 two_gpu_slot"的局面）。
2. 触发 `POST /v2/defrag`，验收全部满足：
   - 迁移计划与 `plan_defrag` 预演一致（先 dry-run 打印再执行）；
   - 被搬 binding 走 delete+create 路径，落点通过 D10 headroom 检查；
   - defrag 全程三个 awake 模型 gateway 零 5xx（后台并发 1 req/s 探针陪跑）；
   - defrag 完成后 14b `feasible_wake` 非空，实际 wake 成功、20/20 请求通过；
   - `POST /v2/reconcile` 无告警。
3. 证据 JSON + `12_realenv_tests.md` N4.4 从 SKIP 翻 **PASS（live）**。

### 4.2 三模型扩缩行为验收（原 10.6 前半）

设计一个 40 分钟的分阶段负载（脚本进 `tre/deploy/scripts/n4b_scale_exercise.py`）：

```
0–10min   三模型各 1 req/s 基线
10–20min  dsqwen-7b 拉到饱和（并发 16），其余不变
20–30min  7b 回落基线；dsllama-8b 拉到饱和
30–40min  全部回落基线
```

验收（全部基于 zm 默认信号，禁止切 queue_len）：
- 7b、8b 在各自饱和段内 awake 从 1 → ≥2（扩容动作 reason 可追溯到 Z_m 分类）；
- 回落段结束后 10 分钟内缩回 awake=1（serving floor 生效，不断流）；
- 扩容目标卡由 headroom 检查选择，无 CrashLoop/OOM；
- 全程 gateway 错误率 < 0.5%（扩缩瞬间的抖动计入）；
- dsqwen-14b 不受干扰（其流量段错误率 0）。

### 4.3 12 小时过夜 soak（原 10.6 后半）

- 负载：4.2 的脚本改为循环模式，周期 2h（每模型每周期各有一次饱和段），跑 12h。
- 输出目录：**76 本地盘** `/root/tre-n4b-soak/<date>/`（禁 NFS）。
- 每 5 分钟采样一轮，全部落 JSONL：SM `/v2/state`、controller/SM 进程 RSS、
  两个 Redis DBSIZE、gateway 探针错误计数、每 GPU 真值驻留（DCGM/Plan B）、
  pod restarts。
- 通过标准：
  1. tre-v2 各组件 restarts 增量 = 0；
  2. controller/SM RSS 增长 < 20%；
  3. gateway 总错误率 < 0.5%，且无连续 > 60s 的不可用窗口；
  4. 每个模型完成 ≥ 5 次正确的扩→缩循环；
  5. 结束时 reconcile 无告警、无双 awake 违例记录；
  6. **泄漏单列**：若出现 `sleep_leak` 告警，soak 不算失败，但要记录发生时刻、
     触发 pod、之前经历的 wake/sleep 次数——这是泄漏发生率的第一手数据；若泄漏
     导致某次扩容失败，记为"泄漏致扩容失败"事件，≤ 2 次可接受（治理靠 hygiene，
     不靠 soak 清零）。超标则回到 E1 加测并考虑 level 2 睡眠决策。
- 早晨复盘：汇总表进 `12_realenv_tests.md` N4.6 翻 **PASS（live, 12h）**。

### 4.4 收尾杂项（与 soak 并行做）

1. **HTTPRoute 消失排查**：grep SM/controller 代码确认我们的路径没有删 HTTPRoute 的
   调用；查 `kubectl get events` 与 aibrix-controller-manager 日志中该 route 的删除记录；
   给出结论记 WORKLOG（若确认是 AIBrix 底座对无 ready endpoint 模型的 GC 行为，则在
   SM 的模型上线路径里加"确保 HTTPRoute 存在"的幂等步骤）。
2. `05_paper_vs_impl.md` 补录 N4 的 7 条真机行为契约（routable 标签路由、awake/bound
   语义、serving floor、cluster-view 启动等待、Redis 断连容错、GPU 标签合法化、
   hidden 清理）+ N4b 新增契约（D7 绑卡、跳过不可行 binding、vLLM 就绪等待、
   泄漏检测、headroom 检查、按模型降级）。
3. ADR 补齐：D7（若 ADR-0006 还没写）、D8/D9/D10 各一条，编号顺延 DECISIONS.md。
4. `12_realenv_tests.md` 全文一致性梳理：删掉过时陈述（如早前"no n4-done tag"残句）。

### 4.5 N4b gate（全过才打 tag）

- [ ] N4.1 / N4.4 / N4.6 三个 SKIP 全部翻 live PASS（N4.1 已于 `4ebab326` 完成）
- [ ] `cd tre && make check` 全绿（预期 ≥ 233 + 新增用例）
- [ ] reconcile 无告警、无泄漏告警的全拓扑稳态
- [ ] 4.4 的文档/ADR 全部落盘并 commit
- [ ] `git tag n4b-done`

---

## 5. 阶段 F4：清场 + 声明式全新部署（N5 前的强制关口，预计 1.5–2 天）

> **为什么必须做（架构师 2026-07-06 复核实况）**：
> - 代码已对齐 **AIBrix 0.7.0**（HEAD = `v0.7.0-198-gxxxxxxx`；`upstream-v0.4.0`
>   只是**旧系统**的对拍基线，不是本系统底座版本）。
> - 但**集群里跑的底座不是干净的 0.7.0**——是旧镜像混装：
>   `aibrix/controller-manager:nightly`、`orchestration-controller:v0.4.1`、
>   `gateway-plugins:<nightly>`、多个 `:latest`。代码在 0.7.0，底座停在 0.4 时代。
> - tre-v2 是一路**手工 patch 出来的**（`kubectl set image` 滚镜像、手工重建
>   HTTPRoute、D8 hygiene 手工删建 Deployment）；**gpu-truth agent 是手工 nohup**
>   拉起的，清单里根本没有它。
> - 结论：**当前所有 N4/N4b live 结论都来自一个不可复现的"雪花集群"**。N5 的论文
>   数字绝不能跑在上面。F4 的产出 = 一个"从干净集群按清单一键拉起、可复现"的
>   AIBrix 0.7.0 + TRE v2 全栈，N5 全部实验在它上面跑。这是决策 **D11**。
>
> **执行顺序**：F3 的路由守卫先在**当前集群**快速复验一次（确认 `f6dce214` 修好了
> defrag route GC），拿到 N4.4 的初步结论；然后做 F4 清场重部署；**N4b 的正式验收
> （N4.2 热切换 / N4.4 defrag 零 5xx / N4.6 三模型扩缩 + 12h soak）在 F4 之后的干净
> 集群上重跑一遍，才是权威 PASS**，`n4b-done` 打在这次干净验收之后。

### 5.1 F4.0 先把手工改动补成声明式（离线，TDD，不动集群）

清场之前，必须先保证"清单里有的 == 手工做过的"，否则重部署会丢功能。逐项审计并补齐：

1. **gpu-truth agent → DaemonSet**（当前最大缺口）。
   - 新建 `tre/deploy/overlays/tre-v2/gpu-truth-daemonset.yaml`：在两台 GPU 节点上
     跑 `gpu_truth_agent.py`，`hostPID: false`、只读挂 `nvidia-smi`（用带 CUDA 的
     基础镜像或把 nvidia-smi 通过 hostPath），env 指 tre-v2 Redis，`SETEX
     tre:gpu_truth:<node>`。nodeSelector 只选带 GPU 的节点。
   - 把脚本打进一个 tre-v2 sidecar 镜像或复用现有镜像；agent 逻辑已在
     `deploy/tests/test_gpu_truth_agent.py` 有覆盖，DaemonSet 清单加进 overlay
     的 `resources:` 和 overlay 测试。
2. **model Deployments + HTTPRoutes 纳入一键部署**。
   - 现状：`tre/deploy/models/` 由 `gen_model_manifests.py` 生成（含 D7 UUID 绑卡 +
     每模型 HTTPRoute），但 `overlays/tre-v2` 没 include models。
   - 决定生成时机：GPU **UUID 是每台机器固定的**，可提前用 `collect_gpu_uuids.py`
     采集写进 registry，再 `gen_model_manifests.py` 生成到 `models/`，纳入版本控制；
     部署时 `kubectl apply -k tre/deploy/models`。把"采 UUID → 生成 → apply"写成
     一个 `tre/deploy/scripts/deploy_models.sh`。
   - HTTPRoute 由生成器产出**初始**版本；运行期由 F3 路由守卫幂等保证存在（双保险）。
3. **env/阈值全部落 kustomize**：`TRE_SLEEP_LEAK_USED_MIB`、`TRE_CREATE_MAX_USED_MIB`、
   `TRE_SIGNAL_SOURCE`（默认 zm）、`TRE_PERCENTILE_MODE`（bucket_upper）、
   `TRE_HIST_BASELINE_LOOKBACK_MS`、`TRE_PAPER_STALE_MAX_WINDOWS`、`TRE_INCOMPLETE_POLICY`
   等 F1/F2 引入的开关，逐一确认已在 `service-manager.yaml`/`controller.yaml` 里显式
   声明（不能只活在手工 set env 里）。
4. **镜像清单固化**：把当前 live 的 SM/controller/ui 镜像 tag + digest 写进
   `docs/refactor/images.lock.md`；overlay 里 image 引用全部用**带 digest 或固定
   tag**，禁 latest/nightly。
5. 全部改完 `cd tre && make check` 全绿，commit。**这一步产出"一键部署包"，是 F4
   的核心交付物。**

### 5.2 F4.1 全量备份（不可跳，红线）

1. 备份当前 tre-v2 全部资源、model Deployments、所有 model HTTPRoute、当前 AIBrix
   底座关键 CR/config：`kubectl get ... -o yaml` 落
   `docs/refactor/p11_evidence/pre_f4_teardown_<date>/`，commit。
2. 备份当前 Redis 状态（SM state + controller EMA）快照，便于对照。
3. **确认旧系统基线备份仍在**：`docs/refactor/p11_evidence/old_system_backup/`
   （R1 要用）——见 5.5 的版本口径提示。

### 5.3 F4.2 清场（第二次动集群，谨慎）

1. 停控制器（已 replicas=0），SM 也缩 0；删 tre-v2 全部 model Deployment + HTTPRoute；
   `nvidia-smi` 双检两台节点 GPU 全部释放（只剩驱动 residue）。
2. 删 tre-v2 namespace（redis/ui/sm/controller 一起走）。
3. **卸载旧 AIBrix 底座**：这是唯一敏感操作。用 AIBrix 0.7.0 官方卸载方式
   （`kubectl delete -k` 对应 install 清单，或 helm uninstall，视仓库 `config/`
   或 `dist/` 提供的方式）。**保留** gpu-operator、prometheus、envoy-gateway CRD
   底座、kube-flannel（这些是集群级依赖，不属于 AIBrix 应用层）——删前用
   `kubectl get` 确认每个要删的对象归属 AIBrix，逐类删，不用 `delete ns` 扫。
   **不确定某对象是否该删就记 Blocked 停手问架构师**，宁可留残余也不误删底座。

### 5.4 F4.3 全新部署 AIBrix 0.7.0 + TRE v2（从清单一键起）

1. **装 AIBrix 0.7.0 底座**：用仓库 `config/`（或 `dist/`）的 0.7.0 安装清单，确认
   装出来的 aibrix-system 组件镜像**全部是 0.7.0 一致版本**（不再有 nightly/v0.4.1
   混装）。等底座 Ready，验 gateway/redis/controller 健康。
2. **部署 TRE v2**：`kubectl apply -k tre/deploy/overlays/tre-v2`（含 F4.0 补进的
   gpu-truth DaemonSet）；`bash tre/deploy/scripts/deploy_models.sh` 起 12 个 model
   binding。全程**不手工 patch**——凡是需要手工插一刀的，都回 F4.0 补进清单再来。
3. **冒烟**：三模型 gateway 各 20/20；`POST /v2/reconcile` warnings=[]；gpu-truth
   两个 key 在刷新；控制器起来后不误扩容。证据落 `p11_evidence/f4_fresh_deploy_<date>/`。
4. **在干净 0.7.0 底座上重新探测 F1/F2 的两个环境假设**（可能变好，要更新决策）：
   - DCGM/prometheus GPU 指标在干净 0.7.0 gpu-operator 下是否**有值了**——若有，
     GPU truth 可从 Plan B(Redis 桥) 升级回 **Plan A(DCGM)**，DaemonSet 可退役，更新
     D8 记录。
   - HTTPRoute GC 行为在干净 0.7.0 gateway 下是否仍出现——若 0.7.0 不再 GC，F3 路由
     守卫从"必需"降为"防御性冗余"，在 12_realenv_tests.md 里注明。

### 5.5 F4.4 干净集群上的 N4b 权威验收（重跑，替代当前雪花上的结论）

在 F4.3 的干净全栈上重跑 N4b 关键验收，作为**权威** PASS：
- N4.2 热切换往返（20 轮，wake P95 记录）
- N4.4 live defrag 零 5xx（这次带 F3 路由守卫，且在干净底座上——才算数）
- N4.6 三模型扩缩（zm 信号，4.2 的分阶段负载）+ 12h 过夜 soak（本地盘输出）
- 每项证据覆盖 12_realenv_tests.md 对应节，全部 live PASS 后 → `git tag n4b-done`。

> **R1 基线的版本口径提示（重要，写给 N5）**：旧系统基线是 AIBrix **0.4.0** 上的旧
> TRE；清场后集群是 0.7.0，忠实还原"旧系统 on 0.4.0"代价高且与新系统不同底座、
> 对比不公平。**建议**：N5 主对照的 `V_static`（静态分配基线）在**同一 0.7.0 底座**
> 上测（新旧同底座才公平）；旧 0.4.0 TRE 系统作为**次要参考**，若要跑就在 F4 清场
> **之前**用 old_system_backup 单独跑一次并记录底座版本差异，或直接以"prior work"
> 口径引用并在 09 报告里注明底座版本 caveat。这条在 N5 的 R1 里对应展开（见 6.1）。

### 5.6 F4 gate

- [ ] F4.0 一键部署包（含 gpu-truth DaemonSet、models apply 脚本、env/镜像固化）
      commit，`make check` 全绿
- [ ] 旧底座 + 雪花 tre-v2 清场完成，GPU 双检释放
- [ ] AIBrix 0.7.0 底座 + TRE v2 从清单**零手工**拉起，冒烟通过
- [ ] F1/F2 两个环境假设在干净底座上复测并更新决策
- [ ] N4b 权威验收（N4.2/N4.4/N4.6）在干净集群 live PASS
- [ ] `git tag n4b-done`（打在干净验收之后，不是雪花上）
- [ ] ADR：D11（清场重部署 + 一键部署包）记入 DECISIONS.md

---

## 6. 阶段 N5：长实验手册（R1–R7 逐项展开；总计算时长 ≈ 35h，建议 4–5 天排完）

依赖顺序不变：`R1 → R3 → R7 → R2 → R4 → R5`，R6 任意空档。每项跑完立即在
`docs/refactor/13_experiments_log.md`（新建）里记一条，格式统一为：

```markdown
## R<x> <名称>  <日期>
- 系统版本: git <sha> / 镜像 <digest 列表> / traceset <tag>
- 命令与参数: <逐条>
- 输出目录: <路径>（本地盘）
- 结果摘要: <关键数字>
- 异常与处理: <无则写无>
```

### 6.1 R1：旧系统基线（约 2h 采集 + 前后各 1h 切换）

**目的**：论文对照组 V_baseline(旧 TRE) 与 V_static（静态分配）在基准负载下的表现。

1. 采集前快照：`kubectl get deploy,svc,httproute -A -o yaml` 存
   `docs/refactor/p11_evidence/pre_r1_snapshot/`；v2 模型 Deployment yaml 单独备份。
2. **停新系统**：controller、service-manager `scale --replicas=0`（redis/ui 保留）；
   删除 v2 全部模型 Deployment；node9/node10 `nvidia-smi` 确认 GPU 清空（只剩底座）。
3. **起旧系统**：用 `docs/refactor/p11_evidence/old_system_backup/` 的 yaml 恢复
   （restore_ready 已 dry-run 过）。等旧系统模型就绪，nvidia-smi + kubectl 双检，
   确认无新旧共管。
4. 跑基准负载：用 replayer 以 **T_bench trace**（R7 之前用现有 p7 冻结 trace 里指定的
   基准场景；R7 重生成后不重跑 R1——基线系统对 trace 变化不敏感的说明记入 13 号文档）
   通过 gateway 打旧系统，采集 SLO 违约、切换次数、GPU 利用（DCGM 导出）。
   同一 trace 再跑一遍 **静态分配**（旧系统关自动策略/固定副本）得 V_static。
5. **撤旧回新**：删旧系统 → GPU 清空双检 → `kubectl apply -k tre/deploy/tre-v2` +
   全拓扑重建（按 2.2 顺序纪律）→ 冒烟（三模型 gateway 各 20/20）。
6. 产出：`/root/tre-experiments/r1/`；13 号文档记录。
**失败处理**：旧系统起不来 → 按 restore_ready 文档排错，2h 内无解则回滚到新系统并记
Blocked（R1 可以最后补跑，不阻塞 R3 起步）。

### 6.2 R3：真机重拟合（θ_m + 容量面；每模型 ≈ 10h，共 3 模型，安排 2–3 个夜间）

> **前置硬护栏（架构师 2026-07-06 定，见 `15_signal_and_window_plan.md` S1.4）**：
> R3 **不许在 S1（TSS 控制窗口改造）的窗口口径最终冻结之前开跑**。θ_m 必须在"最终确定
> 的窗口"上拟合；窗口若还会变，R3 的数据与 θ_m 全部作废、每模型 10h 白跑。且 S4（r3_grid
> 原始日志落盘）也必须在 R3 前完成，以便一次采集支撑多窗口/多口径离线重拟。顺序：
> 先做完 S1 + S4 → 再开 R3。TRS 量纲保持"每窗口 token"不变，不用缩放捷径，就是重新拟合。

**目的**：产出真实 `theta_m` 与容量面 `C_m(i,o)`，替换 registry 中的旧系统继承值；
同时落 `bucket_upper` 与 `interpolated` 两套 percentile 口径（否则 R4 要重跑网格）。

1. 负载网格：对每个模型按 REFACTOR_PLAN.md 第 12 章 ρ 空间设计——
   输入桶 × 输出桶 × 并发档（并发从 1 爬到饱和后再 +2 档），每格稳态 ≥ 5 个窗口。
   驱动脚本 `tre/deploy/scripts/r3_grid.py`（新写，进 git）：直接打模型 Service
   （绕 gateway，排除路由噪声），每格记录窗口级 CSV。
2. CSV 列与 `tre/calibration/tre_calibration/cli.py --input` 要求对齐
   （窗口指标 + trs 信号列；trs 来自 controller 决策 dump 或 metrics_store 离线重算，
   两个 percentile 口径各出一份 CSV——重算用 `TRE_PERCENTILE_MODE` 两次离线跑）。
3. 拟合：对每模型 × 每口径跑 calibration CLI；`--ttft-p95-ms/--tpot-p95-ms` 用**论文
   SLO**（ttft 500ms 一档）与**现网 SLO**（registry 1200ms 一档）各出一版 profile——
   主结果用论文 SLO，现网版进附录；这个双版本决定记 ADR（编号顺延）。
4. 容量面：`capacity.py` 生成 `capacity_<model>.json`（两口径），
   落 `tre/deploy/calibration/`；sync 脚本回填 registry 的 theta_m（bucket_upper 版）。
5. commit + 重建三镜像（registry 打进镜像，`TRE_REGISTRY_PATH=/app/...`）+ roll +
   冒烟。golden 对拍不受影响（golden 用冻结 fixture 不读 registry——跑一遍确认）。
6. 验收：三模型 × 两口径 共 6 份 profile + 6 份容量面；θ_m 与旧值偏差记录
   （偏差 > 30% 时在 13 号文档里解释原因——硬件/引擎版本差异预期内）。

### 6.3 R7：trace 重生成与冻结（≈ 1h）

1. 用 R3 真容量面跑 `tre/replayer` 的 design → `lint.py`（C1/C2/C3 全过）→
   oracle 求解（`oracle.py`）→ 7 条 trace + oracle 值落 `tre/replayer/traces/`。
2. `git tag traceset-v1`。**此后 R2/R4/R5 期间禁改 trace；结果不满意只能改系统不能改题。**
3. 13 号文档记：每条 trace 的 ρ 覆盖、V_static / V_oracle 预算值。

### 6.4 R2：新系统 7-trace 回归（bucket_upper；≈ 8h）

1. 前置：全拓扑稳态、reconcile 干净、controller env `TRE_PERCENTILE_MODE=bucket_upper`
   （现值）。每条 trace 之间执行**状态复位**：全模型缩回 awake=1、清 controller
   EMA 状态（重启 controller pod 即可，启动等待 cluster-view 已有保护）、Redis
   decision 流转存归档后清空。复位脚本 `tre/deploy/scripts/reset_between_traces.sh`。
2. 逐条跑 7 trace（replayer 打 gateway），每条记录：SLO 违约序列、切换/扩缩动作数、
   V_sys，算 oracle 归一化得分 `(V_static − V_sys)/(V_static − V_oracle)`。
3. 产出 `/root/tre-experiments/r2/<trace>/`；13 号文档汇总表（7 行 × 得分列）。
4. 期间任何 `sleep_leak` 告警：trace 之间做 hygiene 重建，trace 之中不动、记录。

### 6.5 R4：interpolated 口径复跑（≈ 8h）

1. `kubectl apply -k tre/deploy/ablation-interpolated`（只切 controller env）+
   registry 侧确认容量面/θ 用 interpolated 版（R3 已产出——如果 registry 只能装一套，
   overlay 里用 env 指向另一套 profile 文件，实现方式执行时按 config.py 现状选最小改动）。
2. 复位纪律、7 trace、同 R2 全套记录。跑完切回 bucket_upper 底座。

### 6.6 R5：消融矩阵（≈ 6h）

- arm 列表：`no-fastloop`、`no-safescale`（现有 overlay）、`signal=queue_len`
  （env TRE_SIGNAL_SOURCE，D9 保留的消融用途）、（percentile 两口径已由 R2/R4 覆盖，
  不重复）。
- 每 arm 跑 7 条 trace 里**预注册**的 3 条代表 trace（在跑之前于 13 号文档写死选哪
  3 条及理由，防止事后挑数据）：默认选 ρ 最低/中/最高各一条。
- 每 arm 跑完恢复默认 overlay 并冒烟，再进下一 arm。

### 6.7 R6：replayer 计时精度（≈ 0.5h，任意空档）

- replayer 双模式（真实 HTTP vs 干跑计时）对同一 trace 的发压时间轴对比，
  P99 偏差 < 50ms 为过；进 13 号文档。

### 6.8 N5 gate

- [ ] 13 号文档含 R1–R7 全部条目，每条可复现（版本/命令/数据齐全）
- [ ] `git tag results-v1`
- [ ] 主对照表：{旧系统, 新系统 bucket_upper, interpolated, 各消融 arm} × 归一化得分

---

## 7. 阶段 F5：论文数据打包与封版（预计 1 天）

1. `09_final_report.md` 终版：叙事从"重构完成"升级为"重构 + 真机验证 + 全量实验"，
   贴 N4b/N5 关键数字；系统边界与已知限制单列（sleep 泄漏治理、AIBrix 指标节奏依赖、
   D7 放弃 k8s GPU 配额的运维含义）。
2. 图表数据导出：每张论文图一个 `paper_data/fig_<n>_<名称>/`（CSV + 生成脚本），
   放 repo `tre/paper_data/`；不出图，只出可复现数据（画图在论文侧做）。
3. 文档一致性终检：00–14 号文档相互引用无死链、DECISIONS.md 与各 ADR 编号连续、
   WORKLOG 无未闭合的 Blocked（每条 Blocked 要么已解决要么在 09 终版"已知限制"挂号）。
4. 仓库卫生：`git status` 干净；/tmp 里有价值的脚本全部收编进
   `tre/deploy/scripts/`；镜像 tag 清单（含 digest）落 `docs/refactor/images.lock.md`。
5. 终 tag：`git tag tre-v2-1.0`。至此 TRE 工作**做完**。

---

## 8. 全局纪律（沿用，此处只列增量）

- `10_next_steps.md` 第 1 章执行原则 v2 全部继续有效（备份后才能删、/root/aibrix-main
  冻结、不动权重/NFS、底座四件套不动、双检 GPU、TDD+tag+WORKLOG、Blocked 不猜）。
- 增量纪律：
  1. 实验输出一律写 76 **本地盘**（`/root/tre-experiments/`、`/root/tre-n4b-soak/`），
     只有最终汇总表和 13 号文档进 NFS git 仓库。
  2. 有价值的验证/驱动脚本**当天收编进 git**（`tre/deploy/scripts/`），不许长住 /tmp
     ——本轮已经因此丢过上下文。
  3. R2/R4/R5 期间 traceset 冻结；controller/SM 代码若必须改 bug，改完要在 13 号
     文档标注"哪些 trace 是修复前跑的"并重跑受影响 trace。
  4. 每个 R 项开跑前 `git status` 必须干净（版本可追溯的前提）。

风险表（增量）：

| 风险 | 缓解 |
|---|---|
| E1 无法复现泄漏（偶发型） | 不阻塞：D8 治理不依赖复现；soak 的泄漏单列统计就是发生率数据 |
| DCGM 指标缺失/无 UUID 标签 | 2.3 Plan B（Redis 真值桥），SM 代码用 provider 接口本来就两态 |
| R1 旧系统恢复失败 | R1 挪到最后补跑，不阻塞 R3–R5；超 2h 记 Blocked 回滚新系统 |
| R3 网格跑一半断 | r3_grid.py 必须支持断点续跑（每格落盘即 checkpoint） |
| 双口径容量面装载改动超预期 | 允许最小实现：env 指 profile 文件路径；禁止为此重构 config 体系 |
| soak 夜间泄漏致扩容失败连发 | 判定规则已写死（≤2 次可接受）；超标走 E1-e level 2 决策，不现场发明 |

---

## 9. 给执行模型的开工指令（原样粘贴即可）

```text
你在接手一个进行到中后期的项目：TRE v2（基于 AIBrix 0.7.0 的 MaaS 多模型热切换自动
扩缩容系统）。工作目录 = 权威工作区：76 服务器 /data/nfs_shared_data/xxy/aibrix
（git 仓库，HEAD 在 main，已对齐 AIBrix 0.7.0）。

━━ 先读文档（按序读完再动手）━━
1. docs/refactor/14_endgame_plan.md 全文——从现在到 TRE 全部做完的唯一任务清单。
   第 0 章=对 N4b 的审计结论，第 1 章=架构决策 D8/D9/D10，第 5 章=D11 清场重部署。
2. docs/refactor/WORKLOG.md 从 "Endgame F1.1" 读到末尾——F1/F2/F3 的全部执行记录。
3. docs/refactor/12_realenv_tests.md 的 N4.4 / N4.6 两节现状。

━━ 当前进度（截至 2026-07-06 架构师复核）━━
- P0–P9、N1–N4、N4b.1–.4 全部完成。
- F1（第2章，泄漏治理）完成：E1 刻画做完；GPU 真值走 Plan B（gpu_truth_agent.py
  手工 nohup 在 node9/node10，SETEX 到 tre-v2 Redis；DCGM/Plan A 在当前脏底座不可用）；
  D8 检测+重建、D10 headroom 检查已实装。
- F2（第3章，zm 信号修复）完成：基线回看 / 按模型降级 / EMA 保持 / ActionQueue
  inflight bug 全修；已验证 zm 能驱动扩容（dsqwen-7b 高负载 1→3 awake）。
- F3（第4章）进行中：live defrag 撞上"AIBrix 删/建 pod 时 HTTPRoute 被 GC"导致
  gateway 5xx 的阻塞；路由守卫 K8sOps.ensure_model_httproute() 已实现并 roll 到
  live（SM 镜像 tre-v2-service-manager:20260705-f6dce214），make check 264 passed，
  【但还没用守卫后的镜像重跑 live defrag 验证】。控制器当前 replicas=0（故意暂停）。

━━ 你的接手顺序 ━━
第一步：把本文档最新版 commit（若未提交）。
① F3 收尾：先在当前集群用路由守卫镜像重跑 live defrag，确认 route GC 修好、
   defrag 期间三模型零 5xx（带 HTTP model 头），拿到 N4.4 初步结论。
② F4（第5章，清场重部署，D11）：这是 N5 前的强制关口。当前集群是"底座陈旧
   (nightly/v0.4.1 混装) + tre-v2 手工捏出来"的雪花，N5 论文数字不能跑在上面。
   F4.0 先把手工改动补成声明式（gpu-truth 做成 DaemonSet、models+HTTPRoute 一键
   apply 脚本、env/镜像固化）→ F4.1 全量备份 → F4.2 清场旧底座+雪花 →
   F4.3 从清单零手工拉起 AIBrix 0.7.0 + TRE v2 → F4.4 干净集群上重跑 N4b 权威
   验收（N4.2/N4.4/N4.6）→ 打 n4b-done（打在干净验收之后）。
③ N5（第6章）：R1→R3→R7→R2→R4→R5，R6 空档插；跑完打 results-v1。
④ F5（第7章）：论文数据打包 + 文档终检；打 tre-v2-1.0。收工。

━━ 纪律 ━━
沿用 P0–N4b：TDD 小步提交、每步 cd tre && make check、WORKLOG 记证据路径、真机 GPU
操作前 nvidia-smi + kubectl 双检、Blocked 记录后继续做不受影响的部分不猜。
新增：实验输出写 76 本地盘（/root/tre-experiments、/root/tre-n4b-soak）不写 NFS；
当天用过的脚本当天收编进 tre/deploy/scripts/（别长住 /tmp，已因此丢过上下文）；
R2 之后 traceset 冻结不许改。

━━ 红线 ━━
/root/aibrix-main 只读；不动模型权重与 NFS 大文件；删 k8s 资源前先备份 yaml 且提交；
镜像 tag 禁 latest/nightly。
【F4 例外】F4.2/F4.3 会显式卸载并重装 AIBrix 0.7.0 应用层底座——这是唯一被授权的
底座操作，但 gpu-operator / prometheus / envoy-gateway CRD / kube-flannel 仍然不动；
删任何底座对象前逐类 kubectl get 确认归属，不确定就记 Blocked 停手问架构师。
```
