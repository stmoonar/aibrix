# TRE v2 重构后续计划 v2（P9 之后 → 清场部署 → 真实环境测试 → 实验）

> 本文档接续 `REFACTOR_PLAN.md`（P0–P9 已全部完成并打 tag）与 `docs/refactor/09_final_report.md`。
> 权威工作区仍是 76 上的 `/data/nfs_shared_data/xxy/aibrix`（见 ADR-0001）。
> 本文档同时存放于 76 的 `docs/refactor/10_next_steps.md`。
>
> **v2 变更（2026-07-04）**：经确认，k8s 集群上只有我们自己的东西（旧 TRE 与 AIBrix 底座），
> 红线放宽——允许删除旧 TRE 部署并部署新系统；新增"真实环境测试"阶段 N4。

---

## 0. 现状评估结论（2026-07-04 架构师审查）

### 0.1 已验证通过的事实（非转述，均为本次亲自复核）

| 项目 | 结果 |
| --- | --- |
| `cd tre && make check` | 176 passed, 1.59s（本次重跑确认） |
| `make smoke` | registry 校验通过 |
| `make manifests` | 12 份模型 Deployment 清单（75/76 各槽位，单副本一 Deployment，GPU 绑定确定化 ✓） |
| git 状态 | 工作区干净，`p0-done` … `p9-done` 十个 tag 齐全，每阶段小步提交，可按 tag 回溯 |
| 文档链 | `docs/refactor/` 下 00–09 阶段文档 + WORKLOG + DECISIONS(ADR) + p0 快照 + p7 trace 报告 + p9 e2e 证据，完整 |
| 红线遵守 | P0–P9 全程零 Kubernetes 写操作；旧系统 `/root/aibrix-main` 未动 |

### 0.2 代码质量抽查结论

- **SlotAllocator**（`tre/service-manager/tre_sm/allocator/slots.py`）：1 卡模型 best-fit 优先填半占用槽位，`feasible_wake` / `plan_defrag` 均已实现——修复了旧系统"1 卡模型占 GPU0 和 GPU2 堵死 2 卡模型"的碎片化缺陷。
- **控制环**：asyncio 任务拆分（metrics / rescue / fairness / safescale / cluster_view）+ ActionQueue 仲裁，Redis 改为 Sorted Set 窗口读（`zrangebyscore`），符合 D2 决策。
- **golden 对拍**：`controller/tests/golden/` 冻结了 legacy_trs / legacy_classify / legacy_planner / legacy_collector，新实现与旧公式逐值对拍，TRS 语义迁移可信。
- **兼容开关**：`TRE_PERCENTILE_MODE` 默认 `bucket_upper`（旧语义），消融开关 `TRE_ABLATION_DISABLE_FAST_LOOP` / `TRE_ABLATION_DISABLE_SAFESCALE` 就位，符合 D5/R4 约定。
- **偏差记录诚实**：`05_paper_vs_impl.md` 明确记录 replica correction 保留、legacy raw-TRS fallback 移除（带 `dropped_legacy_raw_trs` 显式标记）等实现契约。

### 0.3 集群现状清点（2026-07-04，kubectl 实查）

- 两节点 GPU 全空闲（node10/76 四卡均 1 MiB 占用；node9/75 部署前需同样确认）。
- 集群上没有任何模型 pod 在跑。
- `aibrix-system` 内的组件分两类：

| 类别 | 组件 | 处置 |
| --- | --- | --- |
| **旧 TRE（本轮要替换）** | `tre-controller`、`service-management-xxy`、`service-management`、`service-management-lxttest`(0/0 已死) | 备份 yaml 后删除 |
| **AIBrix 底座（保留复用）** | `aibrix-controller-manager`、`aibrix-autoscaling-controller-manager`、`aibrix-gateway-plugins`、`aibrix-metadata-service`、`aibrix-redis-master`、`aibrix-gpu-optimizer`、kuberay ×2、visualizer ×2、`lxtaibrix-gateway-plugins` | 不动（新系统仍走 AIBrix gateway 路由与 Redis 指标链路） |
| **基础设施（绝对不动）** | `envoy-gateway-system`、`gpu-operator`、`prometheus`/`monitoring`、`kube-*` | 不动 |

- 已知环境噪音（不阻塞，顺手可清）：node10 上有若干 `Completed`/`ContainerStatusUnknown` 的僵尸 pod（envoy/prometheus 旧副本）、`prometheus-adapter` 与 `kube-state-metrics` 在 cloud 节点 ImagePullBackOff。清理僵尸 pod 属于删除 pod 级对象、可安全做；ImagePullBackOff 与本项目无关，记录即可。

### 0.4 遗留缺口清单（G1–G7，按修复优先级排序）

| # | 缺口 | 影响 | 修复时机 |
| --- | --- | --- | --- |
| G1 | service-manager v2 **无 `/v2/defrag` 端点**；planner 能产出 `DefragAction`，dispatch 只能返回 unsupported | 碎片化场景下 2 卡模型 CRITICAL 时无法搬迁腾槽 | **N1** |
| G2 | planner "HIGH 同槽先缩、再 defrag" 分支 pending | 缩容优先级次优 | **N1** |
| G3 | controller / service-manager / ui **无 Dockerfile 与镜像** | 部署硬前置 | N2 |
| G4 | 无 ablation-\* overlay（env 开关已有，缺薄封装） | R5 前置，工作量小 | N2 |
| G5 | `registry.yaml` 里 `theta_m: 0.0` 占位；SLO 值（dsqwen-7b ttft 1200ms）与 calibration 论文口径（500ms）不一致 | 分类阈值失真 | N1 定规则，R3 后回填 |
| G6 | P8 UI 截图缺失（76 无 Playwright 浏览器） | 仅证据完整性 | N1 可选 |
| G7 | 现有全部 trace 在 placeholder 容量下 0 条过 lint（主挂 C2/C3） | 正式对比不能用旧 trace 集 | R3 → R7 |

**总体结论：重构达标。** 剩余工作 = 少量离线补齐（N1）+ 构建物（N2）+ 清场部署（N3）+ 真实环境测试（N4）+ 长实验（N5）。

---

## 1. 执行原则（v2 放宽后）

1. **放宽**：允许对集群做写操作，包括删除旧 TRE 部署、部署新系统。集群只有我们的东西，无需再等"空窗确认"。
2. **保留的纪律**（不因放宽而丢）：
   - 删除任何 k8s 资源前先 `kubectl get <res> -o yaml` 备份到 `docs/refactor/p11_evidence/old_system_backup/`——**这不是形式主义：R1 基线实验还要把旧系统临时拉起来跑，备份 yaml 就是恢复手段**。
   - `/root/aibrix-main` 源码仍然冻结只读（它是 R1 基线与 golden 对拍的参照物）。
   - 不删除/移动 `/data/nfs_shared_data/` 下模型权重；不向 NFS 写大量临时文件。
   - 仍禁止 `docker system prune -a`、`kubectl delete ns kube-system` 之类的大范围清理；AIBrix 底座、envoy、gpu-operator、prometheus 不动。
   - 部署模型 pod 前 `nvidia-smi`（两节点）+ `kubectl get pods -A -o wide` 双检 GPU 无人占用。
3. 继续 TDD + 小步提交 + 阶段 tag（`n1-done`…）+ WORKLOG，格式与 P0–P9 一致。
4. 遇到不确定的问题：记录到 WORKLOG 的 `## Blocked` 小节，跳过继续做不受影响的部分，不要猜。

---

## 2. 阶段 N1：离线代码补齐（不碰集群，可立即执行）

### N1.1 打通 defrag 链路（修 G1）

目标：`DefragAction` 从 planner → ActionQueue → controller dispatch → service-manager `/v2/defrag` → `SlotAllocator.plan_defrag` 全链路走通（k8s 操作层用现有 `ops/k8s_ops.py` 抽象打桩，真实执行在 N4 验证）。

步骤（TDD）：
1. **RED**：`service-manager/tests/test_v2_defrag.py`：
   - `POST /v2/defrag {"tp_size": 2}`，在"两个 1 卡模型分别占 (node,0) 和 (node,2)"夹具下，返回迁移计划并执行后 `GET /v2/state` 出现完整空闲 2 卡槽；
   - 迁移中的 serve 先置 `hidden=true`（复用 SafeScale hide-route 语义），失败回滚绑定并恢复可路由；
   - 无解返回 `409 {"reason": "no_feasible_defrag"}`，不做部分迁移。
2. **GREEN**：`tre_sm/api/v2.py` 加端点，编排固定为：`plan_defrag` → 逐个 `hide → sleep → 删旧 serve、新槽起新 serve（新 Deployment 指向新 gpu-ids）→ wake → unhide`；任一步失败即停、回滚可逆步骤、返回已执行/已回滚清单。
3. controller：`sm_client.py` 加 `post_defrag()`；dispatch 把 `DefragAction` 从 unsupported 改为真调用；`/v2/defrag` 404 时降级回 unsupported（兼容旧 SM）。
4. `test_p9_offline_integration.py` 追加"碎片化夹具 → CRITICAL 2 卡模型 → defrag → 扩容成功"离线 case。
5. 文档：`05_controller_design.md` 追加 defrag 时序说明；WORKLOG 记录。

注意：vLLM sleep-mode 的 wake 必须回到原绑定 GPU，所以"换槽"只能走删旧起新，**不能**对同一 serve 换 GPU 再 wake，测试要断言这一点。defrag 与普通 scale 在 ActionQueue 里同模型互斥，复用现有仲裁，不新造锁。

### N1.2 HIGH 同槽先缩分支（修 G2）

架构决策（直接采用）：
- donor 候选 = `state==HIGH` 且缩后不破 `min_replicas` 的 1 卡 serve，且其所在 2 卡槽另一半空闲；多候选取 `Z_m` 最低者；无候选 fallback 到 defrag。
- 逻辑放 `planning/planner.py` 的 cluster-view 分层（与 TP-aware 分支同层），输出数据化 `ShrinkForSlotAction(donor, beneficiary)`，由 SafeScale task 消费，planner 保持纯函数。
- donor 缩成功后**不立即**扩 beneficiary，由下一 rescue tick 自然发现空槽扩容（避免单 tick 两步耦合，5s 延迟可接受）。

步骤：RED（planner 纯函数夹具测试，断言产出 ShrinkForSlotAction 而非 DefragAction）→ GREEN → safescale_task 消费测试 → 文档（此为新行为，旧系统无对应，不进 golden 对拍，记 ADR）。

### N1.3 registry 参数对齐（修 G5 规则部分）

1. `tre/deploy/sync_registry_params.py`：从旧系统 `model_slo_profiles.json` + `seed_calibration.json`（只读）导入 SLO/w_p/w_d/lambda_wait/tau/theta 到 registry.yaml，带 `--dry-run` diff。
2. `make smoke` 增强：`theta_m == 0.0` 打 WARNING；SLO 与 profiles 不一致列 diff。
3. 执行一次 sync（用现有旧 seed 值），把 1200ms 漂移值改回论文口径，WORKLOG 记 before/after。

### N1.4 UI 截图补证据（修 G6，可选）

`npx playwright install chromium`，装不上记录跳过原因，不投入超过 30 分钟。

**N1 验收 gate**：`make check` 全绿（新增 ≥ 10 测试）、`make smoke`、WORKLOG 补全、tag `n1-done`。

---

## 3. 阶段 N2：构建与部署物（build/push 镜像，不改集群状态）

### N2.1 三个 Dockerfile（修 G3）

- `tre/controller/Dockerfile`、`tre/service-manager/Dockerfile`、`tre/ui/Dockerfile`；base 与旧系统 controller 镜像同源（查旧镜像 base，python 版本一致），`tre/common` 作共享层 COPY。
- tag 规范：`tre-v2-<component>:<yyyymmdd>-<git-short-sha>`，**禁止 latest**。（记 ADR）
- 验证：`docker build` + 容器内 `python -c "import tre_controller.app"` + 容器内跑该组件 pytest。
- 76 出网受限时：复用本地已有旧镜像层做 base，或外部机器 build 后 `docker save/load`。

### N2.2 tre-v2 namespace 部署 kustomize

- `tre/deploy/overlays/tre-v2/`：namespace、controller/service-manager/UI Deployment、**独立 Redis 实例**（不复用 `aibrix-redis-master`，避免键空间互踩）、RBAC 最小权限（仅 `tre-v2` 内资源 + 读节点/读 `aibrix-system` gateway 所需对象）。
- 模型 pod 仍部署在模型 manifests 指定的 namespace，gateway 路由链路保持与旧系统一致（新 SM 通过 `routable_models` 语义对接 `aibrix-gateway-plugins`，这一对接在 N4 真机验证）。
- 验证：`kubectl kustomize` 渲染 + `kubectl apply --dry-run=server`（server dry-run 现在允许）。

### N2.3 ablation overlays（修 G4）

`ablation-no-fastloop` / `ablation-no-safescale` / `ablation-bucket-upper` / `ablation-interpolated` 四个薄 overlay，各只 patch controller env。验证同上。

**N2 验收 gate**：三镜像构建成功（digest 记 WORKLOG）、全部 overlay 渲染 + dry-run 通过、tag `n2-done`。

---

## 4. 阶段 N3：清场与部署（第一次动集群）

### N3.1 备份旧系统（先做，不可跳）

```bash
mkdir -p docs/refactor/p11_evidence/old_system_backup
for d in tre-controller service-management-xxy service-management service-management-lxttest; do
  kubectl -n aibrix-system get deploy $d -o yaml > docs/refactor/p11_evidence/old_system_backup/$d.deploy.yaml
done
# 同时备份其关联的 svc / configmap / secret（按 label 或名称逐个 get -o yaml）
kubectl -n aibrix-system get svc,cm,secret -o yaml > docs/refactor/p11_evidence/old_system_backup/aibrix-system-all.yaml
```

备份提交进 git 后才允许执行删除。**这些 yaml 是 R1 基线实验恢复旧系统的唯一部署凭据。**

### N3.2 删除旧 TRE 部署

```bash
kubectl -n aibrix-system delete deploy tre-controller service-management-xxy service-management service-management-lxttest
```

- 只删这四个；AIBrix 底座（controller-manager、gateway-plugins、redis、metadata、kuberay、visualizer、gpu-optimizer）一律保留。
- 顺手清理 node10 上 `Completed`/`ContainerStatusUnknown` 僵尸 pod（`kubectl delete pod` 级操作，安全）。
- 删除后确认：`kubectl get pods -n aibrix-system` 无 TRE 组件、GPU 仍空闲。

### N3.3 部署新系统 + 冒烟

1. `kubectl apply -k tre/deploy/overlays/tre-v2`（controller / SM / UI / 独立 Redis）。
2. 部署 **1 个** dsqwen-7b 模型 Deployment（单槽）。
3. 冒烟清单（结果记 `docs/refactor/11_l3_smoke.md`）：
   - [ ] 模型 pod Running，`nvidia-smi` 显存落在预期 GPU；
   - [ ] `GET /v2/state` 拓扑与 registry 一致；
   - [ ] `PUT /v2/models/dsqwen-7b/target` 完成 sleep/wake 往返，耗时 < 5s；
   - [ ] 经 gateway 发 100 条请求（replayer 短 trace），controller 日志出现完整 `trs_calc_result`，`tre:v2:hist:*` ZSET 有数据、窗口读 < 100ms；
   - [ ] 控制环单 tick 耗时 P95 < 100ms（对照 D2 目标）；
   - [ ] UI `/api/cluster` 反映真实状态。
4. 回退预案：`kubectl delete -k tre/deploy/overlays/tre-v2` + 用 N3.1 备份恢复旧系统（先演练恢复命令的 dry-run）。

**N3 验收 gate**：清单全勾 + tag `n3-done`。任何一项不过：记录、修复、重跑清单，不带病进 N4。

---

## 5. 阶段 N4：真实环境测试（新增，正式实验前的真机功能验证）

> 目的：把 P5–P7 只在离线夹具上验证过的行为，逐条在真集群上复验一遍。每条测试写成
> `tre/scripts/rt_*.sh` 或 replayer 场景配置，结果与日志路径记入 `docs/refactor/12_realenv_tests.md`。
> 这一阶段**只求功能正确与量级合理，不产出论文数据**——论文数据等 R3/R7 之后用冻结 trace 集跑。

### N4.1 全拓扑部署

部署 registry 全部 12 个槽位 Deployment（3 模型 × 75/76 两节点），初始 wake 每模型 1 副本、其余 sleep。验收：`/v2/state` 与 `nvidia-smi` 双向一致；全部 sleep 副本显存占用符合 sleep-mode 预期。

### N4.2 热切换往返压测

脚本循环 20 次对同一 serve 做 sleep→wake：验收 wake P95 < 5s、20 次后无显存泄漏（`nvidia-smi` 对比首末）、wake 后落回原 GPU（读 pod annotation `tre.aibrix.io/gpu-ids` 对 `nvidia-smi` 校验）。

### N4.3 控制环真机行为（对拍旧缺陷场景）

用 replayer 跑三个 10 分钟短场景（现有 trace 裁剪即可，不需要过 lint——本阶段不出论文数据）：
1. **单模型阶跃**：RPS 从低到高一步跳变 → 观察 CRITICAL 触发、rescue tick 5s 内出扩容决策、扩容动作经 ActionQueue 无重复下发；
2. **双模型此消彼长**（Alternating 裁剪版）→ 观察 fairness 环的 LOW-receiver 再平衡、SafeScale 隐藏探测→提交/回滚全流程日志完整；
3. **i/o 漂移小样**（RPS 恒定、输出长度拉长）→ 观察 TRS 响应而队列指标滞后（给 A3 场景的真机预演，顺带验证指标管道端到端语义）。

每个场景验收：决策日志（`tre:v2:decision:latest` 快照链）可完整重建"指标→分类→决策→动作→结果"因果链；无 unexpected exception；tick P95 < 100ms 在有负载时仍成立。

### N4.4 defrag 与同槽先缩真机验证（接 N1.1/N1.2）

人工构造碎片化：wake 两个 1 卡模型到 (node,0) 和 (node,2) → 请求 2 卡模型扩容 → 验收依次触发：HIGH 同槽先缩（若构造了 HIGH donor）或 `/v2/defrag` 搬迁 → 2 卡模型成功起在完整槽；全程 gateway 无 5xx（hide-route 生效）。

### N4.5 故障注入与恢复

- kill controller pod → 重启后从 `tre:v2:sm:state` / state_store 恢复，EMA 与 SafeScale probe journal 不丢（对照 restore/snapshot 接口）；
- kill service-manager pod → 重启后 reconcile 与真实 pod 状态一致（`POST /v2/reconcile`）；
- 停 Redis 30s → controller 降级行为符合预期（不崩、恢复后继续），记录实际行为。

### N4.6 12 小时 soak

低压 trace 循环过夜：验收无内存泄漏（controller/SM RSS 平稳）、Redis 键数量有界（rolling trim 生效）、无累积性 defunct 动作。

**N4 验收 gate**：`12_realenv_tests.md` 全部条目 PASS（或带原因的 SKIP）+ tag `n4-done`。
此后系统视为"可跑正式实验"状态。

---

## 6. 阶段 N5：长实验编排（R1–R7 正确执行顺序）

`09_final_report.md` 的 R1–R7 编号不变，按依赖重排：

```
R1 基线(旧系统, ~2h)      —— 用 N3.1 备份 yaml 临时恢复旧系统跑基线，跑完删除、重新 apply 新系统
   │                        （旧/新系统切换期间双检 GPU，两者不得同时占卡）
R3 重拟合(~10h/模型 ×3)   —— 产出真 theta_m + 容量面 C_m(i,o)；采集时同时落 bucket_upper 与
   │                        interpolated 两套口径参数（否则 R4 要重跑 30h 网格）
   ├─→ sync 脚本回填 registry + capacity_<model>.json
R7 trace 重生成 (~1h)     —— 真容量面跑 design→lint 全过 → git tag 冻结 traceset-v1
R2 新系统 7-trace 回归 (~8h) —— bucket_upper 模式
R4 percentile 切 interpolated 复跑 (~8h)
R5 消融矩阵 (~6h)          —— 用 N2.3 overlays
R6 replayer 计时对比 (~0.5h) —— 无依赖，任意空档插队
```

纪律：
- `traceset-v1` 冻结后 R2/R4/R5 期间禁止改 trace；结果不满意只能改系统不能改题。
- 每个 R 项跑完立即把输出目录、git commit、镜像 digest、trace tag 记入 `docs/refactor/13_experiments_log.md`。
- 主报告指标用 oracle 归一化得分 `(V_static − V_sys)/(V_static − V_oracle)`，附 A6 对照场景证明基线公平（REFACTOR_PLAN.md 12.7）。

---

## 7. 文档与回溯

- 新文档：`10_next_steps.md`（本文）、`11_l3_smoke.md`、`12_realenv_tests.md`、`13_experiments_log.md`；新决策记 ADR（N1.2 donor 规则、N2.1 tag 规范、N3 清场决定）。
- 阶段 tag：`n1-done` … `n4-done`；数据 tag：`traceset-v1`、`results-v1`。
- 回溯：代码任意 `git checkout <tag>`；集群侧新系统 = 删 `tre-v2` overlay 资源，旧系统 = N3.1 备份 yaml 一键恢复。

## 8. 进度复核 v3（2026-07-05）与修订后的下一步

### 8.1 进度结论：N1–N4 全部完成，质量高于预期

亲自复核结果（非转述）：`make check` 220 passed（从 176 增长，N4 修 bug 全部带回归测试）；
`n1-done`…`n4-done` tag 齐全；`tre-v2` 控制面四组件 Running；旧 TRE 四个 Deployment 已删除且
restore-ready 备份通过 server dry-run；真机证据链完整（`11_l3_smoke.md`、`12_realenv_tests.md`）。

真机验证已通过的核心能力：
- 热切换往返 20 轮：wake P95 **0.864s**、sleep P95 1.065s，无绑定漂移、无显存泄漏；
- 真实负载扩容：dsqwen-7b 重负载下 awake 1→3→4，全程 0 错误、endpoint 不断流；
- 双模型交替负载 10 分钟：两模型均正确扩容，0 错误；
- 输出长度漂移小样（A3 预演）：p95 随 max_tokens 单调分层（34.7/430/1246ms），指标管道端到端语义正确；
- 故障注入：controller/SM 重启恢复、Redis 断 30s 降级不崩、reconcile 从真实 pod 重建状态；
- 15 分钟 soak：RSS 平稳、Redis 键数有界、0 重启。

N4 期间发现并修复了 7 个只有真机才暴露的实质 bug（routable 标签路由、planner awake/bound
语义分裂、serving floor、启动期 cluster-view 抢跑、Redis 容错等），全部有回归测试与镜像滚动记录。

### 8.2 修订触发点：三个 justified SKIP 里藏着一个架构级问题

| SKIP | 性质 |
| --- | --- |
| N4.1 全拓扑（12 Deployment 要 16 GPU，只有 8） | **架构问题，见 8.3，必须先修** |
| N4.4 live defrag（`/v2/defrag` 只改状态不真搬 k8s Deployment） | 功能缺口，N4b 补 |
| N4.6 12h soak（只跑了 15 分钟替代） | 补跑即可，可过夜无人值守 |

另有两个小尾巴：dsllama-8b 至今没上过真机；`12_realenv_tests.md` 末尾"“No n4-done tag"陈述已过时（tag 实际已打），顺手修正。

### 8.3 架构决策 D7：模型 pod 改为 env 显式绑卡，放弃 `nvidia.com/gpu` 资源请求（记 ADR）

**问题**：k8s 为 sleeping pod 一样保留 `nvidia.com/gpu` 配额 → bound 总数 ≤ 物理 GPU 数 →
"多模型 pod 绑同一 GPU、同时至多一个 awake"这一 TRE 的核心多路复用能力在 k8s 调度语义下不可能实现。
这不是 manifests 写错，是资源模型选错。

**旧系统证据**：`python/service_manage_aibrix/infra/k8s/runtime_discovery.py:97-129` 显式支持
`requested_gpu_count == 0` 且从 `NVIDIA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` env 解析绑卡的
pod——旧系统正是 env 绑卡、不占 k8s GPU 配额，才能把多模型 bound 到同一批卡上。

**决策（直接执行，不再讨论）**：
1. `gen_model_manifests.py` 改为：去掉 `nvidia.com/gpu` requests/limits；注入
   `NVIDIA_VISIBLE_DEVICES=<GPU-UUID>`（**用 UUID 不用 index**——ADR-0005 已确认 index 不可保证；
   registry.yaml 的 node 条目增加 `gpu_uuids` 列表，由一次性脚本从两节点 `nvidia-smi -L` 采集写入）；
   `nodeName` 钉节点。无资源请求的 pod 会绕过 device-plugin，由 nvidia container runtime 直接按
   env 挂卡，这正是旧系统的工作方式。
2. GPU 互斥从 k8s 调度器移交给 SlotAllocator（它本来就是唯一分配来源，动作经 ActionQueue 串行化）：
   - SlotAllocator 增加并测试硬不变式：**同一 GPU 上至多一个 awake binding**（`feasible_wake` 拒绝
     违例，SM wake 路径强制走该检查，RED 测试先行）；
   - reconcile 增加校验：发现同 GPU 双 awake（外部干预造成）时告警并自动 sleep 后来者。
3. 显存安全：sleep 副本 ~1.1GiB/卡（N4 实测），共卡 bound 数按 40GB 卡预算上限 = 1 awake + ≤2 sleeping，
   `gen_model_manifests.py` 生成时检查每 GPU 的 bound 预算。

### 8.4 修订后的执行顺序

```
N4b GPU 绑定改造 + 补测（先离线后真机，1 个阶段闭环，tag n4b-done）
  1. D7 改造：manifests 生成器 + registry gpu_uuids + SlotAllocator 不变式（离线 TDD）
  2. defrag 补真实 k8s 路径：/v2/defrag 内部走"删旧 Deployment → 新槽生成新 Deployment → wake"
     （复用 N1.1 编排骨架，把打桩的 ops 换成真 k8s 调用；离线用 fake client 测）
  3. 真机：单 GPU 绑 dsqwen-7b + dsllama-8b 双 pod，交替 wake/sleep 20 轮（dsllama-8b 首次上真机）
  4. 真机：全拓扑 12 Deployment 部署（D7 后 16 槽位需求不再受 8 GPU 配额限制），重验 N4.1 清单
  5. 真机：人工碎片化 → live defrag → 2 卡模型起在完整槽（补 N4.4 的 live PASS）
  6. 三模型 alternating 场景 + 12h 过夜 soak（无人值守，次日查报告）
       ↓
N5 长实验（顺序不变：R1 → R3 → R7 → R2 → R4 → R5；R6 任意空档）
  - R1 前置已就绪：restore_ready 备份已通过 server dry-run
  - R3 仍是瓶颈（~30h）：D7 改造不影响单 pod 标定，可与 N4b 的 3–6 步并行排期
```

N4b 期间同步补两处文档：`05_paper_vs_impl.md` 追加 N4 引入的行为契约（TRS 用 awake 副本数、
planner awake/bound 语义分层、serving floor），D7 记 ADR-0006。

**N4b 的逐步执行计划见第 10 章。**

## 9. 风险提示

| 风险 | 缓解 |
| --- | --- |
| 删旧系统后 R1 无基线可跑 | N3.1 备份 yaml 是硬 gate：备份未提交进 git 之前禁止删除 |
| 新 SM 与 aibrix-gateway-plugins 的 routable 对接在真机不通 | N3.3 冒烟第 4 条专门验证；不通则临时保留 v1_compat 路由接口并记 Blocked |
| R3 拿不到 30h 连续机时 | 按模型拆 3 段（~10h/段），train 目录按 scenario 粒度落盘可断点续采 |
| theta 回填后行为变化 | golden 对拍锁公式不锁参数；回填后重跑 `make check` 确认 |
| 76 出网受限拉不到镜像 base | 复用旧镜像层做 base，或外部 build 后 `docker save/load` |
| soak/长实验写爆 NFS | 长操作前 `df -h`；运行数据写本地盘，只把选定结果拷 NFS |

---

## 10. 阶段 N4b 详细执行计划（v3 新增，当前待执行阶段）

> 六个 slice，每个独立可验证、小步提交，沿用 TDD + WORKLOG + 阶段 tag 纪律。
> 10.1/10.2 纯离线；10.3 起动真机。R3 标定与 10.3–10.6 无资源冲突时可并行排期。

### 10.1 N4b.1 D7 绑卡改造（离线）

1. **registry 增加 UUID**：`registry.yaml` 的每个 node 条目增加 `gpu_uuids` 列表；写一次性脚本
   `tre/deploy/collect_gpu_uuids.py` 解析两节点 `nvidia-smi -L` 输出并写入；`load_registry`
   校验 `len(gpu_uuids) == gpus`，缺失时 `make smoke` 报错。
2. **manifests 生成器改造（TDD）**，RED 断言：
   - 生成的 Deployment **不含** `nvidia.com/gpu` requests/limits；
   - env 含 `NVIDIA_VISIBLE_DEVICES=<uuid>`（tp2 为两个 UUID 逗号连接）；`nodeName` 钉节点；
   - 保留 `tre.aibrix.io/gpu-ids` label（逻辑 id 继续供 SlotAllocator/UI 使用），另加
     annotation `tre.aibrix.io/gpu-uuids` 便于审计；
   - **每 GPU bound 预算检查**：同一 GPU 被引用的 Deployment 数 ≤ 3（1 awake + 2 sleeping，
     按 40GB 卡与 N4 实测 sleeping ~1.1GiB 预算），超出直接报错拒绝生成。
3. **SlotAllocator 硬不变式（TDD）**：
   - RED：同 GPU 已有 awake binding 时 `feasible_wake` 返回 False；`bind(awake=True)` 到已有
     awake 的 GPU 被拒绝；
   - SM 所有 wake 路径（target 增加、unhide、defrag 内部 wake）强制过 `feasible_wake`，
     违例返回 409；
   - reconcile 检测到同 GPU 双 awake（外部干预造成）时：告警 + 自动 sleep 后 wake 的那个，
     测试覆盖。
4. 文档：D7 记 **ADR-0006**（含"k8s 调度器不再感知 GPU"的取舍说明）；`04_service_manager.md`
   更新资源模型一节。

验收：`make check` 全绿（新增 ≥ 8 测试）；`make manifests` 人工审查输出（无 GPU requests、
UUID 与节点对应正确、预算检查生效）。

### 10.2 N4b.2 defrag 真实 k8s 路径（离线，fake client）

1. `ops/k8s_ops.py` 增加 `delete_model_deployment` / `create_model_deployment(model, slot)`；
   **create 必须复用 manifests 生成器的同一模板函数**，禁止在 ops 里手写第二份 pod spec
   （否则 D7 改造会漂移出两个真相源）。
2. `/v2/defrag` 编排把打桩 ops 换成真实调用（fake k8s client 测试）：
   `hide → sleep → delete 旧 Deployment → 等 pod 消失（带超时）→ create 新槽 Deployment →
   等 Ready（带超时）→ wake → unhide`；任一步超时/失败即中止，返回已执行/已回滚清单。
3. **幂等与崩溃恢复测试**：defrag 中途 crash 后，`POST /v2/reconcile` 能把 binding 与真实
   pod 状态收敛一致；重复调用 defrag 不产生重复迁移。

验收：`make check`；P9 离线集成 case 更新为走真实 ops 代码路径（fake client），
"碎片夹具 → defrag → 2 卡模型可扩"仍通过。

### 10.3 N4b.3 真机：单卡双模型共卡热切换（dsllama-8b 首次上真机）

> **先做 canary**：D7 无资源请求的 pod 依赖 nvidia container runtime 按 env 挂卡。在
> gpu-operator 环境下先单独 apply 一个 D7 版 dsqwen-7b pod 验证：容器内能看到且只能看到
> 指定 UUID 的卡。若看不到卡，fallback 方案按序尝试：显式 `runtimeClassName: nvidia` →
> privileged + `/dev/nvidia*` hostPath（旧系统等效方式）。canary 结论记 WORKLOG 后再全量。

1. 选 node9 GPU0：部署 `dsqwen-7b` 与 `dsllama-8b` 两个 Deployment 绑同一 UUID（D7 后无配额冲突）。
2. 初始态：7b awake、8b sleeping；校验 `nvidia-smi` 单卡显存 ≈ 1 awake(~37G) + 1 sleeping(~1.1G)。
3. 脚本 20 轮交替：`sleep 7b → wake 8b → 经 gateway 请求 8b 20 条 → sleep 8b → wake 7b →
   请求 7b 20 条`；**wake 前必须确认对方 `/is_sleeping: true`**（sleep 是异步的，SM 编排
   必须串行化这个确认，不能只看下发成功）。
4. 反向测试：对已有 awake 的卡直接调 SM wake → 被 `feasible_wake` 拒绝（409），全程无双 awake。

验收：20 轮无 OOM、无双 awake、wake P95 < 5s、gateway 0 错误；切换耗时分布记入
`12_realenv_tests.md`。

### 10.4 N4b.4 真机：全拓扑部署（重验 N4.1）

1. apply 全部 12 个模型 Deployment（10.1 预算检查通过的布局）；每模型 target 1 awake。
2. 验收清单：全部 pod Running 且绑定正确；`/v2/state` bound=12、awake=3；两节点 `nvidia-smi`
   显存分布与 awake 集合一致；`POST /v2/reconcile` 无 warnings；三模型各经 gateway 发 20 条
   请求 0 错误。
3. `12_realenv_tests.md` 的 N4.1 由 SKIP 更新为 live PASS。

### 10.5 N4b.5 真机：live defrag 与同槽先缩（重验 N4.4）

1. 人工造碎片：两个 1 卡模型分别 wake 到同节点两个 slot 的各一半。
2. 对 dsqwen-14b 加压/直接调 target 请求扩容 → 观察依次触发：HIGH 同槽先缩（若构造了 HIGH
   donor）或 `/v2/defrag` 真实搬迁 → 14b 起在完整 2 卡槽。
3. 验收：搬迁全程 gateway 无 5xx（hide-route 生效）；迁移后 reconcile 无 warnings；
   `12_realenv_tests.md` 的 N4.4 更新为 live PASS。

### 10.6 N4b.6 三模型交替 + 12 小时过夜 soak（重验 N4.6）

1. 三模型 alternating 脚本先跑 15 分钟功能预验（0 错误、三模型均正确扩缩）。
2. 12h soak 无人值守跑：低压三模型轮询，每小时采样 controller/SM RSS、Redis DBSIZE、
   `/v2/state`、pod 重启数；`nohup` 挂后台，输出写本地盘（不写 NFS），次日汇总报告。
3. 验收：无 RSS 增长趋势、无重启、0 请求错误、Redis 键数有界。

### 10.7 N4b 收尾 gate

- `make check` 全绿；`12_realenv_tests.md` 三处 SKIP（N4.1/N4.4/N4.6）全部更新为 live PASS
  （含证据）；顺手修正文末过时的 "No `n4-done` tag" 陈述；WORKLOG 补全；tag **`n4b-done`**。
- 收尾后立即进入 N5（第 6 章顺序不变：R1 → R3 → R7 → R2 → R4 → R5）；R3 若已并行开跑，
  在 `13_experiments_log.md` 登记其输出目录与 commit。

### 10.8 N4b 特有风险

| 风险 | 缓解 |
| --- | --- |
| 无资源请求 pod 在 gpu-operator 环境下看不到卡 | 10.3 的 canary 先行；fallback 链：`runtimeClassName: nvidia` → privileged + hostPath；结论记 ADR-0006 |
| k8s 调度器不再感知 GPU，误调度重叠 | 集群独占 + `nodeName` 钉死 + SlotAllocator 唯一真相源；reconcile 双 awake 自愈兜底 |
| 共卡 wake 时对方 sleep 未完成导致瞬时双占显存 | wake 编排串行化，必须确认 `/is_sleeping: true` 后才 wake，超时则中止并告警 |
| defrag 删建期间容量临时下降 | defrag 前置条件加"该模型 awake ≥ 2 或处于非 CRITICAL"，避免删掉唯一副本 |
