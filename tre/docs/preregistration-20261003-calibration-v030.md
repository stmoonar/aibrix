# θ 重标定（vLLM 0.30 引擎）：预注册（2026-10-03）

> **状态：草案，等用户确认 §9 的 3 处"待确认"后定稿。** 定稿 = 本文件在分支 `calib/theta-20261003` 上的那次提交；之后不改正文，偏离一律追加到文末"偏离记录"（时间、原因、对结论的影响）。
> 本文件只写本轮新增或改变的规则。没写到的方法决定沿用 `docs/preregistration-20260923-calibration-run2.md`（下称"run2 预注册"）和 plan 2026-09-21 §6.11 的 D1–D22。
> 旧冻结目录（`calibration_freeze_20260923/`、`calibration_v1lambda_20260924/` 等）只读，不改。本轮所有产物写到新根目录 `$CALIB_ROOT`（执行计划 `deploy/RUN-calib-theta-20261003.md` 给出取值）。

## 1. 为什么重标定

线上 θ 在 vLLM 0.10.1 上标定。之后引擎升到 0.30 fork（cumem sleep、KV 记账变化、8b tokenizer 修复使 token 数 −18%），网关全走 ext_proc，基线布局改了。D9：换引擎必须重标定。旧数据只能当参考，不进本轮训练或验收。

## 2. 被测系统（采集期间固定）

| 项 | 值 |
|---|---|
| 控制面 | 线上版本 `integ/tre-v2-20261001b`（bdea7a72，镜像源码 ba5b558f） |
| 引擎镜像 | `vllm-openai-tre:0.30.0-ts-8dc0f2a7`（+ reissue sidecar） |
| 采集代码 | 分支 `calib/theta-20261003`（= bdea7a72 + 标定改动）；每个 run 的 manifest 记录 git sha |
| run mode | `observe observe`（controller、SM 都只观测） |
| 副本 | 每模型 1 个可路由副本：7b node9/GPU0、8b node9/GPU1、14b node10/GPU0-1（= 现有基线 awake 集合） |
| 发送端 | 统一客户端，chat + ignore_eos，`--corpus-lang mix --zh-ratio 0.5`，路由头 `routing-strategy: least-gpu-cache`，4 个发送进程 |
| SLO 口径 | 客户端口径（D6′） |

## 3. 标签：先重拟空载 TTFT c/b（新增，硬前提）

- D6′ 主标签：TTFT 阈值 = `max(500 ms, 5·(c_m + b_m·L))`，TPOT 75 ms；固定 500/75 作对照列。
- **c/b 必须在任何打标签的采集（run1）之前，在 0.30 引擎上用专门的空载采集重拟**：
  - 每模型单副本，严格串行（并发 1），输出 16 token（ignore_eos）；
  - L ∈ {128, 256, 512, 1024, 2048, 3072, 4096}，每个 L 约 30 个请求，L 顺序按种子打乱交错，先丢预热请求；
  - 只用孤立请求（发出时无在途请求、prefill 期间无新请求、HTTP 200），客户端口径 TTFT，Huber 回归；
  - 结果写入 `deploy/registry.yaml`（`models[].slo.ttft_idle_c_ms` / `ttft_idle_b_ms_per_token`）并提交；该 commit 的标签定义 sha256 记入每个 run 的 manifest。
- c/b 定下后，本轮不再改。若事后发现 c/b 有误，整轮标签作废，按偏离处理。
- SafeScale 用 labels 模式，读同一组 c/b。

## 4. 采集内容

### 4.1 训练集（D16：θ 只在稳态 hold cell 上拟合）

| 阶段 | 内容 | 用途 |
|---|---|---|
| run1 | `--design primitives`（第一轮的 steps / boundary / ramp / bursts） | 产出 run2 的先验（rho_priors、regime_groups）；H2 训练池；M 的 3 个保留混合 cell |
| run2 | `--design ladder`（7 个训练形状的边界搜索 + hold 阶梯 + ramp + 补点 + 哨兵，规则同 run2 预注册 §7） | 主训练集 |
| S3 补充 | S3 在 run2 ρ* 之上重探 + ρ* 处 300 s smoke（D19） | M / stage3 / T14 容量先验读 S3 锚点 |
| **P1 深度过载** | S2 / S3 / T8 / S4 / S5 × {1.5, 2.0, 3.0}·ρ*_run2，每 cell 150 s，每模型 15 cell（`--training-plan p1-deep-overload`）；另加 3 个 S2 漂移哨兵 **【待确认 3】** | 训练（hold cell） |

- **只补 P1**（用户 10-03 定）。不加 P0 新形状、P2 静态网格、P3 长输出。v2 的形状覆盖仍是 7 个训练形状，这一点作为局限披露。
- P1 cell 全在违规侧，不参加"每个形状健康和违规都要有"的检查；每个 P1 cell 至少要有 1 个有标签的窗，否则退出码 3。
- 每个 P1 cell 开始前等引擎排空（running + waiting = 0，最多 300 s；状态门，不是计时器）。
- P1 不帮助辨识 λ（W 饱和时 TSS→0，与 λ 无关），只补深度过载区的样本。
- **TokenScale 饱和点不放进标定窗口**（用户 10-03 定）。

### 4.2 验收集与测试集（冻结后采集，D22）

- **M**：每模型 13 cell（10 采集 + 3 保留），构成同 `calibration_acceptance.py` 模块说明。保留的 3 个 cell 来自本轮 run1，不是旧引擎数据。
- **T14（仅 14b）**：8 个留出形状（插值 G512x256 / G1200x240 / G640x400 / G1800x160；外推 G3072x96 / G4096x64 / G256x768 / G512x1024）× {0.9, 1.0, 1.1}·Ĉ，每 cell 240 s，共 24 cell（降级版）。**【待确认 1】** 是否改采完整 P0（40 cell，{0.7…1.3} 5 档，约 3.1 h；代码未实现）。
  - Ĉ 来自容量模型 `1/R = c0 + c_in·in + c_out·out`，**只用本轮训练集**（run2 + S3 补充的 D6′ 边界）拟合，在 T14 采集前写进 T14 预注册 JSON 并记 sha256。
  - T14 形状登记进留出名单，代码断言检查；种子与训练集不相交；T14 不做自适应补点。
  - 每组冻结参数在 T14 上只评估一次。

### 4.3 采集期守卫（任一触发即作废该 cell；同一 cell 连续两次作废即停整轮）

- 发送迟到：on-wire p99 > 50 ms → 作废。
- 时钟域：run 开始前检查 gateway/controller 对 Redis TIME 的偏差，失败即拒跑。
- prompt 预检：prompt_tokens 与目标长度不符即拒跑。
- **续发污染**：cell 期间该模型任一 pod 的 sidecar `tre_reissue_total{kind="continue"}` 增量 > 0 → 该 cell 标记受污染，dataset 排除并计数；抓取失败记"未测量"，不当作 0。
- 负载路径（语料、路由、api）不一致的数据不能合并。
- cell 之间排空：引擎 running + waiting = 0（上限 90 s），间隔不少于窗长 30 s。

### 4.4 保留的原始数据（为 §5 的离线比较）

- 每 pod 1 Hz vLLM 指标（`vllm_metrics_1hz/`）：全部 `vllm:*_total` 计数器、running 与 waiting 分开、KV 使用率、`vllm:cache_config_info`（num_gpu_blocks）。
- 网关 token 计数与 controller TSS 记录。
- 原始数据一律保留，不删不覆盖；作废的 attempt 也登记。

## 5. 拟合（D2 / D5 / D13 / D17 / D18 沿用）

- θ 发布三族合并值；w_p 按受约束 1-SE 规则；τ = 10 s；θ CI 半宽门槛 20%（D13，09-24 修订）。
- **两个开放选项在训练集上离线比较，交用户定**（暂停点，执行计划阶段 E 之后）：
  1. **λ_wait**：(a) v2 规则：λ = 1，只有 BA 增益 ≥ .02 才偏离；(b) v1-λ 拟合（`dline_refit wp --lambda-method v1`，线上现行做法）。
  2. **TSS 分子**：(a) 现行 D2：网关 30 s 窗内 token 总量；(b) L3：每 pod vLLM 计数器（prompt + generation tokens）的窗内增量。两者都由 §4.4 的数据离线重算，不重采。选 L3 时，θ 上线前要先改网关写计数器、TRE 读差值。
- 比较只用训练集（含交叉验证），**不看 M / T14**。用户选定后冻结（`dline_refit freeze`），冻结文件记录选项。若用户要求冻结两组参数（例如 λ 两种），每组在 M / T14 上各评估一次，事先写明主参数组和替换规则：备选组只有在 M 上 A、B′ 都过且 BA 不低于主参数组 .02 以上时才替换。

## 6. 验收（plan §6.9f，B 改为 B′）

- **A**：BA ≥ .80；BA CI95 下界 ≥ .75；BA ≥ 训练 BA − .08（不变）。
- **B′（门，替代旧 B）**：
  - 目标族：违规窗中 severity ≥ 该模型**训练集**违规窗 severity 的 .65 分位（severity = 0.8·p95 比 + 0.2·avg 比；与 τ_crit 拟合的临界分位一致）。
  - 分位点只从本轮训练集取，在 `dline_refit freeze` 时计算并写入冻结产物，**在 M / T14 采集前落盘**。
  - 判定：CRITICAL = Z < τ_crit，τ-EMA 10 s，**dwell = 线上值**（现为 `TRE_DWELL_WINDOWS=1`，即不加 dwell）。
  - 门限：召回 ≥ .80，召回 CI95 下界 ≥ .70；健康窗误报 ≤ .05，误报 CI95 上界 ≤ .08。
  - 理由：τ_crit 只针对最严重的 35% 违规拟合，旧 B 考全部违规，口径不一致。这是对齐，不是放宽。
- **披露（不作门）**：dwell 2 下的 B′；全部违规的召回（dwell 1 与 2）；旧 B；落入 LOW 区（τ_crit ≤ Z < 1）被慢环接住的比例；TTFT-only 违规召回（C）。
- **D**：训练停止规则（D13），按冻结时记录。
- 区间：cell 级 bootstrap（独立样本约为窗数 ÷ 6）。
- 验收集 M、T14 先冻结并记 hash，每组参数只评估一次。
- **风险说明（事先写明）**：09-24 的 B′ 门限是在 dwell 2 的训练数据上定的；dwell 1 下召回会更高、误报也会更高，误报门可能更难过。若 B′ 不过，按 §8 处理，不在看过 M 之后改门限。

## 7. 种子与目录

- 设计种子：训练各阶段 `20261003`，T14 `20261004`（与 20260923 / 20260924 不相交）；T14 种子写进 T14 预注册 JSON，并检查与训练集不相交。
- 本轮全部产物在 `$CALIB_ROOT` 下按阶段分目录；失败的 attempt 保留、登记，重跑写新目录。

## 8. 结果处理

- A、B′、D 都过：发布冻结参数（θ、λ、τ、w_p、c/b 一起原子更新，经用户确认后上线）。
- 不过：如实报告，交用户定（发布冻结参数并披露，或保留线上参数）。不允许看过 M / T14 后改规则再评一次。

## 9. 待确认（用户）

1. **【待确认 1】T14 规模**：降级 24 cell（默认，约 2.2–2.9 h）还是完整 P0（40 cell，约 3.1 h，需改代码）。
2. **【待确认 2】是否重跑 run1**：默认重跑（约 5.5 h，先验、H2、M 保留 cell 都来自新引擎）。替代方案：沿用旧 run1 的先验只作搜索起点（不算泄漏，但会失准），省约 5.5 h，代价是 M 改成 10 cell（要改代码）且训练集少一份 H2。
3. **【待确认 3】P1 的 3 个 S2 哨兵**：保留（默认，每模型 18 cell，多约 12 min，用于检查深度过载后引擎有没有漂移），还是严格 15 cell。

（已定、不需再确认：B′ 按线上 dwell 判，现为 1；dwell 2 只披露。若之后线上改 dwell，在偏离记录中写明并按新值补报。）

## 偏离记录

（采集开始后，任何对上文的偏离追加于此。）
