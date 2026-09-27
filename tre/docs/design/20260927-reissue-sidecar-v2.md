# 重试 / 续发 sidecar v2（2026-09-27，计划 P3 / D5 / D6）

分支 `tre/reissue-sidecar-v2`（基于 `tre/transparent-sleep-integration` 47a00dee）。代码 `tre/reissue/tre_reissue/sidecar.py`，
测试 `tre/reissue/tests/`，清单生成 `tre/deploy/gen_model_manifests.py`，registry `reissue:` 段。

来源：`plan-20260927-v2-transparent-sleep-portability.md` 的 D1/D5/D6 与“接口约定”；改编自未合并分支
`feat/reissue-sidecar-20260924`（设计 `20260924-reissue-sidecar.md`）的 sidecar 部分。**没有**带走旧分支的 SM
异步操作 / 排空改动（SM 已在 integration 分支重做），旧 §3.3“只有 SafeScale 排空、其余 drain_s=0”已被 D1 推翻：
所有睡眠路径默认排空，续发只兜底。

## 1. 位置与数据路径

- sidecar 占 pod 的服务端口（默认 8000），vLLM 移到 `127.0.0.1:<reissue.vllm_port>`（默认 8001）。
  Service、网关 `target-pod`、SM（`/sleep` `/wake_up` `/is_sleeping` `/metrics` `/version`）、指标抓取、readiness probe 全部不变。
- 所有路径透明代理（流式保持）；只有下面几类请求有额外行为。
- 只用标准库 + aiohttp（vLLM 镜像自带；0.30 镜像为 python 3.12 + aiohttp 3.14 + uvloop），脚本经 ConfigMap 下发，不需要新镜像。

## 2. sidecar 如何知道本 pod 在睡 / 已隐藏（决定）

**本地状态机，由经过 sidecar 的 `/sleep` 驱动**：
- vLLM 只监听 localhost，唯一能让引擎睡的方式是经 sidecar 的 `/sleep`（或 `/pause`）。
- 该请求必须带 `X-TRE-Hidden: 1`（SM 在 hide → 网关回执之后才发，见 D2），否则 **409**，引擎根本收不到（fail-closed）。所以“带头的 /sleep 到达”本身就证明 pod 已隐藏。
- sidecar 在**转发之前**置 `pending`（abort 输出先于 `/sleep` 返回到达），2xx 后置 `sleeping`；失败回滚；成功的 `/wake_up`（`/resume`）清除。
- 纠偏：每个经代理的 `/is_sleeping` 应答、每 2 s 直连探测、以及每个 `EngineSleeping` 503，都会（受 epoch 保护，`/sleep` 进行中不纠正）把标记同步到引擎实际状态，覆盖 vLLM / sidecar 重启。
- **不用** route-gen / hidden 注解：downward API 的注解文件只按 kubelet 同步周期刷新（可达约 1 分钟），不能用于亚秒级判断；也不需要 sidecar 访问 k8s API。
- `mode=wait` 排空期间（`pending`）新到的请求直接转网关，不打本地引擎；在途请求照常完成；预算耗尽 SM 改发 `mode=abort` 时，被 abort 的请求按 §4 续发。

## 3. 未开始的请求：纯重试（D5）

触发（任一）：
1. 本地已知在睡（`pending` 或 `sleeping`）：`POST /v1/*` 不碰本地引擎，直接转网关；
2. vLLM 返回 503 且 body `error.type == "EngineSleeping"`（fork `--sleep-reject-new`）；
3. 流式响应的**第一个事件**就是 `{"error":{"type":"EngineSleeping"}}`（fork 在 `generate()` 内二次检查时，错误以 SSE 事件出现）；
4. abort 时客户端**一个字节都没收到**（sidecar 延迟到第一个事件才向客户端发响应头）；非流式响应被 abort 且无法续发（不可续发 / 缺 token id）时也是这种情况。

做法：原请求 body **原样**发往 `TRE_GATEWAY_URL`（集群内 DNS），请求头去掉 hop-by-hop、`x-request-id`、`target-pod`、`x-forwarded-*`、`x-envoy-*`，
其余保留（`routing-strategy`、`model` 等）；`x-tre-exclude-pod` = 入站值 ∪ 本 pod 名；`x-tre-reissue-depth` = 入站 + 1。
- 网关 503 / 502 / 连接错误：最多 `retry_attempts`（默认 4）次，退避 0.2 s 起倍增、封顶 2 s，服从 `Retry-After`（同样封顶）。
- 全部失败：客户端收到 **503 + `Retry-After: 1`**（`error.type = ServiceUnavailable`）。
- 成功：透传网关响应，加响应头 `x-tre-retried: <attempts>`。
- 深度超过 `max_depth`（默认 3）：503 + Retry-After（上游 sidecar 会自己退避重试）。

## 4. 已开始的请求：token-id 续发（D6）

触发：本地上游 chunk 出现 `finish_reason == "abort"`，且本地处于 sleeping/pending，客户端仍连着，请求可续发，深度未超限，abort 输出带 token id。

**续发请求**（`build_continuation`）：
- `POST /v1/completions`，`prompt = prompt_token_ids + generated_token_ids`（两者都来自 fork 的 abort 输出，不重新分词，接缝为 0）；
- `max_tokens` = 原上限 − 已生成数（completions 缺省按 vLLM 默认 16 显式化；chat 无上限时用 `max_model_len − prompt_len`，`max_model_len` 取本地 `/v1/models` 并缓存，可用 `TRE_REISSUE_MAX_MODEL_LEN` 覆盖）；预算已用完则直接以 `length` 收尾；
- `min_tokens` 同样扣减；其余采样参数原样（`seed` 保留：固定 seed 在接缝处重启随机流，采样输出有效但与不中断时不逐位相同；`presence/frequency_penalty` 只计输出 token，接缝前的 token 成了 prompt，不再被惩罚——与插件文档同口径）；
- 流式续发强制 `stream_options.include_usage=true`（用于合并 usage，客户端没要时吞掉）；
- 一个 token 都没生成（只发过 chat 的 role chunk）：续发请求就是**原请求**本身。

**chat 的续发方式（决定）：用 completions + 引擎渲染后的 prompt token id**，不用 `continue_final_message`：
- fork 在 abort chunk 里给出的 `prompt_token_ids` 就是引擎按 chat template 渲染并分词后的 id，拼上已生成 id 送 completions，引擎看到的 token 序列与不中断时**逐 token 相同**，接缝精确；
- `continue_final_message=true, add_generation_prompt=false` 要把部分回答作为文本重新渲染、重新分词：接缝处可能切分不同；且 DeepSeek-R1 系模板渲染 assistant 历史时会删掉 `</think>` 之前的内容、generation prompt 与历史 assistant 前缀不同，上下文对不上（旧分支已核实）。
- completions 输出 chunk 改写回 chat 形态（`delta.content`、`object=chat.completion.chunk`、原 id/created/model）。代价：服务端若开 reasoning / tool parser，续发段不经过它们（tool 请求本来就不可续发；当前部署未开 reasoning parser）。

**不可续发**（与网关插件 `treNonContinuable` 同一分类，契约测试逐字段比对）：`n>1`、`best_of>1`、任何 logprobs、`echo`、beam search、
tools/functions（除非 `tool_choice: none`）、结构化输出 / guided decoding（`response_format` 非 text、`guided_*`、`structural_tag`、`structured_outputs`）、
批量 prompt / `prompt_embeds` / `suffix`、completions 与 chat 以外的端点。SM 对这些请求一律等排空；若仍被 abort：
- 流式且已向客户端发过内容：abort 透传，计 `passthrough_abort{reason=non_continuable_*}`；
- 非流式（客户端还什么都没收到）：原请求**从头纯重试**（精确），计 `retry{reason=abort_non_continuable_*}`。
  这是对任务原规则（“不可续发一律透传 abort 并计数”）的有意偏离，已被协调方接受（2026-09-27）：非流式响应在 sidecar 里缓冲，
  客户端没有收到任何字节，重发原请求的结果与不中断时语义相同，只多花一次计算；透传 abort 反而会让客户端看到截断，违反要求 3。
  同理，非流式、可续发但 abort 输出缺 token id 时也走纯重试，而不是 `failed{no_token_ids}`。

## 5. 拼接

**流式**：
- 吞掉第一段的 abort chunk、usage chunk 和 `[DONE]`；abort chunk 里 detokenizer 冲出的尾部文本作为普通 chunk 发出（有 stop 时先扣住，见下）。
- 续发段每个 chunk 的 `id` / `created` / `model` / `object` 改写成原值；chat 去掉重复的 role；fork 的 `generated_token_ids` / `prompt_token_ids` 字段从客户端可见的 chunk 中删除。
- 结尾：合并 usage（`prompt_tokens` = **原** prompt，`completion_tokens` = 各段之和；仅当客户端要了 usage）、SSE 注释行 `: x-tre-continued: <N>`、唯一一个 `[DONE]`。
- **`x-tre-continued` 在流上的形式（决定）**：流式响应头在续发发生前早已发出，HTTP trailer 又不被 aiohttp / Envoy / OpenAI SDK 普遍支持，所以流上用两处承载：
  带 `finish_reason` 的最后一个 chunk 的扩展字段 **`"tre_continued": N`**（OpenAI SDK 放进 `model_extra`，replayer 可直接读），加上结尾注释行。
  非流式响应用真正的响应头 `x-tre-continued: N`。纯重试用响应头 `x-tre-retried`。
- 多级续发：下游 sidecar 在自己的 finish chunk 上写 `tre_continued`，上游读出后改写为 `1 + 下游值`；非流式读下游响应头。
- 续发失败（网关全失败、续发流断开、下游再次 abort）：客户端收到 `finish_reason: abort`（已发内容不回收），计 `failed`。

**非流式**：先缓冲；abort 后用非流式续发，合并文本、`finish_reason`、`stop_reason`、usage，去掉 fork 附加的 token id 字段（客户端自己要了 `return_token_ids` 时保留），加 `x-tre-continued`。

**stop 字符串跨接缝**：续发引擎只在自己的输出上匹配 stop，跨接缝的 stop 两边都看不到。sidecar 取第一段文本末尾 `max_stop_len−1` 个字符
（不含 `include_stop_str_in_output` 时恰为 vLLM 扣住、随 abort chunk 冲出的部分；含时从最近几个已发 chunk 还原）与续发段前 `max_stop_len−1` 个字符拼接匹配：
命中起点落在第一段内的 stop 就在该处截断（`include_stop_str_in_output` 时截到 stop 之后），`finish_reason=stop`，关闭续发连接让下游引擎 abort 余下部分；
完全落在续发段内的 stop 由续发引擎自己处理（续发请求保留原 stop 列表）。非流式在合并文本上做同样检查。

## 6. 控制端点

- `POST /sleep`、`/pause`：必须 `X-TRE-Hidden: 1`，否则 409（`tre_reissue_events_total{event="sleep_rejected_not_hidden"}`）；该头不转发给 vLLM。
- `/wake_up`、`/resume`、`/is_sleeping`、`/health`、`/metrics`、`/version`、`/v1/models` 等：透传（控制类超时 300 s，其它非 `/v1/` 路径 60 s，生成路径不设上限）。
- 自身：`GET /tre-reissue/metrics`、`GET /tre-reissue/state`；每次重试 / 续发 / 透传 abort 在 stdout 打一行 JSON（`event=tre_reissue`）。

## 7. 指标

| 指标 | 含义 |
|---|---|
| `tre_reissue_total{model,kind,reason}` | kind = `retry` / `continue` / `failed` / `passthrough_abort`；reason 细分（`engine_sleeping`、`local_sleeping`、`abort_before_output`、`abort_sleep`、`budget_spent`、`depth_limit`、`retry_exhausted`、`continuation_unavailable`、`continuation_aborted`、`continuation_broken`、`no_token_ids`、`not_sleeping`、`client_gone`、`non_continuable_*`、`abort_non_continuable_*`） |
| `tre_reissue_proxy_added_seconds` | sidecar 在发上游请求前、以及首字节到达后到发给客户端前花的时间（直方图） |
| `tre_reissue_gap_seconds` | abort 到续发首 token（直方图） |
| `tre_reissue_events_total{event}` | sleep 拒绝 / 失败、状态纠偏、`stop_at_seam` 等 |
| `tre_reissue_sleeping` | 本地 sleeping 标记 |

P5 口径：每个 run 报 retry / continue / failed / passthrough_abort；`continue>0` 的 run 标为受污染（计划原文）。

## 8. 配置

sidecar 每个 `Config` 字段 `foo` 都可由环境变量 `TRE_REISSUE_FOO` 覆盖（网关用 `TRE_GATEWAY_URL`，pod 名用 downward API 的 `POD_NAME`）：端口、上游、
路径（`completions_path`、`chat_path`、`sleep_paths`、`wake_paths`、`is_sleeping_path`、`models_path`、自身指标路径）、头名（`hidden_header`、`exclude_header`、
`continued_header`、`retried_header`、`depth_header`）、JSON 字段名（`generated_ids_field`、`prompt_ids_field`、`token_ids_field`、`continued_field`、`sleeping_error_type`）、
重试次数与退避、深度上限、超时。默认值即 fork 与网关插件当前使用的名字。

registry（`deploy/registry.yaml` 与 `overlays/tre-v2/params.yaml` 中的 `tre-v2-registry` 副本保持一致，SM 运行时创建 Deployment 也读它）：

```yaml
reissue:
  enabled: true            # false = 模型 pod 与无 sidecar 时逐字节相同
  gateway_url: null        # null = gateway: 段的稳定 Service（见下）
  vllm_port: 8001
  max_depth: 3
  retry_attempts: 4
  image: null              # null = 模型的 vllm_image
  configmap: tre-reissue-sidecar
  namespace: default
  cpu_request: 50m
  cpu_limit: 500m
  memory_request: 64Mi
  memory_limit: 256Mi
  extra_env: {}            # 额外 TRE_* 环境变量（字段名 / 头名覆盖）
models:
- name: ...
  vllm_features: [sleep_reject_new, abort_return_token_ids]   # 镜像支持时才声明
```

- `vllm_features` 映射为 vLLM 参数 `--sleep-reject-new`、`--abort-return-token-ids`，**只在 `reissue.enabled` 时**渲染（两者只对 sidecar 有意义）。
  当前仓库 registry 的模型仍是 `0.10.1-sleep` 镜像，未声明特性：sidecar 照样部署，能做“已知在睡时转发”和 409 保护，但续发会因缺 token id 计 `failed{no_token_ids}` 并透传 abort，直到换 0.30 fork 镜像（D9）并声明特性。
- **网关地址（2026-09-27 修正）**：Envoy Gateway 自己生成的代理 Service 名带哈希后缀（`envoy-<ns>-<gateway>-<hash>`），不可移植，
  TRE 不再引用它（守卫测试 `test_no_envoy_gateway_hashed_names_in_tre`：`tre/` 下除 `tre/docs/` 外不得出现这类名字）。
  tre-v2 overlay 新增稳定的 ClusterIP `gateway-service.yaml`：`tre-gateway`，位于代理 pod 所在的 namespace（默认 `envoy-gateway-system`，
  Service 只能选同 namespace 的 pod），按 `gateway.envoyproxy.io/owning-gateway-name: tre-aibrix-eg` /
  `owning-gateway-namespace: tre-v2` 选择代理 pod（与 `gateway-stats.yaml` 相同），端口 80 → 10080（EG 把 80 端口监听器映射到容器端口 10080）。
  名字与 namespace 是 kustomize 参数（`gateway-service-params.yaml`，local-config，经 `replacements` 写入 Service）。
  registry `gateway.service_name` / `service_namespace` / `service_port`（默认 `tre-gateway` / `envoy-gateway-system` / 80）渲染出
  sidecar 的 `TRE_GATEWAY_URL = http://tre-gateway.envoy-gateway-system.svc.cluster.local:80`；`reissue.gateway_url` 非空时覆盖。
  外部 NodePort / LoadBalancer 的固定仍是 P4。
```yaml
gateway:
  route_timeout_s: 150
  service_name: tre-gateway
  service_namespace: envoy-gateway-system
  service_port: 80
```

## 9. 清单（`make manifests`）

- 生成 ConfigMap `<reissue.namespace>/<reissue.configmap>`，内容是 `sidecar.py`（守卫测试保证与源码一致，过期时提示 `make manifests`）。
- 每个模型 Deployment：vLLM `--host 127.0.0.1 --port <vllm_port>`，去掉其 ports / readinessProbe；新增容器 `tre-reissue-sidecar`：镜像 = `reissue.image` 或模型镜像，
  `python3 /opt/tre-reissue/sidecar.py`，端口 8000，readiness `/health:8000`（经代理即 vLLM 的 /health），`NVIDIA_VISIBLE_DEVICES=void`，
  `TRE_REISSUE_REQUIRE_HIDDEN_HEADER=true`，CPU 50m/500m、内存 64/256Mi。
- `model.aibrix.ai/port` 仍为 8000；Service targetPort 8000；SM 的 `build_model_deployment` 与渲染结果逐字段相同（有测试）。
- `deploy/scripts/staggered_model_fleet.py` 的离线拉起 `/sleep` 带 `X-TRE-Hidden: 1`（拉起中的 pod 本就 routable=false）。

## 10. 开销（微基准 `tre/reissue/tests/bench_proxy_overhead.py`）

引擎与 sidecar 分进程（同 pod 两容器的情形），客户端逐请求交替直连 / 经 sidecar。76 主机上，于 `vllm/vllm-openai:0.30.0` 镜像内运行（python 3.12 + uvloop + aiohttp 3.14）：

| 指标 | 直连 p50 | 经 sidecar p50 | 增加 p50 | 增加 p99 |
|---|---|---|---|---|
| 非流式往返 | 5.68 ms | 6.04 ms | **0.36 ms** | 0.64 ms |
| 流式 TTFT | 5.85 ms | 6.58 ms | **0.73 ms** | 0.72 ms |
| 流式 token 间隔 | 5.143 ms | 5.149 ms | **0.006 ms** | 0.25 ms |
| 32 路同时发起的 TTFT | 7.5 ms | 11.6 ms | 4.1 ms | 6.2 ms |
| 32 路 token 间隔 | 5.71 ms | 5.77 ms | 0.05 ms | 0.31 ms |

- sidecar CPU ≈ 60–80 µs / chunk（0.5 核上限约 6–8k chunk/s）。
- 纯 aiohttp 代理（`TRE_REISSUE_ENABLED=false`）TTFT +0.62 ms、非流式 +0.57 ms：续发逻辑本身在快路径上几乎不加开销（快路径只做 `b'"abort"' in chunk` 与切分完整事件）。
- 32 路“同一时刻”发起的 TTFT 增加是单线程 sidecar 串行处理一批请求头的排队（~0.13 ms/请求），真实负载不会这样同步到达。
- 主机 python 3.10（无 uvloop、aiohttp 3.11）：非流式 +0.70 ms、TTFT +0.96 ms、token 间隔 +0.005 ms。
- 验收（计划 P3：p50 < 1 ms）在镜像环境满足。TTFT 比非流式多约 0.35 ms，是因为 sidecar 等第一个事件到达才向客户端发响应头（为了能对“流内 EngineSleeping 错误 / 首事件即 abort”做纯重试并给客户端真实的 503）。

## 11. 已知限制 / GPU 验证前的缺口

- 真实 fork 引擎尚未联调：字段名按 fork 分支 `tre/transparent-sleep`（c711cfb1fc / 33a2d082de / a2659293c0）源码核对，未在 GPU 上跑过。需验证：abort chunk 在 chat 中 `prompt_token_ids` 位于顶层、completions 位于 choice；abort 前未出 token 时也发 abort chunk；`--sleep-reject-new` 503 与流内错误两种形态。
- 多字节 UTF-8 字符正好跨接缝时，第一段 detokenizer 扣住的半个字符由 abort 冲出方式决定，续发段从 prompt 尾部重建解码状态；可能出现一个替换字符（未验证）。
- 未做“卡在 pause 后面”的请求检测（旧分支的 stuck scan）：依赖 `--sleep-reject-new`；没有该特性的镜像上，与 /sleep 竞速到达引擎的极少数请求会等到 wake。
- 在途占用翻倍：续发期间原连接（客户端 → Envoy → pod A sidecar）与续发请求（sidecar A → Envoy → pod B）同时占网关配额，gateway_health 的分母被抬高（同旧设计 §5）。
- 续发在 B 上重新 prefill（部署关闭了 prefix caching），gap 计入 TPOT / E2E，TTFT 保持第一段。
- replayer 已记录每请求 `finish_reason` / `tre_continued`（finish chunk 字段、结尾注释或 `x-tre-continued` 头）/ `tre_retried`（`x-tre-retried` 头），run 摘要有 `reissue{abort,retry,continue,continued_segments,by_model}` 与 `reissue_contaminated`（continue>0）。loadgen_v1 未改。
- `tre-gateway` Service 尚未 apply 到集群；在它存在之前，按新清单起的 sidecar 无法重试 / 续发（请求照常透传，只是兜底失效）。
- 端到端验收（`mode=abort` 强制睡眠 + 32 路在途 → 100% 完整响应、接缝 0）需要 GPU 与 fork 镜像，未做。
