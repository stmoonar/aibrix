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
- **回环连接 keep-alive（2026-09-30）**：sidecar 到本地 vLLM 的连接池空闲保留 `upstream_keepalive_s`（默认 2 s），必须小于 vLLM 的
  `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`（vLLM 默认 5 s，registry `vllm.env` 设为 75 s）。此前池用 aiohttp 默认的 15 s：vLLM 5 s 关掉空闲连接的同时
  sidecar 正好复用它，请求在首字节前失败（`Server disconnected` / `Connection reset by peer` / `Can not write request body`），客户端收到 502
  （09-30 smoke 的 17 个 502，与过载无关）。registry 校验两者关系（至少低 1 s：sidecar 限 0.5 核，被限流时池记录的释放时间会滞后于 vLLM 的空闲计时），
  sidecar 启动时按同一规则再查一次（不满足只打 WARNING）。见 §3a。

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
- **只有一层重试（2026-10-05）**：另一个 sidecar 产生的 502 / 503（`error.message` 以 `tre-reissue sidecar:` 开头：深度超限、它自己的重试已用完、
  它的本地引擎连不上）是终态，不再重试——那个 sidecar 已经用完了自己的有界重试，再重试会让每一跳的发送次数相乘
  （嵌套两层时是 `retry_attempts × (1 + retry_attempts)` 次）。Envoy / 网关自己的 502 / 503（没有可路由的 pod、连接失败）照常重试。
  结果计 `failed{retry_exhausted}`（续发路径计 `failed{continuation_unavailable}`），`error` 字段写明是下游 sidecar 的终态应答。
- 全部失败：客户端收到 **503 + `Retry-After: 1`**（`error.type = ServiceUnavailable`）。
- 成功：透传网关响应，加响应头 `x-tre-retried: <attempts>`。
- 深度超过 `max_depth`（默认 3）：503 + Retry-After（上游 sidecar 会自己退避重试）。

## 3a. 本地引擎连接在首字节前失败（2026-09-30）

适用于 sidecar 发往本地 vLLM 的所有请求（生成路径 `_generation`、普通代理 `_proxy_local`、`/sleep` `/wake_up` 等控制调用、`/is_sleeping`），
此时**还没有向客户端写过任何字节**：
1. **复用的池连接**上发生连接级失败——`ServerDisconnectedError`，或 ECONNRESET / EPIPE（含 aiohttp 把写失败包成
   `ClientOSError(errno=None, "Can not write request body")`、原因链上是 reset 的情况，aiohttp 3.11 与 3.14 相同）——用**新连接**
   （`force_close` 的独立 session，不会再拿到同龄的池连接）重发，最多 `local_reconnect_attempts`（默认 1）次。
   是否复用由 aiohttp 的 `on_connection_reuseconn` trace 钩子逐请求标记：keep-alive 竞态只可能发生在复用连接上，而 uvicorn 只在连接空闲时关闭它
   （有数据到达就取消 keep-alive 计时），所以这时请求没有被执行；**新建连接**上的同类失败（例如 vLLM 在长生成中途崩溃）不重发，
   因为服务端可能已经执行了请求。请求体是已读完的 bytes，重发与原请求逐字节相同。
   计数 `tre_reissue_local_reconnect_total{result=ok|fail}`（按次）。GET 等幂等请求 aiohttp 自己已会在同一个池上重发一次，sidecar 的新连接重发叠加在其后。
2. 仍失败时，生成路径：本地已知在睡 → 经网关重试（`retry{local_unavailable_sleeping}`，原有）；**connection refused**（vLLM 没在监听：崩溃 / 重启）
   → 经网关重试（`retry{local_refused}`，带 `x-tre-exclude-pod`）；其余 → **503 + `Retry-After: 1`**，
   `{"error": {"type": "ServiceUnavailable", "layer": "sidecar_upstream"}}`（不再是 502），计 `failed{upstream_unavailable}`。
   refused 的判断假定上游是单地址（清单写死 127.0.0.1）；若改成 `localhost` 这类多地址名，aiohttp 抛无 errno 的合并异常，会走 503 而不是网关。
3. 普通代理路径：连接级失败重发后仍失败 → 同样 503 + `layer=sidecar_upstream`；refused 等其它连接错误仍是 502（带 `layer`），不转网关
   （`/health` 等探测必须反映本 pod）。只有 `POST /v1/*`（`enabled=false` 时的生成请求）计入 `failed{upstream_unavailable}`，探测与 `GET /v1/models` 不计。
4. 日志：计入 `failed{upstream_unavailable}` 的请求每个打一行 `tre_reissue`（带 `status`、`request_id` = `x-request-id`、`error`）；
   另有限频的 WARNING 行（`tre_local_reconnect` / `tre_upstream_unavailable`，每类每 `warn_interval_s`=10 s 至多一行，带 `suppressed`）。
   修复后这类事件很少，逐条打不会刷屏。
5. **已经向客户端写过字节**（流已开始后上游断开）不在此范围：保持原行为（关闭客户端连接；睡眠导致的 abort 走 §4 续发），不本地重发。
6. **客户端实际看到的形态**：上面说的是 sidecar 的响应。TRE 网关插件在响应体阶段会用 ext_proc 的 ImmediateResponse 重写上游 5xx：
   09-30 smoke 里客户端看到的是 `{"error": {"message": "<sidecar 的整段 JSON>", "type": "api_error"}}`，`layer` 只以文本出现在 message 里，
   `Retry-After` 很可能被丢掉（待下次 smoke 经 31094 确认）。E1 分类按 message 中的子串 `sidecar_upstream` 判断。

Envoy → sidecar 这一跳（2026-09-30 补修）：同类竞态，方向相反（sidecar 是服务端）。sidecar 服务端（aiohttp `web.run_app`）默认 75 s 关闭空闲连接，
Envoy 上游连接空闲超时默认 1 h，表现是 Envoy 自己回 503（`UC`/`UF` 标志）。原则同上：**连接由发请求的一方先关**，客户端空闲超时必须小于服务端。
- 推理 POST 走 ext_proc 选 pod，上游是 `EnvoyPatchPolicy` 手写的 ORIGINAL_DST 集群（`overlays/tre-v2/gateway-extproc.yaml`，Envoy 1.33.2 / EG 1.2.8）。
  BackendTrafficPolicy 只作用于 EG 自己从 HTTPRoute 生成的集群，**够不到**这些集群；所以在集群定义里直接设
  `typed_extension_protocol_options.HttpProtocolOptions.common_http_protocol_options.idle_timeout: 60s`（仍是 HTTP/1.1）。
  BackendTrafficPolicy `timeout.http.connectionIdleTimeout: 60s` 同时加在 `gateway-hardening/backendtrafficpolicy-tre-v2.yaml`，覆盖非推理请求走的 Service 路径集群。
  另外 ORIGINAL_DST 集群的主机在 `cleanup_interval`（5 s）内没被用到就会被摘除，连接池随之排空，所以实际暴露面本来就小；显式 idle timeout 是纵深防御。
- sidecar 服务端 keep-alive 可配：registry `reissue.server_keepalive_s`（默认 75，= aiohttp 默认），传给 `web.run_app(keepalive_timeout=...)`。
  registry 校验 `reissue.server_keepalive_s >= gateway.upstream_idle_timeout_s + 1`（默认 75 vs 60）；sidecar 关闭时改校验每个模型的 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`。
  单一来源是 registry `gateway.upstream_idle_timeout_s`（默认 60），守卫测试要求两份手写 YAML 与它相等。sidecar 启动时也做同样的检查（不满足只打 WARNING）。
- 路由上没有 retry_policy（没有 retry_on），本次**不加**：Envoy 1.33.2 已支持 `reset-before-request`，但 POST 生成请求的请求体可能已经写出，
  且 ORIGINAL_DST 集群重试仍会打到同一个 pod（目标由 `target-pod` 头决定），换不了 pod；`reset` / `connect-failure` 更分不清“没执行”和“执行了一半”。
  要不要加留给用户决定。
- **aibrix-system 网关（31592）路径不在本次范围**：它也能路由到这些 pod，Envoy 空闲超时仍是默认 1 h，这一跳的竞态在该路径上依旧存在。
  按 ADR-0008 不改 aibrix-system；实验两臂都走 31094（`campaign_queue.py` 的 GATEWAYS，第 80-90 行附近），所以实验不受影响。
- 重发窗口（P2-1）：`local_reconnect_window_s`（默认 1 s）：拿到复用连接后超过这个时间才失败的请求（例如 vLLM 在长生成中途崩溃）不再重发，避免重复执行。
- `upstream_keepalive_s` / `local_reconnect_attempts` / `local_reconnect_window_s` / `server_keepalive_s`：
  **仓库 registry.yaml 里这四个键保持注释**：线上 controller / SM / UI（20260930-f8ccb0ca 及更早）的 `parse_reissue_config` 遇未知 reissue 键会 raise
  （controller 启动即 crashloop，UI 的 `PUT /api/params` 全部失败）。**controller、SM、UI 三个镜像全部升级到含本提交的版本之后，才可在 live registry 里显式写这些键**
  （不写时用代码默认值，行为相同）；守卫测试断言 registry.yaml 的 reissue 段只含旧版已知键。`gateway.upstream_idle_timeout_s` 旧版会忽略，可直接写。
- 重发窗口之外失败的请求不重发，计入独立指标 `tre_reissue_local_reconnect_skipped_total{reason="outside_window"}` 并写 WARNING `tre_local_reconnect_outside_window`
  （`tre_reissue_local_reconnect_total` 只数实际发生的重发，`result=ok/fail`；2026-10-01 之前的镜像把它记在 `tre_reissue_local_reconnect_total{result="outside_window"}` 里）。
- 指标 `tre_reissue_gap_seconds` 拆出 label `mode`：`stream` = abort 到首个续发 token，`nonstream` = abort 到完整续发响应（含续发生成时间），两者不可比。
- aiohttp 对幂等方法（GET/HEAD/OPTIONS/TRACE/PUT/DELETE）在持久连接失败时已内置重试一次；POST 没有，所以生成请求的重发只靠本层。

验证：`tre/reissue/tests/test_local_keepalive.py` 用真 uvicorn（keep-alive 1 s）+ sidecar 造“空闲略超 keep-alive 后复用”的时序：
main 上 3000 个请求 14 个 502（测试失败）；修复后 0 失败、重发 > 0；只开第 1 层（池 0.5 s、不重发）0 失败；旧配置对照组的失败全为 503。
vLLM 镜像（py3.12 + aiohttp 3.14 + uvloop + httptools，`--network none`）内同样时序 2000 请求：main 17–18 个 502，修复后 0（重发 13–23 次）。
线上证据（09-30 smoke 重跑 `smoke-e1-20260930/rerun-1815/{tre,apa}/errors_evidence.*`）：7 个 502 全部来自同一个 7b pod（在飞最多的那个，
连接池最大），sidecar 报文是 `upstream unavailable: Server disconnected` 或 `[Errno 104] Connection reset by peer`（正是本节第 1 类），
Envoy 耗时 7–112 ms（很快失败、不是超时），发生时 waiting=0、running 正在陡降（如 141→38）——负载回落沿上大批连接变空闲、约 5 s 后被 vLLM 关闭，
与 keep-alive 竞态吻合，与过载无关。

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

**不可续发**（与网关插件 `treNonContinuableReason` 同一分类、同一组 reason 字符串；两边测试都读共享契约 `tre/reissue/contract/non_continuable_cases.json`；两边都忽略 query，网关的请求校验本就按精确路径匹配、带 query 的请求到不了分类）：`n>1`、`best_of>1`、任何 logprobs、`echo`、beam search、
tools/functions（除非 `tool_choice: none`）、结构化输出 / guided decoding（`response_format` 非 text、`guided_*`、`structural_tag`、`structured_outputs`）、
批量 prompt / `prompt_embeds` / `suffix`、chat 的 `messages` 缺失或为空、body 不是 JSON 对象、completions 与 chat 以外的端点。SM 对这些请求一律等排空；若仍被 abort：
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
- 客户端断开（2026-10-05）：sidecar 的 HTTP 服务开 `handler_cancellation`，客户端连接断开时取消 handler，handler 持有的上游请求（本地引擎、经网关的重试或续发）随之关闭，vLLM 见连接关闭即 abort（排队中的请求不再 prefill）。sleep 造成的上游中断（客户端仍在）照常续发。`/sleep`、`/wake_up` 用 `asyncio.shield`，调用方断开也执行到底，sleeping 标记跟随引擎结果；调用方走后该调用若出错，记一条 WARNING（`tre_control_call_failed_after_disconnect`）。已进入重试 / 续发的请求被取消时补记一次 `tre_reissue_total{reason="client_gone"}`（重试为 `kind=retry`，续发为 `kind=passthrough_abort`）。
- 打开文件上限（2026-10-05）：启动时把软 `RLIMIT_NOFILE` 提到 min(`TRE_REISSUE_NOFILE_TARGET`（默认 65535）, 硬上限)，只升不降，失败只记 WARNING；启动日志一行 `event=tre_reissue_nofile`（before / after / hard）。原因：Docker ≥ 25 / containerd ≥ 2.0 的容器默认软 1024，pod spec 不能设 ulimit。
- 自身：`GET /tre-reissue/metrics`、`GET /tre-reissue/state`；每次重试 / 续发 / 透传 abort 在 stdout 打一行 JSON（`event=tre_reissue`）。

## 7. 指标

| 指标 | 含义 |
|---|---|
| `tre_reissue_total{model,kind,reason}` | kind = `retry` / `continue` / `failed` / `passthrough_abort`；reason 细分（`engine_sleeping`、`local_sleeping`、`local_refused`、`local_unavailable_sleeping`、`upstream_unavailable`、`abort_before_output`、`abort_sleep`、`budget_spent`、`depth_limit`、`retry_exhausted`、`continuation_unavailable`、`continuation_aborted`、`continuation_broken`、`no_token_ids`、`not_sleeping`、`client_gone`、`non_continuable_*`、`abort_non_continuable_*`） |
| `tre_reissue_proxy_added_seconds` | sidecar 自身给一个本地应答请求增加的时间（直方图）：forward（读完客户端请求 → 交给上游 HTTP 客户端）+ relay（每个上游响应头 / 数据块从收到到写给客户端，按请求累加）；不含等待上游的时间（响应头，即非流式请求的整个生成过程、块间间隔、经网关的续写请求）；块写入在发送缓冲超过高水位时会包含客户端背压；被转发重试的请求不计入。两部分另见 `tre_reissue_proxy_forward_seconds` / `tre_reissue_proxy_relay_seconds` |
| `tre_reissue_gap_seconds{mode}` | abort 到续发（直方图）：`mode="stream"` = 到首个续发 token，`mode="nonstream"` = 到完整续发响应（含续发生成时间），两者不可比 |
| `tre_reissue_events_total{event}` | sleep 拒绝 / 失败、状态纠偏、`stop_at_seam`、`client_cancel`（客户端拿到完整响应前请求被取消：客户端断开，或服务关闭；上游已关闭。流式 `[DONE]` 已写出后的断开不算）等 |
| `tre_reissue_sleeping` | 本地 sleeping 标记 |
| `tre_reissue_local_reconnect_total{model,result}` | 首字节前连接级失败后的新连接重发次数，`result=ok/fail`（§3a） |
| `tre_reissue_local_reconnect_skipped_total{model,reason}` | 首字节前连接级失败但**没有**重发的次数；`reason=outside_window`：复用连接在交出后超过 `local_reconnect_window_s` 才失败（不是 keep-alive 竞态，避免重复执行）（§3a） |

P5 口径：每个 run 报 retry / continue / failed / passthrough_abort；`continue>0` 的 run 标为受污染（计划原文）。

**与 SM 计数对账（2026-09-30）**：SM 的 `forced_abort_requests`（及 `aborted.in_flight` / `continuable`）是 `/sleep mode=abort`
**发出之前**一次负载读数（引擎 running+waiting、网关 inflight），是被 abort 请求数的**上界**；读数到 abort 生效之间（`waited_s` 加提交阶段，
约几十到一两百 ms）正常结束的请求算在里面，但引擎不会给它们发 abort，sidecar 也就不续发、不记任何事件，客户端拿到的是完整响应。
所以 `continue + passthrough_abort + retry(abort_*) + failed ≤ forced_abort_requests`，差值不是丢请求。实例：09-30 smoke 重跑
（`smoke-e1-20260930/rerun-1815/apa`）SM 计 135、sidecar 续发 133，差的 2 个都在 node9/gpu-2 的第二次睡眠（SM 计 20、续发 18）：
Envoy 访问日志显示该 pod 在 10:42:50.70–.80 在飞 20 个，其中 2 个于 10:42:50.818 以 200 正常结束，abort 生效后在飞 18 个，18 个全部续发成功（时长 ≥10 s，经拼接）。

日志字段：`tre_reissue` 行的 `gap_ms` 是 abort 到续发首 token 的毫秒数（2026-09-30 之前的日志里同一个毫秒值叫 `gap_s`，读旧日志时按毫秒处理）；
指标 `tre_reissue_gap_seconds` 一直是秒。

## 8. 配置

sidecar 每个 `Config` 字段 `foo` 都可由环境变量 `TRE_REISSUE_FOO` 覆盖（网关用 `TRE_GATEWAY_URL`，pod 名用 downward API 的 `POD_NAME`）：端口、上游、
路径（`completions_path`、`chat_path`、`sleep_paths`、`wake_paths`、`is_sleeping_path`、`models_path`、自身指标路径）、头名（`hidden_header`、`exclude_header`、
`continued_header`、`retried_header`、`depth_header`）、JSON 字段名（`generated_ids_field`、`prompt_ids_field`、`token_ids_field`、`continued_field`、`sleeping_error_type`）、
重试次数与退避、深度上限、超时、回环 keep-alive（`upstream_keepalive_s` 默认 2、`upstream_server_keepalive_s` 默认 5 仅用于启动检查、
`local_reconnect_attempts` 默认 1、`local_reconnect_window_s` 默认 1、`server_keepalive_s` 默认 75、`gateway_upstream_idle_s` 仅用于启动检查、`warn_interval_s` 默认 10）。默认值即 fork 与网关插件当前使用的名字。

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
  # 2026-09-30（旧版 controller/SM/UI 拒绝这些键：三个镜像都升级之后才可写进 live registry，仓库里保持注释）：
  # upstream_keepalive_s: 2       # 须小于每个模型的 VLLM_HTTP_TIMEOUT_KEEP_ALIVE（registry 校验）
  # local_reconnect_attempts: 1
  # local_reconnect_window_s: 1   # 拿到复用连接后超过此时间才失败的不重发
  # server_keepalive_s: 75        # sidecar 自己的 HTTP 服务端 keep-alive；须比 gateway.upstream_idle_timeout_s 至少大 1 s
gateway:
  upstream_idle_timeout_s: 60   # Envoy 上游空闲连接超时；手写 YAML 里的值须相等（守卫测试）
vllm:
  env:
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '75'   # vLLM 的 uvicorn keep-alive；改它会改模型 Deployment，须重建 pod
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
  `TRE_REISSUE_REQUIRE_HIDDEN_HEADER=true`，CPU 50m/500m、内存 64/256Mi；`TRE_REISSUE_UPSTREAM_KEEPALIVE_S` / `TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS`
  来自 registry `reissue:`，`TRE_REISSUE_UPSTREAM_SERVER_KEEPALIVE_S` 取该 pod vLLM 容器的 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`（未设为 5）。
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
