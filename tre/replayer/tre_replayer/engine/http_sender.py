"""The sender: the one client every load path in the tree sends through (2026-09-30).

Fires each ScheduledRequest at its scheduled time (via ``dispatch_open_loop``, in this
process or in each worker of :mod:`tre_replayer.engine.procpool`) against an
OpenAI-compatible gateway, streaming the response to capture TTFT and end time, and
records one row per request. Three profiles (:mod:`tre_replayer.engine.profiles`):
``calib`` and ``replay`` (the calibration / trace-replay request and row, unchanged) and
``e1_v1`` (the paper's client: v1's request through the OpenAI SDK, v1's
``performance_metrics.json`` line). ``python3 -m tre_loadgen_v1`` is a shell over this.

The network call is async (:mod:`tre_replayer.engine.transport`: pooled ``httpx``, or the
OpenAI SDK for ``e1_v1``) and runs on the dispatcher's own event loop - no thread per
request. An injectable synchronous seam (``stream_call``) remains for tests and dry runs;
it runs on a thread pool so it never blocks the loop.

Each record also carries ``target_pod``: the pod that served the request, when the
serving path names one. See :class:`StreamingHttpSender` - only the plugin-routed path
(a ``routing-strategy`` header) does; otherwise the field is None rather than guessed.

Timing a request's lateness
---------------------------
Every row carries the gap between "this request is due" and "its bytes are going out",
broken into the three segments that make it up, so a late cell can be attributed rather
than guessed at:

* ``schedule_delay_ms`` - the dispatcher fired the send later than the schedule said.
  The event loop was behind.
* ``pool_wait_ms`` - the fired send then waited before the sender picked it up (with
  the async transport: the loop's own latency, ~0; with a synchronous seam: the wait
  for a thread).
* ``body_build_ms`` - the sender had the request but had not yet called the socket:
  prompt lookup (or, without a materialised store, a full tokenizer fit) plus JSON
  encoding.
* ``on_wire_delay_ms`` - the sum of the three, and the only one of them that answers the
  question the open loop actually asks: **how much later than its scheduled instant did
  this request reach the wire?** (the "send lateness"). It is measured immediately
  before the transport call, so nothing between the schedule and the socket is outside it.

``on_wire_delay_ms`` is what a calibration cell is held to (see
``scripts.openloop.check_cell``). The other three stay because they decompose it, and a
cell that misses its deadline is only actionable once it is known which of the three
segments consumed the time.

Endpoint
--------
``api`` (:mod:`tre_replayer.engine.api`) picks ``/v1/completions`` (the default - the
trace replays send exactly what they always sent) or ``/v1/chat/completions`` (the
calibration drivers, like v1 and the E1 client). The gateway URL must name the matching
path. TTFT is the first chunk that carries generated text in any of the fields an
OpenAI-compatible server puts it in: ``text`` (completions), ``delta.content``, and
``delta.reasoning_content`` / ``delta.reasoning`` (chat with a reasoning parser, where a
R1-style model's first tokens are reasoning). The role-only opening chunk of a chat
stream (``delta: {"role": "assistant", "content": ""}``) carries no token and is not
the first token; the field that carried it is recorded as ``first_token_field``.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

# The sending core lives in driver-free modules; everything is re-exported here so
# every existing import of these names keeps working.
from tre_replayer.engine.api import (  # noqa: F401
    API_CHAT,
    API_COMPLETIONS,
    API_PATHS,
    APIS,
    DEFAULT_API,
    DEFAULT_ROUTING_STRATEGY,
    ROUTING_STRATEGY_HEADER,
    build_request_headers,
    check_api,
    check_api_mode,
    check_api_url,
    request_body,
)
from tre_replayer.engine.in_tokens import RECORD_FIELD as IN_TOKENS_FIELD
from tre_replayer.engine.in_tokens import header_for
from tre_replayer.engine.metrics import dual_fields_ms, strict_view_ms, v1_view_s
from tre_replayer.engine.profiles import (
    PROFILE_E1_V1,
    V1ChatOptions,
    client_provenance,
    fixed_length_timeout_s,
    get_profile,
    profile_for_api,
)
from tre_replayer.engine.stream import (  # noqa: F401
    CHAT_TOKEN_FIELDS,
    CONTINUED_FIELD,
    CONTINUED_HEADER,
    MAX_ERROR_BODY_CHARS,
    POD_HEADER_KEYS,
    RETRIED_HEADER,
    StreamCall,
    StreamResult,
    TTFT_BASIS,
    _chunk_has_content,
    _positive_int,
    chunk_token_field,
    is_client_timeout,
    lower_headers,
    pod_from_headers,
    read_error_body,
    reissue_from_headers,
    result_fields,
    stream_error_message,
    stream_request,
)
from tre_replayer.engine.prompt_store import PromptStore, sender_seed_key
from tre_replayer.engine.prompts import (
    DEFAULT_CORPUS_LANG,
    DEFAULT_MODE,
    DEFAULT_ZH_RATIO,
    build_prompt,
    check_corpus,
)
from tre_replayer.engine.schedule import ScheduledRequest
from tre_replayer.engine.transport import HttpxStreamTransport, OpenAIChatTransport

#: The default synchronous seam (kept under its old name).
_default_stream_call = stream_request


def _now_ms() -> int:
    return int(time.time() * 1000)


def _store_has(store: Any, request_id: str) -> bool:
    if store is None:
        return False
    try:
        return request_id in store
    except TypeError:
        return False


class InFlightCounter:
    """Requests on the wire right now. In-process by default; the multi-process runner
    passes one backed by shared memory so ``in_flight_at_send`` is the run's, not the
    worker's."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def inc(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def dec(self) -> None:
        with self._lock:
            self._n -= 1


class StreamingHttpSender:
    """Fires scheduled requests and records one row per request.

    ``routing_strategy`` selects *which serving path* the request takes, and with it
    whether per-pod attribution is possible at all:

    * ``None`` (the class default, used by the calibration drivers): only the ``model``
      request header is sent, so the per-model HTTPRoute matches and Envoy
      load-balances (LEAST_REQUEST) across the model Service's endpoints. The gateway
      plugin is not in this path, so no answer carries a pod header and ``target_pod``
      is None on every row.
    * a strategy name (e.g. :data:`DEFAULT_ROUTING_STRATEGY`, which ``run_trace`` and the
      campaign send, as the v1 client did): the ``routing-strategy`` header is added. On
      either gateway a route patched in ahead of the per-model routes matches it, the
      ext_proc plugin picks the pod, Envoy forwards to it (ORIGINAL_DST), and the answer
      carries ``target-pod`` / ``target-pod-ip``. The ``model`` header is still sent:
      the tre-v2 gateway keys its per-model ORIGINAL_DST cluster (admission limits, Envoy
      stats) on it, and the aibrix-system gateway ignores it on that route.

    The two options differ in who chooses the pod, which changes the measurement: never
    compare a run made one way against a run made the other.

    ``profile`` (default: ``calib`` for ``api="chat"``, ``replay`` for completions) picks
    the request and the record; ``e1_v1`` needs ``v1_options`` and takes the gateway's
    base URL (``http://host:port``) instead of an endpoint URL. ``max_in_flight`` sizes
    the connection pool (and, with a synchronous seam, its thread pool): an open-loop
    cell at rho > 1 accumulates backlog, so it must cover the peak in-flight.
    ``dual_metrics`` adds both metric bases (:mod:`tre_replayer.engine.metrics`) as extra
    columns of a calibration / replay row (off: those rows stay byte-identical to the
    pre-2026-09-30 sender's).
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        stream_call: StreamCall | None = None,
        input_tokens_default: int = 64,
        output_tokens_default: int = 128,
        max_in_flight: int = 512,
        prompt_mode: str = DEFAULT_MODE,
        prompt_store: PromptStore | None = None,
        corpus_lang: str = DEFAULT_CORPUS_LANG,
        zh_ratio: float = DEFAULT_ZH_RATIO,
        routing_strategy: str | None = None,
        now_ms: Callable[[], int] = _now_ms,
        mono: Callable[[], float] = time.monotonic,
        api: str = DEFAULT_API,
        request_seed: int | None = None,
        profile: str | None = None,
        v1_options: V1ChatOptions | None = None,
        transport: Any = None,
        dual_metrics: bool = False,
        in_flight: Any = None,
        on_record: Callable[[dict], None] | None = None,
        process_id: int = 0,
        send_in_tokens: bool = False,
    ) -> None:
        self._profile = get_profile(profile or profile_for_api(api))
        self._e1 = self._profile.name == PROFILE_E1_V1
        if self._e1:
            api = API_CHAT
            if v1_options is None:
                raise ValueError("the e1_v1 profile needs v1_options (the v1 config's model / client section)")
            if stream_call is not None:
                raise ValueError("the e1_v1 profile sends through the OpenAI SDK; it has no synchronous seam")
        else:
            # The endpoint, checked against the URL and the prompt mode before anything is
            # built: a chat body on the completions path (or a token-id chat message) fails
            # every request, so it is refused here, once.
            check_api_url(gateway_url, api)
            check_api_mode(api, prompt_mode)
        self._v1 = v1_options
        self._api = api
        self._request_seed = None if request_seed is None else int(request_seed)
        self._url = gateway_url
        self._sync_call = stream_call
        self._in = input_tokens_default
        self._out = output_tokens_default
        self._prompt_mode = prompt_mode
        # The inline fallback must build exactly what the materialiser built, so the
        # corpus travels with the sender as well as with the store.
        check_corpus(corpus_lang, zh_ratio)
        self._corpus_lang = corpus_lang
        self._zh_ratio = float(zh_ratio)
        # Prompts built before the run started (tre_replayer.engine.prompt_store). Without
        # one the sender falls back to fitting each prompt inline, which costs milliseconds
        # of GIL-held tokenizer work inside on_wire_delay_ms - see that module's docstring.
        if prompt_store is not None and getattr(prompt_store, "api", None) not in (None, api):
            # A chat prompt is fitted to the templated length, a completions one to the bare
            # string: the other endpoint's prompts are the wrong length on this one.
            raise ValueError(f"the prompt store {getattr(prompt_store, 'path', None)} holds {prompt_store.api} "
                             f"prompts, but this sender sends {api}")
        self._prompt_store = prompt_store
        self._routing_strategy = routing_strategy
        self._now = now_ms
        self._mono = mono
        self._max_in_flight = max(1, int(max_in_flight))
        if transport is None and stream_call is None:
            if self._e1:
                transport = OpenAIChatTransport(
                    gateway_url, api_key=v1_options.api_key, max_retries=v1_options.max_retries,
                    timeout_s=v1_options.timeout_s, routing_strategy=v1_options.routing_strategy,
                    streaming=v1_options.streaming, pool_shards=v1_options.pool_shards,
                )
            else:
                transport = HttpxStreamTransport(max_connections=self._max_in_flight)
        self._transport = transport
        # A synchronous seam blocks for the whole e2e, so it gets its own thread pool
        # sized for the peak in-flight (F5: a smaller pool would silently turn the open
        # loop into a closed loop bounded by the pool).
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._dual = bool(dual_metrics)
        # Requests on the wire right now, recorded on every row as ``in_flight_at_send``.
        # A failure is only interpretable next to how loaded the path was when it was
        # emitted, and this is the one quantity nothing downstream can reconstruct.
        self._in_flight = in_flight if in_flight is not None else InFlightCounter()
        self._on_record = on_record
        self.process_id = int(process_id)
        # Opt-in: the x-tre-bl-in-tokens header (tre_replayer.engine.in_tokens), from the
        # count each request carries. e1_v1 takes it from its v1 options.
        self._send_in_tokens = bool(v1_options.send_in_tokens) if self._e1 else bool(send_in_tokens)
        self.records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ lifecycle

    @property
    def profile(self) -> str:
        return self._profile.name

    def provenance(self, *, processes: int = 1) -> dict:
        """The client part of a run manifest."""
        extra: dict[str, Any] = {"max_in_flight": self._max_in_flight, "dual_metrics": self._dual}
        if self._send_in_tokens:
            extra["send_in_tokens"] = True
        if self._v1 is not None:
            extra["v1_options"] = self._v1.as_dict()
        if self._sync_call is not None:
            extra["seam"] = getattr(self._sync_call, "__name__", repr(self._sync_call))
        return client_provenance(self._profile.name, transport=self._transport, processes=processes, **extra)

    def _pool(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._max_in_flight, thread_name_prefix="trepl-send")
            return self._executor

    async def prepare(self) -> None:
        """Build the transport's client on the running loop ahead of the first send."""
        if self._transport is not None and hasattr(self._transport, "prepare"):
            await self._transport.prepare()

    async def aclose(self) -> None:
        """Close the transport's pool on the loop that ran it (call before the loop ends)."""
        if self._transport is not None and hasattr(self._transport, "aclose"):
            await self._transport.aclose()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def max_pool_wait_ms(self) -> float:
        return max((r.get("pool_wait_ms", 0.0) for r in self.records), default=0.0)

    def max_on_wire_delay_ms(self) -> float:
        return max((r.get("on_wire_delay_ms", 0.0) for r in self.records), default=0.0)

    @property
    def prompt_store_misses(self) -> int:
        """Requests whose prompt was not materialised and had to be built on the send
        path. Non-zero means part of this run paid the tokenizer fit inside its own
        lateness; it is reported per cell so the regression cannot be silent."""
        return 0 if self._prompt_store is None else self._prompt_store.misses

    # ------------------------------------------------------------------ one request

    def _prompt_for(self, request: ScheduledRequest, in_tokens: int):
        """This request's prompt: its own, the materialised one, or an inline fit.

        The inline fit is the fallback, not the design. It uses the same builder and the
        same seed key as the materialiser, so a run that falls back sends byte-identical
        bytes to one that did not - it just pays for them at the wrong moment.
        """
        if request.prompt:
            return request.prompt
        if self._prompt_store is not None:
            materialised = self._prompt_store.get(request.request_id)
            if materialised is not None:
                return materialised
        return build_prompt(
            in_tokens,
            sender_seed_key(request.model, request.request_id),
            mode=self._prompt_mode,
            model=request.model,
            corpus_lang=self._corpus_lang,
            zh_ratio=self._zh_ratio,
            api=self._api,
        )

    async def __call__(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        try:
            if self._e1:
                record = await self._send_e1(request, scheduled_ts, actual_ts)
            else:
                record = await self._send_fixed(request, scheduled_ts, actual_ts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a client fault is this request's record, now
            # Before this, one request's exception surfaced only when the whole schedule
            # had been sent, and took every record of the run with it.
            record = self._client_error_record(request, scheduled_ts, actual_ts, exc)
        self.records.append(record)
        if self._on_record is not None:
            self._on_record(record)

    def _client_error_record(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float,
                             exc: BaseException) -> dict[str, Any]:
        """A failed request whose failure is the client's own (never sent or never read)."""
        now = self._mono()
        text = f"client error: {type(exc).__name__}: {exc}"
        res = StreamResult(0, None, None, error=text, start_epoch_s=time.time(), v1_success=False,
                           v1_error=text, v1_done_ms=0.0, v1_target_pod="", attempts=0)
        lateness = self._lateness(scheduled_ts, actual_ts, now, now)
        if self._e1:
            record = e1_record(request, res, process_id=self.process_id, lateness=lateness, in_flight_at_send=0)
        else:
            out_tokens = request.max_output_tokens or self._out
            record = {
                "request_id": request.request_id, "model": request.model,
                "scheduled_offset_ms": int(scheduled_ts * 1000),
                "scheduled_offset_s": float(request.scheduled_offset_s),
                "actual_send_ts_ms": self._now(), **lateness,
                "input_tokens": request.prompt_tokens or self._in, "output_tokens": out_tokens,
                **result_fields(res), "request_timeout_s": fixed_length_timeout_s(out_tokens),
                "in_flight_at_send": 0, "api": self._api,
            }
        record["client_error"] = text
        if self._send_in_tokens:
            record[IN_TOKENS_FIELD] = request.in_tokens_header
        return record

    def _lateness(self, scheduled_ts: float, actual_ts: float, pickup_ts: float, wire_ts: float,
                  acquire_ms: float = 0.0) -> dict:
        """``acquire_ms`` (the transport's wait for a connection - pool slot, TCP connect,
        a repeated first attempt) is inside the send lateness: it is the time between the
        transport call and the first byte on the wire. It is counted as pool wait."""
        return {
            "schedule_delay_ms": max(0.0, (actual_ts - scheduled_ts) * 1000.0),
            "pool_wait_ms": round(max(0.0, (pickup_ts - actual_ts) * 1000.0) + acquire_ms, 3),
            "body_build_ms": round(max(0.0, (wire_ts - pickup_ts) * 1000.0), 3),
            # Scheduled instant -> first byte on the wire. The guard's deadline; see the
            # module docstring for why the three segments above are not it.
            "on_wire_delay_ms": round(max(0.0, (wire_ts - scheduled_ts) * 1000.0) + acquire_ms, 3),
        }

    async def _send_fixed(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> dict[str, Any]:
        # time from the dispatcher scheduling this send to the sender picking it up; a
        # large p99 here means the loop (or, with a synchronous seam, its pool) starved.
        pickup_ts = self._mono()
        out_tokens = request.max_output_tokens or self._out
        in_tokens = request.prompt_tokens or self._in
        # A trace may carry its own prompt text; otherwise use the one materialised for
        # this request before the run began. A constant prompt would be served from the
        # prefix cache on any engine that has it enabled, making prefill free and the
        # measurement worthless (see tre_replayer.engine.prompts); the materialiser keeps
        # one distinct prompt per request and takes the cost of building it off this path
        # (see tre_replayer.engine.prompt_store).
        if request.prompt or _store_has(self._prompt_store, request.request_id):
            prompt = self._prompt_for(request, in_tokens)
        else:
            # An inline fit is milliseconds of tokenizer work: done on a thread so it
            # delays only this request, never the loop every other request is timed on.
            loop = asyncio.get_running_loop()
            prompt = await loop.run_in_executor(self._pool(), self._prompt_for, request, in_tokens)
        body = json.dumps(
            request_body(request.model, prompt, out_tokens, api=self._api, seed=self._request_seed)
        ).encode("utf-8")
        headers = build_request_headers(request.model, self._routing_strategy)
        if self._send_in_tokens:
            headers.update(header_for(request.in_tokens_header))
        timeout_s = fixed_length_timeout_s(out_tokens)
        # Last instant before the transport call: everything the driver does between the
        # scheduled instant and here is inside on_wire_delay_ms, prompt work included.
        wire_ts = self._mono()
        send_ts_ms = self._now()
        in_flight_at_send = self._in_flight.inc()
        # Exactly one call per request, and no retry on any outcome. A retried request
        # would be counted once as offered and twice as sent, which biases goodput
        # upwards, and it would re-offer load the schedule never planned - so the cell
        # would no longer be the open loop it claims to be.
        try:
            if self._sync_call is not None:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(self._pool(), self._sync_call, self._url, headers, body, timeout_s)
            else:
                res = await self._transport.send(self._url, headers, body, timeout_s)
        finally:
            self._in_flight.dec()
        lateness = self._lateness(scheduled_ts, actual_ts, pickup_ts, wire_ts, res.conn_acquire_ms or 0.0)
        record = {
            "request_id": request.request_id,
            "model": request.model,
            "scheduled_offset_ms": int(scheduled_ts * 1000),  # dispatcher monotonic clock, NOT epoch
            # The request's own place in the schedule, in the schedule's time base. Kept
            # verbatim so the achieved arrival series can be binned on the same grid as
            # the nominal one (tre_replayer.engine.rps_timeline).
            "scheduled_offset_s": float(request.scheduled_offset_s),
            "actual_send_ts_ms": send_ts_ms,
            **lateness,
            # The prompt length asked for - for chat the templated total, i.e. what
            # usage.prompt_tokens (``prompt_tokens``) must equal.
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            **result_fields(res),
            "request_timeout_s": timeout_s,
            "in_flight_at_send": in_flight_at_send,
            "api": self._api,
            # Connection evidence (None from a synthetic seam): see StreamResult.
            "stream_complete": res.stream_complete,
            "connection_reused": res.connection_reused,
            "conn_acquire_ms": None if res.conn_acquire_ms is None else round(res.conn_acquire_ms, 3),
            "transport_retries": res.transport_retries,
        }
        if self._dual:
            record.update(dual_fields_ms(res))
        if self._send_in_tokens:
            record[IN_TOKENS_FIELD] = request.in_tokens_header
        return record

    async def _send_e1(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> dict[str, Any]:
        pickup_ts = self._mono()
        if not request.prompt:
            raise ValueError(f"{request.request_id}: the e1_v1 profile sends the trace's own prompt; it has none")
        kwargs = self._v1.kwargs_for(request.model, request.prompt, request.max_output_tokens,
                                     request.in_tokens_header)
        wire_ts = self._mono()
        in_flight_at_send = self._in_flight.inc()
        try:
            res = await self._transport.send_chat(kwargs)
        finally:
            self._in_flight.dec()
        lateness = self._lateness(scheduled_ts, actual_ts, pickup_ts, wire_ts)
        record = e1_record(request, res, process_id=self.process_id, lateness=lateness,
                           in_flight_at_send=in_flight_at_send)
        if self._send_in_tokens:
            record[IN_TOKENS_FIELD] = request.in_tokens_header
        return record

    def write_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return len(self.records)


#: v1's ``performance_metrics.json`` fields, in v1's order (``phase_type`` is the
#: trace's, filled in by the shell), then the audit fields the port added.
V1_RECORD_FIELDS = (
    "request_id", "model_name", "timestamp", "start_time", "end_time", "e2e_latency", "ttft", "tpot",
    "input_tokens", "output_tokens", "total_tokens", "success", "error_message", "http_status",
    "phase_type", "target_pod", "process_id",
)
V1_AUDIT_FIELDS = ("attempts", "stream_interrupted", "stream_error", "finish_reason", "attempt_log")
#: Added by the unified client: the strict basis (``*_strict*``), retries kept out of it,
#: the send lateness and the reissue sidecar's marks.
E1_EXTRA_FIELDS = (
    "success_strict", "failure_strict", "ttft_strict_s", "tpot_strict_s", "e2e_strict_s", "ttft_missing_strict",
    "retries", "retry_wait_s", "http_status_strict", "first_token_field", "send_lateness_ms",
    "schedule_delay_ms", "in_flight_at_send", "tre_continued", "tre_retried",
)


def e1_record(request: ScheduledRequest, res: StreamResult, *, process_id: int, lateness: dict,
              in_flight_at_send: int, phase_type: Optional[str] = None) -> dict[str, Any]:
    """One ``performance_metrics.json`` line: v1's fields with v1's formulas, v1's audit
    fields, then the strict basis and the lateness (:data:`E1_EXTRA_FIELDS`)."""
    v1 = v1_view_s(res)
    strict = strict_view_ms(res)
    start = float(res.start_epoch_s or 0.0)
    s = (lambda ms: None if ms is None else ms / 1000.0)
    return {
        "request_id": request.request_id,
        "model_name": request.model,
        "timestamp": request.scheduled_offset_s,
        "start_time": start,
        "end_time": start + (v1["e2e_s"] or 0.0),
        "e2e_latency": v1["e2e_s"],
        "ttft": v1["ttft_s"],
        "tpot": v1["tpot_s"],
        "input_tokens": v1["input_tokens"],
        "output_tokens": v1["output_tokens"],
        "total_tokens": v1["total_tokens"],
        "success": v1["success"],
        "error_message": None if v1["success"] else res.v1_error,
        "http_status": res.v1_status,
        "phase_type": phase_type,
        "target_pod": res.v1_target_pod,
        "process_id": process_id,
        "attempts": res.attempts,
        "stream_interrupted": bool(res.stream_interrupted),
        "stream_error": res.interrupt_error,
        "finish_reason": res.v1_finish_reason,
        "attempt_log": list(res.attempt_log or []),
        "success_strict": strict["success"],
        "failure_strict": strict["failure"],
        "ttft_strict_s": s(strict["ttft_ms"]),
        "tpot_strict_s": s(strict["tpot_ms"]),
        "e2e_strict_s": s(strict["e2e_ms"]),
        "ttft_missing_strict": strict["ttft_missing"],
        "retries": strict["retries"],
        "retry_wait_s": s(strict["retry_wait_ms"]),
        "http_status_strict": res.status,
        "first_token_field": res.first_token_field,
        "send_lateness_ms": lateness["on_wire_delay_ms"],
        "schedule_delay_ms": round(lateness["schedule_delay_ms"], 3),
        "in_flight_at_send": in_flight_at_send,
        "tre_continued": res.tre_continued,
        "tre_retried": res.tre_retried,
    }
