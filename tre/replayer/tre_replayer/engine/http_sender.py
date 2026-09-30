"""Open-loop streaming HTTP sender for the replayer (audit blocker B3).

Fires each ScheduledRequest at its scheduled time (via dispatch_open_loop) against an
OpenAI-compatible gateway, streaming the response to capture TTFT and end time, and records
one per-request JSONL line. That per-request record is also the S4 raw-logger format, so this
is the single sender both R2 (trace runs) and R3 (raw capacity logging) use.

The actual network call is an injectable seam (`stream_call`) so tests run with a fake and
never touch the network. The default seam uses urllib + SSE parsing.

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
* ``pool_wait_ms`` - the fired send then waited for a sender thread. The pool starved,
  i.e. the open loop had degenerated into a closed loop bounded by the driver.
* ``body_build_ms`` - the worker had the request but had not yet called the socket:
  prompt lookup (or, without a materialised store, a full tokenizer fit) plus JSON
  encoding.
* ``on_wire_delay_ms`` - the sum of the three, and the only one of them that answers the
  question the open loop actually asks: **how much later than its scheduled instant did
  this request reach the wire?** It is measured immediately before the transport call,
  so nothing between the schedule and the socket is outside it.

``on_wire_delay_ms`` is what a calibration cell is held to (see
``scripts.openloop.check_cell``). The other three stay because they decompose it, and a
cell that misses its deadline is only actionable once it is known which of the three
segments consumed the time.

Layout
------
This module is the open-loop *driver-facing* sender: scheduling hand-off, the worker pool,
prompt lookup and the per-request record with its lateness decomposition. The sending
core it is built on has no driver in it and is shared: :mod:`tre_replayer.engine.api`
builds the request (body and headers) and :mod:`tre_replayer.engine.stream` makes one
streamed call (timing, SSE parsing, usage, failure evidence; ``result_fields`` is the
answer's part of the record). Their names are re-exported here.

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

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# The sending core lives in two driver-free modules; everything is re-exported here so
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

#: The default seam (kept under its old name).
_default_stream_call = stream_request


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    ) -> None:
        # The endpoint, checked against the URL and the prompt mode before anything is
        # built: a chat body on the completions path (or a token-id chat message) fails
        # every request, so it is refused here, once.
        check_api_url(gateway_url, api)
        check_api_mode(api, prompt_mode)
        self._api = api
        self._request_seed = None if request_seed is None else int(request_seed)
        self._url = gateway_url
        self._call = stream_call or _default_stream_call
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
        # F5: each streamed request blocks a worker for its whole e2e. asyncio.to_thread's
        # default executor is capped at ~32, which would silently turn the open-loop replay
        # into a closed-loop-32 and under-drive the system under saturation. Use a dedicated
        # pool sized for the peak in-flight, and record pool_wait_ms so starvation is visible.
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_in_flight), thread_name_prefix="trepl-send")
        # Requests on the wire right now, recorded on every row as ``in_flight_at_send``.
        # A failure is only interpretable next to how loaded the path was when it was
        # emitted, and this is the one quantity nothing downstream can reconstruct: the
        # raw log keeps send and done timestamps, but not the driver's own view of
        # concurrency at the instant of the send.
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    async def __call__(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(self._executor, self._send_one, request, scheduled_ts, actual_ts)
        self.records.append(record)

    def close(self) -> None:
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

    def _send_one(self, request: ScheduledRequest, scheduled_ts: float, actual_ts: float) -> dict[str, Any]:
        # time from the dispatcher scheduling this send to a worker actually picking it up;
        # a large p99 here means the pool starved and the replay under-drove the target (F5).
        pickup_ts = self._mono()
        pool_wait_ms = max(0.0, (pickup_ts - actual_ts) * 1000.0)
        out_tokens = request.max_output_tokens or self._out
        in_tokens = request.prompt_tokens or self._in
        # A trace may carry its own prompt text; otherwise use the one materialised for
        # this request before the run began. A constant prompt would be served from the
        # prefix cache on any engine that has it enabled, making prefill free and the
        # measurement worthless (see tre_replayer.engine.prompts); the materialiser keeps
        # one distinct prompt per request and takes the cost of building it off this path
        # (see tre_replayer.engine.prompt_store).
        prompt = self._prompt_for(request, in_tokens)
        body = json.dumps(
            request_body(request.model, prompt, out_tokens, api=self._api, seed=self._request_seed)
        ).encode("utf-8")
        headers = build_request_headers(request.model, self._routing_strategy)
        timeout_s = max(30.0, out_tokens / 4.0)
        # Last instant before the transport call: everything the driver does between the
        # scheduled instant and here is inside on_wire_delay_ms, prompt work included.
        wire_ts = self._mono()
        send_ts_ms = self._now()
        with self._in_flight_lock:
            self._in_flight += 1
            in_flight_at_send = self._in_flight
        # Exactly one call per request, and no retry on any outcome. A retried request
        # would be counted once as offered and twice as sent, which biases goodput
        # upwards, and it would re-offer load the schedule never planned - so the cell
        # would no longer be the open loop it claims to be.
        try:
            res = self._call(self._url, headers, body, timeout_s)
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1
        return {
            "request_id": request.request_id,
            "model": request.model,
            "scheduled_offset_ms": int(scheduled_ts * 1000),  # dispatcher monotonic clock, NOT epoch
            # The request's own place in the schedule, in the schedule's time base. Kept
            # verbatim so the achieved arrival series can be binned on the same grid as
            # the nominal one (tre_replayer.engine.rps_timeline).
            "scheduled_offset_s": float(request.scheduled_offset_s),
            "actual_send_ts_ms": send_ts_ms,
            "schedule_delay_ms": max(0.0, (actual_ts - scheduled_ts) * 1000.0),
            "pool_wait_ms": round(pool_wait_ms, 3),
            "body_build_ms": round(max(0.0, (wire_ts - pickup_ts) * 1000.0), 3),
            # Scheduled instant -> socket call. The guard's deadline; see the module
            # docstring for why the three segments above are not it.
            "on_wire_delay_ms": round(max(0.0, (wire_ts - scheduled_ts) * 1000.0), 3),
            # The prompt length asked for - for chat the templated total, i.e. what
            # usage.prompt_tokens (``prompt_tokens``) must equal.
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            **result_fields(res),
            "request_timeout_s": timeout_s,
            "in_flight_at_send": in_flight_at_send,
            "api": self._api,
        }

    def write_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return len(self.records)
