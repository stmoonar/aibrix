#!/usr/bin/env python3
"""TRE retry / continuation sidecar (plan 2026-09-27 D5, D6).

Runs in every model pod next to vLLM. The sidecar owns the pod's serving port (default
8000) and vLLM moves to a local port (default 127.0.0.1:8001), so the Service, the gateway
(target-pod = podIP:8000), the service-manager, metric scrapers and probes are unchanged.
Every path is proxied transparently (streaming kept). On top of that:

* Not-yet-started requests are RETRIED through the TRE gateway, unchanged, with
  ``x-tre-exclude-pod: <this pod>`` merged into the request: when vLLM answers
  ``503 {"error": {"type": "EngineSleeping"}}`` (fork flag ``--sleep-reject-new``, also as
  the first event of an SSE stream), when the local engine is known to be asleep, or when
  a generation is aborted before a single byte reached the client. Bounded attempts; if
  all fail the client gets ``503`` + ``Retry-After``.
* Started requests aborted by a sleep (``finish_reason == "abort"`` after a ``/sleep`` call
  reached this sidecar since the request started) are CONTINUED on another instance through the gateway with a token-id prompt:
  ``prompt = prompt_token_ids + generated_token_ids`` (both from the abort output of the
  fork flag ``--abort-return-token-ids``), ``max_tokens`` reduced by the generated count,
  the same sampling parameters. Chat requests are continued as ``/v1/completions`` over the
  engine-rendered prompt ids (exact seam) and the chunks are reshaped back into chat
  chunks. The two segments are stitched (one id / created / model, one [DONE], merged
  usage with the ORIGINAL prompt_tokens, stop strings re-checked across the seam).
* The loopback hop to vLLM is kept healthy: pooled connections idle at most
  ``upstream_keepalive_s`` (below vLLM's own keep-alive), a request whose pooled
  connection is dropped before the first response byte (and within
  ``local_reconnect_window_s`` of being handed out) is re-sent once on a fresh
  connection, and if the engine still cannot be reached before anything was sent the
  client gets ``503`` + ``Retry-After`` (``error.layer = "sidecar_upstream"``), or the
  request goes through the gateway when nothing listens locally / the pod is asleep.
* The hop Envoy -> sidecar is kept healthy the same way round: the sidecar's own server
  keeps an idle connection ``server_keepalive_s`` (default 75 s), which must stay above
  Envoy's upstream idle timeout (``gateway.upstream_idle_timeout_s``, default 60 s), so
  Envoy - the side that sends requests - always closes an idle connection first.
* ``POST /sleep`` (and ``/pause``) must carry ``X-TRE-Hidden: 1`` (sent by the
  service-manager after it hid the pod), else 409: fail closed.
* A client that goes away cancels its request: the server cancels the handler on a lost
  connection and every upstream request it holds (local engine, gateway retry or
  continuation) is closed, so vLLM aborts it (a queued request is not prefilled). An
  upstream break caused by a sleep, with the client still there, is continued as above.
* At startup the soft open-files limit is raised to ``nofile_target`` (capped at the hard
  limit), as vLLM does: some container runtimes start processes with soft 1024.
* ``GET /tre-reissue/metrics`` (Prometheus text), ``GET /tre-reissue/state``, one JSON log
  line per retry / continuation on stdout.

How the sidecar knows the pod is asleep: the only way to put the engine to sleep is a
``/sleep`` through this sidecar (vLLM listens on localhost), so the sidecar marks the pod
sleeping BEFORE it forwards the call (the abort outputs arrive before /sleep returns) and
clears the mark on a successful ``/wake_up``. The mark is re-synced from every proxied
``/is_sleeping`` answer, a periodic direct ``/is_sleeping`` probe and every EngineSleeping
503 (vLLM or sidecar restarts). The route-gen / hidden pod annotation is not used: the
downward API refreshes annotation files only on the kubelet sync period (up to a minute).

Design and field names: tre/docs/design/20260927-reissue-sidecar-v2.md. Every endpoint,
port, header and JSON field name is configurable through ``TRE_REISSUE_*`` environment
variables (``TRE_GATEWAY_URL`` for the gateway). Only the standard library and aiohttp
are used (both ship in the vLLM image), so the script is delivered through a ConfigMap.
Python >= 3.10.
"""
from __future__ import annotations

import asyncio
import dataclasses
import errno
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

try:
    import resource
except ImportError:  # not POSIX: the open-files limit is left alone
    resource = None  # type: ignore[assignment]

#: The stable ClusterIP Service in front of the tre-v2 Envoy proxy (tre-v2 overlay
#: gateway-service.yaml; registry gateway.service_name / service_namespace). Override with
#: TRE_GATEWAY_URL, which the manifests render from the registry.
DEFAULT_GATEWAY_URL = "http://tre-gateway.envoy-gateway-system.svc.cluster.local:80"

#: vLLM's /v1/completions default when max_tokens is absent or null.
COMPLETIONS_DEFAULT_MAX_TOKENS = 16

#: Never copied between hops (RFC 7230 hop-by-hop plus what aiohttp recomputes).
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
#: Additionally dropped from requests sent to the gateway: per-hop identity and the
#: routing decision made for the ORIGINAL request.
_GATEWAY_DROP_BASE = HOP_BY_HOP | frozenset(
    {
        "x-request-id",
        "target-pod",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-real-ip",
    }
)
_RESPONSE_DROP = HOP_BY_HOP | frozenset({"server", "date"})

#: Sampling / output parameters that mean the same on chat and completions requests;
#: copied when a chat request is continued as a token-id completion.
CHAT_TO_COMPLETION_KEYS = (
    "model",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "length_penalty",
    "logit_bias",
    "bad_words",
    "allowed_token_ids",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "include_stop_str_in_output",
    "min_tokens",
    "user",
    "priority",
    "cache_salt",
    "vllm_xargs",
    "return_token_ids",
    "stream_options",
)
#: Removed from a completions request continued with a token-id prompt.
COMPLETION_CONT_DROP = ("prompt", "prompt_embeds", "echo", "suffix", "truncate_prompt_tokens", "add_special_tokens")

_ABORT_MARK = b'"abort"'

#: Start of the message of every error response the sidecar itself produces (``_error``).
#: A 502 / 503 carrying it came from another sidecar and is final for the gateway retry
#: loop (``ReissueSidecar._gateway_request``).
SIDECAR_ERROR_PREFIX = "tre-reissue sidecar:"
SIDECAR_ERROR_MARK = SIDECAR_ERROR_PREFIX.encode()



def _request_key(name: str, kind: type) -> Any:
    """A ``web.Request`` item key (aiohttp < 3.12 has no ``web.RequestKey``: a string)."""
    return web.RequestKey(name, kind) if hasattr(web, "RequestKey") else name


#: Per-request: the tre_reissue_total kind owed for a request whose retry / continuation
#: is under way; ``_account`` clears it. If the client goes away first, the request is
#: accounted as ``(kind, "client_gone")``.
_REISSUE_PENDING = _request_key("tre_reissue_pending", str)
#: Per-request: the client has the complete response (the stream's ``[DONE]`` is
#: written); a cancel after that (the engine had not ended its response yet) is not a
#: client cancel.
_DELIVERED = _request_key("tre_reissue_delivered", bool)

OVERHEAD_BUCKETS_S = (
    0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.25, 1.0,
)
GAP_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


# ---------------------------------------------------------------------------- config


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Every field ``foo`` is read from ``TRE_REISSUE_FOO`` (see ``from_env``)."""

    listen_host: str = "0.0.0.0"
    #: The pod's serving port (Service targetPort, gateway target-pod, SM, probes).
    listen_port: int = 8000
    #: The local vLLM (moved to the internal port).
    upstream_url: str = "http://127.0.0.1:8001"
    #: TRE gateway, in-cluster DNS name (env TRE_GATEWAY_URL).
    gateway_url: str = DEFAULT_GATEWAY_URL
    model: str = ""
    #: This pod's name (env POD_NAME, downward API): the x-tre-exclude-pod value.
    pod_name: str = ""
    #: False = pure transparent proxy (still tracks sleep; never retries / continues).
    enabled: bool = True
    #: Max retry / continuation hops a request may take (x-tre-reissue-depth).
    max_depth: int = 3
    #: Gateway attempts per retry or continuation request (503 / 502 / connect errors).
    retry_attempts: int = 4
    retry_backoff_s: float = 0.2
    #: Cap of the backoff and of an honoured Retry-After between attempts.
    retry_max_backoff_s: float = 2.0
    #: Refuse /sleep and /pause without ``X-TRE-Hidden: 1`` (409).
    require_hidden_header: bool = True
    #: Context limit for chat continuations without a token limit; 0 = ask vLLM
    #: (GET /v1/models, cached).
    max_model_len: int = 0
    probe_interval_s: float = 2.0
    #: Idle keep-alive of the pooled connections to the local vLLM. Must stay BELOW vLLM's
    #: own keep-alive (``upstream_server_keepalive_s``): uvicorn closes an idle connection
    #: after that, and a pooled connection reused just as the server closes it fails before
    #: the first response byte (Server disconnected / Connection reset by peer).
    upstream_keepalive_s: float = 2.0
    #: vLLM's keep-alive (its VLLM_HTTP_TIMEOUT_KEEP_ALIVE; vLLM's default is 5 s). The
    #: manifests render it from the vLLM container env; only used for the startup check.
    upstream_server_keepalive_s: float = 5.0
    #: Re-sends on a FRESH connection when a request to the local vLLM failed at the
    #: connection level (disconnect / ECONNRESET / EPIPE) before any response byte and
    #: before anything was written to the client. 0 = off.
    local_reconnect_attempts: int = 1
    #: A fresh-connection re-send happens only when the failure came within this many
    #: seconds of the pooled connection being handed to the request. The keep-alive race
    #: is instantaneous (the peer closed the idle connection a moment ago); a reused
    #: connection that dies later (engine crash mid-generation) may have run the request,
    #: so it is not re-sent. Must be >= 0.
    local_reconnect_window_s: float = 1.0
    #: Keep-alive of THIS sidecar's HTTP server (aiohttp ``keepalive_timeout``): how long
    #: it keeps an idle client connection (Envoy's upstream connection) open. The peer
    #: that sends requests must close first, so this must stay ABOVE Envoy's upstream idle
    #: timeout (``gateway_upstream_idle_s``), else Envoy reuses a connection the sidecar is
    #: closing (503 UC / reset). aiohttp's own default is 75 s.
    server_keepalive_s: float = 75.0
    #: Envoy's upstream connection idle timeout (registry ``gateway.upstream_idle_timeout_s``,
    #: set on the tre-v2 gateway clusters). Only used for the startup check against
    #: ``server_keepalive_s``; 0 = unknown, no check.
    gateway_upstream_idle_s: float = 0.0
    #: Minimum seconds between two WARNING lines of the same kind (rate limit).
    warn_interval_s: float = 10.0
    connect_timeout_s: float = 6.0
    #: Total timeout of proxied control / metadata calls (/health, /metrics, ...).
    proxy_timeout_s: float = 60.0
    #: Total timeout of /sleep, /pause, /wake_up, /resume.
    control_timeout_s: float = 300.0
    client_max_size: int = 64 * 1024 * 1024
    #: At startup the soft RLIMIT_NOFILE is raised to min(this, hard limit), never lowered
    #: (vLLM does the same and gets 65535). Each request in flight holds two or three
    #: sockets, and some container runtimes start processes with soft 1024 (Docker >= 25 /
    #: containerd >= 2.0); a pod spec cannot set ulimits. 0 = leave the limit alone.
    nofile_target: int = 65535
    # --- protocol names (defaults = vLLM >= 0.30 fork + TRE gateway plugin) ---
    completions_path: str = "/v1/completions"
    chat_path: str = "/v1/chat/completions"
    #: POSTs under this prefix are retried when the engine is asleep.
    retry_path_prefix: str = "/v1/"
    sleep_paths: tuple[str, ...] = ("/sleep", "/pause")
    wake_paths: tuple[str, ...] = ("/wake_up", "/resume")
    is_sleeping_path: str = "/is_sleeping"
    models_path: str = "/v1/models"
    metrics_path: str = "/tre-reissue/metrics"
    state_path: str = "/tre-reissue/state"
    hidden_header: str = "X-TRE-Hidden"
    exclude_header: str = "x-tre-exclude-pod"
    continued_header: str = "x-tre-continued"
    retried_header: str = "x-tre-retried"
    depth_header: str = "x-tre-reissue-depth"
    #: Extension field on the final (finish_reason) chunk of a continued stream.
    continued_field: str = "tre_continued"
    #: Error type of the fork's 503 for requests reaching a sleeping engine.
    sleeping_error_type: str = "EngineSleeping"
    #: Abort chunk (stream): all generated token ids, in choices[i].
    generated_ids_field: str = "generated_token_ids"
    #: Abort output: prompt token ids (chat: top level; completions: in choices[i]).
    prompt_ids_field: str = "prompt_token_ids"
    #: Aborted non-streaming response: generated token ids in choices[i].
    token_ids_field: str = "token_ids"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        env = dict(os.environ if env is None else env)
        values: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            raw = env.get("TRE_REISSUE_" + f.name.upper())
            if f.name == "gateway_url":
                raw = env.get("TRE_GATEWAY_URL") or raw
            elif f.name == "pod_name":
                raw = env.get("POD_NAME") or raw or env.get("HOSTNAME")
            if raw is None or raw == "":
                continue
            default = f.default
            if isinstance(default, bool):
                values[f.name] = _truthy(raw)
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            elif isinstance(default, tuple):
                values[f.name] = tuple(p.strip() for p in raw.split(",") if p.strip())
            else:
                values[f.name] = raw.strip()
        cfg = cls(**values)
        cfg = dataclasses.replace(
            cfg, upstream_url=cfg.upstream_url.rstrip("/"), gateway_url=cfg.gateway_url.rstrip("/")
        )
        if cfg.max_depth < 0 or cfg.retry_attempts < 1:
            raise ValueError("TRE_REISSUE_MAX_DEPTH must be >= 0 and TRE_REISSUE_RETRY_ATTEMPTS >= 1")
        if not cfg.upstream_keepalive_s > 0 or cfg.local_reconnect_attempts < 0:
            raise ValueError("TRE_REISSUE_UPSTREAM_KEEPALIVE_S must be > 0 and "
                             "TRE_REISSUE_LOCAL_RECONNECT_ATTEMPTS >= 0")
        if cfg.local_reconnect_window_s < 0:
            raise ValueError("TRE_REISSUE_LOCAL_RECONNECT_WINDOW_S must be >= 0")
        if not cfg.server_keepalive_s > 0 or cfg.gateway_upstream_idle_s < 0:
            raise ValueError("TRE_REISSUE_SERVER_KEEPALIVE_S must be > 0 and "
                             "TRE_REISSUE_GATEWAY_UPSTREAM_IDLE_S >= 0")
        if cfg.nofile_target < 0:
            raise ValueError("TRE_REISSUE_NOFILE_TARGET must be >= 0")
        if not cfg.gateway_url.startswith(("http://", "https://")):
            raise ValueError(f"TRE_GATEWAY_URL must be an http(s) URL, got {cfg.gateway_url!r}")
        return cfg


# --------------------------------------------------------------------------- metrics


class Histogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1

    def render(self, name: str, labels: str) -> list[str]:
        lines = [f'{name}_bucket{{{labels},le="{b}"}} {c}' for b, c in zip(self.buckets, self.counts)]
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum{{{labels}}} {self.sum:.6f}")
        lines.append(f"{name}_count{{{labels}}} {self.count}")
        return lines


class AddedTime:
    """Time the sidecar itself adds to one request (``tre_reissue_proxy_added_seconds``).

    * ``forward``: from the full client request read to the moment the request is
      handed to the upstream HTTP client (JSON parse, classification, headers);
    * ``relay``: summed over the upstream response - from receiving the upstream
      headers / each upstream chunk to finishing writing it (or the rewritten
      events) to the client, including the sidecar's own SSE / abort processing.

    Waiting on upstream is excluded: connection setup and body upload inside the
    upstream client, the time to the response headers (for a non-streaming
    request: the whole generation) and the gaps between chunks, and the
    continuation request through the gateway. A chunk write returns once aiohttp
    has buffered it; it only waits for the socket when the transport's write
    buffer is above its high-water mark, so a slow client's backpressure can show
    up in ``relay``. Only requests answered by the local engine are observed; a
    retried request (sent on through the gateway) is not.
    """

    __slots__ = ("forward", "relay", "relayed", "_started", "_mark")

    def __init__(self) -> None:
        self.forward = 0.0
        self.relay = 0.0
        self.relayed = False
        self._started = time.perf_counter()
        self._mark: float | None = None

    def forwarded(self) -> None:
        """The request is handed to the upstream client now."""
        self.forward = time.perf_counter() - self._started

    def start(self) -> None:
        """Upstream data (headers or a chunk) received: sidecar work starts."""
        self.relayed = True
        self._mark = time.perf_counter()

    def stop(self) -> None:
        """That data was written to the client (or the sidecar stopped to wait on
        upstream again)."""
        if self._mark is not None:
            self.relay += time.perf_counter() - self._mark
            self._mark = None

    def discard(self) -> None:
        """The request is retried elsewhere: not a locally answered request."""
        self.relayed = False
        self._mark = None

    @property
    def total(self) -> float:
        return self.forward + self.relay


class Metrics:
    KINDS = ("retry", "continue", "failed", "passthrough_abort")

    def __init__(self, model: str) -> None:
        self.model = model
        self.reissue: dict[tuple[str, str], int] = {}
        self.events: dict[str, int] = {}
        #: forward + relay per request (see :class:`AddedTime`), and the two parts.
        self.overhead = Histogram(OVERHEAD_BUCKETS_S)
        self.forward = Histogram(OVERHEAD_BUCKETS_S)
        self.relay = Histogram(OVERHEAD_BUCKETS_S)
        #: Abort -> continuation, per ``mode``: "stream" = to the FIRST continuation token
        #: (the client sees it at once); "nonstream" = to the COMPLETE continuation
        #: response (the client only sees anything then), so it also contains the
        #: continued generation time. The two are not comparable.
        self.gap = {"stream": Histogram(GAP_BUCKETS_S), "nonstream": Histogram(GAP_BUCKETS_S)}
        #: Fresh-connection re-sends to the local engine, per attempt (see
        #: ``ReissueSidecar._local_request``).
        self.reconnect: dict[str, int] = {"ok": 0, "fail": 0}
        #: Stale-connection failures on a reused connection NOT re-sent because they came
        #: later than ``local_reconnect_window_s`` after the connection was handed out.
        self.reconnect_outside_window = 0

    def observe_added(self, added: "AddedTime") -> None:
        added.stop()
        if not added.relayed:
            return
        self.overhead.observe(added.total)
        self.forward.observe(added.forward)
        self.relay.observe(added.relay)

    def count(self, kind: str, reason: str) -> None:
        key = (kind, reason)
        self.reissue[key] = self.reissue.get(key, 0) + 1

    def total(self, kind: str) -> int:
        return sum(v for (k, _), v in self.reissue.items() if k == kind)

    def event(self, name: str) -> None:
        self.events[name] = self.events.get(name, 0) + 1

    def render(self, state: "SleepState") -> str:
        model = _label(self.model)
        lines = [
            "# HELP tre_reissue_total Requests the sidecar retried, continued, failed to "
            "save, or passed an abort through for.",
            "# TYPE tre_reissue_total counter",
        ]
        for (kind, reason), value in sorted(self.reissue.items()):
            lines.append(f'tre_reissue_total{{model="{model}",kind="{kind}",reason="{reason}"}} {value}')
        lines += [
            "# HELP tre_reissue_proxy_added_seconds Time the sidecar itself adds to a request "
            "answered by the local engine: forward (full client request read -> request handed "
            "to the upstream client) + relay (summed over the response: upstream headers / "
            "chunk received -> written to the client). Excludes waiting on upstream (response "
            "headers, i.e. the generation of a non-streaming request, and gaps between chunks); "
            "a chunk write can include client backpressure.",
            "# TYPE tre_reissue_proxy_added_seconds histogram",
        ]
        lines += self.overhead.render("tre_reissue_proxy_added_seconds", f'model="{model}"')
        lines += [
            "# HELP tre_reissue_proxy_forward_seconds Forward part of "
            "tre_reissue_proxy_added_seconds (client request read -> handed upstream).",
            "# TYPE tre_reissue_proxy_forward_seconds histogram",
        ]
        lines += self.forward.render("tre_reissue_proxy_forward_seconds", f'model="{model}"')
        lines += [
            "# HELP tre_reissue_proxy_relay_seconds Relay part of tre_reissue_proxy_added_seconds "
            "(upstream data received -> written to the client, summed per request).",
            "# TYPE tre_reissue_proxy_relay_seconds histogram",
        ]
        lines += self.relay.render("tre_reissue_proxy_relay_seconds", f'model="{model}"')
        lines += [
            "# HELP tre_reissue_gap_seconds Abort to continuation: mode=stream is abort to the first "
            "continuation token; mode=nonstream is abort to the complete continuation response "
            "(includes the continued generation), so the two modes are not comparable.",
            "# TYPE tre_reissue_gap_seconds histogram",
        ]
        for mode, hist in self.gap.items():
            lines += hist.render("tre_reissue_gap_seconds", f'model="{model}",mode="{mode}"')
        lines += [
            "# HELP tre_reissue_events_total Sidecar events (sleep rejections, state corrections, "
            "client_cancel: the request was cancelled before the client had the complete response - "
            "the client went away, or the server shut down - and its upstream calls were closed, ...).",
            "# TYPE tre_reissue_events_total counter",
        ]
        for name, value in sorted(self.events.items()):
            lines.append(f'tre_reissue_events_total{{model="{model}",event="{name}"}} {value}')
        lines += [
            "# HELP tre_reissue_local_reconnect_total Re-sends to the local engine on a fresh "
            "connection after the pooled connection failed before any response byte (keep-alive "
            "race: disconnect / ECONNRESET / EPIPE), per attempt.",
            "# TYPE tre_reissue_local_reconnect_total counter",
        ]
        for result in ("ok", "fail"):
            lines.append(f'tre_reissue_local_reconnect_total{{model="{model}",result="{result}"}} '
                         f'{self.reconnect[result]}')
        lines += [
            "# HELP tre_reissue_local_reconnect_skipped_total Connection-level failures before "
            "any response byte that were NOT re-sent; reason=outside_window: the reused "
            "connection failed later than local_reconnect_window_s after it was handed out "
            "(not the keep-alive race).",
            "# TYPE tre_reissue_local_reconnect_skipped_total counter",
            f'tre_reissue_local_reconnect_skipped_total{{model="{model}",reason="outside_window"}} '
            f'{self.reconnect_outside_window}',
        ]
        lines += [
            "# HELP tre_reissue_sleeping 1 while the local engine is (going to) sleep.",
            "# TYPE tre_reissue_sleeping gauge",
            f'tre_reissue_sleeping{{model="{model}"}} {1 if state.active else 0}',
        ]
        return "\n".join(lines) + "\n"


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# ----------------------------------------------------------------------------- state


class SleepState:
    def __init__(self) -> None:
        self.sleeping = False
        #: /sleep (/pause) calls in progress: the engine pauses at their start.
        self.pending = 0
        #: /wake_up (/resume) calls in progress.
        self.waking = 0
        #: Bumped on every transition; an observed state is applied only if unchanged.
        self.epoch = 0
        #: /sleep (/pause) calls forwarded to the engine so far, counted BEFORE forwarding
        #: and never rolled back. A generation snapshots it at its start; an abort is
        #: sleep-caused iff the count moved since (``ReissueSidecar._abort_decision``):
        #: evidence by state, not by a time window, and it holds for an abort output that
        #: arrives after a /sleep the engine failed and rolled back.
        self.sleep_calls = 0
        self.last_sleep: dict[str, Any] | None = None

    def observe(self, is_sleeping: bool, epoch: int) -> str | None:
        """Apply an engine state observed while ``epoch`` was current. Returns the
        correction made ("to_sleeping" / "to_awake") or None."""
        if self.pending or self.waking or self.epoch != epoch or bool(is_sleeping) == self.sleeping:
            return None
        self.sleeping = bool(is_sleeping)
        self.epoch += 1
        return "to_sleeping" if self.sleeping else "to_awake"

    @property
    def active(self) -> bool:
        return self.sleeping or self.pending > 0


# ----------------------------------------------------------------- pure helpers (SSE)


def split_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Complete SSE events (each including its blank-line terminator) and the rest."""
    events: list[bytes] = []
    start = 0
    while True:
        index = buffer.find(b"\n\n", start)
        if index < 0:
            break
        events.append(buffer[start : index + 2])
        start = index + 2
    return events, buffer[start:]


def event_data(event: bytes) -> bytes | None:
    """The joined ``data:`` payload of one event, or None for a comment-only event."""
    if event.startswith(b"data: ") and event.count(b"\n") == 2 and b"\r" not in event:
        return event[6:-2]
    parts = []
    for line in event.split(b"\n"):
        line = line.rstrip(b"\r")
        if line.startswith(b"data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(b" ") else value)
    return b"\n".join(parts) if parts else None


def event_obj(event: bytes) -> dict | None:
    payload = event_data(event)
    if payload is None or payload.strip() == b"[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def is_usage_only(obj: dict) -> bool:
    return not obj.get("choices") and isinstance(obj.get("usage"), dict)


def has_abort(obj: dict) -> bool:
    return any(isinstance(c, dict) and c.get("finish_reason") == "abort" for c in obj.get("choices") or ())


def finish_of(obj: dict) -> str | None:
    for choice in obj.get("choices") or ():
        if isinstance(choice, dict) and choice.get("finish_reason"):
            return choice["finish_reason"]
    return None


def choice_text(choice: dict, chat: bool) -> str:
    if chat:
        for key in ("delta", "message"):
            part = choice.get(key)
            if isinstance(part, dict):
                content = part.get("content")
                return content if isinstance(content, str) else ""
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def events_text(blocks: "deque[bytes] | list[bytes]", chat: bool) -> str:
    parts: list[str] = []
    for block in blocks:
        for event in split_events(block)[0]:
            obj = event_obj(event)
            if obj is None:
                continue
            for choice in obj.get("choices") or ():
                if isinstance(choice, dict):
                    parts.append(choice_text(choice, chat))
    return "".join(parts)


def is_sleeping_error(obj: Any, error_type: str) -> bool:
    error = obj.get("error") if isinstance(obj, dict) else None
    return isinstance(error, dict) and error.get("type") == error_type


def stop_strings(body: dict) -> list[str]:
    stop = body.get("stop")
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str) and s]
    return []


def find_seam_stop(tail: str, text: str, stops: list[str], include: bool) -> int | None:
    """Cut position in ``tail + text`` for the earliest stop string that STARTS inside
    ``tail`` (the end of the first segment's text) and ends inside ``text`` (the
    continuation), i.e. spans the seam; None if there is none. Stops wholly inside
    ``text`` are the continuation engine's job (it has the same stop list)."""
    combined = tail + text
    best: tuple[int, int] | None = None
    for stop in stops:
        start = combined.find(stop, max(0, len(tail) - len(stop) + 1))
        while start != -1 and start < len(tail):
            if start + len(stop) > len(tail):
                cut = start + len(stop) if include else start
                if best is None or start < best[0]:
                    best = (start, cut)
                break
            start = combined.find(stop, start + 1)
    return None if best is None else best[1]


def merged_usage(prompt_tokens: int, generated: int, cont_usage: dict | None) -> dict:
    """Client-facing usage of a stitched request: the ORIGINAL prompt, all completion
    tokens of all segments (a nested continuation's usage is already merged)."""
    completion = int(generated) + int((cont_usage or {}).get("completion_tokens") or 0)
    return {"prompt_tokens": int(prompt_tokens), "completion_tokens": completion,
            "total_tokens": int(prompt_tokens) + completion}


def completion_chunk_as_chat(obj: dict, *, stream: bool) -> dict:
    """A /v1/completions chunk (or response) reshaped as a chat chunk (or response)."""
    out = {k: v for k, v in obj.items() if k != "choices"}
    choices = []
    for choice in obj.get("choices") or ():
        entry: dict[str, Any] = {"index": choice.get("index", 0)}
        text = choice.get("text") or ""
        if stream:
            entry["delta"] = {"content": text}
        else:
            entry["message"] = {"role": "assistant", "content": text}
        entry["logprobs"] = None
        entry["finish_reason"] = choice.get("finish_reason")
        if "stop_reason" in choice:
            entry["stop_reason"] = choice.get("stop_reason")
        choices.append(entry)
    out["choices"] = choices
    out["object"] = "chat.completion.chunk" if stream else "chat.completion"
    return out


def _present(value: Any) -> bool:
    return value is not None


def _num_above(value: Any, limit: float) -> bool:
    """Present and > limit; present but not a number counts as above (conservative)."""
    if value is None:
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return value > limit


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def parse_body(raw: bytes) -> Any:
    """The request body as JSON, or None when it cannot be parsed, accepting what the
    gateway's parser accepts: invalid UTF-8 and raw control characters inside strings
    pass, NaN / Infinity / a byte-order mark do not; too deep nesting is None too."""
    try:
        return json.loads(raw.decode("utf-8", "replace"), strict=False, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None


def non_continuable_reason(path: str, body: Any, cfg: "Config") -> str | None:
    """Why a request cannot be resumed from its emitted tokens (None = it can). The same
    rules and reason strings as the gateway plugin's treNonContinuableReason
    (pkg/plugins/gateway/tre_transparent_sleep.go), which makes the service-manager drain
    such requests instead of aborting them; both are checked against the shared contract
    tre/reissue/contract/non_continuable_cases.json. ``path`` excludes the query string
    and ``body`` is :func:`parse_body` of the raw request body."""
    if path not in (cfg.completions_path, cfg.chat_path):
        return "endpoint"
    if not isinstance(body, dict):
        return "body"
    if _num_above(body.get("n"), 1) or _num_above(body.get("best_of"), 1):
        return "n"
    logprobs = body.get("logprobs")
    if _present(logprobs) and logprobs is not False:
        return "logprobs"
    if _num_above(body.get("top_logprobs"), 0) or _present(body.get("prompt_logprobs")):
        return "logprobs"
    if body.get("echo") is True:
        return "echo"
    if body.get("use_beam_search") is True:
        return "beam_search"
    choice = body.get("tool_choice") if _present(body.get("tool_choice")) else body.get("function_call")
    if choice != "none":
        offered = any(_nonempty_list(body.get(k)) for k in ("tools", "functions"))
        if offered or (_present(choice) and choice != "auto"):
            return "tools"
    fmt = body.get("response_format")
    if _present(fmt) and not (isinstance(fmt, dict) and fmt.get("type") in (None, "", "text")):
        return "structured_output"
    for key in ("guided_json", "guided_regex", "guided_choice", "guided_grammar", "structural_tag",
                "structured_outputs"):
        if _present(body.get(key)):
            return "structured_output"
    if _present(body.get("guided_json_object")) and body.get("guided_json_object") is not False:
        return "structured_output"
    if path == cfg.completions_path:
        if _present(body.get("prompt_embeds")) or _present(body.get("suffix")):
            return "prompt_form"
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return None
        if isinstance(prompt, list) and prompt:
            if all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
                return None
            if len(prompt) == 1 and isinstance(prompt[0], (str, list)):
                return None
        return "prompt_form"
    messages = body.get("messages")
    return None if isinstance(messages, list) and messages else "prompt_form"


def _nonempty_list(value: Any) -> bool:
    if value is None:
        return False
    return not isinstance(value, list) or bool(value)


class NoBudget(ValueError):
    """A chat continuation needs a token limit and none is known."""


@dataclass
class ContPlan:
    path: str
    body: dict
    #: The original is a chat request continued as a completion: reshape the output.
    as_chat: bool


def build_continuation(
    path: str,
    body: dict,
    prompt_ids: list[int] | None,
    generated_ids: list[int],
    *,
    stream: bool,
    max_model_len: int | None,
    cfg: Config,
) -> ContPlan | None:
    """The continuation request, or None when the token budget is already spent.

    Nothing generated yet: the ORIGINAL request, unchanged (pure retry semantics).
    Otherwise a /v1/completions request with ``prompt = prompt_ids + generated_ids``
    (exact seam: no re-tokenization), ``max_tokens`` / ``min_tokens`` reduced by the
    generated count and the same sampling parameters (``seed`` included: a fixed seed
    restarts its random stream at the seam, so sampled output is valid but not
    bit-identical to an uninterrupted run)."""
    generated = len(generated_ids)
    if generated == 0:
        out = dict(body)
        out["stream"] = stream
        if stream:
            out["stream_options"] = dict(body.get("stream_options") or {}, include_usage=True)
        return ContPlan(path, out, as_chat=False)
    if prompt_ids is None:
        raise ValueError("prompt token ids missing")
    chat = path == cfg.chat_path
    if chat:
        limit = body.get("max_completion_tokens")
        if limit is None:
            limit = body.get("max_tokens")
        if limit is None:
            if not max_model_len:
                raise NoBudget("chat continuation without max_tokens needs max_model_len")
            limit = int(max_model_len) - len(prompt_ids)
        out = {k: body[k] for k in CHAT_TO_COMPLETION_KEYS if k in body}
    else:
        limit = body.get("max_tokens")
        limit = COMPLETIONS_DEFAULT_MAX_TOKENS if limit is None else int(limit)
        out = {k: v for k, v in body.items() if k not in COMPLETION_CONT_DROP}
    left = int(limit) - generated
    if left <= 0:
        return None
    out["prompt"] = [int(t) for t in prompt_ids] + [int(t) for t in generated_ids]
    out["max_tokens"] = left
    out.pop("max_completion_tokens", None)
    if out.get("min_tokens"):
        out["min_tokens"] = max(0, int(out["min_tokens"]) - generated)
    out["stream"] = stream
    if stream:
        out["stream_options"] = dict(body.get("stream_options") or {}, include_usage=True)
    else:
        out.pop("stream_options", None)
    return ContPlan(cfg.completions_path, out, as_chat=chat)


def forward_headers(headers, drop: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in headers.items():
        lower = name.lower()
        if lower in drop or lower.startswith("x-envoy-"):
            continue
        out[name] = value
    return out


def response_headers(headers) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in _RESPONSE_DROP}


def merge_exclude(values: list[str], pod: str) -> str:
    seen: list[str] = []
    for value in values + [pod]:
        for name in (value or "").split(","):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)
    return ",".join(seen)


def _int_header(value: str | None) -> int:
    try:
        return max(0, int(value)) if value is not None else 0
    except ValueError:
        return 0


def _retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


#: The sidecar's pool keep-alive must stay at least this far below vLLM's: the pool
#: timestamps a connection when the sidecar releases it, which lags the server's own
#: idle clock under CPU throttling (the sidecar runs with a 0.5-core limit).
KEEPALIVE_MARGIN_S = 1.0


def keepalive_is_safe(pool_keepalive_s: float, server_keepalive_s: float) -> bool:
    """The connection is closed by the side that SENDS requests first: the client's idle
    limit must be at least ``KEEPALIVE_MARGIN_S`` below the server's."""
    return pool_keepalive_s <= server_keepalive_s - KEEPALIVE_MARGIN_S


def server_options(cfg: "Config") -> dict[str, Any]:
    """Options of the sidecar's own HTTP server (aiohttp ``AppRunner`` / ``Server``).

    ``handler_cancellation``: when the client's connection is lost, aiohttp cancels the
    handler. A generation handler then closes every upstream request it holds (the local
    engine, the gateway retry or continuation; ``ClientResponse.release`` closes a
    connection whose body was not read to the end), and vLLM aborts a request whose
    connection closed, so a queued request of a client that went away is not prefilled.
    Sleep / wake calls are shielded from it (see ``handle``). Needs aiohttp >= 3.9."""
    return {"keepalive_timeout": cfg.server_keepalive_s, "handler_cancellation": True}


def serve_kwargs(cfg: "Config") -> dict[str, Any]:
    """Keyword arguments of ``web.run_app`` (the sidecar's own HTTP server)."""
    return {"host": cfg.listen_host, "port": cfg.listen_port, "access_log": None, "print": None,
            "backlog": 2048, "handle_signals": True, **server_options(cfg)}


def raise_nofile_limit(target: int) -> dict | None:
    """Raise this process's soft RLIMIT_NOFILE to ``min(target, hard)``; never lowers it.
    Logs the before / after values once (one JSON line) and returns that record; a
    failure is logged as a WARNING and never raised. ``target`` 0 = do nothing."""
    if target <= 0 or resource is None:
        return None
    record: dict[str, Any] = {"event": "tre_reissue_nofile", "target": target}
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        record.update(before=soft, after=soft, hard=hard)
        unlimited = resource.RLIM_INFINITY
        want = target if hard == unlimited else min(target, hard)
        if soft != unlimited and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            record["after"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception as exc:  # noqa: BLE001 - the sidecar must start anyway
        record.update(level="WARNING", error=f"{type(exc).__name__}: {exc}"[:300])
    _log(record)
    return record


def is_stale_connection_error(exc: BaseException) -> bool:
    """A request on an ESTABLISHED (pooled) connection failed at the connection level
    before any response byte: the server dropped it (``Server disconnected``), reset it
    (ECONNRESET) or the write hit a closed socket (EPIPE / aiohttp's "Cannot write to
    closing transport"). Typical cause: the connection was reused just as the server's
    keep-alive timer closed it. Only the exception is classified here; whether the
    connection was a REUSED one (the only case where that race exists) is tracked by
    ``ReissueSidecar._local_request``. A failed CONNECT (``ClientConnectorError``:
    refused, unreachable) is not one of these. aiohttp wraps a failed body write as
    ``ClientOSError(errno=None, "Can not write request body")`` with the reset as its
    ``__cause__`` (3.11 and 3.14), so the cause chain is followed."""
    if isinstance(exc, aiohttp.ClientConnectorError):
        return False
    seen = 0
    current: BaseException | None = exc
    while current is not None and seen < 4:
        if isinstance(current, aiohttp.ServerDisconnectedError):
            return True
        if isinstance(current, (ConnectionResetError, BrokenPipeError)):
            return True
        if isinstance(current, OSError) and current.errno in (errno.ECONNRESET, errno.EPIPE):
            return True
        current = current.__cause__
        seen += 1
    return False


def is_connection_refused(exc: BaseException) -> bool:
    """Nothing listens on the local engine's port (vLLM not (yet / any more) up).
    Assumes a single-address upstream (the manifests use 127.0.0.1): for a name that
    resolves to several addresses (``localhost``: v4 + v6) aiohttp raises one combined
    ``OSError("Multiple exceptions")`` without an errno, which is NOT recognised here
    (the request then gets the 503 instead of the gateway retry)."""
    if not isinstance(exc, aiohttp.ClientConnectorError):
        return False
    os_error = getattr(exc, "os_error", None)
    return isinstance(os_error, ConnectionRefusedError) or exc.errno == errno.ECONNREFUSED


def _client_gone(request: web.Request) -> bool:
    transport = request.transport
    return transport is None or transport.is_closing()


def _log(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------- sidecar


@dataclass
class AbortInfo:
    """What the first segment left behind when it was aborted."""

    obj: dict
    prompt_ids: list[int] | None
    generated_ids: list[int] | None
    text: str


class ReissueSidecar:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = SleepState()
        self.metrics = Metrics(cfg.model)
        self.local: aiohttp.ClientSession | None = None
        #: One connection per request (force_close): the re-send after a stale pooled
        #: connection must not pick another pooled connection of the same age.
        self.local_fresh: aiohttp.ClientSession | None = None
        self.gateway: aiohttp.ClientSession | None = None
        #: Rate-limited WARNING lines: kind -> (last emitted, monotonic; suppressed since).
        self._warned: dict[str, tuple[float, int]] = {}
        #: Requests that got an idle pooled connection (diagnostics / tests).
        self.local_reused = 0
        self._monitor_task: asyncio.Task | None = None
        self._max_model_len: int | None = cfg.max_model_len or None
        self._sleep_error_mark = json.dumps(cfg.sleeping_error_type).encode()
        self._gateway_drop = _GATEWAY_DROP_BASE | frozenset(
            h.lower() for h in (cfg.depth_header, cfg.hidden_header, cfg.exclude_header)
        )
        self._local_drop = HOP_BY_HOP | frozenset({cfg.depth_header.lower()})

    # ---------------------------------------------------------------- lifecycle

    async def on_startup(self, app: web.Application) -> None:
        cfg = self.cfg
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=cfg.connect_timeout_s)
        if not keepalive_is_safe(cfg.upstream_keepalive_s, cfg.upstream_server_keepalive_s):
            _log({"level": "WARNING", "event": "tre_upstream_keepalive_unsafe", "model": cfg.model,
                  "pod": cfg.pod_name, "upstream_keepalive_s": cfg.upstream_keepalive_s,
                  "upstream_server_keepalive_s": cfg.upstream_server_keepalive_s,
                  "detail": "the sidecar's keep-alive to the local vLLM must be at least "
                            f"{KEEPALIVE_MARGIN_S:g} s below vLLM's (VLLM_HTTP_TIMEOUT_KEEP_ALIVE), else "
                            "pooled connections are reused while vLLM closes them; fresh-connection "
                            "re-sends still cover it"})
        if cfg.gateway_upstream_idle_s > 0 and not keepalive_is_safe(
                cfg.gateway_upstream_idle_s, cfg.server_keepalive_s):
            _log({"level": "WARNING", "event": "tre_server_keepalive_unsafe", "model": cfg.model,
                  "pod": cfg.pod_name, "server_keepalive_s": cfg.server_keepalive_s,
                  "gateway_upstream_idle_s": cfg.gateway_upstream_idle_s,
                  "detail": "the sidecar's own server keep-alive must be at least "
                            f"{KEEPALIVE_MARGIN_S:g} s above Envoy's upstream idle timeout "
                            "(gateway.upstream_idle_timeout_s), else Envoy reuses connections the "
                            "sidecar is closing (503 UC / reset)"})
        # keepalive_timeout < vLLM's keep-alive: the pool drops an idle connection before
        # the server can close it under a reused request.
        reuse = aiohttp.TraceConfig()
        reuse.on_connection_reuseconn.append(self._on_reuseconn)
        self.local = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=cfg.upstream_keepalive_s),
            timeout=timeout, auto_decompress=False, trace_configs=[reuse],
        )
        self.local_fresh = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=True), timeout=timeout, auto_decompress=False
        )
        self.gateway = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0), timeout=timeout, auto_decompress=False
        )
        self._monitor_task = asyncio.ensure_future(self._monitor())

    async def on_cleanup(self, app: web.Application) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
        for session in (self.local, self.local_fresh, self.gateway):
            if session is not None:
                await session.close()

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=self.cfg.client_max_size)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app

    def _timeout(self, total: float | None) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=total, sock_connect=self.cfg.connect_timeout_s)

    async def _monitor(self) -> None:
        """Periodic direct /is_sleeping probe of the engine: corrects the sleeping mark
        after a vLLM or sidecar restart."""
        while True:
            try:
                await self._probe_once()
                if self._max_model_len is None:
                    await self._fetch_max_model_len()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the monitor must never die
                _log({"event": "tre_monitor_error", "error": repr(exc)[:300]})
            await asyncio.sleep(self.cfg.probe_interval_s)

    async def _probe_once(self) -> None:
        epoch = self.state.epoch
        try:
            async with self.local.get(
                self.cfg.upstream_url + self.cfg.is_sleeping_path, timeout=self._timeout(5.0)
            ) as resp:
                if resp.status != 200:
                    return
                payload = await resp.json(content_type=None)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - engine not up (yet / any more)
            return
        if isinstance(payload, dict) and "is_sleeping" in payload:
            self._apply_observed(bool(payload["is_sleeping"]), epoch, source="probe")

    async def _fetch_max_model_len(self) -> int | None:
        try:
            async with self.local.get(
                self.cfg.upstream_url + self.cfg.models_path, timeout=self._timeout(5.0)
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            for card in payload.get("data") or ():
                value = card.get("max_model_len") if isinstance(card, dict) else None
                if isinstance(value, int) and value > 0:
                    self._max_model_len = value
                    return value
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return None
        return None

    def _apply_observed(self, is_sleeping: bool, epoch: int, *, source: str) -> None:
        change = self.state.observe(is_sleeping, epoch)
        if change:
            self.metrics.event(f"state_corrected_{change}")
            _log({"event": "tre_state_corrected", "model": self.cfg.model, "pod": self.cfg.pod_name,
                  "change": change, "source": source})

    # ------------------------------------------------------ local engine calls

    async def _on_reuseconn(self, session: aiohttp.ClientSession, ctx: Any, params: Any) -> None:
        """aiohttp trace hook: this request got an idle pooled connection."""
        self.local_reused += 1
        marker = getattr(ctx, "trace_request_ctx", None)
        if isinstance(marker, dict):
            marker["reused"] = True
            marker["reused_at"] = time.monotonic()

    async def _local_request(self, method: str, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """``self.local.request`` (the response headers are in when it returns), re-sent
        on a fresh connection up to ``local_reconnect_attempts`` times when the attempt
        ran on a REUSED pooled connection, failed with
        :func:`is_stale_connection_error` and did so within ``local_reconnect_window_s``
        of getting the connection - the keep-alive race: uvicorn closes a connection only
        while it is idle (data arriving cancels its keep-alive timer), so the request was
        not processed, and the failure shows up at once. A failure on a newly opened
        connection, or on a reused one only after a long time (engine crash mid-
        generation, so the request may have run), is not re-sent.

        Layering note: aiohttp itself already retries once, inside ``request()``, when a
        persistent connection fails for an IDEMPOTENT method (GET / HEAD / OPTIONS / TRACE /
        PUT / DELETE); it never does for POST, which is what generation requests are, so
        this method is the only retry for them.
        Callers invoke it before writing anything to the client, and the body is
        ``bytes``, so the re-send is exact. Raises the last error."""
        marker: dict[str, Any] = {"reused": False}
        try:
            return await self.local.request(method, url, trace_request_ctx=marker, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            if (not marker["reused"] or not is_stale_connection_error(exc)
                    or self.cfg.local_reconnect_attempts < 1):
                raise
            if time.monotonic() - marker["reused_at"] > self.cfg.local_reconnect_window_s:
                self.metrics.reconnect_outside_window += 1
                self._warn("local_reconnect_outside_window", {
                    "event": "tre_local_reconnect_outside_window", "path": url,
                    "window_s": self.cfg.local_reconnect_window_s,
                    "error": f"{type(exc).__name__}: {exc}"[:300]})
                raise
            first = exc
        last: BaseException = first
        for _ in range(self.cfg.local_reconnect_attempts):
            try:
                resp = await self.local_fresh.request(method, url, **kwargs)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                self.metrics.reconnect["fail"] += 1
                last = exc
                if not is_stale_connection_error(exc):
                    break  # refused / timed out: another fresh connection does not help
                continue
            self.metrics.reconnect["ok"] += 1
            self._warn("local_reconnect", {"event": "tre_local_reconnect", "result": "ok", "path": url,
                                           "error": f"{type(first).__name__}: {first}"[:300]})
            return resp
        raise last

    def _warn(self, kind: str, record: dict) -> None:
        """A WARNING line, at most one per ``warn_interval_s`` per ``kind``; the next line
        says how many were suppressed in between."""
        now = time.monotonic()
        last, suppressed = self._warned.get(kind, (None, 0))
        if last is not None and now - last < self.cfg.warn_interval_s:
            self._warned[kind] = (last, suppressed + 1)
            return
        self._warned[kind] = (now, 0)
        _log({"level": "WARNING", **record, "model": self.cfg.model, "pod": self.cfg.pod_name,
              "suppressed": suppressed})

    def _upstream_failed(self, request: web.Request, depth: int, exc: BaseException, *,
                         status: int, account: bool = True) -> web.Response:
        """The local engine could not be reached before anything was sent to the client.
        503 + Retry-After (``error.layer = sidecar_upstream``): the request MAY have been
        executed by the engine (a connection that died mid-request cannot tell), so it is
        retryable for generation requests (they carry no side effects), not for anything
        else; 502 only where kept for compatibility (connection refused etc.
        on a plain proxied path). Counted requests get one ``tre_reissue`` line each
        (with the request id) besides the rate-limited WARNING."""
        error = f"{type(exc).__name__}: {exc}"[:300]
        if account:
            self._account("failed", "upstream_unavailable", request, depth, error=error, status=status,
                          request_id=request.headers.get("x-request-id"))
        if account or status == 503:  # not: a probe finding the engine not (yet) listening
            self._warn(f"upstream_{status}", {"event": "tre_upstream_unavailable", "status": status,
                                              "path": request.path, "error": error})
        if status == 503:
            return _error(503, f"local engine connection failed before any response; the request may "
                               f"have been executed, generation requests can be retried ({error})",
                          "ServiceUnavailable", headers={"Retry-After": "1"}, layer="sidecar_upstream")
        return _error(status, f"upstream unavailable: {error}", "BadGateway", layer="sidecar_upstream")

    # ------------------------------------------------------------------ routing

    async def handle(self, request: web.Request) -> web.StreamResponse:
        cfg = self.cfg
        path = request.path
        method = request.method
        if method == "POST":
            # Sleep / wake run to the end even if the caller disconnects (the server
            # cancels handlers on a lost connection, ``server_options``): the engine
            # finishes the call anyway, and the sleeping mark must follow its answer.
            if path in cfg.sleep_paths:
                return await self._run_to_end(self._handle_sleep(request), request)
            if path in cfg.wake_paths:
                return await self._run_to_end(self._handle_wake(request), request)
            if cfg.enabled and path.startswith(cfg.retry_path_prefix):
                return await self._handle_generation(request, path)
        elif method == "GET":
            if path == cfg.metrics_path:
                return web.Response(text=self.metrics.render(self.state), content_type="text/plain")
            if path == cfg.state_path:
                return web.json_response(self._state_view())
            if path == cfg.is_sleeping_path:
                return await self._handle_is_sleeping(request)
        body = await request.read()
        timeout = None if path.startswith(cfg.retry_path_prefix) else cfg.proxy_timeout_s
        return await self._proxy_local(request, body, timeout=timeout)

    async def _run_to_end(self, coro, request: web.Request) -> web.StreamResponse:
        """``coro`` shielded from the handler's cancellation; if the caller went away, an
        error of the call that carries on is logged (WARNING), not left to the GC."""
        task = asyncio.ensure_future(coro)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            path = request.path

            def report(done: asyncio.Task) -> None:
                if not done.cancelled() and done.exception() is not None:
                    self._warn("detached_control_call", {
                        "event": "tre_control_call_failed_after_disconnect", "path": path,
                        "error": f"{type(done.exception()).__name__}: {done.exception()}"[:300]})

            task.add_done_callback(report)
            raise

    def _state_view(self) -> dict:
        return {
            "model": self.cfg.model,
            "pod": self.cfg.pod_name,
            "enabled": self.cfg.enabled,
            "sleeping": self.state.sleeping,
            "sleep_pending": self.state.pending,
            "waking": self.state.waking,
            "epoch": self.state.epoch,
            "last_sleep": self.state.last_sleep,
            "gateway_url": self.cfg.gateway_url,
            "max_depth": self.cfg.max_depth,
            "max_model_len": self._max_model_len,
            "counts": {kind: self.metrics.total(kind) for kind in Metrics.KINDS},
        }

    # -------------------------------------------------------------- plain proxy

    async def _proxy_local(self, request: web.Request, body: bytes, *, timeout: float | None):
        try:
            resp = await self._local_request(
                request.method, self.cfg.upstream_url + request.path_qs, data=body if body else None,
                headers=forward_headers(request.headers, HOP_BY_HOP), allow_redirects=False,
                timeout=self._timeout(timeout),
            )
        except asyncio.TimeoutError:
            return _error(504, "upstream timed out", "GatewayTimeout")
        except (aiohttp.ClientError, OSError) as exc:
            # Only client API requests (POST /v1/*) are counted: probes of a starting /
            # dead engine (/health, /metrics, GET /v1/models, ...) would flood
            # tre_reissue_total{kind="failed"}.
            api = request.method == "POST" and request.path.startswith(self.cfg.retry_path_prefix)
            depth = _int_header(request.headers.get(self.cfg.depth_header))
            if is_stale_connection_error(exc):
                return self._upstream_failed(request, depth, exc, status=503, account=api)
            return self._upstream_failed(request, depth, exc, status=502, account=api)
        return await self._relay(request, resp)

    async def _relay(
        self, request: web.Request, resp: aiohttp.ClientResponse, *, extra: dict[str, str] | None = None,
        added: AddedTime | None = None,
    ) -> web.StreamResponse:
        """Relay ``resp`` unchanged. With ``added``: time the sidecar's part (headers
        received -> client headers sent, each chunk received -> written)."""
        if added is not None:
            added.start()
        try:
            headers = response_headers(resp.headers)
            if extra:
                headers.update(extra)
            out = web.StreamResponse(status=resp.status, reason=resp.reason, headers=headers)
            if resp.content_length is not None and "chunked" not in resp.headers.get("Transfer-Encoding", "").lower():
                out.content_length = resp.content_length
            try:
                await out.prepare(request)
            except (ConnectionResetError, OSError):
                resp.close()
                return out
            if added is not None:
                added.stop()
            try:
                async for data in resp.content.iter_any():
                    if added is not None:
                        added.start()
                    await out.write(data)
                    if added is not None:
                        added.stop()
            except ConnectionResetError:
                # The client went away (aiohttp's ClientConnectionResetError is also a
                # ClientError, so this clause comes first): drop the upstream.
                resp.close()
                return out
            except (aiohttp.ClientError, asyncio.TimeoutError):
                resp.close()
                if request.transport is not None:
                    request.transport.close()
                return out
            try:
                await out.write_eof()
            except ConnectionResetError:
                pass
            return out
        finally:
            resp.release()

    # ------------------------------------------------------------- sleep / wake

    async def _control_call(self, request: web.Request, body: bytes, drop: frozenset[str]):
        """(status, payload, headers) or (None, error, {})."""
        try:
            async with await self._local_request(
                "POST", self.cfg.upstream_url + request.path_qs, data=body if body else None,
                headers=forward_headers(request.headers, drop), timeout=self._timeout(self.cfg.control_timeout_s),
            ) as resp:
                return resp.status, await resp.read(), response_headers(resp.headers)
        except asyncio.TimeoutError:
            return None, f"{request.path} timed out", {}
        except (aiohttp.ClientError, OSError) as exc:
            return None, f"{request.path} failed: {exc}", {}

    async def _handle_sleep(self, request: web.Request) -> web.StreamResponse:
        cfg, state = self.cfg, self.state
        body = await request.read()
        if cfg.require_hidden_header and request.headers.get(cfg.hidden_header, "").strip() != "1":
            self.metrics.event("sleep_rejected_not_hidden")
            _log({"event": "tre_sleep_rejected", "model": cfg.model, "pod": cfg.pod_name, "path": request.path,
                  "reason": f"missing {cfg.hidden_header}: 1"})
            return _error(409, f"{request.path} refused: hide the pod first (the service-manager sends "
                               f"{cfg.hidden_header}: 1 once the gateway stopped routing to it)", "Conflict")
        # Mark BEFORE forwarding: the engine aborts in-flight requests inside the call and
        # their abort outputs reach us before the call returns. Idempotent.
        transition = not state.active
        state.pending += 1
        state.sleep_calls += 1
        if transition:
            state.epoch += 1
        started = time.time()
        try:
            status, payload, headers = await self._control_call(
                request, body, HOP_BY_HOP | {cfg.hidden_header.lower()}
            )
        finally:
            state.pending -= 1
        if status is None or not 200 <= status < 300:
            # The engine did not confirm: roll back (a probe corrects it if it slept anyway).
            if not state.active:
                state.epoch += 1
            self.metrics.event("sleep_failed")
            _log({"event": "tre_sleep_failed", "model": cfg.model, "pod": cfg.pod_name, "status": status,
                  "error": payload if status is None else None})
            if status is None:
                return _error(502, str(payload), "BadGateway")
            return web.Response(body=payload, status=status, headers=headers)
        if not state.sleeping:
            state.sleeping = True
        state.last_sleep = {"ts": started, "path": request.path, "query": request.query_string,
                            "duration_s": round(time.time() - started, 3)}
        _log({"event": "tre_sleep", "model": cfg.model, "pod": cfg.pod_name, "path": request.path_qs,
              "duration_s": state.last_sleep["duration_s"]})
        return web.Response(body=payload, status=status, headers=headers)

    async def _handle_wake(self, request: web.Request) -> web.StreamResponse:
        state = self.state
        body = await request.read()
        state.waking += 1
        try:
            status, payload, headers = await self._control_call(request, body, HOP_BY_HOP)
        finally:
            state.waking -= 1
        if status is None:
            return _error(502, str(payload), "BadGateway")
        if 200 <= status < 300:
            state.sleeping = False
            state.epoch += 1
        return web.Response(body=payload, status=status, headers=headers)

    async def _handle_is_sleeping(self, request: web.Request) -> web.StreamResponse:
        """Proxied /is_sleeping; its answer also re-syncs the sleeping mark."""
        epoch = self.state.epoch
        try:
            async with await self._local_request(
                "GET", self.cfg.upstream_url + request.path_qs, headers=forward_headers(request.headers, HOP_BY_HOP),
                timeout=self._timeout(self.cfg.proxy_timeout_s),
            ) as resp:
                payload, status, headers = await resp.read(), resp.status, response_headers(resp.headers)
        except asyncio.TimeoutError:
            return _error(504, "upstream timed out", "GatewayTimeout")
        except (aiohttp.ClientError, OSError) as exc:
            return _error(502, f"upstream unavailable: {exc}", "BadGateway")
        if status == 200:
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and "is_sleeping" in parsed:
                self._apply_observed(bool(parsed["is_sleeping"]), epoch, source="is_sleeping")
        return web.Response(body=payload, status=status, headers=headers)

    # ------------------------------------------------------------ gateway calls

    def _gateway_headers(self, request: web.Request, *, model: Any, depth: int) -> dict[str, str]:
        cfg = self.cfg
        headers = forward_headers(request.headers, self._gateway_drop)
        if isinstance(model, str) and model and not any(n.lower() == "model" for n in headers):
            headers["model"] = model
        headers[cfg.depth_header] = str(depth)
        exclude = merge_exclude(request.headers.getall(cfg.exclude_header, []), cfg.pod_name)
        if exclude:
            headers[cfg.exclude_header] = exclude
        return headers

    async def _gateway_request(
        self, method: str, path_qs: str, data: bytes, headers: dict[str, str]
    ) -> tuple[aiohttp.ClientResponse | None, int, str]:
        """Send a not-yet-started request to the gateway, retrying 503 / 502 / connect
        errors (bounded, backoff, Retry-After honoured). (response, attempts, last error).

        One retry layer: a 502 / 503 produced by ANOTHER sidecar (its error message starts
        with ``SIDECAR_ERROR_MARK``: hop limit, its own retries exhausted, its engine
        unreachable) is final, not retried - that sidecar already spent its own bounded
        attempts, and retrying it would multiply the sends per hop. Envoy's / the
        gateway's own 502 / 503 (no routable pod, connect failure) are retried."""
        cfg = self.cfg
        error = ""
        delay = 0.0
        for attempt in range(1, cfg.retry_attempts + 1):
            if delay > 0:
                await asyncio.sleep(delay)
            backoff = min(cfg.retry_backoff_s * (2 ** (attempt - 1)), cfg.retry_max_backoff_s)
            try:
                resp = await self.gateway.request(method, cfg.gateway_url + path_qs, data=data, headers=headers)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                error = f"gateway: {type(exc).__name__}: {exc}"
                delay = backoff
                continue
            if resp.status in (502, 503):
                try:
                    payload = await resp.read()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    payload = b""
                finally:
                    resp.release()
                detail = payload[:300].decode("utf-8", "replace")
                if SIDECAR_ERROR_MARK in payload:
                    return None, attempt, f"sidecar HTTP {resp.status} (final, not retried): {detail}"
                error = f"gateway HTTP {resp.status}: {detail}"
                advised = _retry_after(resp.headers.get("Retry-After"))
                delay = min(advised if advised is not None else backoff, cfg.retry_max_backoff_s)
                continue
            return resp, attempt, ""
        return None, cfg.retry_attempts, error

    # --------------------------------------------------------------- retry path

    async def _retry(
        self, request: web.Request, raw: bytes, depth: int, reason: str, added: AddedTime | None = None
    ) -> web.StreamResponse:
        """Forward a request that has not started (nothing sent to the client) to the
        gateway, unchanged apart from the exclude / depth headers."""
        if added is not None:
            added.discard()  # answered elsewhere: not a locally relayed request
        request[_REISSUE_PENDING] = "retry"
        cfg = self.cfg
        if depth + 1 > cfg.max_depth:
            self._account("failed", "depth_limit", request, depth, retry_of=reason)
            return self._unavailable("retry hop limit reached")
        model = _body_model(raw)
        headers = self._gateway_headers(request, model=model, depth=depth + 1)
        resp, attempts, error = await self._gateway_request(request.method, request.path_qs, raw, headers)
        if resp is None:
            self._account("failed", "retry_exhausted", request, depth, retry_of=reason, attempts=attempts,
                          error=error)
            return self._unavailable(f"every instance is asleep or unreachable ({error})")
        self._account("retry", reason, request, depth, attempts=attempts, target=_target(resp))
        return await self._relay(request, resp, extra={cfg.retried_header: str(attempts)})

    def _unavailable(self, message: str) -> web.Response:
        return _error(503, message, "ServiceUnavailable", headers={"Retry-After": "1"})

    # --------------------------------------------------------------- generation

    async def _handle_generation(self, request: web.Request, path: str) -> web.StreamResponse:
        raw = await request.read()
        added = AddedTime()  # the full client request is read: the sidecar's clock starts
        try:
            return await self._generation(request, path, raw, added)
        except asyncio.CancelledError:
            # The client went away (or the server shuts down): the upstream requests
            # were closed on the way out. A request whose retry / continuation was cut
            # short still gets its one tre_reissue_total line. Once the client had the
            # complete response it is a completed request (normal accounting).
            if not request.get(_DELIVERED):
                added.discard()
                self.metrics.event("client_cancel")
                kind = request.pop(_REISSUE_PENDING, None)
                if kind is not None:
                    self._account(kind, "client_gone", request,
                                  _int_header(request.headers.get(self.cfg.depth_header)))
            raise
        finally:
            self.metrics.observe_added(added)

    async def _generation(
        self, request: web.Request, path: str, raw: bytes, added: AddedTime
    ) -> web.StreamResponse:
        cfg = self.cfg
        depth = _int_header(request.headers.get(cfg.depth_header))
        if self.state.active:
            return await self._retry(request, raw, depth, "local_sleeping")
        body: Any = None
        generation = path in (cfg.completions_path, cfg.chat_path)
        if generation:
            body = parse_body(raw)
        nc_reason = non_continuable_reason(path, body, cfg) if generation else "endpoint"
        epoch = self.state.epoch
        sleeps = self.state.sleep_calls  # an abort after a /sleep call since now is sleep-caused
        headers = forward_headers(request.headers, self._local_drop)
        added.forwarded()
        try:
            # Nothing has been written to the client yet: a stale pooled connection is
            # re-sent on a fresh one inside _local_request.
            resp = await self._local_request(
                "POST", cfg.upstream_url + request.path_qs, data=raw, headers=headers,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            if self.state.active:
                return await self._retry(request, raw, depth, "local_unavailable_sleeping", added)
            if is_connection_refused(exc):
                # The engine is not listening (crashed / restarting): another instance.
                return await self._retry(request, raw, depth, "local_refused", added)
            return self._upstream_failed(request, depth, exc, status=503)
        if resp.status == 503:
            try:
                payload = await resp.read()
            finally:
                resp.release()
            if self._sleep_error_mark in payload and _is_sleeping_payload(payload, cfg.sleeping_error_type):
                self._apply_observed(True, epoch, source="engine_503")
                return await self._retry(request, raw, depth, "engine_sleeping")
            return web.Response(body=payload, status=503, headers=response_headers(resp.headers))
        if not generation or resp.status != 200 or not isinstance(body, dict):
            return await self._relay(request, resp, added=added)
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            return await self._stream(request, path, raw, body, resp, depth, nc_reason, added, sleeps)
        if "json" in ctype:
            return await self._non_stream(request, path, raw, body, resp, depth, nc_reason, added, sleeps)
        return await self._relay(request, resp, added=added)

    def _abort_decision(self, request: web.Request, depth: int, nc_reason: str | None,
                        info: AbortInfo, sent_any: bool, sleeps: int) -> tuple[str, str]:
        """(action, reason); action = retry | continue | passthrough_abort | failed.

        ``sleeps``: ``state.sleep_calls`` when the generation started. The abort is
        sleep-caused iff a /sleep call was forwarded since - whether it is still running,
        succeeded, or failed and was rolled back (the engine may abort its requests and
        then fail the sleep; their abort outputs can arrive after the call returned). The
        current sleeping mark is not needed: the only way to put the engine to sleep is a
        /sleep through this sidecar, and a generation starts only while the mark is
        clear. An abort with no sleep call in between is the engine's own: passed through."""
        if self.state.sleep_calls == sleeps:
            return "passthrough_abort", "not_sleeping"
        if _client_gone(request):
            return "passthrough_abort", "client_gone"
        if not sent_any:
            # Nothing reached the client: resending the original is exact.
            return "retry", "abort_before_output"
        if nc_reason is not None:
            return "passthrough_abort", f"non_continuable_{nc_reason}"
        if depth + 1 > self.cfg.max_depth:
            return "failed", "depth_limit"
        if info.generated_ids is None or (info.generated_ids and info.prompt_ids is None):
            return "failed", "no_token_ids"
        return "continue", "abort_sleep"

    def _abort_info(self, obj: dict, chat: bool, *, stream: bool) -> AbortInfo:
        cfg = self.cfg
        choices = [c for c in obj.get("choices") or () if isinstance(c, dict)]
        choice = choices[0] if choices else {}
        ids_field = cfg.generated_ids_field if stream else cfg.token_ids_field
        generated = choice.get(ids_field)
        prompt = obj.get(cfg.prompt_ids_field)
        if prompt is None:
            prompt = choice.get(cfg.prompt_ids_field)
        return AbortInfo(
            obj=obj,
            prompt_ids=list(prompt) if isinstance(prompt, list) else None,
            generated_ids=list(generated) if isinstance(generated, list) else None,
            text=choice_text(choice, chat),
        )

    # ------------------------------------------------------------ streaming path

    async def _stream(
        self, request: web.Request, path: str, raw: bytes, body: dict, resp: aiohttp.ClientResponse,
        depth: int, nc_reason: str | None, added: AddedTime, sleeps: int,
    ) -> web.StreamResponse:
        cfg = self.cfg
        chat = path == cfg.chat_path
        stops = stop_strings(body) if nc_reason is None else []
        window = max((len(s) for s in stops), default=1) - 1
        recent: deque[bytes] | None = deque(maxlen=window + 4) if window > 0 else None
        client: web.StreamResponse | None = None
        buffer = b""
        abort: AbortInfo | None = None
        first_usage: dict | None = None
        upstream_failed = False
        sleep_mark = self._sleep_error_mark

        async def open_client() -> web.StreamResponse:
            out = web.StreamResponse(status=resp.status, reason=resp.reason, headers=response_headers(resp.headers))
            await out.prepare(request)
            return out

        try:
            try:
                async for data in resp.content.iter_any():
                    # Each chunk: received -> processed / written (sidecar-added time).
                    added.start()
                    if abort is not None:
                        # The first segment's tail after its abort chunk: usage + [DONE].
                        buffer += data
                        added.stop()
                        continue
                    buffer += data
                    cut = buffer.rfind(b"\n\n")
                    if cut < 0:
                        added.stop()  # a partial event: wait for upstream
                        continue
                    complete = buffer[: cut + 2]
                    buffer = buffer[cut + 2 :]
                    if _ABORT_MARK not in complete and (client is not None or sleep_mark not in complete):
                        # Fast path: whole events that cannot matter, forwarded untouched.
                        if client is None:
                            client = await open_client()
                        if recent is not None:
                            recent.append(complete)
                        await client.write(complete)
                        if complete.endswith(SSE_DONE):
                            request[_DELIVERED] = True
                        added.stop()
                        continue
                    out: list[bytes] = []
                    events = split_events(complete)[0]
                    for index, event in enumerate(events):
                        obj = event_obj(event)
                        if obj is not None and client is None and not out and is_sleeping_error(
                            obj, cfg.sleeping_error_type
                        ):
                            resp.close()
                            self._apply_observed(True, self.state.epoch, source="engine_stream_error")
                            return await self._retry(request, raw, depth, "engine_sleeping", added)
                        if obj is not None and has_abort(obj):
                            info = self._abort_info(obj, chat, stream=True)
                            action, reason = self._abort_decision(
                                request, depth, nc_reason, info, sent_any=client is not None or bool(out),
                                sleeps=sleeps,
                            )
                            if action == "retry":
                                resp.close()
                                return await self._retry(request, raw, depth, reason, added)
                            if action == "continue":
                                request[_REISSUE_PENDING] = "passthrough_abort"
                                abort = info
                                buffer = b"".join(events[index + 1 :]) + buffer
                                break
                            self._account(action, reason, request, depth)
                        out.append(event)
                    if out:
                        if client is None:
                            client = await open_client()
                        block = b"".join(out)
                        if recent is not None:
                            recent.append(block)
                        await client.write(block)
                        if block.endswith(SSE_DONE):
                            request[_DELIVERED] = True
                    added.stop()
            except ConnectionResetError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError):
                upstream_failed = True
            # Waiting on upstream (the next chunk / end of stream) is not sidecar time;
            # a chunk left mid-processing (break, abort tail) is closed here.
            added.stop()
            if abort is not None:
                for event in split_events(buffer)[0]:
                    obj = event_obj(event)
                    if obj is not None and is_usage_only(obj):
                        first_usage = obj["usage"]
                buffer = b""
            elif client is None and not upstream_failed:
                added.start()
                client = await open_client()
                added.stop()
        except ConnectionResetError:
            resp.close()
            return client if client is not None else web.Response(status=499)
        finally:
            resp.release()
        if abort is not None and client is not None:
            await self._continue_stream(request, client, path, body, depth, abort, first_usage, recent, stops,
                                        window, added)
            return client
        if client is None:
            return _error(502, "upstream stream failed before the first event", "BadGateway")
        try:
            if upstream_failed:
                if request.transport is not None:
                    request.transport.close()
                return client
            added.start()
            if buffer:
                await client.write(buffer)
            await client.write_eof()
        except ConnectionResetError:
            pass
        finally:
            added.stop()
        return client

    async def _continue_stream(
        self, request: web.Request, client: web.StreamResponse, path: str, body: dict, depth: int,
        abort: AbortInfo, first_usage: dict | None, recent: deque[bytes] | None, stops: list[str], window: int,
        added: AddedTime | None = None,
    ) -> None:
        """``added``: each continuation chunk's processing / write counts as relay
        time; the continuation request through the gateway is upstream time."""
        cfg = self.cfg
        chat = path == cfg.chat_path
        added = added or AddedTime()
        t_abort = time.monotonic()
        generated = len(abort.generated_ids or ())
        prompt_tokens = (first_usage or {}).get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = len(abort.prompt_ids or ())
        wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        base = {k: abort.obj.get(k) for k in ("id", "object", "created", "model")}
        include = bool(body.get("include_stop_str_in_output"))
        held = abort.text
        # Seam stop matching: the end of the first segment (text already sent plus the
        # withheld tail the abort chunk flushed) against the continuation's first chars.
        matching = bool(stops) and window > 0
        tail = ""
        if matching:
            emitted = events_text(recent or (), chat)
            tail = (emitted + held)[-max(window, len(held)):] if (emitted or held) else ""
        emitted_in_tail = len(tail) - len(held)

        def with_text(template: dict, text: str, finish: str | None) -> dict:
            out = {k: v for k, v in base.items() if v is not None}
            if cfg.continued_field in template:
                out[cfg.continued_field] = template[cfg.continued_field]
            choice = {"index": 0, "logprobs": None, "finish_reason": finish}
            if chat:
                choice["delta"] = {"content": text}
            else:
                choice["text"] = text
            out["choices"] = [choice]
            return out

        pending: list[dict] = []

        def flush_pending() -> list[bytes]:
            """The chunks held back for seam matching, the withheld first-segment text
            prepended to the first of them."""
            first = pending[0]
            first_text = "".join(choice_text(c, chat) for c in first.get("choices") or ())
            return [sse(with_text(first, held + first_text, finish_of(first)))] + [sse(i) for i in pending[1:]]

        parts: list[bytes] = []
        if held and not matching:
            parts.append(sse(with_text(abort.obj, held, None)))
            held_sent = True
        else:
            held_sent = not held
        try:
            plan = build_continuation(
                path, body, abort.prompt_ids, abort.generated_ids or [], stream=True,
                max_model_len=self._max_model_len or await self._fetch_max_model_len(), cfg=cfg,
            )
            plan_error = ""
        except ValueError as exc:
            plan, plan_error = None, str(exc)
        if plan is None and not plan_error:
            # Budget spent: the abort hit the very last token.
            text = "" if held_sent else held
            parts.append(sse(with_text(abort.obj, text, "length") | {cfg.continued_field: 0}))
            await self._finish_stream(client, parts, base, merged_usage(prompt_tokens, generated, None),
                                      wants_usage, 0)
            self._account("continue", "budget_spent", request, depth, generated=generated)
            return
        cont_resp = None
        attempts = 0
        error = plan_error
        if plan is not None:
            headers = self._gateway_headers(request, model=body.get("model"), depth=depth + 1)
            headers["Content-Type"] = "application/json"
            cont_resp, attempts, error = await self._gateway_request(
                "POST", plan.path, json.dumps(plan.body).encode("utf-8"), headers
            )
            if cont_resp is not None and cont_resp.status != 200:
                try:
                    detail = (await cont_resp.read())[:300].decode("utf-8", "replace")
                finally:
                    cont_resp.release()
                error = f"gateway HTTP {cont_resp.status}: {detail}"
                cont_resp = None
        if cont_resp is None:
            text = "" if held_sent else held
            parts.append(sse(with_text(abort.obj, text, "abort")))
            await self._finish_stream(client, parts, base, merged_usage(prompt_tokens, generated, None),
                                      wants_usage, None)
            self._account("failed", "continuation_unavailable", request, depth, generated=generated,
                          attempts=attempts, error=error)
            return
        target = _target(cont_resp)
        cont_usage: dict | None = None
        cont_finish: str | None = None
        segments = 1
        gap_s: float | None = None
        pending_text = ""
        consumed = 0
        stopped_at_seam = False
        cont_failed = False
        error = ""
        buf = b""
        try:
            if parts:
                await client.write(b"".join(parts))
                parts = []
            try:
                async for data in cont_resp.content.iter_any():
                    added.start()
                    buf += data
                    events, buf = split_events(buf)
                    out: list[bytes] = []
                    for event in events:
                        obj = event_obj(event)
                        if obj is None:
                            continue  # [DONE], comments (a nested sidecar's), junk
                        if "error" in obj and not obj.get("choices"):
                            cont_failed, error = True, f"continuation error event: {str(obj.get('error'))[:200]}"
                            continue
                        if plan.as_chat:
                            obj = completion_chunk_as_chat(obj, stream=True)
                        if is_usage_only(obj):
                            cont_usage = obj["usage"]
                            continue
                        nested = obj.pop(cfg.continued_field, None)
                        if isinstance(nested, int) and nested > 0:
                            segments = 1 + nested
                        self._scrub(obj)
                        obj.update({k: v for k, v in base.items() if v is not None})
                        meaningful = False
                        for choice in obj.get("choices") or ():
                            delta = choice.get("delta")
                            if chat and isinstance(delta, dict):
                                delta.pop("role", None)
                                if delta.get("content") or delta.get("reasoning_content"):
                                    meaningful = True
                            elif choice_text(choice, chat):
                                meaningful = True
                            if choice.get("finish_reason"):
                                meaningful = True
                        if isinstance(obj.get("usage"), dict):
                            obj["usage"] = merged_usage(prompt_tokens, generated, obj["usage"])
                            meaningful = True
                        if not meaningful:
                            continue
                        text = "".join(choice_text(c, chat) for c in obj.get("choices") or ())
                        if gap_s is None:
                            gap_s = time.monotonic() - t_abort
                        finish = finish_of(obj)
                        if finish:
                            cont_finish = finish
                            if finish != "abort":
                                obj[cfg.continued_field] = segments
                        if not matching:
                            out.append(sse(obj))
                            continue
                        pending.append(obj)
                        pending_text += text
                        if text:
                            consumed += 1
                        cut = find_seam_stop(tail, pending_text, stops, include)
                        if cut is not None:
                            combined = tail + pending_text
                            final = with_text(pending[0], combined[max(emitted_in_tail, 0):max(cut, emitted_in_tail)],
                                              "stop")
                            final[cfg.continued_field] = segments
                            out.append(sse(final))
                            cont_finish, stopped_at_seam, matching = "stop", True, False
                            break
                        if len(pending_text) >= window or finish:
                            out.extend(flush_pending())
                            pending, matching, held_sent = [], False, True
                    if out:
                        await client.write(b"".join(out))
                    added.stop()
                    if stopped_at_seam:
                        cont_resp.close()  # the downstream engine aborts the rest
                        break
            except ConnectionResetError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                cont_failed, error = True, f"continuation stream: {type(exc).__name__}: {exc}"
            added.stop()
            if matching and pending:
                # The continuation ended inside the window without completing a seam stop.
                await client.write(b"".join(flush_pending()))
                held_sent = True
        except ConnectionResetError:
            cont_resp.close()
            self._account("passthrough_abort", "client_gone", request, depth, generated=generated, target=target)
            return
        finally:
            cont_resp.release()
        if stopped_at_seam:
            self.metrics.event("stop_at_seam")
            cont_usage = {"completion_tokens": consumed}
        if gap_s is not None:
            self.metrics.gap["stream"].observe(gap_s)
        tail_parts: list[bytes] = []
        if cont_failed or cont_finish is None or cont_finish == "abort":
            outcome, reason = "failed", ("continuation_aborted" if cont_finish == "abort" else "continuation_broken")
            if cont_finish is None:
                # No finish chunk reached the client: close the stream as aborted.
                tail_parts.append(sse(with_text(abort.obj, "" if held_sent else held, "abort")))
        else:
            outcome, reason = "continue", "abort_sleep"
        await self._finish_stream(client, tail_parts, base, merged_usage(prompt_tokens, generated, cont_usage),
                                  wants_usage, segments if outcome == "continue" else None)
        self._account(outcome, reason, request, depth, generated=generated, attempts=attempts, target=target,
                      gap_ms=_ms(gap_s), segments=segments, error=error, stop_at_seam=stopped_at_seam)

    async def _finish_stream(self, client: web.StreamResponse, parts: list[bytes], base: dict, usage: dict,
                             wants_usage: bool, segments: int | None) -> None:
        if wants_usage:
            parts.append(sse(dict(base, choices=[], usage=usage)))
        if segments:
            parts.append(f": {self.cfg.continued_header}: {segments}\n\n".encode())
        parts.append(SSE_DONE)
        try:
            await client.write(b"".join(parts))
            await client.write_eof()
        except ConnectionResetError:
            pass

    # -------------------------------------------------------- non-streaming path

    async def _non_stream(
        self, request: web.Request, path: str, raw: bytes, body: dict, resp: aiohttp.ClientResponse,
        depth: int, nc_reason: str | None, added: AddedTime, sleeps: int,
    ) -> web.StreamResponse:
        """Relay time: the whole body received -> the response handed back to aiohttp
        (which sends it after the handler returns; that socket write is not timed),
        minus the wait for a continuation through the gateway."""
        cfg = self.cfg
        try:
            payload = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if self.state.active:
                return await self._retry(request, raw, depth, "local_broken_sleeping", added)
            return _error(502, f"upstream failed: {exc}", "BadGateway")
        finally:
            resp.release()
        added.start()
        headers = response_headers(resp.headers)
        obj = None
        if _ABORT_MARK in payload:
            try:
                obj = json.loads(payload)
            except ValueError:
                obj = None
        if not isinstance(obj, dict) or not has_abort(obj):
            return web.Response(body=payload, status=resp.status, headers=headers)
        chat = path == cfg.chat_path
        info = self._abort_info(obj, chat, stream=False)
        # Non-streaming: nothing reached the client yet, so resending the original request
        # is always exact. A continuation only saves the tokens already generated, so it
        # is used when possible (continuable, token ids present); otherwise pure retry.
        continuable = nc_reason is None and bool(info.generated_ids) and info.prompt_ids is not None
        action, reason = self._abort_decision(request, depth, nc_reason, info, sent_any=continuable,
                                              sleeps=sleeps)
        if action == "retry" and nc_reason is not None:
            reason = f"abort_non_continuable_{nc_reason}"
        if action == "retry":
            return await self._retry(request, raw, depth, reason, added)
        if action != "continue":
            self._account(action, reason, request, depth)
            return web.Response(body=payload, status=resp.status, headers=headers)
        request[_REISSUE_PENDING] = "passthrough_abort"
        generated = len(info.generated_ids or ())
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = len(info.prompt_ids or ())
        first_text = choice_text((obj.get("choices") or [{}])[0], chat)
        merged = _strip_ids(obj, cfg, body)
        try:
            plan = build_continuation(
                path, body, info.prompt_ids, info.generated_ids or [], stream=False,
                max_model_len=self._max_model_len or await self._fetch_max_model_len(), cfg=cfg,
            )
            error = ""
        except ValueError as exc:
            plan, error = None, str(exc)
        t_abort = time.monotonic()
        if plan is None and not error:
            _set_choice(merged, chat, first_text, "length", None)
            merged["usage"] = merged_usage(prompt_tokens, generated, None)
            self._account("continue", "budget_spent", request, depth, generated=generated)
            return web.json_response(merged, headers={cfg.continued_header: "0"})
        second = None
        attempts = 0
        target = ""
        nested = 0
        if plan is not None:
            gw_headers = self._gateway_headers(request, model=body.get("model"), depth=depth + 1)
            gw_headers["Content-Type"] = "application/json"
            added.stop()  # the continuation through the gateway is upstream time
            cont_resp, attempts, error = await self._gateway_request(
                "POST", plan.path, json.dumps(plan.body).encode("utf-8"), gw_headers
            )
            if cont_resp is None:
                added.start()
            else:
                try:
                    cont_payload = await cont_resp.read()
                    target = _target(cont_resp)
                    nested = _int_header(cont_resp.headers.get(cfg.continued_header))
                    if cont_resp.status != 200:
                        error = f"gateway HTTP {cont_resp.status}: {cont_payload[:300].decode('utf-8', 'replace')}"
                    else:
                        second = json.loads(cont_payload)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                    error = f"continuation: {type(exc).__name__}: {exc}"
                finally:
                    cont_resp.release()
                    added.start()
        second_choice = ((second or {}).get("choices") or [{}])[0] if isinstance(second, dict) else {}
        finish = second_choice.get("finish_reason") if isinstance(second_choice, dict) else None
        if second is None or finish in (None, "abort"):
            self._account("failed", "continuation_aborted" if finish == "abort" else "continuation_unavailable",
                          request, depth, generated=generated, attempts=attempts, target=target, error=error)
            return web.Response(body=payload, status=resp.status, headers=headers)
        gap_s = time.monotonic() - t_abort
        self.metrics.gap["nonstream"].observe(gap_s)
        second_text = second_choice.get("text") if plan.as_chat or not chat else None
        if second_text is None:
            second_text = choice_text(second_choice, chat)
        text = first_text + (second_text or "")
        stop_reason = second_choice.get("stop_reason")
        stop_at_seam = False
        stops = stop_strings(body)
        if stops:
            window = max(len(s) for s in stops) - 1
            tail_start = max(0, len(first_text) - window)
            cut = find_seam_stop(first_text[tail_start:], second_text or "", stops,
                                 bool(body.get("include_stop_str_in_output")))
            if cut is not None:
                text = first_text[:tail_start] + (first_text[tail_start:] + (second_text or ""))[:cut]
                finish, stop_reason, stop_at_seam = "stop", None, True
                self.metrics.event("stop_at_seam")
        _set_choice(merged, chat, text, finish, stop_reason)
        cont_usage = second.get("usage") if isinstance(second.get("usage"), dict) else None
        merged["usage"] = merged_usage(prompt_tokens, generated, cont_usage)
        segments = 1 + nested
        self._account("continue", "abort_sleep", request, depth, generated=generated, attempts=attempts,
                      target=target, gap_ms=_ms(gap_s), segments=segments, stop_at_seam=stop_at_seam)
        return web.json_response(merged, headers={cfg.continued_header: str(segments)})

    # ------------------------------------------------------------------ logging

    def _scrub(self, obj: dict) -> None:
        """Drop the abort-only token-id fields of the fork from a chunk the client sees."""
        cfg = self.cfg
        obj.pop(cfg.prompt_ids_field, None)
        for choice in obj.get("choices") or ():
            if isinstance(choice, dict):
                choice.pop(cfg.generated_ids_field, None)
                choice.pop(cfg.prompt_ids_field, None)

    def _account(self, kind: str, reason: str, request: web.Request, depth: int, **extra: Any) -> None:
        request.pop(_REISSUE_PENDING, None)
        self.metrics.count(kind, reason)
        record = {"event": "tre_reissue", "ts": round(time.time(), 3), "model": self.cfg.model,
                  "pod": self.cfg.pod_name, "kind": kind, "reason": reason, "path": request.path, "depth": depth}
        for key, value in extra.items():
            if value is None or value == "" or value is False:
                continue
            record[key] = value
        _log(record)


def _ms(seconds: float | None) -> float | None:
    """Log field ``gap_ms`` (abort -> first continuation token, milliseconds). Logs written
    before 2026-09-30 carry the same millisecond value under the misleading name
    ``gap_s``; readers of old logs must treat ``gap_s`` as milliseconds too. The metric
    ``tre_reissue_gap_seconds`` is (and was) in seconds."""
    return None if seconds is None else round(seconds * 1000.0, 1)


def _strip_ids(obj: dict, cfg: Config, body: dict) -> dict:
    """The aborted response without the token-id fields the fork added for us (kept
    when the client asked for them with return_token_ids)."""
    merged = dict(obj)
    if body.get("return_token_ids"):
        return merged
    merged.pop(cfg.prompt_ids_field, None)
    choices = []
    for choice in obj.get("choices") or ():
        if isinstance(choice, dict):
            choice = {k: v for k, v in choice.items()
                      if k not in (cfg.prompt_ids_field, cfg.token_ids_field, cfg.generated_ids_field)}
        choices.append(choice)
    merged["choices"] = choices
    return merged


def _set_choice(obj: dict, chat: bool, text: str, finish: str | None, stop_reason: Any) -> None:
    choices = obj.get("choices") or [{"index": 0}]
    choice = dict(choices[0])
    if chat:
        message = dict(choice.get("message") or {"role": "assistant"})
        message["content"] = text
        choice["message"] = message
    else:
        choice["text"] = text
    choice["finish_reason"] = finish
    choice["stop_reason"] = stop_reason
    obj["choices"] = [choice] + list(choices[1:])


def _is_sleeping_payload(payload: bytes, error_type: str) -> bool:
    try:
        return is_sleeping_error(json.loads(payload), error_type)
    except ValueError:
        return False


def _body_model(raw: bytes) -> str | None:
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return body.get("model") if isinstance(body, dict) and isinstance(body.get("model"), str) else None


def _target(resp: aiohttp.ClientResponse) -> str:
    return resp.headers.get("target-pod") or resp.headers.get("target-pod-ip") or ""


def _error(status: int, message: str, err_type: str, *, headers: dict[str, str] | None = None,
           layer: str | None = None) -> web.Response:
    """An OpenAI-style error. ``layer`` names the hop that failed (``sidecar_upstream``:
    sidecar -> local vLLM) so clients / replayers can tell it from engine errors."""
    error: dict[str, Any] = {"message": f"{SIDECAR_ERROR_PREFIX} {message}", "type": err_type, "code": status}
    if layer:
        error["layer"] = layer
    return web.json_response({"error": error}, status=status, headers=headers)


def build_app(cfg: Config) -> web.Application:
    return ReissueSidecar(cfg).build_app()


def main() -> None:
    try:  # present in the vLLM image; optional
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    cfg = Config.from_env()
    raise_nofile_limit(cfg.nofile_target)
    _log({"event": "tre_reissue_start", "model": cfg.model, "pod": cfg.pod_name, "listen": cfg.listen_port,
          "upstream": cfg.upstream_url, "gateway": cfg.gateway_url, "enabled": cfg.enabled,
          "max_depth": cfg.max_depth, "retry_attempts": cfg.retry_attempts,
          "upstream_keepalive_s": cfg.upstream_keepalive_s,
          "upstream_server_keepalive_s": cfg.upstream_server_keepalive_s,
          "local_reconnect_attempts": cfg.local_reconnect_attempts,
          "local_reconnect_window_s": cfg.local_reconnect_window_s,
          "server_keepalive_s": cfg.server_keepalive_s,
          "gateway_upstream_idle_s": cfg.gateway_upstream_idle_s})
    web.run_app(build_app(cfg), **serve_kwargs(cfg))


if __name__ == "__main__":
    main()
