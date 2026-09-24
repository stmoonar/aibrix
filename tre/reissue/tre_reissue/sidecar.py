#!/usr/bin/env python3
"""TRE reissue sidecar: continue requests that a vLLM /sleep cut off.

Runs inside every model pod next to vLLM (which moves to 127.0.0.1:8001) and owns the
pod port 8000, so the Service, the gateway plugin (target-pod = podIP:8000), the
service-manager (/sleep, /wake_up, /is_sleeping), metric scrapers and probes all keep
talking to :8000 unchanged. Every path is proxied transparently with streaming kept.

What it adds (design: tre/docs/design/20260924-reissue-sidecar.md):

* ``POST /sleep`` is intercepted: the pod is marked sleeping BEFORE the call is forwarded
  (the custom vLLM image aborts in-flight requests inside /sleep, so the abort chunks
  reach us before /sleep returns); the returned abort snapshot is recorded. A 2xx
  ``/wake_up`` clears the mark.
* A generation request (``/v1/completions``, ``/v1/chat/completions``) whose local
  upstream ends with ``finish_reason == "abort"`` while the pod is sleeping and while the
  client is still connected is CONTINUED: a text-based continuation (original prompt +
  the text generated so far, token budget reduced by the tokens already generated) is
  sent through the tre-v2 gateway (never to a pod or to the model Service), carrying the
  original routing headers plus ``X-TRE-Reissue-Depth``. The continuation is spliced into
  the client stream: the first segment's abort chunk / usage / [DONE] are swallowed,
  chunk ids are rewritten to the original id, usage is merged, and a ``tre_reissue``
  extension field plus an SSE comment line describe what happened.
* A request stuck in vLLM behind the pause (it reached vLLM after the pause, so /sleep
  did not abort it and it would hang until wake) is detected after /sleep returns and
  reissued as a fresh request through the gateway.
* While the pod is sleeping, NEW generation requests are forwarded to the gateway
  instead of hanging on the paused engine (hop-limited by ``X-TRE-Forward-Hops``).
* ``GET /tre-reissue/metrics`` (Prometheus text) and ``GET /tre-reissue/state``; one
  JSON log line per reissue on stdout.

Only the standard library and aiohttp are used (both are in the vLLM image), so the
script ships through a ConfigMap and needs no image build. Python >= 3.10.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

COMPLETIONS_PATH = "/v1/completions"
CHAT_PATH = "/v1/chat/completions"
GENERATION_PATHS = frozenset({COMPLETIONS_PATH, CHAT_PATH})

DEPTH_HEADER = "X-TRE-Reissue-Depth"
HOPS_HEADER = "X-TRE-Forward-Hops"
ORIGIN_HEADER = "X-TRE-Reissue-Origin"
REQUEST_ID_HEADER = "X-Request-Id"

METRICS_PATH = "/tre-reissue/metrics"
STATE_PATH = "/tre-reissue/state"

#: Never copied between hops (RFC 7230 hop-by-hop plus what aiohttp recomputes).
_HOP_BY_HOP = frozenset(
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
#: Additionally dropped from requests the sidecar sends to the gateway: per-hop identity
#: and the plugin's routing decision for the ORIGINAL request.
_GATEWAY_DROP = _HOP_BY_HOP | frozenset(
    {
        "x-request-id",
        "target-pod",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-real-ip",
        DEPTH_HEADER.lower(),
        HOPS_HEADER.lower(),
        ORIGIN_HEADER.lower(),
    }
)
#: Response headers the sidecar's own server sets.
_RESPONSE_DROP = _HOP_BY_HOP | frozenset({"server", "date"})

#: Sampling parameters that mean the same on /v1/chat/completions and /v1/completions,
#: copied when a chat request is continued as a completion over the rendered prompt.
_SHARED_SAMPLING_KEYS = (
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
    "stream",
)

GAP_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


def _truthy(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    upstream_url: str = "http://127.0.0.1:8001"
    gateway_url: str = ""
    model: str = ""
    pod_name: str = ""
    #: False = pure transparent proxy (still tracks sleep; never reissues or forwards).
    enabled: bool = True
    forward_while_sleeping: bool = True
    max_depth: int = 3
    #: "render": chat continued as a completion over vLLM's own rendering of the chat
    #: prompt (exact context); "continue_final_message": chat continued as a chat request
    #: with the partial answer as the final assistant message.
    chat_mode: str = "render"
    connect_timeout_s: float = 6.0
    #: After /sleep returns, how long to wait before declaring a request that has not
    #: produced a byte (and was not in the abort snapshot) stuck behind the pause.
    stuck_grace_s: float = 0.2
    client_max_size: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        env = dict(os.environ if env is None else env)
        chat_mode = env.get("TRE_REISSUE_CHAT_MODE", "render").strip() or "render"
        if chat_mode not in {"render", "continue_final_message"}:
            raise ValueError(f"TRE_REISSUE_CHAT_MODE must be render|continue_final_message, got {chat_mode!r}")
        max_depth = int(env.get("TRE_REISSUE_MAX_DEPTH", "3"))
        if max_depth < 0:
            raise ValueError("TRE_REISSUE_MAX_DEPTH must be >= 0")
        return cls(
            listen_host=env.get("TRE_REISSUE_LISTEN_HOST", "0.0.0.0"),
            listen_port=int(env.get("TRE_REISSUE_LISTEN_PORT", "8000")),
            upstream_url=env.get("TRE_REISSUE_UPSTREAM", "http://127.0.0.1:8001").rstrip("/"),
            gateway_url=env.get("TRE_REISSUE_GATEWAY_URL", "").rstrip("/"),
            model=env.get("TRE_REISSUE_MODEL", ""),
            pod_name=env.get("POD_NAME", env.get("HOSTNAME", "")),
            enabled=_truthy(env.get("TRE_REISSUE_ENABLED"), True),
            forward_while_sleeping=_truthy(env.get("TRE_REISSUE_FORWARD_WHILE_SLEEPING"), True),
            max_depth=max_depth,
            chat_mode=chat_mode,
            connect_timeout_s=float(env.get("TRE_REISSUE_CONNECT_TIMEOUT_S", "6")),
            stuck_grace_s=float(env.get("TRE_REISSUE_STUCK_GRACE_S", "0.2")),
        )


# --------------------------------------------------------------------------- metrics


class Metrics:
    def __init__(self, model: str) -> None:
        self.model = model
        self.reissue: dict[tuple[str, str], int] = {}
        self.forward: dict[str, int] = {}
        self.gap_buckets = [0] * len(GAP_BUCKETS_S)
        self.gap_sum = 0.0
        self.gap_count = 0

    def count_reissue(self, kind: str, outcome: str) -> None:
        key = (kind, outcome)
        self.reissue[key] = self.reissue.get(key, 0) + 1

    def count_forward(self, outcome: str) -> None:
        self.forward[outcome] = self.forward.get(outcome, 0) + 1

    def observe_gap(self, seconds: float) -> None:
        self.gap_sum += seconds
        self.gap_count += 1
        for index, bound in enumerate(GAP_BUCKETS_S):
            if seconds <= bound:
                self.gap_buckets[index] += 1

    def render(self, state: "SleepState") -> str:
        model = _label(self.model)
        lines = [
            "# HELP tre_reissue_total Aborted-by-sleep requests handled by the reissue sidecar.",
            "# TYPE tre_reissue_total counter",
        ]
        for (kind, outcome), value in sorted(self.reissue.items()):
            lines.append(f'tre_reissue_total{{model="{model}",kind="{kind}",outcome="{outcome}"}} {value}')
        lines += [
            "# HELP tre_reissue_gap_seconds Abort to first continuation token.",
            "# TYPE tre_reissue_gap_seconds histogram",
        ]
        for bound, value in zip(GAP_BUCKETS_S, self.gap_buckets):
            lines.append(f'tre_reissue_gap_seconds_bucket{{model="{model}",le="{bound}"}} {value}')
        lines.append(f'tre_reissue_gap_seconds_bucket{{model="{model}",le="+Inf"}} {self.gap_count}')
        lines.append(f'tre_reissue_gap_seconds_sum{{model="{model}"}} {self.gap_sum:.6f}')
        lines.append(f'tre_reissue_gap_seconds_count{{model="{model}"}} {self.gap_count}')
        lines += [
            "# HELP tre_reissue_sleep_forward_total New requests forwarded to the gateway while sleeping.",
            "# TYPE tre_reissue_sleep_forward_total counter",
        ]
        for outcome, value in sorted(self.forward.items()):
            lines.append(f'tre_reissue_sleep_forward_total{{model="{model}",outcome="{outcome}"}} {value}')
        lines += [
            "# HELP tre_reissue_sleeping 1 while the local engine is (going to) sleep.",
            "# TYPE tre_reissue_sleeping gauge",
            f'tre_reissue_sleeping{{model="{model}"}} {1 if state.active else 0}',
            "# HELP tre_reissue_local_inflight Requests in flight to the local engine.",
            "# TYPE tre_reissue_local_inflight gauge",
            f'tre_reissue_local_inflight{{model="{model}"}} {len(state.inflight)}',
        ]
        return "\n".join(lines) + "\n"


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# ----------------------------------------------------------------------------- state


class Inflight:
    """One request in flight to the local engine."""

    __slots__ = ("rid", "stream", "got_data", "stuck", "task", "resp")

    def __init__(self, rid: str, stream: bool) -> None:
        self.rid = rid
        self.stream = stream
        self.got_data = False
        self.stuck = False
        self.task: asyncio.Future | None = None
        self.resp: aiohttp.ClientResponse | None = None


class SleepState:
    def __init__(self) -> None:
        self.sleeping = False
        self.pending = 0
        self.epoch = 0
        self.last_sleep: dict[str, Any] | None = None
        self.last_wake_ts: float | None = None
        self.inflight: dict[int, Inflight] = {}

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
    parts = []
    for line in event.split(b"\n"):
        line = line.rstrip(b"\r")
        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            parts.append(value)
    if not parts:
        return None
    return b"\n".join(parts)


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def is_usage_only(obj: dict) -> bool:
    return not obj.get("choices") and isinstance(obj.get("usage"), dict)


def choice_text(choice: dict, chat: bool) -> str:
    if chat:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            return content if isinstance(content, str) else ""
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            return content if isinstance(content, str) else ""
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


class Segment:
    """What the client has been sent from the local engine so far."""

    __slots__ = ("chat", "parts", "id", "created", "object", "model", "usage", "finish", "abort_obj", "chunks")

    def __init__(self, chat: bool) -> None:
        self.chat = chat
        self.parts: list[str] = []
        self.id: str | None = None
        self.created: int | None = None
        self.object: str | None = None
        self.model: str | None = None
        self.usage: dict | None = None
        self.finish: str | None = None
        self.abort_obj: dict | None = None
        self.chunks = 0

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def observe(self, obj: dict) -> None:
        if self.id is None and obj.get("id"):
            self.id = obj.get("id")
            self.created = obj.get("created")
            self.object = obj.get("object")
            self.model = obj.get("model")
        for choice in obj.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            text = choice_text(choice, self.chat)
            if text:
                self.parts.append(text)
            if choice.get("finish_reason"):
                self.finish = choice.get("finish_reason")
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self.usage = usage
        self.chunks += 1

    @property
    def completion_tokens(self) -> int:
        if self.usage and isinstance(self.usage.get("completion_tokens"), int):
            return int(self.usage["completion_tokens"])
        # No usage (should not happen: the sidecar forces include_usage). Fall back to the
        # number of content chunks, which is exact for vLLM's one-token-per-chunk stream.
        return len(self.parts)

    @property
    def prompt_tokens(self) -> int | None:
        if self.usage and isinstance(self.usage.get("prompt_tokens"), int):
            return int(self.usage["prompt_tokens"])
        return None


def has_abort(obj: dict) -> bool:
    return any(isinstance(c, dict) and c.get("finish_reason") == "abort" for c in obj.get("choices") or ())


def strip_finish(obj: dict, chat: bool) -> dict | None:
    """The abort chunk minus its abort: its text (vLLM flushes held-back detokenizer text
    into the final chunk) as an ordinary chunk, or None when it carries no text."""
    choices = []
    for choice in obj.get("choices") or ():
        if not isinstance(choice, dict) or not choice_text(choice, chat):
            continue
        copy = dict(choice)
        copy["finish_reason"] = None
        if "stop_reason" in copy:
            copy["stop_reason"] = None
        choices.append(copy)
    if not choices:
        return None
    out = dict(obj)
    out["choices"] = choices
    if "usage" in out:
        out["usage"] = None
    return out


def blank_abort(obj: dict | None, chat: bool, *, seg: Segment) -> dict:
    """An abort chunk with no text (for passing a failed reissue through)."""
    if obj is None:
        obj = {
            "id": seg.id,
            "object": seg.object or ("chat.completion.chunk" if chat else "text_completion"),
            "created": seg.created if seg.created is not None else int(time.time()),
            "model": seg.model,
            "choices": [{"index": 0, ("delta" if chat else "text"): ({} if chat else ""), "finish_reason": "abort"}],
        }
    out = dict(obj)
    choices = []
    for choice in obj.get("choices") or ():
        copy = dict(choice)
        if chat:
            copy["delta"] = {"content": ""} if isinstance(copy.get("delta"), dict) else copy.get("delta")
        else:
            copy["text"] = ""
        copy["finish_reason"] = "abort"
        choices.append(copy)
    out["choices"] = choices
    if "usage" in out:
        out["usage"] = None
    return out


def merge_usage(first: dict | None, second: dict | None, *, generated_before: int) -> dict | None:
    """Client-facing usage of a spliced request: the ORIGINAL prompt, all completion
    tokens of all segments."""
    if first is None and second is None:
        return None
    second = second or {}
    completion = int(generated_before) + int(second.get("completion_tokens") or 0)
    prompt = None
    if first is not None and isinstance(first.get("prompt_tokens"), int):
        prompt = int(first["prompt_tokens"])
    elif isinstance(second.get("prompt_tokens"), int):
        # Stuck request: nothing was generated locally, the continuation IS the original.
        prompt = int(second["prompt_tokens"]) - int(generated_before)
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion}
    usage["total_tokens"] = (prompt or 0) + completion
    return usage


def remaining_budget(body: dict, generated: int, keys: tuple[str, ...]) -> dict[str, int] | None:
    """Reduced token limits, or None when the budget is exhausted."""
    updates: dict[str, int] = {}
    for key in keys:
        value = body.get(key)
        if value is None:
            continue
        left = int(value) - int(generated)
        if left <= 0:
            return None
        updates[key] = left
    return updates


def _usage_opts(body: dict) -> dict:
    opts = body.get("stream_options")
    opts = dict(opts) if isinstance(opts, dict) else {}
    opts["include_usage"] = True
    return opts


def completion_continuation(body: dict, text: str, generated: int) -> dict | None:
    budget = remaining_budget(body, generated, ("max_tokens",))
    if budget is None:
        return None
    out = dict(body)
    out.update(budget)
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = prompt[0]
    out["prompt"] = prompt + text
    out["echo"] = False
    if out.get("min_tokens"):
        out["min_tokens"] = max(0, int(out["min_tokens"]) - int(generated))
    if out.get("stream"):
        out["stream_options"] = _usage_opts(body)
    return out


def chat_cfm_continuation(body: dict, text: str, generated: int) -> dict | None:
    """Chat continued as chat: the partial answer becomes the final assistant message and
    vLLM continues it (continue_final_message=true, add_generation_prompt=false).
    Caveat: templates that rewrite assistant history (DeepSeek-R1 drops everything up to
    ``</think>``) break this; ``render`` mode is the default for that reason."""
    budget = remaining_budget(body, generated, ("max_tokens", "max_completion_tokens"))
    if budget is None:
        return None
    out = dict(body)
    out.update(budget)
    messages = [dict(m) if isinstance(m, dict) else m for m in body.get("messages") or []]
    if text:
        last = messages[-1] if messages else None
        if (
            body.get("continue_final_message")
            and isinstance(last, dict)
            and last.get("role") == "assistant"
            and isinstance(last.get("content"), str)
        ):
            last["content"] = last["content"] + text
        else:
            messages.append({"role": "assistant", "content": text})
        out["continue_final_message"] = True
        out["add_generation_prompt"] = False
    out["messages"] = messages
    if out.get("min_tokens"):
        out["min_tokens"] = max(0, int(out["min_tokens"]) - int(generated))
    if out.get("stream"):
        out["stream_options"] = _usage_opts(body)
    return out


def chat_render_continuation(
    body: dict, rendered_prompt: str, text: str, generated: int, *, max_model_len: int | None, prompt_len: int | None
) -> dict | None:
    """Chat continued as a /v1/completions request over vLLM's own rendering of the
    original chat prompt plus the partial answer (special tokens are text in the rendered
    prompt, so add_special_tokens=false keeps them single)."""
    limit = body.get("max_completion_tokens")
    if limit is None:
        limit = body.get("max_tokens")
    if limit is None and max_model_len is not None and prompt_len is not None:
        limit = int(max_model_len) - int(prompt_len)
    out: dict[str, Any] = {"model": body.get("model"), "prompt": rendered_prompt + text, "add_special_tokens": False}
    for key in _SHARED_SAMPLING_KEYS:
        if key in body:
            out[key] = body[key]
    if limit is not None:
        left = int(limit) - int(generated)
        if left <= 0:
            return None
        out["max_tokens"] = left
    if out.get("min_tokens"):
        out["min_tokens"] = max(0, int(out["min_tokens"]) - int(generated))
    if out.get("stream"):
        out["stream_options"] = _usage_opts(body)
    return out


def completion_chunk_as_chat(obj: dict) -> dict:
    """A /v1/completions stream chunk (or response) reshaped as a chat chunk."""
    out = {k: v for k, v in obj.items() if k != "choices"}
    choices = []
    for choice in obj.get("choices") or ():
        entry = {
            "index": choice.get("index", 0),
            "delta": {"content": choice.get("text") or ""},
            "logprobs": None,
            "finish_reason": choice.get("finish_reason"),
        }
        if "stop_reason" in choice:
            entry["stop_reason"] = choice.get("stop_reason")
        choices.append(entry)
    out["choices"] = choices
    out["object"] = "chat.completion.chunk"
    return out


def reissuable(path: str, body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    try:
        if int(body.get("n") or 1) != 1 or int(body.get("best_of") or 1) != 1:
            return False
    except (TypeError, ValueError):
        return False
    if path == COMPLETIONS_PATH:
        if body.get("echo"):
            return False
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            return len(prompt) == 1 and isinstance(prompt[0], str)
        return isinstance(prompt, str)
    if path == CHAT_PATH:
        return isinstance(body.get("messages"), list) and bool(body.get("messages"))
    return False


def forward_headers(headers, drop: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in headers.items():
        lower = name.lower()
        if lower in drop or lower.startswith("x-envoy-"):
            continue
        out[name] = value
    return out


def response_headers(headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in _RESPONSE_DROP:
            continue
        out[name] = value
    return out


# ---------------------------------------------------------------------------- sidecar


class ReissueSidecar:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = SleepState()
        self.metrics = Metrics(cfg.model)
        self.local: aiohttp.ClientSession | None = None
        self.gateway: aiohttp.ClientSession | None = None
        self._probe_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- lifecycle

    async def on_startup(self, app: web.Application) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.cfg.connect_timeout_s)
        self.local = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0), timeout=timeout, auto_decompress=False
        )
        self.gateway = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0), timeout=timeout, auto_decompress=False
        )
        self._probe_task = asyncio.ensure_future(self._probe_initial_sleep_state())

    async def on_cleanup(self, app: web.Application) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
        for session in (self.local, self.gateway):
            if session is not None:
                await session.close()

    async def _probe_initial_sleep_state(self) -> None:
        """A restarted sidecar may find vLLM already asleep."""
        while self.state.epoch == 0:
            try:
                async with self.local.get(self.cfg.upstream_url + "/is_sleeping") as resp:
                    if resp.status == 200:
                        payload = await resp.json(content_type=None)
                        if self.state.epoch == 0 and isinstance(payload, dict):
                            self.state.sleeping = bool(payload.get("is_sleeping"))
                        return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - engine not up yet
                pass
            await asyncio.sleep(2.0)

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=self.cfg.client_max_size)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app

    # ------------------------------------------------------------------ routing

    async def handle(self, request: web.Request) -> web.StreamResponse:
        path = request.path
        if path == METRICS_PATH:
            return web.Response(text=self.metrics.render(self.state), content_type="text/plain")
        if path == STATE_PATH:
            return web.json_response(self._state_view())
        body = await request.read()
        if request.method == "POST" and path == "/sleep":
            return await self._handle_sleep(request, body)
        if request.method == "POST" and path == "/wake_up":
            return await self._handle_wake(request, body)
        if request.method == "POST" and path in GENERATION_PATHS and self.cfg.enabled:
            if self.state.active and self.cfg.forward_while_sleeping and self.cfg.gateway_url:
                return await self._forward_while_sleeping(request, body)
            return await self._handle_generation(request, path, body)
        return await self._proxy(request, body, self.local, self.cfg.upstream_url)

    def _state_view(self) -> dict:
        return {
            "model": self.cfg.model,
            "pod": self.cfg.pod_name,
            "enabled": self.cfg.enabled,
            "sleeping": self.state.sleeping,
            "sleep_pending": self.state.pending,
            "epoch": self.state.epoch,
            "last_sleep": self.state.last_sleep,
            "local_inflight": len(self.state.inflight),
            "gateway_url": self.cfg.gateway_url,
            "max_depth": self.cfg.max_depth,
        }

    # -------------------------------------------------------------- plain proxy

    async def _proxy(
        self,
        request: web.Request,
        body: bytes,
        session: aiohttp.ClientSession,
        base: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> web.StreamResponse:
        url = base + request.path_qs
        if headers is None:
            headers = forward_headers(request.headers, _HOP_BY_HOP)
        try:
            resp = await session.request(
                request.method, url, data=body if body else None, headers=headers, allow_redirects=False
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: upstream unavailable: {exc}", "type": "BadGateway"}},
                status=502,
            )
        return await self._relay(request, resp)

    async def _relay(self, request: web.Request, resp: aiohttp.ClientResponse) -> web.StreamResponse:
        try:
            out = web.StreamResponse(status=resp.status, reason=resp.reason, headers=response_headers(resp.headers))
            if resp.content_length is not None and "chunked" not in resp.headers.get("Transfer-Encoding", "").lower():
                out.content_length = resp.content_length
            try:
                await out.prepare(request)
            except ConnectionResetError:
                resp.close()
                return out
            try:
                async for data in resp.content.iter_any():
                    await out.write(data)
            except ConnectionResetError:
                # The client went away (aiohttp's ClientConnectionResetError is also a
                # ClientError, so this clause must come first): drop the upstream so the
                # engine aborts the request.
                resp.close()
                return out
            except (aiohttp.ClientError, asyncio.TimeoutError):
                resp.close()
                if request.transport is not None:
                    request.transport.close()
                return out
            await out.write_eof()
            return out
        finally:
            resp.release()

    # ------------------------------------------------------------- sleep / wake

    async def _handle_sleep(self, request: web.Request, body: bytes) -> web.StreamResponse:
        state = self.state
        state.pending += 1
        state.epoch += 1
        epoch = state.epoch
        started = time.time()
        try:
            resp = await self.local.post(
                self.cfg.upstream_url + request.path_qs,
                data=body if body else None,
                headers=forward_headers(request.headers, _HOP_BY_HOP),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            state.pending -= 1
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: upstream unavailable: {exc}", "type": "BadGateway"}},
                status=502,
            )
        try:
            payload = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            state.pending -= 1
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: /sleep failed: {exc}", "type": "BadGateway"}},
                status=502,
            )
        finally:
            resp.release()
        state.pending -= 1
        if 200 <= resp.status < 300:
            state.sleeping = True
            aborted = _parse_snapshot(payload)
            state.last_sleep = {
                "ts": started,
                "duration_s": round(time.time() - started, 3),
                "aborted": aborted,
                "local_inflight_at_return": len(state.inflight),
            }
            _log({"event": "tre_sleep", "model": self.cfg.model, "pod": self.cfg.pod_name,
                  "aborted": len(aborted) if isinstance(aborted, list) else None,
                  "duration_s": state.last_sleep["duration_s"]})
            if self.cfg.enabled:
                asyncio.get_running_loop().call_later(
                    self.cfg.stuck_grace_s, self._mark_stuck, epoch, aborted
                )
        return web.Response(body=payload, status=resp.status, headers=response_headers(resp.headers))

    def _mark_stuck(self, epoch: int, aborted: Any) -> None:
        """Requests that reached vLLM after the pause are not in the abort snapshot and
        would hang until wake: cut them loose so their handler reissues them."""
        if not self.state.active or self.state.epoch != epoch:
            return
        aborted_ids = [str(item.get("request_id", "")) for item in aborted if isinstance(item, dict)] if isinstance(aborted, list) else None
        for entry in list(self.state.inflight.values()):
            if entry.got_data or entry.stuck:
                continue
            if aborted_ids is None:
                if not entry.stream:
                    continue
            elif any(entry.rid in rid for rid in aborted_ids):
                continue
            entry.stuck = True
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
            elif entry.resp is not None:
                entry.resp.close()

    async def _handle_wake(self, request: web.Request, body: bytes) -> web.StreamResponse:
        try:
            resp = await self.local.post(
                self.cfg.upstream_url + request.path_qs,
                data=body if body else None,
                headers=forward_headers(request.headers, _HOP_BY_HOP),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: upstream unavailable: {exc}", "type": "BadGateway"}},
                status=502,
            )
        try:
            payload = await resp.read()
        finally:
            resp.release()
        if 200 <= resp.status < 300:
            self.state.sleeping = False
            self.state.epoch += 1
            self.state.last_wake_ts = time.time()
        return web.Response(body=payload, status=resp.status, headers=response_headers(resp.headers))

    # ------------------------------------------------------- sleeping: forward

    async def _forward_while_sleeping(self, request: web.Request, body: bytes) -> web.StreamResponse:
        hops = _int_header(request.headers.get(HOPS_HEADER))
        if hops >= self.cfg.max_depth:
            self.metrics.count_forward("hop_limit")
            return web.json_response(
                {"error": {"message": "tre-reissue sidecar: engine sleeping and forward hop limit reached",
                           "type": "ServiceUnavailable", "code": 503}},
                status=503,
            )
        headers = self._gateway_headers(request, body_model=_body_model(body))
        headers[HOPS_HEADER] = str(hops + 1)
        depth = request.headers.get(DEPTH_HEADER)
        if depth is not None:
            headers[DEPTH_HEADER] = depth
        try:
            resp = await self.gateway.post(self.cfg.gateway_url + request.path_qs, data=body, headers=headers)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            self.metrics.count_forward("error")
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: gateway unavailable: {exc}", "type": "BadGateway"}},
                status=502,
            )
        self.metrics.count_forward("ok" if resp.status == 200 else f"http_{resp.status}")
        return await self._relay(request, resp)

    def _gateway_headers(self, request: web.Request, *, body_model: str | None) -> dict[str, str]:
        headers = forward_headers(request.headers, _GATEWAY_DROP)
        if body_model and not any(name.lower() == "model" for name in headers):
            headers["model"] = body_model
        if self.cfg.pod_name:
            headers[ORIGIN_HEADER] = self.cfg.pod_name
        return headers

    # --------------------------------------------------------------- generation

    async def _handle_generation(self, request: web.Request, path: str, raw: bytes) -> web.StreamResponse:
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not reissuable(path, body):
            return await self._proxy(request, raw, self.local, self.cfg.upstream_url)
        stream = bool(body.get("stream"))
        wants_usage = bool(isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage"))
        send_body = raw
        if stream and not wants_usage:
            forced = dict(body)
            forced["stream_options"] = _usage_opts(body)
            send_body = json.dumps(forced).encode("utf-8")
        headers = forward_headers(request.headers, _HOP_BY_HOP)
        rid = request.headers.get(REQUEST_ID_HEADER)
        if not rid:
            rid = uuid.uuid4().hex
            headers[REQUEST_ID_HEADER] = rid
        depth = _int_header(request.headers.get(DEPTH_HEADER))
        entry = Inflight(rid, stream)
        key = id(entry)
        self.state.inflight[key] = entry
        try:
            task = asyncio.ensure_future(
                self.local.post(self.cfg.upstream_url + request.path_qs, data=send_body, headers=headers)
            )
            entry.task = task
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.cancel()
                raise
            entry.task = None
            if task.cancelled():
                if entry.stuck:
                    return await self._reissue_stuck(request, path, body, depth, stream, wants_usage, client=None)
                raise asyncio.CancelledError()
            try:
                resp = task.result()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                return web.json_response(
                    {"error": {"message": f"tre-reissue sidecar: upstream unavailable: {exc}", "type": "BadGateway"}},
                    status=502,
                )
            entry.resp = resp
            ctype = resp.headers.get("Content-Type", "")
            if not stream or resp.status != 200 or "text/event-stream" not in ctype:
                entry.got_data = True
                if stream or resp.status != 200 or "json" not in ctype:
                    return await self._relay(request, resp)
                return await self._non_stream(request, path, body, resp, depth, entry)
            return await self._stream(request, path, body, resp, depth, wants_usage, entry)
        finally:
            self.state.inflight.pop(key, None)

    def _abort_decision(self, request: web.Request, depth: int) -> str:
        if not self.state.active:
            return "abort_not_sleeping"
        if not self.cfg.gateway_url:
            return "no_gateway"
        if _client_gone(request):
            return "client_disconnected"
        if depth + 1 > self.cfg.max_depth:
            return "depth_limit"
        return "reissue"

    # ------------------------------------------------------------ streaming path

    async def _stream(
        self,
        request: web.Request,
        path: str,
        body: dict,
        resp: aiohttp.ClientResponse,
        depth: int,
        wants_usage: bool,
        entry: Inflight,
    ) -> web.StreamResponse:
        chat = path == CHAT_PATH
        seg = Segment(chat)
        client = web.StreamResponse(status=resp.status, reason=resp.reason, headers=response_headers(resp.headers))
        try:
            await client.prepare(request)
        except (ConnectionResetError, OSError):
            resp.close()
            return client
        reissue = False
        t_abort = 0.0
        buffer = b""
        upstream_failed = False
        try:
            try:
                async for data in resp.content.iter_any():
                    entry.got_data = True
                    buffer += data
                    events, buffer = split_events(buffer)
                    if not events:
                        continue
                    out: list[bytes] = []
                    for event in events:
                        payload = event_data(event)
                        if payload is None:
                            if not reissue:
                                out.append(event)
                            continue
                        if payload.strip() == b"[DONE]":
                            if not reissue:
                                out.append(event)
                            continue
                        try:
                            obj = json.loads(payload)
                        except ValueError:
                            out.append(event)
                            continue
                        if not isinstance(obj, dict):
                            out.append(event)
                            continue
                        seg.observe(obj)
                        if reissue:
                            continue  # the first segment's usage chunk: merged later
                        if has_abort(obj):
                            decision = self._abort_decision(request, depth)
                            if decision == "reissue":
                                reissue = True
                                t_abort = time.monotonic()
                                seg.abort_obj = obj
                                carried = strip_finish(obj, chat)
                                if carried is not None:
                                    out.append(sse(carried))
                                continue
                            self.metrics.count_reissue("abort", decision)
                            _log(self._log_record("abort", decision, path, True, depth, seg))
                        if is_usage_only(obj) and not wants_usage:
                            continue
                        out.append(event)
                    if out:
                        await client.write(b"".join(out))
            except ConnectionResetError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError):
                upstream_failed = True
            if buffer and not reissue and not upstream_failed and not entry.stuck:
                await client.write(buffer)
        except ConnectionResetError:
            resp.close()
            return client
        finally:
            resp.release()
        if entry.stuck and seg.chunks == 0:
            return await self._reissue_stuck(request, path, body, depth, True, wants_usage, client=client)
        if upstream_failed and not reissue:
            if request.transport is not None:
                request.transport.close()
            return client
        if reissue:
            await self._continue_stream(request, client, path, body, seg, depth, t_abort, wants_usage, kind="abort")
            return client
        try:
            await client.write_eof()
        except ConnectionResetError:
            pass
        return client

    async def _continue_stream(
        self,
        request: web.Request,
        client: web.StreamResponse,
        path: str,
        body: dict,
        seg: Segment,
        depth: int,
        t_abort: float,
        wants_usage: bool,
        *,
        kind: str,
    ) -> None:
        chat = path == CHAT_PATH
        next_depth = depth + 1
        generated = seg.completion_tokens if seg.chunks else 0
        text = seg.text
        target = ""
        gap_s: float | None = None
        cont_usage: dict | None = None
        nested: dict | None = None
        cont_finish: str | None = None
        as_chat = False
        outcome = "failed"
        error = ""
        cont_path, cont_body, as_chat = await self._continuation_request(path, body, text, generated, stream=True)
        if cont_body is None and cont_path is None:
            # Budget already spent: the abort hit the very last token. Close it as "length".
            outcome = "ok"
            final = blank_abort(seg.abort_obj, chat, seg=seg)
            for choice in final["choices"]:
                choice["finish_reason"] = "length"
            await self._finish_stream(client, seg, final_chunk=final, usage=seg.usage, wants_usage=wants_usage,
                                      ext=self._ext(1, 0.0, "", next_depth, kind, outcome))
            self._account(kind, outcome, path, True, depth, seg, gap_s=None, target="", generated=generated)
            return
        if cont_body is None:
            error = cont_path or "continuation unavailable"
        cont_resp: aiohttp.ClientResponse | None = None
        if cont_body is not None:
            headers = self._gateway_headers(request, body_model=body.get("model"))
            headers[DEPTH_HEADER] = str(next_depth)
            headers["Content-Type"] = "application/json"
            try:
                cont_resp = await self.gateway.post(
                    self.cfg.gateway_url + cont_path, data=json.dumps(cont_body).encode("utf-8"), headers=headers
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                error = f"gateway: {type(exc).__name__}: {exc}"
        if cont_resp is not None and cont_resp.status != 200:
            detail = (await cont_resp.read())[:300].decode("utf-8", "replace")
            cont_resp.release()
            error = f"gateway HTTP {cont_resp.status}: {detail}"
            cont_resp = None
        if cont_resp is None:
            await self._finish_stream(
                client, seg, final_chunk=blank_abort(seg.abort_obj, chat, seg=seg), usage=seg.usage,
                wants_usage=wants_usage, ext=self._ext(1, None, "", next_depth, kind, "failed", error=error),
            )
            self._account(kind, "failed", path, True, depth, seg, gap_s=None, target="", generated=generated, error=error)
            return
        target = cont_resp.headers.get("target-pod") or cont_resp.headers.get("target-pod-ip") or ""
        need_role = chat and seg.chunks == 0
        buffer = b""
        cont_failed = False
        try:
            try:
                async for data in cont_resp.content.iter_any():
                    buffer += data
                    events, buffer = split_events(buffer)
                    out: list[bytes] = []
                    for event in events:
                        payload = event_data(event)
                        if payload is None or payload.strip() == b"[DONE]":
                            continue
                        try:
                            obj = json.loads(payload)
                        except ValueError:
                            out.append(event)
                            continue
                        if not isinstance(obj, dict):
                            out.append(event)
                            continue
                        if as_chat:
                            obj = completion_chunk_as_chat(obj)
                        if is_usage_only(obj):
                            cont_usage = obj["usage"]
                            if isinstance(obj.get("tre_reissue"), dict):
                                nested = obj["tre_reissue"]
                            continue
                        if "error" in obj and not obj.get("choices"):
                            out.append(event)
                            continue
                        obj = self._rewrite_cont_chunk(obj, seg, chat, generated, keep_role=need_role and not as_chat)
                        if obj is None:
                            continue
                        if need_role and as_chat:
                            role = dict(obj)
                            role["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                "logprobs": None, "finish_reason": None}]
                            role.pop("usage", None)
                            out.append(sse(role))
                        need_role = False
                        for choice in obj.get("choices") or ():
                            if gap_s is None and (choice_text(choice, chat) or choice.get("finish_reason")):
                                gap_s = time.monotonic() - t_abort
                            if choice.get("finish_reason"):
                                cont_finish = choice.get("finish_reason")
                        out.append(sse(obj))
                    if out:
                        await client.write(b"".join(out))
            except ConnectionResetError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                cont_failed = True
                error = f"continuation stream: {type(exc).__name__}: {exc}"
        except ConnectionResetError:
            cont_resp.close()
            self._account(kind, "client_disconnected", path, True, depth, seg, gap_s=gap_s, target=target,
                          generated=generated)
            return
        finally:
            cont_resp.release()
        if cont_failed or cont_finish is None:
            outcome = "failed"
            error = error or "continuation ended without a finish_reason"
        elif cont_finish == "abort":
            outcome = "failed"
            error = "continuation aborted downstream"
        else:
            outcome = "ok"
        if gap_s is not None:
            self.metrics.observe_gap(gap_s)
        n = 1 + int((nested or {}).get("n") or 0)
        gap_ms = None if gap_s is None else gap_s * 1000.0 + float((nested or {}).get("gap_ms") or 0.0)
        usage = merge_usage(seg.usage if seg.chunks else None, cont_usage, generated_before=generated)
        final_chunk = None
        if cont_failed or cont_finish is None:
            final_chunk = blank_abort(seg.abort_obj, chat, seg=seg)
        await self._finish_stream(
            client, seg, final_chunk=final_chunk, usage=usage, wants_usage=wants_usage,
            ext=self._ext(n, gap_ms, (nested or {}).get("target") or target, next_depth, kind, outcome, error=error),
        )
        self._account(kind, outcome, path, True, depth, seg, gap_s=gap_s, target=target, generated=generated,
                      error=error)

    def _rewrite_cont_chunk(self, obj: dict, seg: Segment, chat: bool, generated: int, *, keep_role: bool) -> dict | None:
        if seg.id is not None:
            obj["id"] = seg.id
            if seg.created is not None:
                obj["created"] = seg.created
            if seg.object is not None:
                obj["object"] = seg.object
        if chat and not keep_role:
            meaningful = False
            for choice in obj.get("choices") or ():
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    delta.pop("role", None)
                    if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
                        meaningful = True
                if choice.get("finish_reason"):
                    meaningful = True
            if not meaningful:
                return None
        usage = obj.get("usage")
        if isinstance(usage, dict):
            obj["usage"] = merge_usage(seg.usage if seg.chunks else None, usage, generated_before=generated)
        return obj

    async def _finish_stream(
        self,
        client: web.StreamResponse,
        seg: Segment,
        *,
        final_chunk: dict | None,
        usage: dict | None,
        wants_usage: bool,
        ext: dict,
    ) -> None:
        parts: list[bytes] = []
        if final_chunk is not None:
            parts.append(sse(final_chunk))
        if wants_usage and usage is not None:
            parts.append(
                sse(
                    {
                        "id": seg.id,
                        "object": seg.object or ("chat.completion.chunk" if seg.chat else "text_completion"),
                        "created": seg.created if seg.created is not None else int(time.time()),
                        "model": seg.model,
                        "choices": [],
                        "usage": usage,
                        "tre_reissue": ext,
                    }
                )
            )
        parts.append(b": tre-reissue " + json.dumps(ext, separators=(",", ":")).encode("utf-8") + b"\n\n")
        parts.append(SSE_DONE)
        try:
            await client.write(b"".join(parts))
            await client.write_eof()
        except ConnectionResetError:
            pass

    @staticmethod
    def _ext(n: int, gap_ms: float | None, target: str, depth: int, kind: str, outcome: str, *, error: str = "") -> dict:
        ext: dict[str, Any] = {
            "n": n,
            "gap_ms": None if gap_ms is None else round(gap_ms, 1),
            "target": target,
            "depth": depth,
            "kind": kind,
            "outcome": outcome,
        }
        if error:
            ext["error"] = error[:300]
        return ext

    async def _continuation_request(
        self, path: str, body: dict, text: str, generated: int, *, stream: bool
    ) -> tuple[str | None, dict | None, bool]:
        """(path, body, completion-shaped-for-chat). (None, None, _) = budget spent;
        (reason, None, _) = cannot build."""
        body = dict(body)
        body["stream"] = stream
        if path == COMPLETIONS_PATH:
            cont = completion_continuation(body, text, generated)
            return (None, None, False) if cont is None else (COMPLETIONS_PATH, cont, False)
        if not text and not generated:
            cont = dict(body)
            if stream:
                cont["stream_options"] = _usage_opts(body)
            return CHAT_PATH, cont, False
        if self.cfg.chat_mode == "render":
            rendered = await self._render_chat_prompt(body)
            if rendered is not None:
                prompt_text, prompt_len, max_len = rendered
                cont = chat_render_continuation(
                    body, prompt_text, text, generated, max_model_len=max_len, prompt_len=prompt_len
                )
                return (None, None, True) if cont is None else (COMPLETIONS_PATH, cont, True)
        cont = chat_cfm_continuation(body, text, generated)
        return (None, None, False) if cont is None else (CHAT_PATH, cont, False)

    async def _render_chat_prompt(self, body: dict) -> tuple[str, int | None, int | None] | None:
        """vLLM's own rendering of the chat prompt, via the local API server's /tokenize +
        /detokenize (tokenizer-only: they work while the engine is paused/asleep)."""
        request: dict[str, Any] = {
            "model": body.get("model"),
            "messages": body.get("messages"),
            "add_generation_prompt": body.get("add_generation_prompt", True),
            "continue_final_message": body.get("continue_final_message", False),
            "add_special_tokens": body.get("add_special_tokens", False),
        }
        for key in ("chat_template", "chat_template_kwargs", "tools"):
            if body.get(key) is not None:
                request[key] = body[key]
        try:
            async with self.local.post(self.cfg.upstream_url + "/tokenize", json=request) as resp:
                if resp.status != 200:
                    return None
                tokenized = await resp.json(content_type=None)
            tokens = tokenized.get("tokens")
            if not isinstance(tokens, list):
                return None
            async with self.local.post(
                self.cfg.upstream_url + "/detokenize", json={"model": body.get("model"), "tokens": tokens}
            ) as resp:
                if resp.status != 200:
                    return None
                detok = await resp.json(content_type=None)
            prompt = detok.get("prompt")
            if not isinstance(prompt, str):
                return None
            return prompt, len(tokens), tokenized.get("max_model_len")
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, AttributeError):
            return None

    async def _reissue_stuck(
        self,
        request: web.Request,
        path: str,
        body: dict,
        depth: int,
        stream: bool,
        wants_usage: bool,
        *,
        client: web.StreamResponse | None,
    ) -> web.StreamResponse:
        decision = self._abort_decision(request, depth)
        seg = Segment(path == CHAT_PATH)
        if decision != "reissue":
            self.metrics.count_reissue("stuck", decision)
            _log(self._log_record("stuck", decision, path, stream, depth, seg))
            if client is not None:
                if request.transport is not None:
                    request.transport.close()
                return client
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: request stuck behind sleep ({decision})",
                           "type": "ServiceUnavailable", "code": 503}},
                status=503,
            )
        t0 = time.monotonic()
        if not stream:
            return await self._continue_non_stream(request, path, body, None, seg, depth, t0, kind="stuck")
        if client is None:
            client = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream; charset=utf-8"})
            await client.prepare(request)
        await self._continue_stream(request, client, path, body, seg, depth, t0, wants_usage, kind="stuck")
        return client

    # -------------------------------------------------------- non-streaming path

    async def _non_stream(
        self,
        request: web.Request,
        path: str,
        body: dict,
        resp: aiohttp.ClientResponse,
        depth: int,
        entry: Inflight,
    ) -> web.StreamResponse:
        try:
            payload = await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return web.json_response(
                {"error": {"message": f"tre-reissue sidecar: upstream failed: {exc}", "type": "BadGateway"}},
                status=502,
            )
        finally:
            resp.release()
        headers = response_headers(resp.headers)
        try:
            obj = json.loads(payload)
        except ValueError:
            obj = None
        if not isinstance(obj, dict) or not has_abort(obj):
            return web.Response(body=payload, status=resp.status, headers=headers)
        chat = path == CHAT_PATH
        seg = Segment(chat)
        seg.observe(obj)
        seg.abort_obj = obj
        decision = self._abort_decision(request, depth)
        if decision != "reissue":
            self.metrics.count_reissue("abort", decision)
            _log(self._log_record("abort", decision, path, False, depth, seg))
            return web.Response(body=payload, status=resp.status, headers=headers)
        return await self._continue_non_stream(request, path, body, obj, seg, depth, time.monotonic(), kind="abort")

    async def _continue_non_stream(
        self,
        request: web.Request,
        path: str,
        body: dict,
        first: dict | None,
        seg: Segment,
        depth: int,
        t_abort: float,
        *,
        kind: str,
    ) -> web.StreamResponse:
        chat = path == CHAT_PATH
        next_depth = depth + 1
        generated = seg.completion_tokens if first is not None else 0
        text = seg.text
        cont_path, cont_body, as_chat = await self._continuation_request(path, body, text, generated, stream=False)
        error = ""
        merged: dict | None = None
        target = ""
        gap_s = None
        nested: dict | None = None
        if cont_body is None and cont_path is None and first is not None:
            merged = dict(first)
            merged["choices"] = [dict(c, finish_reason="length") for c in first.get("choices") or ()]
            outcome = "ok"
        else:
            outcome = "failed"
            if cont_body is None:
                error = cont_path or "continuation unavailable"
            else:
                headers = self._gateway_headers(request, body_model=body.get("model"))
                headers[DEPTH_HEADER] = str(next_depth)
                headers["Content-Type"] = "application/json"
                try:
                    async with self.gateway.post(
                        self.cfg.gateway_url + cont_path, data=json.dumps(cont_body).encode("utf-8"), headers=headers
                    ) as cont_resp:
                        cont_payload = await cont_resp.read()
                        target = cont_resp.headers.get("target-pod") or cont_resp.headers.get("target-pod-ip") or ""
                        status = cont_resp.status
                    gap_s = time.monotonic() - t_abort
                    if status != 200:
                        error = f"gateway HTTP {status}: {cont_payload[:300].decode('utf-8', 'replace')}"
                    else:
                        second = json.loads(cont_payload)
                        merged, cont_finish, nested = _merge_non_stream(first, second, chat=chat, as_chat=as_chat,
                                                                        generated=generated)
                        outcome = "failed" if cont_finish in (None, "abort") else "ok"
                        if outcome == "failed":
                            error = "continuation aborted downstream"
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
                    error = f"gateway: {type(exc).__name__}: {exc}"
        if merged is None:
            if first is None:
                self._account(kind, "failed", path, False, depth, seg, gap_s=gap_s, target=target, generated=0,
                              error=error)
                return web.json_response(
                    {"error": {"message": f"tre-reissue sidecar: reissue failed: {error}", "type": "BadGateway"}},
                    status=502,
                )
            merged = dict(first)
        if gap_s is not None and outcome == "ok":
            self.metrics.observe_gap(gap_s)
        n = 1 + int((nested or {}).get("n") or 0)
        gap_ms = None if gap_s is None else gap_s * 1000.0 + float((nested or {}).get("gap_ms") or 0.0)
        merged["tre_reissue"] = self._ext(n, gap_ms, (nested or {}).get("target") or target, next_depth, kind,
                                          outcome, error=error)
        self._account(kind, outcome, path, False, depth, seg, gap_s=gap_s, target=target, generated=generated,
                      error=error)
        return web.json_response(merged)

    # ------------------------------------------------------------------ logging

    def _account(
        self,
        kind: str,
        outcome: str,
        path: str,
        stream: bool,
        depth: int,
        seg: Segment,
        *,
        gap_s: float | None,
        target: str,
        generated: int,
        error: str = "",
    ) -> None:
        self.metrics.count_reissue(kind, outcome)
        record = self._log_record(kind, outcome, path, stream, depth, seg)
        record.update(
            {
                "generated_tokens": generated,
                "gap_ms": None if gap_s is None else round(gap_s * 1000.0, 1),
                "target": target,
                "next_depth": depth + 1,
            }
        )
        if error:
            record["error"] = error[:300]
        _log(record)

    def _log_record(self, kind: str, outcome: str, path: str, stream: bool, depth: int, seg: Segment) -> dict:
        return {
            "event": "tre_reissue",
            "ts": round(time.time(), 3),
            "model": self.cfg.model,
            "pod": self.cfg.pod_name,
            "kind": kind,
            "outcome": outcome,
            "path": path,
            "stream": stream,
            "depth": depth,
            "id": seg.id,
        }


def _merge_non_stream(
    first: dict | None, second: dict, *, chat: bool, as_chat: bool, generated: int
) -> tuple[dict, str | None, dict | None]:
    nested = second.get("tre_reissue") if isinstance(second.get("tre_reissue"), dict) else None
    second_choices = second.get("choices") or [{}]
    second_choice = second_choices[0] if isinstance(second_choices[0], dict) else {}
    second_text = second_choice.get("text") if as_chat or not chat else (second_choice.get("message") or {}).get("content")
    second_text = second_text if isinstance(second_text, str) else ""
    finish = second_choice.get("finish_reason")
    if first is None:
        merged = dict(second)
        if as_chat:  # not reached: a stuck chat request is resent as chat
            merged = second
        return merged, finish, nested
    merged = dict(first)
    choices = []
    for index, choice in enumerate(first.get("choices") or ()):
        copy = dict(choice)
        if index == 0:
            if chat:
                message = dict(copy.get("message") or {})
                message["content"] = (message.get("content") or "") + second_text
                copy["message"] = message
            else:
                copy["text"] = (copy.get("text") or "") + second_text
            copy["finish_reason"] = finish
            if "stop_reason" in copy or "stop_reason" in second_choice:
                copy["stop_reason"] = second_choice.get("stop_reason")
        choices.append(copy)
    merged["choices"] = choices
    merged["usage"] = merge_usage(first.get("usage"), second.get("usage"), generated_before=generated)
    return merged, finish, nested


def _parse_snapshot(payload: bytes) -> Any:
    try:
        return json.loads(payload) if payload else []
    except ValueError:
        return None


def _int_header(value: str | None) -> int:
    try:
        return max(0, int(value)) if value is not None else 0
    except ValueError:
        return 0


def _body_model(raw: bytes) -> str | None:
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return body.get("model") if isinstance(body, dict) and isinstance(body.get("model"), str) else None


def _client_gone(request: web.Request) -> bool:
    transport = request.transport
    return transport is None or transport.is_closing()


def _log(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def build_app(cfg: Config) -> web.Application:
    return ReissueSidecar(cfg).build_app()


def main() -> None:
    cfg = Config.from_env()
    _log({"event": "tre_reissue_start", "model": cfg.model, "pod": cfg.pod_name, "listen": cfg.listen_port,
          "upstream": cfg.upstream_url, "gateway": cfg.gateway_url, "enabled": cfg.enabled,
          "max_depth": cfg.max_depth, "chat_mode": cfg.chat_mode})
    web.run_app(
        build_app(cfg),
        host=cfg.listen_host,
        port=cfg.listen_port,
        access_log=None,
        print=None,
        backlog=2048,
        handle_signals=True,
    )


if __name__ == "__main__":
    main()
