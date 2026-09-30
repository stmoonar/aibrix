"""What a request is, per client profile - the one place the three load paths differ.

The sending core (:class:`tre_replayer.engine.http_sender.StreamingHttpSender` on a
:mod:`~tre_replayer.engine.transport`, scheduled by :mod:`~tre_replayer.engine.dispatcher`
/ :mod:`~tre_replayer.engine.procpool`) is shared; a profile fixes the request and the
record:

``calib``
    The calibration request, byte for byte what the 2026-09-30 chat sender sent
    (:func:`tre_replayer.engine.api.request_body` with ``api="chat"``: ``temperature 0``,
    ``ignore_eos``, ``max_tokens``, ``stream`` + ``include_usage``, optional ``seed``;
    headers :func:`~tre_replayer.engine.api.build_request_headers`), prompts fitted to the
    templated length. Raw ``httpx`` transport, pooled connections, **no retry**, timeout
    ``max(30, max_tokens / 4)`` s per socket operation. Record: the calibration row
    (``result_fields`` + lateness), unchanged.
``replay``
    The same body on ``/v1/completions`` (the replayer's default, ``run_trace`` /
    ``campaign_queue``) - request bytes unchanged.
``e1_v1``
    The paper's client (v1 ``CustomTraceGenerator``, ported as ``tre/loadgen_v1``):
    ``chat.completions.create`` through the OpenAI SDK with v1's options - ``messages``
    = one user turn holding the trace's prompt, ``temperature`` from the model's config
    (unset = JSON ``null``), **no** ``ignore_eos``, ``max_tokens`` = the trace's
    ``max_output_tokens`` (else the model's config, else absent), ``stream`` +
    ``include_usage``; ``routing-strategy`` header; SDK retries (default 2, as v1);
    timeout 300 s. Record: v1's ``performance_metrics.json`` line (v1 fields, then v1's
    audit fields, then the strict and lateness fields).

Every profile reads the answer through the same parser
(:class:`tre_replayer.engine.stream.StreamParser`) and so recognises the reissue
sidecar's ``x-tre-continued`` / ``x-tre-retried`` marks and computes both metric bases
(:mod:`tre_replayer.engine.metrics`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from tre_replayer.engine.api import API_CHAT, API_COMPLETIONS
from tre_replayer.engine.transport import TRANSPORT_HTTPX, TRANSPORT_OPENAI_SDK

PROFILE_CALIB = "calib"
PROFILE_REPLAY = "replay"
PROFILE_E1_V1 = "e1_v1"
PROFILES = (PROFILE_CALIB, PROFILE_REPLAY, PROFILE_E1_V1)

RECORD_CALIBRATION = "calibration_row"
RECORD_V1_PERFORMANCE = "v1_performance_metrics"

#: Processes the calibration / replay drivers send from when not told otherwise
#: (``--sender-processes``). Sized on the local benchmark of 2026-09-30
#: (replayer/README.md): the peak of the committed schedules is ~200 rps bursts with up
#: to ~4k requests in flight.
DEFAULT_SENDER_PROCESSES = 4


@dataclass(frozen=True)
class ClientProfile:
    name: str
    api: str
    transport: str
    body: str
    ignore_eos: bool
    temperature: str
    retries: str
    timeout: str
    record: str

    def as_dict(self) -> dict:
        return asdict(self)


_FIXED_BODY = "tre_replayer.engine.api.request_body (temperature 0, ignore_eos, max_tokens, stream, include_usage)"

PROFILE_TABLE: dict[str, ClientProfile] = {
    PROFILE_CALIB: ClientProfile(
        PROFILE_CALIB, API_CHAT, TRANSPORT_HTTPX, _FIXED_BODY, True, "0", "none",
        "max(30, max_tokens/4) s per connect/read/write", RECORD_CALIBRATION),
    PROFILE_REPLAY: ClientProfile(
        PROFILE_REPLAY, API_COMPLETIONS, TRANSPORT_HTTPX, _FIXED_BODY, True, "0", "none",
        "max(30, max_tokens/4) s per connect/read/write", RECORD_CALIBRATION),
    PROFILE_E1_V1: ClientProfile(
        PROFILE_E1_V1, API_CHAT, TRANSPORT_OPENAI_SDK,
        "openai chat.completions.create(model, messages=[user], temperature, stream, stream_options, max_tokens)",
        False, "model config (unset -> JSON null)", "OpenAI SDK max_retries (default 2)",
        "SDK timeout (default 300 s)", RECORD_V1_PERFORMANCE),
}


def get_profile(name: str) -> ClientProfile:
    try:
        return PROFILE_TABLE[name]
    except KeyError:
        raise ValueError(f"unknown client profile {name!r} (expected one of {PROFILES})") from None


def profile_for_api(api: str) -> str:
    """The fixed-length profile of an endpoint: chat = calibration, completions = replay."""
    return PROFILE_CALIB if api == API_CHAT else PROFILE_REPLAY


def fixed_length_timeout_s(max_tokens: int) -> float:
    """Per-socket-operation timeout of a fixed-length request (calib / replay)."""
    return max(30.0, max_tokens / 4.0)


#: v1's fallback when a request names a model its config does not list
#: (``ModelConfig(name, "", max_tokens=128, temperature=0.0)``).
V1_UNKNOWN_MODEL_PARAMS = {"max_tokens": 128, "temperature": 0.0}


@dataclass
class V1ChatOptions:
    """The ``e1_v1`` request parameters - v1's config: per-model ``max_tokens`` /
    ``temperature`` (None = not set in the config) and the client section."""

    model_params: dict[str, dict] = field(default_factory=dict)
    api_key: str = ""
    max_retries: int = 2
    timeout_s: float = 300.0
    routing_strategy: Optional[str] = "least-gpu-cache"
    streaming: bool = True
    #: SDK clients per worker process (1 = v1's single client, the default; None = up
    #: to transport.E1_MAX_POOL_SHARDS, opened as load needs them). Not a request parameter.
    pool_shards: Optional[int] = 1

    def kwargs_for(self, model: str, prompt: str, max_output_tokens: Optional[int]) -> dict[str, Any]:
        """``chat.completions.create`` keyword arguments, as v1 built them."""
        params = self.model_params.get(model)
        if params is None:
            params = V1_UNKNOWN_MODEL_PARAMS
        max_tokens = params.get("max_tokens")
        if max_output_tokens is not None:  # the trace's per-request value wins
            max_tokens = max_output_tokens
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # None is sent as JSON null (v1 passed the config value through unchanged).
            "temperature": params.get("temperature"),
        }
        if self.streaming:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("api_key", None)  # never recorded
        return out


def client_provenance(profile: str, *, transport: Any = None, processes: int = 1,
                      schedule: str = "absolute (per-process asyncio, pre-sharded)", **extra) -> dict:
    """What a run's requests were and how they were sent, for its manifest."""
    out = {"profile": get_profile(profile).as_dict(), "processes": int(processes), "scheduling": schedule}
    if transport is not None and hasattr(transport, "provenance"):
        out["wire"] = transport.provenance()
    out.update(extra)
    return out
