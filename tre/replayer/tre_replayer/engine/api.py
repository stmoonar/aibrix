"""Request construction: which OpenAI-compatible endpoint, the body, the headers.

Part of the driver-free sending core (with :mod:`tre_replayer.engine.stream`, which makes
the call): :func:`request_body` and :func:`build_request_headers` are all a client needs
to put one fixed-length streamed request on the wire.

Two endpoints, two prefill paths
--------------------------------
* ``completions`` (``/v1/completions``): the prompt string is tokenised as is. On the
  fleet's vLLM 0.30 engines that adds **no** BOS token, so ``usage.prompt_tokens`` is the
  plain token count of the string.
* ``chat`` (``/v1/chat/completions``): the engine renders ``messages`` through the
  model's chat template - for the three DeepSeek-R1-Distill models
  ``<BOS><｜User｜>{content}<｜Assistant｜><think>\\n`` - and tokenises the result, so
  ``usage.prompt_tokens`` is the content's tokens plus the template's (5 for all three).

v1 (the paper's client) and the E1 client (``tre/loadgen_v1``) send ``chat``; the
calibration drivers (``deploy/scripts/r3_grid.py`` and the campaign) send ``chat`` too
since 2026-09-30, so the prefill a calibrated theta was measured on - template, BOS,
tokenizer path - is the one the experiments put on the engine. The only difference left
is the output: calibration pins it with ``ignore_eos`` + ``max_tokens``.

The replayer's own default stays ``completions`` (:data:`DEFAULT_API`): the trace runs
(``run_trace`` / ``campaign_queue``) keep sending exactly what they sent before; a caller
that wants ``chat`` asks for it.

Exact length under ``chat``: the natural-prompt fitter counts the *templated* prompt
(:func:`tre_replayer.engine.model_tokenizer.for_api`), so the content is the requested
total minus the template, and ``usage.prompt_tokens`` equals the requested total.
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

API_COMPLETIONS = "completions"
API_CHAT = "chat"
APIS = (API_COMPLETIONS, API_CHAT)

#: What :class:`~tre_replayer.engine.http_sender.StreamingHttpSender` sends unless told
#: otherwise. Unchanged: the trace replays keep their request shape.
DEFAULT_API = API_COMPLETIONS

#: The URL path each endpoint is served on.
API_PATHS = {API_COMPLETIONS: "/v1/completions", API_CHAT: "/v1/chat/completions"}


def check_api(api: str) -> str:
    if api not in APIS:
        raise ValueError(f"unknown API: {api!r} (expected one of {APIS})")
    return api


def check_api_url(url: str, api: str) -> None:
    """Refuse a gateway URL that names the other endpoint.

    A ``chat`` sender must be given the chat path (a body with ``messages`` sent to
    ``/v1/completions`` is a 400 on every request, and a silent URL rewrite would hide a
    launch script that was never updated). A ``completions`` sender refuses only the chat
    path, so the trace replays' URLs keep working unchanged.
    """
    check_api(api)
    path = urlparse(str(url)).path.rstrip("/")
    if api == API_CHAT:
        if not path.endswith(API_PATHS[API_CHAT]):
            raise ValueError(
                f"API {api!r} needs a gateway URL ending in {API_PATHS[API_CHAT]}, got {url!r}"
            )
        return
    if path.endswith(API_PATHS[API_CHAT]):
        raise ValueError(
            f"API {api!r} sends a completions body, but {url!r} is the chat endpoint; pass the "
            f"{API_PATHS[API_COMPLETIONS]} URL or select the chat API"
        )


def check_api_mode(api: str, prompt_mode: str) -> None:
    """``chat`` carries text only, and only the natural mode makes its length exact.

    A token-id list cannot be a chat message, and the ``text`` mode's nominal length
    would be off by the template (the whole point of the chat path is an exact
    ``usage.prompt_tokens``). Refused before any work.
    """
    check_api(api)
    if api == API_CHAT and prompt_mode != "natural":
        raise ValueError(
            f"the chat API needs prompt mode 'natural' (exact templated length); "
            f"{prompt_mode!r} cannot be sent as a chat message of an exact length"
        )


def request_body(
    model: str,
    prompt: Any,
    max_tokens: int,
    *,
    api: str = DEFAULT_API,
    seed: Optional[int] = None,
) -> dict:
    """The streamed, fixed-length generation request for ``api``.

    Both carry ``temperature 0``, ``ignore_eos`` (the output length is ``max_tokens``,
    whatever the model would have said) and ``stream_options.include_usage`` (the final
    chunk carries ``usage``, the only source of the token counts). ``seed`` is sent only
    when given; the completions body is byte-for-byte the one the replayer always sent.
    """
    check_api(api)
    if api == API_CHAT:
        if not isinstance(prompt, str):
            raise ValueError("a chat request carries text; got a token-id prompt")
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    if seed is not None:
        body["seed"] = int(seed)
    return body


def request_prompt(body: dict) -> Any:
    """The prompt of a body built by :func:`request_body` (tests, evidence)."""
    if "messages" in body:
        return body["messages"][0]["content"]
    return body["prompt"]


#: Request header that makes the AIBrix gateway plugin route (and therefore report the
#: pod it routed to). See :class:`tre_replayer.engine.http_sender.StreamingHttpSender`.
ROUTING_STRATEGY_HEADER = "routing-strategy"

#: What the v1 client sent on every request (OpenAI SDK ``default_headers``, all v1
#: configs: ``client.routing_algorithm: least-gpu-cache``): the plugin picks the awake pod
#: with the lowest ``vllm:kv_cache_usage_perc`` (``gpu_cache_usage_perc`` before 0.11).
#: The trace replayer (``run_trace``) and the campaign default to it so both arms are
#: routed the way v1 routed them.
DEFAULT_ROUTING_STRATEGY = "least-gpu-cache"


def build_request_headers(model: str, routing_strategy: str | None = None) -> dict[str, str]:
    """Request headers for one completion, and with them the serving path.

    The ``model`` header is always sent. Without a routing strategy it is what the
    per-model HTTPRoute matches (Service path). With one, the ``routing-strategy`` header
    is added and the plugin-routed route - patched in AHEAD of the per-model routes on
    both gateways, so it wins regardless of the ``model`` header - takes the request; on
    tre-v2 that route is per model and matches the ``model`` header too (per-model
    ORIGINAL_DST cluster). The plugin itself reads the model from the JSON body.

    (Until 2026-09-24 the ``model`` header was dropped on the routed path, on the belief
    that it would make the per-model HTTPRoute win; the patched route sits at index 0 of
    the route table, so it never did.)
    """
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "model": model}
    if routing_strategy:
        headers[ROUTING_STRATEGY_HEADER] = routing_strategy
    return headers
