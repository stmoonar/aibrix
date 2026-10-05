"""The exact prompt length a client tells the gateway: header ``x-tre-bl-in-tokens``.

The baseline policies' gateway hook (``pkg/plugins/gateway/tre_bl_req_events.go``,
``HeaderBLInTokens``) needs each request's prompt length when it arrives. It reads this
header when a client sends it and otherwise falls back to a character estimate. The
header carries ``usage.prompt_tokens`` as the engine will report it, so it is counted
with the model's own tokenizer (:mod:`tre_replayer.engine.model_tokenizer`, which finds it
from ``TRE_TOKENIZER_PATHS`` / the registry's ``weights_path``):

* ``chat`` (``/v1/chat/completions``): the content rendered inside the model's chat
  template (``model_tokenizer.for_api(tok, "chat").count``, the same view the calibration
  prompt fitter counts with);
* ``completions``: the plain token count of the string (the fleet's vLLM 0.30 engines add
  no BOS to a completions prompt, see :mod:`tre_replayer.engine.api`); a token-id prompt
  is its length.

Counted once per request when the schedule is loaded (:func:`precount_in_tokens`), before
any worker forks: tokenising on the send path would put milliseconds of GIL-held work
inside each request's send lateness. A request whose count cannot be made (no tokenizer,
no template the module can reproduce, no prompt) is sent **without** the header - never
with a guess - and is counted in the summary. Opt-in on every client, default off: TRE
and APA do not read the header, so sending it changes nothing for them.

``scripts/check_in_tokens_header.py`` compares the sent value with ``usage.prompt_tokens``
over a run's records.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Optional

from tre_replayer.engine.api import API_CHAT, check_api

#: The request header (the gateway's ``HeaderBLInTokens``): a positive decimal integer.
IN_TOKENS_HEADER = "x-tre-bl-in-tokens"

#: Row field holding the value sent (None: the flag was on but the header was omitted).
#: Present only on runs with the flag on, so a default run's rows are unchanged.
RECORD_FIELD = "in_tokens_header"


def prompt_token_count(tok: Any, prompt: Any, api: str) -> int:
    """``usage.prompt_tokens`` of ``prompt`` sent to ``api``, counted with ``tok``."""
    check_api(api)
    if isinstance(prompt, (list, tuple)):
        if api == API_CHAT:
            raise ValueError("a chat request carries text; got a token-id prompt")
        return len(prompt)
    if api == API_CHAT:
        from tre_replayer.engine.model_tokenizer import for_api

        return int(for_api(tok, API_CHAT).count(prompt))
    return len(tok.encode_plain(prompt))


def header_for(count: Optional[int]) -> dict[str, str]:
    """The header to add for ``count`` (none when it is unknown)."""
    if count is None or int(count) <= 0:
        return {}
    return {IN_TOKENS_HEADER: str(int(count))}


def precount_in_tokens(
    events: Iterable[Any],
    *,
    api: str,
    prompt_of: Optional[Callable[[Any], Any]] = None,
    tokenizer_paths: Optional[Mapping[str, str]] = None,
    load: Optional[Callable[..., Any]] = None,
) -> tuple[list[Any], dict]:
    """``events`` with ``in_tokens_header`` set to each request's exact prompt length,
    and a summary for the run's manifest.

    ``prompt_of(event)`` is the prompt that will be sent (default: ``event.prompt``).
    ``tokenizer_paths`` maps a model to its tokenizer directory and wins over the
    resolution chain of :func:`~tre_replayer.engine.model_tokenizer.load_tokenizer`.
    A model whose tokenizer cannot be loaded, or a request that cannot be counted, keeps
    ``in_tokens_header=None`` (the header is then omitted) and is counted under
    ``omitted``; the first reason per model is kept in ``errors``.
    """
    check_api(api)
    if load is None:
        from tre_replayer.engine import model_tokenizer

        load = model_tokenizer.load_tokenizer
    if prompt_of is None:
        prompt_of = lambda event: event.prompt  # noqa: E731
    paths = dict(tokenizer_paths or {})
    tokenizers: dict[str, Any] = {}
    errors: dict[str, str] = {}
    omitted: dict[str, int] = {}
    counted = 0
    started = time.perf_counter()
    out: list[Any] = []
    for event in events:
        model = event.model
        count: Optional[int] = None
        if model not in tokenizers and model not in errors:
            try:
                tokenizers[model] = load(model, tokenizer_path=paths.get(model))
            except Exception as exc:  # noqa: BLE001 - TokenizerUnavailable, OSError, ...
                errors[model] = f"{type(exc).__name__}: {exc}"
        tok = tokenizers.get(model)
        if tok is not None:
            prompt = prompt_of(event)
            if prompt:
                try:
                    count = prompt_token_count(tok, prompt, api)
                except Exception as exc:  # noqa: BLE001 - e.g. no reproducible chat template
                    errors.setdefault(model, f"{type(exc).__name__}: {exc}")
                    count = None
            else:
                errors.setdefault(model, "a request has no prompt to count")
        if count is None or count <= 0:
            count = None
            omitted[model] = omitted.get(model, 0) + 1
        else:
            counted += 1
        out.append(replace(event, in_tokens_header=count))
    summary = {
        "header": IN_TOKENS_HEADER,
        "api": api,
        "counted": counted,
        "omitted": sum(omitted.values()),
        "omitted_by_model": dict(sorted(omitted.items())),
        "errors": dict(sorted(errors.items())),
        "tokenizers": {model: getattr(tok, "path", None) for model, tok in sorted(tokenizers.items())},
        "precount_s": round(time.perf_counter() - started, 3),
    }
    return out, summary

