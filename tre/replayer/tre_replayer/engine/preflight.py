"""Does the engine prefill exactly what the client fitted? One request says.

A calibration cell is indexed by its prompt length, and since 2026-09-30 that length is
the *templated* one (``/v1/chat/completions``: the chat template around the user content,
see :mod:`tre_replayer.engine.api`). The client fits every prompt to it with its local
tokenizer and template; this module checks, before any load, that the engine agrees:
one request of the run's exact kind - same builder, tokenizer, corpus, endpoint, URL,
routing header and seed - must come back with

* ``usage.prompt_tokens`` equal to the length the prompt was fitted to,
* ``usage.completion_tokens`` equal to ``max_tokens`` (``ignore_eos`` honoured),
* a first token in a field the SSE parser recognises
  (:data:`tre_replayer.engine.stream.CHAT_TOKEN_FIELDS` or completions ``text``),
* for a mixed-corpus natural prompt, a Chinese token share of the user content within
  :data:`ZH_RATIO_TOLERANCE` of the target (from 128 tokens up; below that one token is
  already more than the tolerance).

Driver-free (the sending core: :mod:`~tre_replayer.engine.api` +
:mod:`~tre_replayer.engine.stream`); the campaign's run-level check
(``scripts.calibration_campaign.require_prompt_preflight``), standalone ``r3_grid`` and
the CLI ``scripts.calib_preflight`` all call :func:`preflight_prompt_tokens`.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

#: The default request: long enough that the corpus mix and the template both matter,
#: short enough to cost nothing on any fleet engine.
PREFLIGHT_INPUT_TOKENS = 512
PREFLIGHT_OUTPUT_TOKENS = 8
PREFLIGHT_TIMEOUT_S = 60.0

#: How far the user content's Chinese token share may sit from the target.
ZH_RATIO_TOLERANCE = 0.01
#: Below this many tokens the share is not checked (one token > the tolerance).
ZH_RATIO_MIN_TOKENS = 128

#: Fields a first token may arrive in (completions ``text`` plus the chat delta fields).
FIRST_TOKEN_FIELDS = ("text", "content", "reasoning_content", "reasoning")


def preflight_prompt_tokens(
    gateway_url: str,
    model: str,
    *,
    api: str,
    prompt_mode: str = "natural",
    corpus_lang: Optional[str] = None,
    zh_ratio: Optional[float] = None,
    routing_strategy: Optional[str] = None,
    request_seed: Optional[int] = None,
    input_tokens: int = PREFLIGHT_INPUT_TOKENS,
    output_tokens: int = PREFLIGHT_OUTPUT_TOKENS,
    stream_call: Optional[Callable] = None,
    tokenizer: Any = None,
    timeout_s: float = PREFLIGHT_TIMEOUT_S,
    seed_key: Optional[str] = None,
) -> dict:
    """One request of the run's exact kind to ``model``; its verdict as a dict.

    ``ok`` is True only when every check of the module docstring holds; ``reasons`` lists
    the ones that did not. Never raises: a request that cannot be built or sent is a
    failed verdict with its reason. The dict also carries what the caller records: the
    template overhead the local tokenizer applied, the SSE field of the first token, the
    user content's Chinese token share (``zh_token_ratio``, None when not measured).
    """
    from tre_replayer.engine import corpus as corpus_mod
    from tre_replayer.engine.api import build_request_headers, check_api_mode, check_api_url, request_body
    from tre_replayer.engine.model_tokenizer import for_api, load_tokenizer
    from tre_replayer.engine.prompts import DEFAULT_CORPUS_LANG, DEFAULT_ZH_RATIO, build_prompt
    from tre_replayer.engine.stream import stream_request

    lang = corpus_lang if corpus_lang is not None else DEFAULT_CORPUS_LANG
    ratio = float(zh_ratio if zh_ratio is not None else DEFAULT_ZH_RATIO)
    verdict: dict = {
        "model": model, "api": api, "gateway_url": gateway_url,
        "routing_strategy": routing_strategy, "prompt_mode": prompt_mode,
        "corpus_lang": lang, "zh_ratio": ratio, "request_seed": request_seed,
        "target": int(input_tokens), "expected_prompt_tokens": int(input_tokens),
        "max_tokens": int(output_tokens), "template_overhead": None,
        "http_status": None, "prompt_tokens": None, "completion_tokens": None,
        "first_token_field": None, "first_token_ms": None, "zh_token_ratio": None,
        "error": None, "ok": False, "reasons": [],
    }
    reasons = verdict["reasons"]
    try:
        check_api_url(gateway_url, api)
        check_api_mode(api, prompt_mode)
        tok = None
        if prompt_mode == "natural":
            tok = tokenizer if tokenizer is not None else load_tokenizer(model)
            verdict["template_overhead"] = for_api(tok, api).overhead
        prompt = build_prompt(int(input_tokens), seed_key or f"preflight|{model}", mode=prompt_mode, model=model,
                              tokenizer=tok, corpus_lang=lang, zh_ratio=ratio, api=api)
        if tok is not None and isinstance(prompt, str):
            plain = len(tok.encode_plain(prompt))
            verdict["zh_token_ratio"] = round(tok.cjk_token_count(prompt) / plain, 6) if plain else None
        body = json.dumps(request_body(model, prompt, int(output_tokens), api=api,
                                       seed=request_seed)).encode("utf-8")
        call = stream_call or stream_request
        res = call(gateway_url, build_request_headers(model, routing_strategy), body, timeout_s)
    except Exception as exc:  # noqa: BLE001 - any failure is a refusal, with its reason
        verdict["error"] = f"{type(exc).__name__}: {exc}"
        reasons.append(f"preflight request not made or failed: {verdict['error']}")
        return verdict
    verdict.update({
        "http_status": res.status, "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "first_token_field": getattr(res, "first_token_field", None),
        "first_token_ms": res.first_token_ms, "error": res.error,
    })
    if res.status != 200:
        reasons.append(f"HTTP {res.status}: {(res.error_body or res.error or '')[:200]}")
    else:
        if res.prompt_tokens != int(input_tokens):
            reasons.append(f"usage.prompt_tokens {res.prompt_tokens} != {int(input_tokens)} the prompt "
                           f"was fitted to (local {api} count: template overhead "
                           f"{verdict['template_overhead']}): the local tokenizer / template does not "
                           "match the engine's")
        if res.completion_tokens != int(output_tokens):
            reasons.append(f"usage.completion_tokens {res.completion_tokens} != max_tokens "
                           f"{int(output_tokens)}: ignore_eos was not honoured")
        if res.first_token_ms is None:
            reasons.append("no chunk carried a token (text / content / reasoning_content / "
                           "reasoning): TTFT cannot be measured on this stream")
        elif verdict["first_token_field"] is not None and verdict["first_token_field"] not in FIRST_TOKEN_FIELDS:
            reasons.append(f"first token in an unrecognised field {verdict['first_token_field']!r}")
    target_share = corpus_mod.effective_zh_ratio(lang, ratio)
    share = verdict["zh_token_ratio"]
    if (share is not None and lang == corpus_mod.LANG_MIX and int(input_tokens) >= ZH_RATIO_MIN_TOKENS
            and abs(share - target_share) > ZH_RATIO_TOLERANCE):
        reasons.append(f"Chinese token share of the content {share} is not {target_share} +- "
                       f"{ZH_RATIO_TOLERANCE}")
    verdict["ok"] = not reasons
    return verdict
