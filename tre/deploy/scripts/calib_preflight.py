#!/usr/bin/env python3
"""Calibration preflight, standalone: is every model's prompt length exact through chat?

For each ``--models`` model and each ``--targets`` length, one request of the calibration
kind - a natural prompt (``--corpus-lang`` / ``--zh-ratio``, default the 1:1 zh/en mix)
fitted with the model's own tokenizer so the *chat-templated* prompt is exactly the
target, sent to ``--gateway-url`` (the ``/v1/chat/completions`` path) with the campaign's
routing header, ``ignore_eos`` and ``max_tokens`` - must come back with
``usage.prompt_tokens == target``, ``usage.completion_tokens == max_tokens``, a first
token in a recognised SSE field and, for the mix, the content's Chinese token share
within 0.01 of the target (``tre_replayer.engine.preflight``; the same check the campaign
runs before its first cell).

Writes one JSON object per line to ``--out`` (``{model, target, prompt_tokens,
completion_tokens, zh_token_ratio, ok, reasons, first_token_field, template_overhead,
...}``) and exits 0 only when every row is exact, 1 when any is not, 2 on a usage error.

Run from ``tre/deploy`` (it imports ``scripts.*``) with the replayer on the path::

    cd tre/deploy && PYTHONPATH=../common:.:../replayer python3 -m scripts.calib_preflight \\
        --models dsqwen-7b,dsllama-8b,dsqwen-14b \\
        --gateway-url http://<gateway>/v1/chat/completions --out /tmp/calib_preflight.jsonl

Sends ``len(models) x len(targets)`` requests of ``max_tokens`` (default 8) output tokens;
nothing else touches the cluster.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from scripts import prompt_corpus

#: The campaign's routing header (calibration_campaign.DEFAULT_ROUTING_STRATEGY).
DEFAULT_ROUTING_STRATEGY = "least-gpu-cache"
DEFAULT_TARGETS = "512"
DEFAULT_MAX_TOKENS = 8


def _split(values: Sequence[str]) -> list[str]:
    return [v.strip() for value in values for v in str(value).split(",") if v.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="append", required=True,
                    help="model names, comma separated and/or repeated")
    ap.add_argument("--gateway-url", required=True,
                    help="the gateway's endpoint URL; its path must match --api")
    ap.add_argument("--out", type=Path, required=True, help="JSON Lines file, one row per model x target")
    ap.add_argument("--targets", action="append", default=None,
                    help=f"templated prompt lengths to check (comma separated / repeated; default {DEFAULT_TARGETS})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--api", default=prompt_corpus.CALIBRATION_API, choices=list(prompt_corpus.APIS))
    ap.add_argument("--corpus-lang", default=prompt_corpus.DEFAULT_CORPUS_LANG, choices=list(prompt_corpus.CORPUS_LANGS))
    ap.add_argument("--zh-ratio", type=float, default=prompt_corpus.DEFAULT_ZH_RATIO)
    ap.add_argument("--routing-strategy", default=DEFAULT_ROUTING_STRATEGY,
                    help="routing-strategy header ('' or 'none' = none, the per-model HTTPRoute)")
    ap.add_argument("--request-seed", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    args = ap.parse_args(argv)
    args.models = _split(args.models)
    try:
        args.targets = [int(t) for t in _split(args.targets or [DEFAULT_TARGETS])]
    except ValueError as exc:
        ap.error(f"--targets: {exc}")
    if not args.models:
        ap.error("--models names no model")
    if args.max_tokens < 1 or any(t < 1 for t in args.targets):
        ap.error("--targets and --max-tokens must be positive")
    if not 0.0 <= args.zh_ratio <= 1.0:
        ap.error("--zh-ratio must be within [0, 1]")
    try:
        prompt_corpus.check_gateway_url(args.gateway_url, args.api)
    except ValueError as exc:
        ap.error(str(exc))
    args.routing_strategy = prompt_corpus.normalize_routing(args.routing_strategy)
    return args


def run(args, *, stream_call=None, tokenizers: Optional[dict] = None) -> list[dict]:
    """Every (model, target) verdict, in order. ``stream_call`` / ``tokenizers`` are the
    tests' seams (default: the network and each model's tokenizer on local disk)."""
    from tre_replayer.engine.preflight import preflight_prompt_tokens

    rows = []
    for model in args.models:
        for target in args.targets:
            rows.append(preflight_prompt_tokens(
                args.gateway_url, model, api=args.api, corpus_lang=args.corpus_lang,
                zh_ratio=args.zh_ratio, routing_strategy=args.routing_strategy,
                request_seed=args.request_seed, input_tokens=target, output_tokens=args.max_tokens,
                stream_call=stream_call, tokenizer=(tokenizers or {}).get(model),
                timeout_s=args.timeout_s, seed_key=f"calib-preflight|{model}|{target}",
            ))
    return rows


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None, *, stream_call=None, tokenizers: Optional[dict] = None) -> int:
    args = parse_args(argv)
    rows = run(args, stream_call=stream_call, tokenizers=tokenizers)
    write_rows(args.out, rows)
    for row in rows:
        print(f"{'OK  ' if row['ok'] else 'FAIL'} {row['model']:12} target {row['target']:5} "
              f"prompt_tokens {row['prompt_tokens']} completion {row['completion_tokens']}/{row['max_tokens']} "
              f"first token in {row['first_token_field']!r} zh {row['zh_token_ratio']} "
              f"template +{row['template_overhead']}"
              + ("" if row["ok"] else f"  <- {'; '.join(row['reasons'])}"))
    failed = sum(1 for row in rows if not row["ok"])
    print(f"{len(rows) - failed}/{len(rows)} exact -> {args.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
