"""Measure what the natural-prompt corpus really sends, per model, on CPU.

For each model: ``--samples`` prompts at lengths drawn log-uniformly from
``[--min-tokens, --max-tokens]`` (seeded), built through the same path a run uses -
:func:`tre_replayer.engine.prompt_store.materialize_prompts` into a scratch file, then
looked up the way the sender does - and for each prompt:

* the token-count error against the target (``usage.prompt_tokens`` semantics: the
  model's own tokenizer, special tokens included) - must be 0;
* the Chinese share of its plain tokens, *measured* by decoding every token on its own
  (:meth:`~tre_replayer.engine.model_tokenizer.ModelTokenizer.cjk_token_count`: CJK
  text or a byte-level piece of a character), independently of how the corpus budgeted
  it;
* the store's miss count after looking every request up - must be 0.

Needs only the tokenizers on local disk (no GPU, no network, no cluster)::

    PYTHONPATH=replayer python3 -m scripts.prompt_corpus_report   # from tre/replayer: -m scripts...

Exit status is 1 if any prompt misses its target or any lookup misses.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tre_replayer.engine import model_tokenizer
from tre_replayer.engine.corpus import CORPUS_LANGS, DEFAULT_CORPUS_LANG, DEFAULT_ZH_RATIO
from tre_replayer.engine.prompt_store import materialize_prompts

#: The models whose tokenizers resolve without configuration (override with --models;
#: any model resolvable by tre_replayer.engine.model_tokenizer works).
DEFAULT_MODELS = tuple(model_tokenizer.FLEET_TOKENIZER_PATHS)


@dataclass
class _Request:
    request_id: str
    model: str
    prompt_tokens: int
    prompt: str = ""


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))]

    return {
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.pstdev(ordered), 4),
        "min": round(ordered[0], 4),
        "p5": round(q(0.05), 4),
        "p50": round(q(0.50), 4),
        "p95": round(q(0.95), 4),
        "max": round(ordered[-1], 4),
    }


def report_model(model: str, *, samples: int, lo: int, hi: int, seed: int,
                 corpus_lang: str, zh_ratio: float, workdir: Path) -> dict:
    tok = model_tokenizer.load_tokenizer(model)
    rng = random.Random(f"{seed}|{model}")
    requests = [
        _Request(f"report-{i}", model, int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))))))
        for i in range(samples)
    ]
    started = time.perf_counter()
    store = materialize_prompts(
        requests, path=workdir / f"{model}.prompts.jsonl", processes=1,
        corpus_lang=corpus_lang, zh_ratio=zh_ratio,
    )
    build_ms = (time.perf_counter() - started) * 1000.0
    errors, shares, by_bucket = [], [], {}
    for request in requests:
        text = store.get(request.request_id)
        if text is None:
            continue
        errors.append(tok.count(text) - request.prompt_tokens)
        plain = tok.count(text) - tok.overhead
        share = tok.cjk_token_count(text) / plain if plain else 0.0
        shares.append(share)
        bucket = "<256" if request.prompt_tokens < 256 else "256-1023" if request.prompt_tokens < 1024 else ">=1024"
        by_bucket.setdefault(bucket, []).append(share)
    return {
        "model": model,
        "tokenizer": tok.path,
        "samples": samples,
        "target_tokens": {"min": min(r.prompt_tokens for r in requests),
                          "max": max(r.prompt_tokens for r in requests)},
        "token_count_error": {"nonzero": sum(1 for e in errors if e), "max_abs": max((abs(e) for e in errors), default=0)},
        "prompt_store_misses": store.misses,
        "zh_token_share": _quantiles(shares),
        "zh_token_share_by_length": {k: {**_quantiles(v), "n": len(v)} for k, v in sorted(by_bucket.items())},
        "build_ms_per_prompt": round(build_ms / max(1, samples), 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--corpus-lang", default=DEFAULT_CORPUS_LANG, choices=list(CORPUS_LANGS))
    ap.add_argument("--zh-ratio", type=float, default=DEFAULT_ZH_RATIO)
    args = ap.parse_args(argv)
    ok = True
    with tempfile.TemporaryDirectory(prefix="prompt-corpus-report-") as tmp:
        for model in [m for m in args.models.split(",") if m]:
            row = report_model(model, samples=args.samples, lo=args.min_tokens, hi=args.max_tokens,
                               seed=args.seed, corpus_lang=args.corpus_lang, zh_ratio=args.zh_ratio,
                               workdir=Path(tmp))
            ok = ok and row["token_count_error"]["nonzero"] == 0 and row["prompt_store_misses"] == 0
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
