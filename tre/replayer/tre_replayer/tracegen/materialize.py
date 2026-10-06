"""Design plan -> effective plan (what the client replays), and the effective == design check.

The effective file is the v1 per-request format ``tre_loadgen_v1 --trace-file`` reads:
``request_id, timestamp, model_name, prompt, prompt_length, phase_type, max_output_tokens``.
``prompt`` is a natural prompt (``tre_replayer.engine.prompts``, mode natural, ``api=chat``,
because the e1_v1 client sends chat) fitted with the model's own tokenizer so that the
engine's ``usage.prompt_tokens`` - chat template included - equals ``prompt_length``.
The seed key is ``<trace>|seed<k>|<request_id>``: prompts are unique per request and per
trace file, and the same design always yields byte-identical prompts.

:func:`verify` re-reads both files and checks, request by request: same ids, order,
timestamps, models, phase types; ``max_output_tokens`` identical and a positive int (never
null: a null would make the client fall back to the config's ``max_tokens``); the prompt's
templated token count equals ``prompt_length`` (re-counted with the tokenizer, unless
``recount=False``); prompts pairwise distinct.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tre_replayer.engine.api import API_CHAT
from tre_replayer.engine.prompt_store import PromptSpec, build_prompts

from .generate import DESIGN_FILE, MANIFEST_FILE, dumps_plan, sha256_bytes

EFFECTIVE_FILE = "traces_tre.effective.json"
EFFECTIVE_FIELDS = ("request_id", "timestamp", "model_name", "prompt", "prompt_length", "phase_type",
                    "max_output_tokens")


def seed_key(trace: str, seed: int, request_id: str) -> str:
    return f"{trace}|seed{seed}|{request_id}"


def effective_rows(design: list[dict], prompts: dict) -> list[dict]:
    return [{"request_id": r["request_id"], "timestamp": r["timestamp"], "model_name": r["model_name"],
             "prompt": prompts[r["request_id"]], "prompt_length": r["prompt_length"],
             "phase_type": r["phase_type"], "max_output_tokens": r["max_output_tokens"]} for r in design]


def materialize(out_dir: str | Path, *, processes: int | None = 4, tokenizers: dict | None = None,
                tokenizer_paths: dict | None = None, recount: bool = True) -> dict:
    """Build ``traces_tre.effective.json`` next to ``design.json``, verify it, update the manifest.

    ``tokenizers`` {model: tokenizer} is the tests' seam (no pool); otherwise each model's
    tokenizer is resolved by ``tre_replayer.engine.model_tokenizer`` (``tokenizer_paths``,
    ``TRE_TOKENIZER_PATHS``, the registry)."""
    out = Path(out_dir)
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    design = json.loads((out / DESIGN_FILE).read_bytes())
    trace, seed = manifest["trace"], int(manifest["seed"])
    prompts: dict[str, Any] = {}
    for model in sorted({r["model_name"] for r in design}):
        specs = [PromptSpec(r["request_id"], model, r["prompt_length"], seed_key(trace, seed, r["request_id"]))
                 for r in design if r["model_name"] == model]
        tok = (tokenizers or {}).get(model)
        path = (tokenizer_paths or {}).get(model)
        prompts.update(build_prompts(specs, processes=1 if tok is not None else processes, tokenizer=tok,
                                     tokenizer_path=path, api=API_CHAT))
    data = dumps_plan(effective_rows(design, prompts))
    (out / EFFECTIVE_FILE).write_bytes(data)
    result = verify(out, tokenizers=tokenizers, tokenizer_paths=tokenizer_paths, recount=recount)
    manifest["effective"] = {"file": EFFECTIVE_FILE, "sha256": sha256_bytes(data), "bytes": len(data),
                             "prompt_mode": "natural", "prompt_api": API_CHAT,
                             "prompt_seed_key": "<trace>|seed<k>|<request_id>", "verify": result}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def _counter(model: str, tokenizers, tokenizer_paths):
    from tre_replayer.engine.model_tokenizer import for_api, load_tokenizer
    tok = (tokenizers or {}).get(model)
    if tok is None:
        tok = load_tokenizer(model, tokenizer_path=(tokenizer_paths or {}).get(model))
    return for_api(tok, API_CHAT)


def verify(out_dir: str | Path, *, tokenizers: dict | None = None, tokenizer_paths: dict | None = None,
           recount: bool = True) -> dict:
    """effective == design (raises AssertionError on the first mismatch); returns a summary."""
    out = Path(out_dir)
    design = json.loads((out / DESIGN_FILE).read_bytes())
    eff = json.loads((out / EFFECTIVE_FILE).read_bytes())
    if len(eff) != len(design):
        raise AssertionError(f"effective has {len(eff)} requests, design {len(design)}")
    counters: dict = {}
    seen = set()
    for d, e in zip(design, eff):
        if tuple(e.keys()) != EFFECTIVE_FIELDS:
            raise AssertionError(f"{e.get('request_id')}: effective fields {tuple(e.keys())}")
        for k in ("request_id", "timestamp", "model_name", "prompt_length", "phase_type", "max_output_tokens"):
            if e[k] != d[k]:
                raise AssertionError(f"{d['request_id']}: {k} effective {e[k]!r} != design {d[k]!r}")
        mot = e["max_output_tokens"]
        if not isinstance(mot, int) or isinstance(mot, bool) or mot < 1:
            raise AssertionError(f"{d['request_id']}: max_output_tokens {mot!r} is not a positive int")
        if not isinstance(e["prompt"], str) or not e["prompt"]:
            raise AssertionError(f"{d['request_id']}: empty prompt")
        h = hashlib.blake2b(e["prompt"].encode(), digest_size=16).digest()
        if h in seen:
            raise AssertionError(f"{d['request_id']}: duplicate prompt")
        seen.add(h)
        if recount:
            c = counters.get(e["model_name"])
            if c is None:
                c = counters[e["model_name"]] = _counter(e["model_name"], tokenizers, tokenizer_paths)
            n = int(c.count(e["prompt"]))
            if n != e["prompt_length"]:
                raise AssertionError(f"{d['request_id']}: prompt counts {n} tokens (chat), design {e['prompt_length']}")
    return {"requests": len(eff), "prompt_tokens_recounted": bool(recount), "max_output_tokens_null": 0,
            "design_sha256": sha256_bytes((out / DESIGN_FILE).read_bytes())}
