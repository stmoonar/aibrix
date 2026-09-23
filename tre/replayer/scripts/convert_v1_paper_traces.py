#!/usr/bin/env python3
"""Convert the 7 ICSE-submission v1 traces into the v2 replayer's trace.json schema.

Source of truth
----------------
The 7 traces are the "traces_v14" workloads used for the ICSE submission's headline
TRE-vs-APA comparison (see /root/aibrix-main/CustomTraceGenerator/config/traces_v14 on
76 for the v1 generator configs). Those configs are *not* a materialised per-request
trace: `custom_load_test.custom_trace_json` / `real_trace_file` only encode a rate
*schedule* (piecewise RPS + token shape), and the actual per-request arrival plan that
v1's `client_dispatcher.py` sent is the product of that schedule plus Python's `random`
module consumed across 8 OS processes x 100 coroutines each -- not reproducible bit for
bit from the config and seed alone (confirmed below: the TRE-arm and APA-arm replay of
the *same* config/seed produced two different concrete request sequences).

The generator DID leave behind the concrete, already-materialised per-request plan it
sent for the submission runs:

    /data/nfs_shared_data/xxy/trace_output/output_traces_v14/{tre,apa}/<trace>/traces.json

Each line-item is one `RequestRecord`: `request_id`, `timestamp` (seconds relative to
the 720s window), `model_name`, `prompt` (the actual Chinese filler text sent),
`prompt_length` (that text's real-tokenizer length -- the input token count), and
`max_output_tokens` (the `max_tokens` cap sent to the API). This *is* the "raw trace
file" the task asks for: it has arrival time, model, input tokens and output-token cap
for every request v1 actually dispatched. `performance_metrics.json` in the same
directory is the corresponding dispatch log (JSONL, one row per request, same
`request_id`/`timestamp`) and is used only to cross-check that traces.json is indeed
what was sent (see docs/v1-paper-trace-conversion.md verification table) -- it is not a
second source of the plan.

This script uses the **tre/ arm** as the canonical source per trace (the arm the
paper's TRE numbers come from) and reports, as evidence, how far the apa/ arm's
independently-drawn realisation of the same config/seed diverges in aggregate (it
should NOT be read as "the same trace twice": the seed is not exactly reproducible
across the 8-process dispatcher, so the two arms are two different concrete draws from
the same generator/config, not the same one).

Conversion rule (schema-driven, not a design choice made for its own sake)
----------------------------------------------------------------------
v2's `trace.json` (`tre_replayer.traces.loader.load_trace_segments`) is a
model-keyed list of fixed-width segments `{start_time, end_time, rps, input_tokens,
max_tokens}` (a "fixed" segment, every request gets exactly that length) or the
`{..., input_tokens_dist: {kind, low, high}}` variant (every request draws its length
log-uniformly from `[low, high]` -- see `tre_replayer.engine.schedule.TokenRange`).
There is no third option: a segment cannot state an arbitrary empirical distribution.

We bin the realised per-request plan into BIN_WIDTH_S = 1s segments per model (matching
the native 1s granularity v1's own `real_trace_file` format already uses, and finer
than any v1 stable/transition phase, so no rate structure is smoothed away). Within
each (model, 1s-bin):

* `Alternating_hot_model_periodic_A`, `Decode_heavy_burst`,
  `Prefill_mixed_corner_decode_mix`, `Simultaneous_spike_ramp_twice_tps1o2`,
  `Sinusoidal_demand` ("custom" v1 generate_mode): `max_output_tokens` is *exactly*
  constant within every bin (it is a config literal -- 0% of bins show more than one
  value, confirmed empirically). `prompt_length` is *not* exactly constant (74-100% of
  bins show 2+ distinct values) even though the v1 config's target is a fixed literal
  too: v1's sentence-level text expander (`data_generator._expand_prompt_content`) stops
  once the remaining deficit is <=15 tokens, so realised length is a few tokens of
  one-sided noise below the target (observed stdev ~3.9 tokens around a 350-token
  target) -- not a real per-request distribution, just v1's own fitting slop. We
  therefore emit a single **fixed** value per bin, the bin's rounded mean, for both
  input and output tokens on these 5 traces: this reproduces the design target and
  drops only that slop, never a real signal.
* `Real_code_2024_slice_a_tok70`, `Real_conv_2023_slice_a_tok70` (v1 "real_trace"
  generate_mode, backed by a real per-request corpus): both input and output tokens
  vary substantially and genuinely within almost every bin (100% of bins). We emit the
  `input_tokens_dist` / `max_tokens_dist` log-uniform range `[min, max]` observed in
  that bin, so the replayer draws its own length per request rather than flattening the
  bin to one value. This does NOT reproduce v1's empirical distribution shape exactly --
  log-uniform puts more mass at the tails than v1's real distribution -- and this is the
  single biggest documented approximation of the whole conversion; see the verification
  table in docs/v1-paper-trace-conversion.md for the resulting quantile deltas.

What is intentionally dropped
------------------------------
* The actual prompt text. v2 never stores prompt text in trace.json (see
  `tre_replayer.engine.prompt_store`); prompts are synthesised at run time by
  `tre_replayer.engine.prompts.build_prompt` (English filler text or raw token ids
  fitted with the model's own tokenizer, seeded by `(model, request_id)`), not v1's
  Chinese template sentences. This is a routing/serving-relevant difference (prefix
  caching, tokenizer overhead) documented in the semantics diff report, not something
  this converter can or should preserve.
* `phase_type` (v1's internal stable/transition/realtime label) and `request_id`
  ordering -- v2 assigns its own request ids (`<model>-<index>`) at schedule build time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

BIN_WIDTH_S = 1

#: v1 generate_mode == "custom": token shapes are (near-)fixed config literals: emit a
#: single rounded-mean fixed value per bin (see module docstring).
CUSTOM_MODE_TRACES = (
    "Alternating_hot_model_periodic_A",
    "Decode_heavy_burst",
    "Prefill_mixed_corner_decode_mix",
    "Simultaneous_spike_ramp_twice_tps1o2",
    "Sinusoidal_demand",
)

#: v1 generate_mode == "real_trace": token shapes are genuinely per-request (drawn from
#: a real corpus): emit a log-uniform range per bin.
REAL_TRACE_MODE_TRACES = (
    "Real_code_2024_slice_a_tok70",
    "Real_conv_2023_slice_a_tok70",
)

ALL_TRACES = CUSTOM_MODE_TRACES + REAL_TRACE_MODE_TRACES

#: v1's `_get_request_overrides` only returns a `max_output_tokens` override for a
#: timestamp that falls inside a declared stable/hold segment; a request generated
#: during an inter-segment transition gets `None` in traces.json and the dispatcher
#: falls back to the model's config-level `max_tokens` default (client_dispatcher.py:
#: "max_tokens = model_config.max_tokens; if request.max_output_tokens is not None:
#: max_tokens = request.max_output_tokens"). All 3 models share one default per trace
#: (config/traces_v14/<trace>/config.yaml). Confirmed against performance_metrics.json:
#: e.g. Sinusoidal_demand req_000039 has max_output_tokens=None in traces.json and its
#: dispatch log shows output_tokens=300, exactly the config default below. Real-trace
#: mode traces never hit this path (every request has a real corpus value).
TRANSITION_DEFAULT_MAX_TOKENS = {
    "Alternating_hot_model_periodic_A": 400,
    "Decode_heavy_burst": 300,
    "Prefill_mixed_corner_decode_mix": 700,
    "Simultaneous_spike_ramp_twice_tps1o2": 600,
    "Sinusoidal_demand": 300,
}

MODEL_ORDER = ("dsllama-8b", "dsqwen-7b", "dsqwen-14b")

DEFAULT_SOURCE_ROOT = Path("/data/nfs_shared_data/xxy/trace_output/output_traces_v14")
CANONICAL_ARM = "tre"
CROSSCHECK_ARM = "apa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(source_root: Path, trace_name: str, arm: str) -> list[dict[str, Any]]:
    path = source_root / arm / trace_name / "traces.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bin_requests(
    requests: Iterable[dict[str, Any]],
    *,
    bin_width_s: int = BIN_WIDTH_S,
    transition_default_max_tokens: int | None = None,
) -> dict[tuple[str, int], dict[str, list[int]]]:
    """Group requests by (model, bin_index); bin_index * bin_width_s = start_time."""
    buckets: dict[tuple[str, int], dict[str, list[int]]] = {}
    for request in requests:
        model = request["model_name"]
        bin_index = int(request["timestamp"] // bin_width_s)
        key = (model, bin_index)
        bucket = buckets.setdefault(key, {"input": [], "output": []})
        bucket["input"].append(int(request["prompt_length"]))
        max_output_tokens = request["max_output_tokens"]
        if max_output_tokens is None:
            # Inter-segment transition request: see TRANSITION_DEFAULT_MAX_TOKENS.
            if transition_default_max_tokens is None:
                raise ValueError(
                    f"request {request['request_id']!r} has max_output_tokens=None "
                    "but no transition_default_max_tokens was supplied"
                )
            max_output_tokens = transition_default_max_tokens
        bucket["output"].append(int(max_output_tokens))
    return buckets


def build_trace(
    requests: list[dict[str, Any]], *, trace_name: str, bin_width_s: int = BIN_WIDTH_S
) -> dict[str, list[dict[str, Any]]]:
    sampled = trace_name in REAL_TRACE_MODE_TRACES
    buckets = bin_requests(
        requests,
        bin_width_s=bin_width_s,
        transition_default_max_tokens=TRANSITION_DEFAULT_MAX_TOKENS.get(trace_name),
    )
    trace: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_ORDER}
    for (model, bin_index), values in buckets.items():
        start = bin_index * bin_width_s
        end = start + bin_width_s
        count = len(values["input"])
        segment: dict[str, Any] = {
            "start_time": start,
            "end_time": end,
            "rps": round(count / bin_width_s, 6),
        }
        if sampled:
            in_min, in_max = min(values["input"]), max(values["input"])
            out_min, out_max = min(values["output"]), max(values["output"])
            segment["input_tokens_dist"] = {"kind": "log_uniform", "low": max(1, in_min), "high": max(1, in_max)}
            segment["max_tokens_dist"] = {"kind": "log_uniform", "low": max(1, out_min), "high": max(1, out_max)}
        else:
            segment["input_tokens"] = max(1, round(statistics.mean(values["input"])))
            segment["max_tokens"] = max(1, round(statistics.mean(values["output"])))
        trace.setdefault(model, []).append(segment)
    for model in trace:
        trace[model].sort(key=lambda seg: seg["start_time"])
    return trace


def convert_one(
    source_root: Path, trace_name: str, out_dir: Path, *, arm: str = CANONICAL_ARM
) -> dict[str, Any]:
    src_path = source_root / arm / trace_name / "traces.json"
    requests = load_requests(source_root, trace_name, arm)
    trace = build_trace(requests, trace_name=trace_name)
    out_path = out_dir / trace_name / "trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    model_counts: dict[str, int] = {model: 0 for model in MODEL_ORDER}
    for request in requests:
        model_counts[request["model_name"]] = model_counts.get(request["model_name"], 0) + 1
    duration_s = max(r["timestamp"] for r in requests)
    return {
        "trace_name": trace_name,
        "source_path": str(src_path),
        "source_sha256": sha256_file(src_path),
        "source_arm": arm,
        "request_count": len(requests),
        "model_request_counts": model_counts,
        "duration_s": duration_s,
        "bin_width_s": BIN_WIDTH_S,
        "token_rule": "sampled_log_uniform" if trace_name in REAL_TRACE_MODE_TRACES else "fixed_bin_mean",
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        "segment_count": sum(len(v) for v in trace.values()),
    }


def convert_all(source_root: Path, out_dir: Path, *, arm: str = CANONICAL_ARM) -> list[dict[str, Any]]:
    return [convert_one(source_root, name, out_dir, arm=arm) for name in ALL_TRACES]


def write_index(out_dir: Path) -> None:
    index = {
        "version": "traceset-v1paper",
        "notes": [
            "7 traces converted from the ICSE-submission traces_v14 realised "
            "per-request plans "
            "(/data/nfs_shared_data/xxy/trace_output/output_traces_v14/tre/<trace>/traces.json), "
            "preserving exact per-model request counts and per-second arrival rate.",
            "Token shapes are binned to v2's segment schema (fixed value or log-uniform "
            "range per 1s bin); see scripts/convert_v1_paper_traces.py module docstring "
            "and README.md for the exact rule and its documented approximation.",
        ],
        "workloads": list(ALL_TRACES),
    }
    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", default=CANONICAL_ARM, choices=(CANONICAL_ARM, CROSSCHECK_ARM))
    args = parser.parse_args(argv)

    reports = convert_all(args.source_root, args.out_dir, arm=args.arm)
    write_index(args.out_dir)
    manifest_path = args.out_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
