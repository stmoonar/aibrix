#!/usr/bin/env python3
"""Idle TTFT(L) capture: the data the D6' label's ``c_m`` / ``b_m`` are fitted on.

The primary calibration label is ``TTFT_slo(L) = max(500 ms, 5 * (c_m + b_m * L))``
(plan 2026-09-21 6.11 D6'), so ``c_m`` / ``b_m`` must be measured on the engine being
calibrated *before* any labelled capture. This entry point measures them directly instead
of mining isolated requests out of loaded cells:

* one model at a time on **one routable replica** (``--expect-replicas``);
* prompt lengths ``--lengths`` (default 128..4096), ``--per-length`` requests each, in a
  **block-randomised** order (every block holds each length once, shuffled with
  ``--seed``), so slow drift of the engine cannot line up with the length;
* requests **one at a time**: a deterministic schedule fires one request every ``--gap-s``
  (>= 0.5 s), longer than one request's service time at the largest length, so nothing
  overlaps; the fit (:mod:`scripts.ttft_idle_fit`) still keeps only isolated requests;
* a few ``--warmup`` requests first, in their own cell (raw under ``raw/warmup``, which
  the fit excludes);
* ``--max-tokens`` (16) output with ``ignore_eos``, so a request is short.

Everything else is the calibration campaign's load path and guards, reused as is: the
open-loop driver of :mod:`scripts.r3_grid` (same sender, corpus, chat endpoint, routing
header, raw schema with client-side TTFT and ``scheduled_send_ts_ms`` /
``on_wire_delay_ms``), the run-mode check (controller and SM both ``observe``), the
clock-domain pre-flight, the prompt-length pre-flight (on every length of this run) and
the capture of the system side. The window label is ``fixed`` (500/75 ms): ``c``/``b`` are
not known yet, and nothing is fitted on these windows anyway.

Layout under ``--out-dir``::

    <model>/idle_plan.json      provenance, the order sent, pods / images / engine version
    <model>/prompt_preflight.json
    <model>/schedules/{warmup,idle}.json
    <model>/online/{warmup,idle}.csv  (+ online/cells/ capture)
    <model>/raw/{warmup,idle}/<cell>.jsonl
    <model>/idle_status.json

Then ``python3 -m scripts.ttft_idle_fit --root <out-dir> --models ...``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from tre_common import slo_labels

from scripts import calibration_campaign as campaign
from scripts import calibration_capture as capture
from scripts import openloop
from scripts import prompt_corpus as corpus_record
from scripts import r3_grid

DEFAULT_MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048, 3072, 4096)
DEFAULT_PER_LENGTH = 30
DEFAULT_WARMUP = 5
DEFAULT_MAX_TOKENS = 16
DEFAULT_GAP_S = 2.0
#: Below this the "idle" engine may still be finishing the previous request.
MIN_GAP_S = 0.5
DEFAULT_SEED = 20261003
#: Rough fixed cost per r3_grid invocation (prompt materialisation, sidecar start, the
#: capture's gateway flush wait) for the wall-clock estimate.
INVOCATION_OVERHEAD_S = 30.0
PLAN_FILE = "idle_plan.json"
STATUS_FILE = "idle_status.json"
WARMUP = "warmup"
IDLE = "idle"


def block_order(lengths: Sequence[int], per_length: int, seed: int) -> list[int]:
    """``per_length`` blocks; each block holds every length once, shuffled by ``seed``."""
    rng = random.Random(int(seed))
    order: list[int] = []
    for _ in range(int(per_length)):
        block = [int(n) for n in lengths]
        rng.shuffle(block)
        order.extend(block)
    return order


def build_trace(model: str, order: Sequence[int], *, gap_s: float, max_tokens: int) -> dict:
    """A replayer trace with one single-request segment per entry of ``order``: segment k
    spans ``[k*gap, (k+1)*gap)`` at ``1/gap`` rps, which the deterministic arrival process
    turns into exactly one request at ``k*gap``."""
    gap = float(gap_s)
    return {model: [
        {"start_time": round(k * gap, 6), "end_time": round((k + 1) * gap, 6),
         "rps": 1.0 / gap, "input_tokens": int(n), "max_tokens": int(max_tokens)}
        for k, n in enumerate(order)
    ]}


def cell_id(kind: str, max_tokens: int) -> str:
    """A GridCell id (``rewindow_from_raw`` parses it): mixed lengths -> ``i0``; the load
    code tells the warmup (0) from the measured cell (1)."""
    return f"i0_o{int(max_tokens)}_c{0 if kind == WARMUP else 1}"


def estimate_wall_clock_s(n_requests: int, gap_s: float, invocations: int = 2) -> float:
    return float(n_requests) * float(gap_s) + invocations * INVOCATION_OVERHEAD_S


def fixed_label(args, model: str) -> slo_labels.LabelDefinition:
    return slo_labels.label_def_for_model(
        model, ttft_p95_ms=args.ttft_slo_ms, tpot_p95_ms=args.tpot_slo_ms,
        mode=slo_labels.TTFT_SLO_MODE_FIXED)


def r3_command(args, model: str, kind: str, schedule: Path, model_dir: Path) -> list[str]:
    """The r3_grid invocation of one cell: the campaign's per-cell discipline
    (:func:`scripts.calibration_campaign.cell_command`) with deterministic arrivals and the
    fixed label."""
    command = [
        sys.executable, "-m", "scripts.r3_grid",
        "--model", model,
        "--gateway-url", args.gateway_url,
        "--schedule", str(schedule),
        "--arrivals", openloop.ARRIVALS_DETERMINISTIC,
        "--cell-id", cell_id(kind, args.max_tokens),
        "--prompt-key", f"ttft-idle-{kind}-{args.seed}",
        "--output", str(model_dir / "online" / f"{kind}.csv"),
        "--raw-dir", str(model_dir / "raw"),
        "--window-ms", str(args.window_ms),
        "--instant-sample-ms", str(args.instant_sample_ms),
        "--metrics-schema", "v2",
        "--namespace", args.model_namespace,
        "--guard-mode", "fail",
        "--min-slo-windows", "0",
        "--prompt-dir", str(model_dir / "prompts"),
        *campaign.prompt_corpus_cli_args(args),
        "--api", campaign.api_for(args),
        "--prompt-preflight", "skip",
        "--shed-policy", openloop.SHED_POLICY_VOID,
        "--max-p99-delay-ms", str(openloop.CALIBRATION_MAX_P99_DELAY_MS),
        "--ttft-slo-ms", str(args.ttft_slo_ms),
        "--tpot-slo-ms", str(args.tpot_slo_ms),
        *slo_labels.label_mode_cli_args(fixed_label(args, model)),
    ]
    if not args.no_capture_extras:
        command += ["--capture-dir", str(model_dir / "online" / capture.CELLS_DIRNAME),
                    "--control-namespace", args.controller_namespace,
                    "--clock-domain-check", "flag"]
    for flag, value in (("--envoy-stats-url", args.envoy_stats_url),
                        ("--envoy-cluster-filter", args.envoy_cluster_filter),
                        ("--registry", args.registry), ("--redis-url", args.redis_url)):
        if value:
            command += [flag, value]
    if args.request_seed is not None:
        command += ["--request-seed", str(int(args.request_seed))]
    routing = campaign.routing_strategy_for(args)
    if routing:
        command += ["--routing-strategy", routing]
    return command


def engine_version(target: dict, timeout_s: float = 3.0) -> Optional[str]:
    """vLLM's ``/version`` on the pod's metrics port; None when it does not answer."""
    url = str(target.get("url") or "").rsplit("/metrics", 1)[0] + "/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - pod URL
            return json.loads(resp.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def require_replicas(args, model: str) -> list[dict]:
    targets = capture.discover_pod_targets(model, args.model_namespace, args.pod_metrics_port)
    if len(targets) != args.expect_replicas:
        raise SystemExit(
            f"refusing to run {model}: {len(targets)} routable pod(s) "
            f"({', '.join(t['name'] for t in targets) or 'none'}), expected {args.expect_replicas}")
    for t in targets:
        t["engine_version"] = engine_version(t)
    return targets


def plan_for(args, model: str) -> dict:
    order = block_order(args.lengths, args.per_length, args.seed)
    blocks = -(-args.warmup // len(args.lengths))
    warm = block_order(args.lengths, blocks, args.seed + 1)[: args.warmup]
    return {
        "model": model,
        "lengths": list(args.lengths), "per_length": args.per_length,
        "warmup_lengths": warm, "order": order,
        "gap_s": args.gap_s, "max_tokens": args.max_tokens, "ignore_eos": True,
        "seed": args.seed, "arrivals": openloop.ARRIVALS_DETERMINISTIC,
        "cells": {WARMUP: cell_id(WARMUP, args.max_tokens), IDLE: cell_id(IDLE, args.max_tokens)},
        "estimated_wall_clock_s": estimate_wall_clock_s(len(order) + len(warm), args.gap_s),
    }


def run_model(args, model: str, provenance: dict) -> int:
    model_dir = Path(args.out_dir) / model
    model_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_for(args, model)
    plan["pods"] = require_replicas(args, model)
    campaign.require_prompt_preflight(args, [model], out_dir=model_dir, lengths=args.lengths)
    plan["provenance"] = provenance
    plan["started_at_utc"] = campaign.utc_iso()
    (model_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = {"model": model, "cells": {}}
    rc = 0
    for kind, order in ((WARMUP, plan["warmup_lengths"]), (IDLE, plan["order"])):
        if not order:
            continue
        schedule = model_dir / "schedules" / f"{kind}.json"
        schedule.parent.mkdir(parents=True, exist_ok=True)
        schedule.write_text(json.dumps(build_trace(model, order, gap_s=args.gap_s,
                                                   max_tokens=args.max_tokens)) + "\n", encoding="utf-8")
        command = r3_command(args, model, kind, schedule, model_dir)
        print(f"[{model}] {kind}: {len(order)} request(s), gap {args.gap_s}s: {' '.join(command)}", flush=True)
        rc = subprocess.run(command, check=False).returncode
        status["cells"][kind] = {"exit_code": rc, "finished_at_utc": campaign.utc_iso()}
        if rc != 0:
            break
    status["status"] = "complete" if rc == 0 else "failed"
    (model_dir / STATUS_FILE).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rc


def provenance_for(args) -> dict:
    return {
        "code": campaign.git_state(Path(__file__).resolve().parents[2]),
        "registry_path": str(campaign.registry_path_for(args)),
        "registry_sha256": campaign.file_sha256(campaign.registry_path_for(args)),
        "prompt": campaign.prompt_corpus(args),
        "routing_strategy": campaign.routing_strategy_for(args),
        "api": campaign.api_record(args),
        "label": slo_labels.label_definition(fixed_label(args, args.models[0])),
        "run_mode": getattr(args, "run_mode", None),
        "argv": list(sys.argv),
    }


def _lengths(text: str) -> list[int]:
    values = sorted({int(x) for x in text.split(",") if x.strip()})
    if len(values) < 2 or values[0] < 1:
        raise argparse.ArgumentTypeError("need at least two positive lengths")
    return values


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--gateway-url", default=os.environ.get(campaign.GATEWAY_URL_ENV) or None,
                    help=f"chat-completions URL of the gateway (or ${campaign.GATEWAY_URL_ENV})")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lengths", type=_lengths, default=list(DEFAULT_LENGTHS))
    ap.add_argument("--per-length", type=int, default=DEFAULT_PER_LENGTH)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--gap-s", type=float, default=DEFAULT_GAP_S)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--expect-replicas", type=int, default=1)
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=75.0)
    ap.add_argument("--window-ms", type=int, default=30000)
    ap.add_argument("--instant-sample-ms", type=int, default=1000)
    ap.add_argument("--corpus-lang", default=r3_grid.CORPUS_LANG_DEFAULT, choices=list(r3_grid.CORPUS_LANGS))
    ap.add_argument("--zh-ratio", type=r3_grid._unit_interval, default=r3_grid.ZH_RATIO_DEFAULT)
    ap.add_argument("--api", default=corpus_record.CALIBRATION_API, choices=list(corpus_record.APIS))
    ap.add_argument("--routing-strategy", type=campaign.normalize_routing_strategy,
                    default=campaign.DEFAULT_ROUTING_STRATEGY)
    ap.add_argument("--request-seed", type=int, default=None)
    ap.add_argument("--prompt-preflight", default="refuse", choices=["refuse", "skip"])
    ap.add_argument("--registry", default=None)
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--envoy-stats-url", default=None)
    ap.add_argument("--envoy-cluster-filter", default=None)
    ap.add_argument("--model-namespace", default="default")
    ap.add_argument("--controller-namespace", default="tre-v2")
    ap.add_argument("--pod-metrics-port", type=int, default=8000)
    ap.add_argument("--no-capture-extras", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate; send nothing")
    args = ap.parse_args(argv)
    args.models = [m for m in args.models.split(",") if m]
    if args.gap_s < MIN_GAP_S:
        ap.error(f"--gap-s must be >= {MIN_GAP_S}")
    if args.per_length < 1 or args.max_tokens < 1 or args.warmup < 0:
        ap.error("--per-length and --max-tokens must be >= 1, --warmup >= 0")
    if not args.dry_run and not args.gateway_url:
        ap.error(f"--gateway-url (or ${campaign.GATEWAY_URL_ENV}) is required")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        for model in args.models:
            plan = plan_for(args, model)
            print(json.dumps({k: plan[k] for k in ("model", "lengths", "per_length", "warmup_lengths",
                                                    "gap_s", "max_tokens", "estimated_wall_clock_s")}))
        return 0
    args.run_mode = campaign.require_calibration_run_mode(args.controller_namespace)
    campaign.require_capture_clock_domains(args, args.models)
    provenance = provenance_for(args)
    for model in args.models:
        rc = run_model(args, model, provenance)
        if rc != 0:
            print(f"[{model}] FAILED (exit {rc}); stopping", file=sys.stderr)
            return rc
    print("idle TTFT capture complete; fit: python3 -m scripts.ttft_idle_fit --root "
          f"{args.out_dir} --models {','.join(args.models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
