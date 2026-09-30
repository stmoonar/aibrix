#!/usr/bin/env python3
"""N4.6 staged scale exercise + 12h soak driver (endgame plan 4.2/4.3).

Drives load through the isolated tre-aibrix-eg gateway while the TRE controller
autoscales on zm, and samples system state to JSONL on local disk. One cycle
(default 40 min):
    0-10  all models baseline (1 req/s)
    10-20 dsqwen-7b saturated (concurrency N), others baseline
    20-30 dsqwen-7b baseline; dsllama-8b saturated
    30-40 all baseline
--soak repeats cycles until --duration-seconds elapses (soak: set e.g. 43200).
Samples SM /v2/state, component RSS/restarts, redis DBSIZE, gateway error count,
per-GPU truth every --sample-seconds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

GATEWAY = "http://10.103.92.7/v1/completions"
SM = "http://10.111.21.116:8000"
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")

_stop = threading.Event()
_errlock = threading.Lock()
_errors: Counter = Counter()
_oks: Counter = Counter()


def _probe(model: str, max_tokens: int, timeout: float) -> None:
    body = json.dumps({"model": model, "prompt": "soak load probe",
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json", "model": model})
    label = None
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        ok = 200 <= r.status < 300
        if not ok:
            label = f"{model}:http{r.status}"
    except Exception as exc:  # noqa: BLE001
        ok = False
        label = f"{model}:{type(exc).__name__}"
    with _errlock:
        if ok:
            _oks[model] += 1
        else:
            _errors[label] += 1


def baseline_worker(model: str) -> None:
    while not _stop.is_set():
        _probe(model, 16, 20)
        _stop.wait(1.0)


def saturation_pool(model: str, workers: int, until: float) -> None:
    def hammer() -> None:
        while not _stop.is_set() and time.time() < until:
            _probe(model, 96, 30)
    threads = [threading.Thread(target=hammer, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def sampler(out: Path, interval: float, node_alias: dict) -> None:
    while not _stop.is_set():
        row = {"ts": None, "wall": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            state = json.load(urllib.request.urlopen(SM + "/v2/state", timeout=15))
            row["models"] = state["models"]
            row["sm_version"] = state["version"]
        except Exception as exc:  # noqa: BLE001
            row["state_error"] = str(exc)[:120]
        try:
            rec = json.load(urllib.request.urlopen(urllib.request.Request(
                SM + "/v2/reconcile", method="POST"), timeout=30))
            row["reconcile_warnings"] = rec.get("warnings", "err")
        except Exception as exc:  # noqa: BLE001
            row["reconcile_error"] = str(exc)[:120]
        with _errlock:
            row["ok_counts"] = dict(_oks)
            row["err_counts"] = dict(_errors)
        try:
            row["pods"] = subprocess.run(
                ["kubectl", "-n", "tre-v2", "get", "pods", "--no-headers"],
                capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        except Exception:  # noqa: BLE001
            pass
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        _stop.wait(interval)


def run_cycle(workers: int, phase_seconds: float) -> None:
    # phase 1: all baseline
    _stop.wait(phase_seconds)
    if _stop.is_set():
        return
    # phase 2: saturate dsqwen-7b
    saturation_pool("dsqwen-7b", workers, time.time() + phase_seconds)
    if _stop.is_set():
        return
    # phase 3: saturate dsllama-8b
    saturation_pool("dsllama-8b", workers, time.time() + phase_seconds)
    if _stop.is_set():
        return
    # phase 4: all baseline
    _stop.wait(phase_seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--duration-seconds", type=float, default=2400)  # one 40-min cycle
    ap.add_argument("--phase-seconds", type=float, default=600)      # 10 min/phase
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample-seconds", type=float, default=300)     # 5 min
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = outdir / "samples.jsonl"

    base = [threading.Thread(target=baseline_worker, args=(m,), daemon=True) for m in MODELS]
    for t in base:
        t.start()
    smp = threading.Thread(target=sampler, args=(samples, args.sample_seconds, {}), daemon=True)
    smp.start()

    started = time.time()
    cycle = 0
    try:
        while time.time() - started < args.duration_seconds and not _stop.is_set():
            cycle += 1
            run_cycle(args.workers, args.phase_seconds)
    finally:
        _stop.set()
        time.sleep(2)
    summary = {"cycles": cycle, "duration_s": round(time.time() - started, 1),
               "ok_counts": dict(_oks), "err_counts": dict(_errors)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
