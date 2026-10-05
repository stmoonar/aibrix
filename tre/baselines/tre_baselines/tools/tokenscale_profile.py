"""Baseline parameter profiling: TokenScale-colocated V_b / V_P and PreServe-oracle mu in
one run (decision 2026-10-06, "参数测量").

The hot alt trace has one request shape (input ~492, output 400), so TokenScale's 3x3
buckets collapse into one cell: one shape is measured and written to all nine cells
(lookups clamp to the nearest bucket; disclose "buckets degenerate to one cell").

Per model, on its **single awake replica** (``--sm-url`` checks exactly one is awake; the
models run in parallel, each on its own replica), a closed-loop load at each concurrency
of the ladder (default 1, 2, 4, 8, 12, 16, 24, 32, 48, 64; extended by x1.5 steps while
the last step still gains more than ``--extend-gain`` (5 %), up to ``--max-concurrency``,
with a warning if it still gains there). Each step lasts ``--step-s`` (60) of which the
first ``--warmup-s`` (15) are dropped. Requests are the calibration chat sender's
(:func:`scripts.r3_grid.drive_cell`): ``/v1/chat/completions``, a distinct natural prompt of
``in`` tokens after the chat template per request, ``max_tokens = out``, ``ignore_eos``,
temperature 0, streamed with usage. Per step:

* prefill / decode tok/s from the replica's own counters ``vllm:prompt_tokens_total`` /
  ``vllm:generation_tokens_total``, scraped at the end of the warm-up and at the end of
  the step (not from completed requests);
* p95 TTFT and p95 TPOT (client side) of the requests that finished inside the window.

Outputs (``--out-dir``): ``profile.yaml`` and ``profile_raw.csv``:

* TokenScale ``velocity``: ``V_b`` = the highest (prefill + decode) tok/s of any step (a
  saturation indicator; no SLO), ``V_P`` = the highest prefill tok/s of a separate ladder
  with ``out = 1``;
* PreServe ``mu`` (``p`` / ``d`` / ``t``): prefill / decode / total tok/s of the highest
  step whose p95 TTFT <= the model's TTFT SLO at this input length
  (``max(floor, k (c + b L))`` from the registry, the SLO labels use) and p95 TPOT <= its
  TPOT SLO (75 ms). mu_p : mu_d is fixed by the shape (one free quantity; disclose). A
  closed loop keeps shorter queues than an open one, so mu is slightly optimistic
  (disclose; cross-check with the calibration rho* x token mix).

Cost: (len(ladder) x 2) x step_s per model, models in parallel (default ~20 min).
Sending needs ``--i-have-user-approval``; ``--dry-run`` uses a synthetic stub.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.request import Request, urlopen

import yaml

#: Closed-loop concurrency ladder (decision 2026-10-06).
DEFAULT_LADDER = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
DEFAULT_EXTEND_GAIN = 0.05
DEFAULT_MAX_CONCURRENCY = 256
PROMPT_COUNTER = "vllm:prompt_tokens_total"
GEN_COUNTER = "vllm:generation_tokens_total"


@dataclass(frozen=True)
class StepMeasure:
    concurrency: int
    window_s: float
    completed: int
    #: Engine counters' rates over the measured window (None when unreadable / reset).
    prefill_tok_s: Optional[float]
    decode_tok_s: Optional[float]
    #: Client-side p95 of the requests that finished inside the window (ms).
    ttft_p95_ms: Optional[float]
    tpot_p95_ms: Optional[float]

    @property
    def tok_s(self) -> Optional[float]:
        if self.prefill_tok_s is None or self.decode_tok_s is None:
            return None
        return self.prefill_tok_s + self.decode_tok_s


#: measure(model, in_tokens, out_tokens, concurrency, step_s, warmup_s) -> StepMeasure
Measure = Callable[[str, int, int, int, float, float], StepMeasure]


# ------------------------------------------------------------------ pieces


def p95(values: Sequence[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, max(0, math.ceil(0.95 * len(vals)) - 1))]


def counter_sum(text: str, name: str) -> Optional[float]:
    from tre_baselines.sources import parse_prometheus_text

    vals = [s.value for s in parse_prometheus_text(text) if s.name == name and math.isfinite(s.value)]
    return sum(vals) if vals else None


def engine_rates(before: str, after: str, dt_s: float) -> tuple[Optional[float], Optional[float]]:
    """(prefill tok/s, decode tok/s) from two scrapes of the replica; None on a missing
    counter, a reset (engine restart) or a zero-length window."""
    out = []
    for name in (PROMPT_COUNTER, GEN_COUNTER):
        a, b = counter_sum(before, name), counter_sum(after, name)
        out.append(None if a is None or b is None or b < a or dt_s <= 0 else (b - a) / dt_s)
    return out[0], out[1]


def window_latencies(records: Sequence[dict], lo_ms: int, hi_ms: int) -> tuple[int, Optional[float], Optional[float]]:
    """(completed, p95 TTFT ms, p95 TPOT ms) of the requests that finished with 200 and no
    stream error inside ``[lo, hi]`` (raw records of ``r3_grid.drive_cell``)."""
    ok = [r for r in records
          if r.get("http_status") == 200 and not r.get("stream_error") and r.get("done_ts_ms") is not None
          and lo_ms <= r["done_ts_ms"] <= hi_ms]
    return len(ok), p95([r.get("ttft_ms") for r in ok]), p95([r.get("tpot_ms") for r in ok])


def http_get_text(url: str, timeout_s: float = 5.0) -> str:
    with urlopen(Request(url, headers={"accept": "text/plain"}), timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", "replace")


def make_http_measure(gateway_url: str, metrics_urls: Callable[[str], Sequence[str]], *, api: str = "chat",
                      prompt_mode: str = "natural", routing_strategy: Optional[str] = None,
                      run_key: str = "bl-profile", stream_call: Optional[Callable] = None,
                      fetch: Callable[[str], str] = http_get_text, raw_dir: Optional[Path] = None,
                      now_ms: Callable[[], int] = lambda: int(time.time() * 1000)) -> Measure:
    """One step = ``r3_grid.drive_cell`` (the calibration chat sender) for ``step_s`` with
    ``concurrency`` closed-loop workers, the replica's ``/metrics`` (``metrics_urls(model)``)
    scraped when the warm-up ends and when the step ends."""
    from scripts import r3_grid  # lazy: tre/deploy on PYTHONPATH (as for the calibration tools)

    def scrape(model: str) -> str:
        return "\n".join(fetch(url) for url in metrics_urls(model))

    def measure(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float,
                warmup_s: float) -> StepMeasure:
        cell = r3_grid.GridCell(int(in_tokens), int(out_tokens), int(concurrency))
        marks: dict[str, tuple[float, int, Optional[str]]] = {}

        def mark(name: str) -> None:
            try:
                text: Optional[str] = scrape(model)
            except Exception:  # noqa: BLE001 - an unreadable scrape is unknown, not 0
                text = None
            marks[name] = (time.monotonic(), now_ms(), text)

        timers = [threading.Timer(warmup_s, mark, ("a",)), threading.Timer(step_s, mark, ("b",))]
        with tempfile.TemporaryDirectory(prefix="bl-profile-") as tmp:
            base = Path(raw_dir) if raw_dir is not None else Path(tmp)
            base.mkdir(parents=True, exist_ok=True)
            raw = base / f"{model}_{cell.scenario_id}.jsonl"
            raw.unlink(missing_ok=True)
            for t in timers:
                t.start()
            r3_grid.drive_cell(gateway_url, model, cell, step_s, raw_path=raw, stream_call=stream_call,
                               prompt_mode=prompt_mode, routing_strategy=routing_strategy,
                               run_key=f"{run_key}|{model}", api=api)
            for t in timers:
                t.join()
            records = [json.loads(ln) for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip()] \
                if raw.exists() else []
        (ta, wa, sa), (tb, wb, sb) = marks["a"], marks["b"]
        prefill, decode = engine_rates(sa, sb, tb - ta) if sa is not None and sb is not None else (None, None)
        completed, ttft, tpot = window_latencies(records, wa, wb)
        return StepMeasure(int(concurrency), round(tb - ta, 3), completed, prefill, decode, ttft, tpot)

    return measure


def make_stub_measure(vmax_tok_s: float = 10000.0, c_half: float = 4.0, vmax_prefill_tok_s: float = 30000.0,
                      ttft_ms_per_c: float = 30.0, tpot_ms_per_c: float = 2.0) -> Measure:
    """Synthetic saturating throughput and latencies that grow with concurrency."""

    def measure(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float,
                warmup_s: float) -> StepMeasure:
        vmax = vmax_prefill_tok_s if out_tokens <= 1 else vmax_tok_s
        rate = vmax * concurrency / (concurrency + c_half)
        share = in_tokens / (in_tokens + out_tokens)
        window = step_s - warmup_s
        return StepMeasure(concurrency, window, int(rate * window / (in_tokens + out_tokens)), rate * share,
                           rate * (1 - share), 100.0 + ttft_ms_per_c * concurrency, 20.0 + tpot_ms_per_c * concurrency)

    return measure


# ------------------------------------------------------------------ the ladder


def run_ladder(measure: Measure, model: str, in_tokens: int, out_tokens: int, step_s: float, warmup_s: float,
               ladder: Sequence[int] = DEFAULT_LADDER, *, extend_gain: float = DEFAULT_EXTEND_GAIN,
               max_concurrency: int = DEFAULT_MAX_CONCURRENCY, rate: str = "tok_s",
               warn: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr)) -> list[StepMeasure]:
    """Every step of the ladder; then, while the last step still gained more than
    ``extend_gain`` over the one before, further steps at x1.5 up to ``max_concurrency``."""
    steps = [measure(model, in_tokens, out_tokens, int(c), step_s, warmup_s) for c in ladder]

    def gain() -> float:
        if len(steps) < 2:
            return 0.0
        a, b = getattr(steps[-2], rate), getattr(steps[-1], rate)
        return 0.0 if not a or b is None else (b - a) / a

    while gain() > extend_gain:
        nxt = int(math.ceil(steps[-1].concurrency * 1.5))
        if nxt > max_concurrency:
            warn(f"warning: {model} in={in_tokens} out={out_tokens}: still +{gain():.0%} at concurrency "
                 f"{steps[-1].concurrency} (max {max_concurrency}); the peak may be higher")
            break
        steps.append(measure(model, in_tokens, out_tokens, nxt, step_s, warmup_s))
    return steps


def _best(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def preserve_mu(steps: Sequence[StepMeasure], ttft_slo_ms: float, tpot_slo_ms: float) -> Optional[dict[str, float]]:
    """mu from the highest-concurrency step meeting both p95 SLOs (None if none does)."""
    ok = [s for s in steps if s.completed > 0 and s.tok_s is not None and s.ttft_p95_ms is not None
          and s.tpot_p95_ms is not None and s.ttft_p95_ms <= ttft_slo_ms and s.tpot_p95_ms <= tpot_slo_ms]
    if not ok:
        return None
    best = max(ok, key=lambda s: s.concurrency)
    return {"p": round(best.prefill_tok_s, 1), "d": round(best.decode_tok_s, 1), "t": round(best.tok_s, 1),
            "concurrency": best.concurrency}


def profile_model(measure: Measure, model: str, in_tokens: int, out_tokens: int, slo: Any, *, step_s: float,
                  warmup_s: float, ladder: Sequence[int], extend_gain: float = DEFAULT_EXTEND_GAIN,
                  max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> tuple[dict, list[dict]]:
    """Both ladders of one model -> ({velocity, mu, slo}, raw rows)."""
    kw = dict(extend_gain=extend_gain, max_concurrency=max_concurrency)
    mixed = run_ladder(measure, model, in_tokens, out_tokens, step_s, warmup_s, ladder, rate="tok_s", **kw)
    prefill = run_ladder(measure, model, in_tokens, 1, step_s, warmup_s, ladder, rate="prefill_tok_s", **kw)
    ttft_slo = float(slo.ttft_slo_ms(in_tokens))
    tpot_slo = float(slo.tpot_p95_ms)
    v_b, v_p = _best([s.tok_s for s in mixed]), _best([s.prefill_tok_s for s in prefill])
    result = {
        "velocity": None if v_b is None or v_p is None else
        {"buckets": [[round(v_b, 1)] * 3 for _ in range(3)], "v_prefill": round(v_p, 1)},
        "mu": preserve_mu(mixed, ttft_slo, tpot_slo),
        "slo": {"ttft_p95_ms": round(ttft_slo, 1), "tpot_p95_ms": tpot_slo, "in_tokens": in_tokens},
    }
    rows = [dict(model=model, kind=kind, in_tokens=in_tokens, out_tokens=out, **asdict(s), tok_s=s.tok_s)
            for kind, out, steps in (("mixed", out_tokens, mixed), ("prefill", 1, prefill)) for s in steps]
    return result, rows


# ------------------------------------------------------------------ cluster checks (read only)


def check_one_awake(sm_url: str, model: str, timeout_s: float = 5.0) -> int:
    """Awake replicas of ``model`` per the SM's ``GET /v2/state`` (read only)."""
    with urlopen(Request(sm_url.rstrip("/") + "/v2/state", headers={"accept": "application/json"}),
                 timeout=timeout_s) as resp:
        state = json.loads(resp.read().decode("utf-8"))
    return int(((state.get("models") or {}).get(model) or {}).get("awake", 0))


def model_slo(model: str, registry_path: Optional[str]) -> Any:
    """The model's SLO definition from the registry with its fitted c/b (strict: refuses a
    registry without them, as an actuating shell does)."""
    from tre_common.registry import load_registry

    from tre_baselines.config import _slo_definition

    registry = load_registry(registry_path)
    return _slo_definition(model, registry.model(model), registry, strict=True)


def _ladder(text: str) -> tuple[int, ...]:
    try:
        ladder = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not ladder or any(c < 1 for c in ladder):
        raise argparse.ArgumentTypeError("concurrencies must be >= 1")
    return ladder


def main(argv: Optional[list[str]] = None, *, measure: Optional[Measure] = None,
         slo_for: Optional[Callable[[str], Any]] = None) -> int:
    ap = argparse.ArgumentParser(description="TokenScale V_b/V_P + PreServe mu profiling (one awake replica per model)")
    ap.add_argument("--models", required=True, help="comma list; profiled in parallel, one awake replica each")
    ap.add_argument("--in-tokens", type=int, default=492, help="request input length after the chat template")
    ap.add_argument("--out-tokens", type=int, default=400, help="max_tokens (ignore_eos)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gateway-url", default=None, help="the chat endpoint URL (.../v1/chat/completions)")
    ap.add_argument("--sm-url", default=None, help="required for a real run: checks exactly one awake replica")
    ap.add_argument("--model-namespace", default="default", help="namespace of the model pods")
    ap.add_argument("--metrics-port", type=int, default=8000, help="pod port serving vLLM /metrics")
    ap.add_argument("--registry", default=None, help="registry with the fitted c/b (default: the shared one)")
    ap.add_argument("--routing-strategy", default=None)
    ap.add_argument("--raw-dir", default=None, help="keep the per-request raw JSONL here")
    ap.add_argument("--dry-run", action="store_true", help="synthetic stub; nothing is sent")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required to send real requests")
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=15.0)
    ap.add_argument("--concurrency", type=_ladder, default=DEFAULT_LADDER)
    ap.add_argument("--extend-gain", type=float, default=DEFAULT_EXTEND_GAIN)
    ap.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.warmup_s >= args.step_s:
        print("error: --warmup-s must be < --step-s", file=sys.stderr)
        return 2
    minutes = 2 * len(args.concurrency) * args.step_s / 60
    print(f"estimated time: ~{minutes:.0f} min per model (models in parallel), ladder {list(args.concurrency)}")
    slo_for = slo_for or (lambda m: model_slo(m, args.registry))
    try:
        slos = {m: slo_for(m) for m in models}
    except (ValueError, KeyError, SystemExit) as exc:
        print(f"error: SLO: {exc}", file=sys.stderr)
        return 2
    if measure is None:
        if args.dry_run:
            measure = make_stub_measure()
        else:
            if not args.i_have_user_approval:
                print("refusing to send requests without --i-have-user-approval (use --dry-run)", file=sys.stderr)
                return 3
            if not args.gateway_url or not args.sm_url:
                print("error: a real run needs --gateway-url and --sm-url", file=sys.stderr)
                return 2
            wrong = {m: n for m in models if (n := check_one_awake(args.sm_url, m)) != 1}
            if wrong:
                print(f"error: profile with exactly one awake replica per model, got {wrong}", file=sys.stderr)
                return 4
            from scripts.r3_grid import discover_pod_metrics_endpoints

            urls = {m: discover_pod_metrics_endpoints(m, args.model_namespace, args.metrics_port) for m in models}
            if any(len(u) != 1 for u in urls.values()):
                print(f"error: expected one routable pod per model, got {urls}", file=sys.stderr)
                return 4
            measure = make_http_measure(args.gateway_url, lambda m: urls[m], routing_strategy=args.routing_strategy,
                                        raw_dir=Path(args.raw_dir) if args.raw_dir else None)

    def one(model: str) -> tuple[dict, list[dict]]:
        return profile_model(measure, model, args.in_tokens, args.out_tokens, slos[model], step_s=args.step_s,
                             warmup_s=args.warmup_s, ladder=args.concurrency, extend_gain=args.extend_gain,
                             max_concurrency=args.max_concurrency)

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        results = dict(zip(models, pool.map(one, models)))
    doc = {
        "shape": {"in_tokens": args.in_tokens, "out_tokens": args.out_tokens,
                  "note": "TokenScale buckets degenerate to one cell (all nine = V_b); disclose"},
        "velocity": {m: r["velocity"] for m, (r, _) in results.items()},
        "mu": {m: r["mu"] for m, (r, _) in results.items()},
        "slo": {m: r["slo"] for m, (r, _) in results.items()},
    }
    missing = [m for m, (r, _) in results.items() if r["mu"] is None or r["velocity"] is None]
    if missing:
        print(f"warning: no SLO-meeting step / unreadable counters for {missing}: see profile_raw.csv", file=sys.stderr)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    (out / "profile.yaml").write_text(text, encoding="utf-8")
    rows = [row for _, (_, rs) in results.items() for row in rs]
    with (out / "profile_raw.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
