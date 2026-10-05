"""TokenScale-colocated offline token-velocity profiling (TokenScale SS IV-B2).

Measured on the current engine image, one model at a time, with exactly **one awake
replica** of that model (the operator arranges it; ``--sm-url`` makes the tool check it
first). For every 3x3 bucket center (in, out) from ``tokenscale_buckets`` (bucket edges
derived from the replayed trace) a closed-loop load runs at each concurrency of the
ladder (default 1, 2, 4, 8, 16, 32); each step lasts ``--step-s`` (60) of which the first
``--warmup-s`` (15) are dropped. The step's rate is (input + output) tokens of the
requests that completed inside the measured window, per second; ``V_b`` is the peak over
the ladder. ``V_P`` is measured the same way with ``out = 1`` at the trace's median input
length (``median_in`` of the centers file, else the middle bucket's input, else
``--prefill-in``) and reported as input tokens/s.

A *sender* runs one step::

    sender(model, in_tokens, out_tokens, concurrency, step_s, warmup_s)
        -> (completed, window_s, in_tokens_sum, out_tokens_sum)

(``completed`` / token sums over the requests that finished inside the measured window;
a sender may return only ``(completed, window_s)``, then the nominal lengths are used.)

* ``--sender http`` (``make_http_sender``): wraps the calibration chat sender,
  :func:`scripts.r3_grid.drive_cell` - ``concurrency`` closed-loop workers sending
  ``/v1/chat/completions`` with ``ignore_eos``, ``temperature 0`` and ``max_tokens = out``
  (``tre_replayer.engine.api.request_body``), each request with its own natural prompt of
  ``in`` tokens after the chat template, streamed with usage; token counts are the
  engine's usage counts. Needs ``--gateway-url`` and ``--i-have-user-approval``.
* ``stub``: a synthetic saturating curve (``--dry-run`` and tests); ``module:callable``
  imports another sender.

Output (``--out-dir``): ``velocity.yaml`` in the policy's ``velocity`` param format and
``profile_raw.csv`` with every step. Cost: (9 buckets + prefill) x len(ladder) x step_s
per model (default 10 x 6 x 60 s = 1 h of one GPU per model).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.request import Request, urlopen

import yaml

Sender = Callable[[str, int, int, int, float, float], tuple]

#: Closed-loop concurrency ladder (decision 2026-10-05).
DEFAULT_LADDER = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Step:
    concurrency: int
    tok_s: float
    in_tok_s: float
    completed: int
    window_s: float


def make_stub_sender(vmax_tok_s: float = 10000.0, c_half: float = 2.0, vmax_prefill_tok_s: float = 30000.0) -> Sender:
    """Synthetic saturating curve ``V * c / (c + c_half)`` (tok/s); out == 1 uses the prefill cap."""

    def send(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float, warmup_s: float):
        vmax = vmax_prefill_tok_s if out_tokens <= 1 else vmax_tok_s
        rate = vmax * concurrency / (concurrency + c_half)
        window = max(step_s - warmup_s, 0.0)
        n = int(rate * window / (in_tokens + out_tokens))
        return n, window, n * in_tokens, n * out_tokens

    return send


# ------------------------------------------------------------------ the HTTP sender


def measure_window(records: Sequence[dict], start_ms: int, end_ms: int, warmup_s: float,
                   in_tokens: int, out_tokens: int) -> tuple[int, float, int, int]:
    """(completed, window_s, input tokens, output tokens) of the requests that finished
    with HTTP 200 and no stream error inside ``[start + warmup, end]`` (raw records of
    ``r3_grid.drive_cell``). Usage counts are used; a missing count falls back to the
    nominal length."""
    lo = start_ms + int(warmup_s * 1000)
    window_s = max(0.0, (end_ms - lo) / 1000.0)
    completed = tin = tout = 0
    for rec in records:
        done = rec.get("done_ts_ms")
        if rec.get("http_status") != 200 or rec.get("stream_error") or done is None or not lo <= done <= end_ms:
            continue
        completed += 1
        tin += int(rec["input_tokens"]) if rec.get("input_tokens") is not None else in_tokens
        tout += int(rec["output_tokens"]) if rec.get("output_tokens") is not None else out_tokens
    return completed, window_s, tin, tout


def make_http_sender(gateway_url: str, *, api: str = "chat", prompt_mode: str = "natural",
                     routing_strategy: Optional[str] = None, run_key: str = "tokenscale",
                     stream_call: Optional[Callable] = None, raw_dir: Optional[Path] = None) -> Sender:
    """One step = :func:`scripts.r3_grid.drive_cell` (the calibration chat sender) for
    ``step_s`` with ``concurrency`` closed-loop workers; raw per-request records go to
    ``raw_dir`` (a temporary directory when None)."""
    from scripts import r3_grid  # lazy: tre/deploy on PYTHONPATH (as for the calibration tools)

    def send(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float, warmup_s: float):
        cell = r3_grid.GridCell(int(in_tokens), int(out_tokens), int(concurrency))
        with tempfile.TemporaryDirectory(prefix="ts-profile-") as tmp:
            base = Path(raw_dir) if raw_dir is not None else Path(tmp)
            base.mkdir(parents=True, exist_ok=True)
            raw = base / f"{model}_{cell.scenario_id}.jsonl"
            raw.unlink(missing_ok=True)
            start_ms, end_ms = r3_grid.drive_cell(
                gateway_url, model, cell, step_s, raw_path=raw, stream_call=stream_call,
                prompt_mode=prompt_mode, routing_strategy=routing_strategy,
                run_key=f"{run_key}|{model}", api=api,
            )
            records = [json.loads(ln) for ln in raw.read_text(encoding="utf-8").splitlines() if ln.strip()] \
                if raw.exists() else []
        return measure_window(records, start_ms, end_ms, warmup_s, int(in_tokens), int(out_tokens))

    return send


def load_sender(spec: str, **http_kwargs: Any) -> Sender:
    if spec == "stub":
        return make_stub_sender()
    if spec == "http":
        url = http_kwargs.pop("gateway_url", None)
        if not url:
            raise ValueError("--sender http needs --gateway-url")
        return make_http_sender(url, **http_kwargs)
    if ":" not in spec:
        raise ValueError(f"--sender must be 'http', 'stub' or 'module:callable', got {spec!r}")
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def check_one_awake(sm_url: str, model: str, timeout_s: float = 5.0) -> int:
    """Awake replicas of ``model`` per the SM's ``GET /v2/state`` (read only)."""
    with urlopen(Request(sm_url.rstrip("/") + "/v2/state", headers={"accept": "application/json"}),
                 timeout=timeout_s) as resp:
        state = json.loads(resp.read().decode("utf-8"))
    return int(((state.get("models") or {}).get(model) or {}).get("awake", 0))


# ------------------------------------------------------------------ the ladder


def _unpack(result: tuple, in_tokens: int, out_tokens: int) -> tuple[int, float, int, int]:
    if len(result) >= 4:
        return int(result[0]), float(result[1]), int(result[2]), int(result[3])
    completed, window = int(result[0]), float(result[1])
    return completed, window, completed * in_tokens, completed * out_tokens


def profile_point(
    sender: Sender, model: str, in_tokens: int, out_tokens: int, step_s: float, warmup_s: float,
    ladder: Sequence[int] = DEFAULT_LADDER,
) -> tuple[float, list[Step]]:
    """Run every concurrency of the ladder; returns (peak (in+out) tok/s, all steps)."""
    steps: list[Step] = []
    for c in ladder:
        completed, window, tin, tout = _unpack(sender(model, in_tokens, out_tokens, int(c), step_s, warmup_s),
                                               in_tokens, out_tokens)
        tok_s = (tin + tout) / window if window > 0 else 0.0
        in_tok_s = tin / window if window > 0 else 0.0
        steps.append(Step(int(c), tok_s, in_tok_s, completed, window))
    return max(s.tok_s for s in steps), steps


def estimate_gpu_time_s(n_points: int, step_s: float, ladder: Sequence[int] = DEFAULT_LADDER) -> float:
    return n_points * len(ladder) * step_s


def _rows(model: str, kind: str, bucket: str, tin: int, tout: int, steps: list[Step]) -> list[dict[str, Any]]:
    return [dict(model=model, kind=kind, bucket=bucket, in_tokens=tin, out_tokens=tout, concurrency=s.concurrency,
                 tok_s=round(s.tok_s, 2), in_tok_s=round(s.in_tok_s, 2), completed=s.completed,
                 window_s=s.window_s) for s in steps]


def run_profile(
    sender: Sender, model: str, centers: list[list[Optional[list[int]]]], step_s: float = 60.0,
    warmup_s: float = 15.0, ladder: Sequence[int] = DEFAULT_LADDER, prefill_in: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grid: list[list[float]] = []
    for i in range(3):
        grid.append([])
        for j in range(3):
            center = centers[i][j]
            if center is None:
                raise ValueError(f"bucket ({i},{j}) has no center (no requests fell in it); cannot profile")
            v, steps = profile_point(sender, model, int(center[0]), int(center[1]), step_s, warmup_s, ladder)
            grid[i].append(round(v, 1))
            rows += _rows(model, "bucket", f"{i}{j}", int(center[0]), int(center[1]), steps)
    if prefill_in is None:
        mid = centers[1][1]
        if mid is None:
            raise ValueError("no median input length; pass --prefill-in")
        prefill_in = int(mid[0])
    _v, steps = profile_point(sender, model, int(prefill_in), 1, step_s, warmup_s, ladder)
    # out == 1: V_P is the prefill velocity in input tokens/s.
    vp_in = max(s.in_tok_s for s in steps)
    rows += _rows(model, "prefill", "P", int(prefill_in), 1, steps)
    return {model: {"buckets": grid, "v_prefill": round(vp_in, 1)}}, rows


def _load_centers(path: str, model: str) -> tuple[list[list[Optional[list[int]]]], Optional[int]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    centers = (data or {}).get("bucket_centers") if isinstance(data, dict) else None
    if not isinstance(centers, dict):
        raise ValueError(f"{path}: no 'bucket_centers' mapping (output of tokenscale_buckets)")
    c = centers.get(model, centers.get("*"))
    if c is None:
        raise ValueError(f"{path}: no centers for model {model!r} (have {sorted(centers)})")
    medians = (data or {}).get("median_in") or {}
    median = medians.get(model, medians.get("*")) if isinstance(medians, dict) else None
    return c, (int(median) if median is not None else None)


def _ladder(text: str) -> tuple[int, ...]:
    try:
        ladder = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not ladder or any(c < 1 for c in ladder):
        raise argparse.ArgumentTypeError("concurrencies must be >= 1")
    return ladder


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TokenScale-colocated velocity profiling (one awake replica)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--centers", required=True, help="YAML from tokenscale_buckets (bucket_centers, median_in)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sender", default=None, help="'http', 'stub' or module:callable")
    ap.add_argument("--gateway-url", default=None, help="http sender: gateway base URL (from config, not code)")
    ap.add_argument("--api", default="chat", help="http sender: chat (default, as the experiments) | completions")
    ap.add_argument("--routing-strategy", default=None, help="http sender: routing-strategy header")
    ap.add_argument("--raw-dir", default=None, help="http sender: keep per-request raw JSONL here")
    ap.add_argument("--sm-url", default=None, help="check via GET /v2/state that exactly one replica is awake")
    ap.add_argument("--dry-run", action="store_true", help="use the stub sender; nothing is sent")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required to send real requests")
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=15.0)
    ap.add_argument("--concurrency", type=_ladder, default=DEFAULT_LADDER, help="ladder, e.g. 1,2,4,8,16,32")
    ap.add_argument("--prefill-in", type=int, default=None, help="V_P input length (default: median_in)")
    args = ap.parse_args(argv)
    if args.warmup_s >= args.step_s:
        print("error: --warmup-s must be < --step-s", file=sys.stderr)
        return 2
    try:
        centers, median_in = _load_centers(args.centers, args.model)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    gpu_s = estimate_gpu_time_s(10, args.step_s, args.concurrency)
    print(f"estimated GPU time: {gpu_s / 60:.0f} min (9 buckets + prefill, {len(args.concurrency)} steps "
          f"of {args.step_s:.0f}s each, concurrency {list(args.concurrency)})")
    if args.dry_run:
        sender: Sender = make_stub_sender()
    else:
        if not args.i_have_user_approval:
            print("refusing to send requests without --i-have-user-approval (use --dry-run to test)", file=sys.stderr)
            return 3
        if not args.sender:
            print("error: --sender is required for a real run (http | module:callable)", file=sys.stderr)
            return 2
        if args.sm_url:
            awake = check_one_awake(args.sm_url, args.model)
            if awake != 1:
                print(f"error: {args.model} has {awake} awake replicas; profile with exactly one", file=sys.stderr)
                return 4
        try:
            sender = load_sender(args.sender, gateway_url=args.gateway_url, api=args.api,
                                 routing_strategy=args.routing_strategy,
                                 raw_dir=Path(args.raw_dir) if args.raw_dir else None)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    prefill_in = args.prefill_in if args.prefill_in is not None else median_in
    try:
        velocity, rows = run_profile(sender, args.model, centers, args.step_s, args.warmup_s, args.concurrency,
                                     prefill_in)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"velocity": velocity}, sort_keys=False, default_flow_style=None)
    (out / "velocity.yaml").write_text(text, encoding="utf-8")
    with (out / "profile_raw.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
