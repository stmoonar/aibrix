"""TokenScale offline token-velocity profiling (SS IV-B2). DRY-RUN ONLY in this round.

For each bucket center (in, out) the concurrency is stepped 1, 2, 4, 8, ...; each step
lasts ``--step-s`` (default 60) of which the first ``--warmup-s`` (15) are dropped; the
measured rate is (in+out) tokens/s. Profiling stops when two consecutive steps each improve
by < 3 % over their predecessor; ``V_b`` is the peak rate. ``V_P`` is measured the same
way with ``out = 1`` at ``--prefill-in`` tokens (default: the median input center).

A *sender* is an injected callable::

    sender(model, in_tokens, out_tokens, concurrency, step_s, warmup_s) -> (completed, window_s)

``completed`` = requests finished inside the measured window (after warmup), ``window_s`` =
that window's length. ``--sender module:callable`` imports one; ``stub`` is a synthetic
saturating curve (used by ``--dry-run`` and tests). Sending real requests needs
``--i-have-user-approval``.

TODO(real sender): no HTTP sender is implemented here. Reuse the calibration chat sender
(chat completions with ignore_eos + temperature 0) from branch
``feat/calib-chat-sender-20260930``, ``tre/replayer/tre_replayer/engine/api.py``, wrapped
into the callable above.

Output (``--out-dir``): ``velocity.yaml`` in the policy's ``velocity`` param format and
``profile_raw.csv`` with every step.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

Sender = Callable[[str, int, int, int, float, float], "tuple[int, float]"]

STOP_IMPROVEMENT = 0.03  # paper: not given; ours
STOP_CONSECUTIVE = 2


@dataclass(frozen=True)
class Step:
    concurrency: int
    tok_s: float
    completed: int
    window_s: float


def make_stub_sender(vmax_tok_s: float = 10000.0, c_half: float = 2.0, vmax_prefill_tok_s: float = 30000.0) -> Sender:
    """Synthetic saturating curve ``V * c / (c + c_half)`` (tok/s); out == 1 uses the prefill cap."""

    def send(model: str, in_tokens: int, out_tokens: int, concurrency: int, step_s: float, warmup_s: float):
        vmax = vmax_prefill_tok_s if out_tokens <= 1 else vmax_tok_s
        rate = vmax * concurrency / (concurrency + c_half)
        window = max(step_s - warmup_s, 0.0)
        return int(rate * window / (in_tokens + out_tokens)), window

    return send


def load_sender(spec: str) -> Sender:
    if spec == "stub":
        return make_stub_sender()
    if ":" not in spec:
        raise ValueError(f"--sender must be 'stub' or 'module:callable', got {spec!r}")
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def profile_point(
    sender: Sender, model: str, in_tokens: int, out_tokens: int, step_s: float, warmup_s: float,
    max_concurrency: int = 256,
) -> tuple[float, list[Step]]:
    """Step concurrency until saturation; returns (peak tok/s, all steps)."""
    steps: list[Step] = []
    flat = 0
    c = 1
    while c <= max_concurrency:
        completed, window = sender(model, in_tokens, out_tokens, c, step_s, warmup_s)
        tok_s = completed * (in_tokens + out_tokens) / window if window > 0 else 0.0
        if steps:
            prev = steps[-1].tok_s
            flat = flat + 1 if (prev <= 0 or (tok_s - prev) / prev < STOP_IMPROVEMENT) else 0
        steps.append(Step(c, tok_s, completed, window))
        if flat >= STOP_CONSECUTIVE:
            break
        c *= 2
    return max(s.tok_s for s in steps), steps


def n_steps_max(max_concurrency: int) -> int:
    n, c = 0, 1
    while c <= max_concurrency:
        n += 1
        c *= 2
    return n


def estimate_gpu_time_s(n_points: int, step_s: float, max_concurrency: int) -> tuple[float, float]:
    """(typical, worst case) seconds of GPU time: typical ~ 2/3 of the worst-case steps."""
    worst = n_points * n_steps_max(max_concurrency) * step_s
    return worst * 2 / 3, worst


def run_profile(
    sender: Sender, model: str, centers: list[list[Optional[list[int]]]], step_s: float = 60.0,
    warmup_s: float = 15.0, max_concurrency: int = 256, prefill_in: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    grid: list[list[float]] = []
    for i in range(3):
        grid.append([])
        for j in range(3):
            center = centers[i][j]
            if center is None:
                raise ValueError(f"bucket ({i},{j}) has no center (no requests fell in it); cannot profile")
            v, steps = profile_point(sender, model, int(center[0]), int(center[1]), step_s, warmup_s, max_concurrency)
            grid[i].append(round(v, 1))
            rows += [dict(model=model, kind="bucket", bucket=f"{i}{j}", in_tokens=center[0], out_tokens=center[1],
                          concurrency=s.concurrency, tok_s=round(s.tok_s, 2), completed=s.completed, window_s=s.window_s)
                     for s in steps]
    if prefill_in is None:
        mid = centers[1][1]
        if mid is None:
            raise ValueError("no median input center; pass --prefill-in")
        prefill_in = int(mid[0])
    vp, steps = profile_point(sender, model, prefill_in, 1, step_s, warmup_s, max_concurrency)
    # out == 1: (in+out) tok/s ~ input tok/s; report input tokens/s.
    vp_in = max(s.completed * prefill_in / s.window_s for s in steps if s.window_s > 0)
    rows += [dict(model=model, kind="prefill", bucket="P", in_tokens=prefill_in, out_tokens=1, concurrency=s.concurrency,
                  tok_s=round(s.tok_s, 2), completed=s.completed, window_s=s.window_s) for s in steps]
    return {model: {"buckets": grid, "v_prefill": round(vp_in, 1)}}, rows


def _load_centers(path: str, model: str) -> list[list[Optional[list[int]]]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    centers = (data or {}).get("bucket_centers") if isinstance(data, dict) else None
    if not isinstance(centers, dict):
        raise ValueError(f"{path}: no 'bucket_centers' mapping (output of tokenscale_buckets)")
    c = centers.get(model, centers.get("*"))
    if c is None:
        raise ValueError(f"{path}: no centers for model {model!r} (have {sorted(centers)})")
    return c


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TokenScale velocity profiling (dry-run only this round)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--centers", required=True, help="YAML from tokenscale_buckets (bucket_centers)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sender", default=None, help="'stub' or module:callable")
    ap.add_argument("--dry-run", action="store_true", help="use the stub sender; nothing is sent")
    ap.add_argument("--i-have-user-approval", action="store_true", help="required to send real requests")
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=15.0)
    ap.add_argument("--max-concurrency", type=int, default=256)
    ap.add_argument("--prefill-in", type=int, default=None)
    args = ap.parse_args(argv)
    if args.warmup_s >= args.step_s:
        print("error: --warmup-s must be < --step-s", file=sys.stderr)
        return 2
    try:
        centers = _load_centers(args.centers, args.model)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    typ, worst = estimate_gpu_time_s(10, args.step_s, args.max_concurrency)
    print(f"estimated GPU time: typical ~{typ / 60:.0f} min, worst case {worst / 60:.0f} min "
          f"(9 buckets + prefill, <= {n_steps_max(args.max_concurrency)} steps of {args.step_s:.0f}s each)")
    if args.dry_run:
        sender: Sender = make_stub_sender()
    else:
        if not args.i_have_user_approval:
            print("refusing to send requests without --i-have-user-approval (use --dry-run to test)", file=sys.stderr)
            return 3
        if not args.sender:
            print("error: --sender is required for a real run", file=sys.stderr)
            return 2
        sender = load_sender(args.sender)
    try:
        velocity, rows = run_profile(sender, args.model, centers, args.step_s, args.warmup_s, args.max_concurrency, args.prefill_in)
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
