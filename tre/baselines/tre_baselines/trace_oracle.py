"""PreServe Tier-1 oracle: per-model, per-window token totals of the replayed trace.

PreServe (§4.1, Alg.2) forecasts, per service and per 10 min window, the prompt and the
response token counts ``P`` and ``D`` with an mLSTM. We do not reproduce the forecaster;
the replayed trace itself is the forecast (D8, ``tier1`` modes in
:mod:`tre_baselines.policies.preserve`). This module turns a trace file into
``P_w = sum(in_tokens)`` / ``D_w = sum(max_tokens)`` per model per window, where the window
index of a request is ``floor(arrival_offset_s / window_s)`` and ``arrival_offset_s`` is
relative to the replay start ``ReplayInfo.t0_ms``.

Everything here is a pure function of the trace file (plus a seed); the only I/O is
reading that file once (:func:`load_oracle`, called by the policy factory).

Trace formats
-------------
``segments`` - the replayer's ``trace.json`` (``tre_replayer.traces.loader``): a
    model-keyed object of ``{start_time, end_time, rps, input_tokens|input_tokens_dist,
    max_tokens|max_tokens_dist}`` segments. The replayer does not send the segments, it
    sends a seeded arrival schedule built from them (``tre_replayer.run_trace`` ->
    ``build_poisson_schedule(segments, seed=--seed)``). We rebuild the very same schedule
    with the replayer's own functions, so ``seed`` must be the run's ``--seed``
    (campaign ``RunSpec.seed``); a wrong seed moves arrivals and lengths.
``requests`` - a per-request list (v1 recorded ``traces.json`` / ``*.effective.json``, or a
    list of ``ScheduledRequest``-shaped objects, optionally under a ``"requests"`` key).
    Fields read (first present wins): arrival ``timestamp`` | ``scheduled_offset_s`` |
    ``offset_s`` | ``arrival_s`` (seconds from replay start); model ``model_name`` |
    ``model``; input ``prompt_tokens`` | ``prompt_length`` | ``input_tokens`` |
    ``in_tokens``; output ``max_output_tokens`` | ``max_tokens``.

A request without an input count adds 0 to ``P`` and one without ``max_tokens`` adds 0 to
``D``; both are counted per window (``missing_in`` / ``missing_max``).

Our traces are tens of minutes long, so at the paper's 600 s window a run has only a few
Tier-1 windows. The last one is usually partial; its effective length (the ``W`` the
policy divides by) is ``min(window_s, duration_s - start)``, floored at
``min(window_s, MIN_EFFECTIVE_WINDOW_S)`` so a stray arrival just past a boundary cannot
turn into a huge rate. ``duration_s`` is the segments' end for a segment trace and the
last arrival for a request list.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from tre_baselines.snapshot import ReplayInfo

FORMAT_SEGMENTS = "segments"
FORMAT_REQUESTS = "requests"
SCHEDULE_POISSON = "poisson"  # what tre_replayer.run_trace sends
SCHEDULE_DETERMINISTIC = "deterministic"
SCHEDULES = (SCHEDULE_POISSON, SCHEDULE_DETERMINISTIC)

#: Floor of a partial window's effective length (ours, not in the paper).
MIN_EFFECTIVE_WINDOW_S = 60.0

_OFFSET_KEYS = ("timestamp", "scheduled_offset_s", "offset_s", "arrival_s")
_MODEL_KEYS = ("model_name", "model")
_IN_KEYS = ("prompt_tokens", "prompt_length", "input_tokens", "in_tokens")
_MAX_KEYS = ("max_output_tokens", "max_tokens")


@dataclass(frozen=True)
class TraceRequest:
    offset_s: float
    model: str
    in_tokens: Optional[int]
    max_tokens: Optional[int]


@dataclass(frozen=True)
class WindowTotals:
    P: int = 0
    D: int = 0
    n: int = 0
    missing_in: int = 0
    missing_max: int = 0


def _first(raw: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if raw.get(key) is not None:
            return raw[key]
    return None


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _is_segment_doc(data: Any) -> bool:
    if not isinstance(data, dict) or not data or "requests" in data:
        return False
    for value in data.values():
        if not isinstance(value, list):
            return False
        if value and not (isinstance(value[0], dict) and "start_time" in value[0]):
            return False
    return True


def requests_from_segments(path: str | Path, *, seed: int = 0,
                           schedule: str = SCHEDULE_POISSON) -> tuple[list[TraceRequest], float]:
    """The replayer's own arrival schedule for a segment ``trace.json`` (and its span)."""
    from tre_replayer.engine.schedule import build_deterministic_schedule, build_poisson_schedule
    from tre_replayer.traces.loader import load_trace_segments

    if schedule not in SCHEDULES:
        raise ValueError(f"trace schedule must be one of {SCHEDULES}, got {schedule!r}")
    segments = load_trace_segments(path)
    build = build_poisson_schedule if schedule == SCHEDULE_POISSON else build_deterministic_schedule
    events = build(segments, seed=int(seed))
    out = [
        TraceRequest(float(e.scheduled_offset_s), str(e.model), _opt_int(e.prompt_tokens),
                     _opt_int(e.max_output_tokens))
        for e in events
    ]
    duration = max((float(s.end_s) for s in segments), default=0.0)
    return out, duration


def requests_from_records(records: Iterable[Any]) -> tuple[list[TraceRequest], float, int]:
    """(requests, span, records skipped for lacking an arrival or a model)."""
    out: list[TraceRequest] = []
    skipped = 0
    for raw in records:
        if not isinstance(raw, Mapping):
            skipped += 1
            continue
        offset = _first(raw, _OFFSET_KEYS)
        model = _first(raw, _MODEL_KEYS)
        try:
            offset_f = float(offset)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if model is None or not math.isfinite(offset_f):
            skipped += 1
            continue
        out.append(TraceRequest(offset_f, str(model), _opt_int(_first(raw, _IN_KEYS)),
                                _opt_int(_first(raw, _MAX_KEYS))))
    out.sort(key=lambda r: r.offset_s)
    duration = max((r.offset_s for r in out), default=0.0)
    return out, duration, skipped


def bucket(requests: Iterable[TraceRequest], window_s: float) -> dict[str, dict[int, WindowTotals]]:
    """``{model: {window index: totals}}``; requests before offset 0 are dropped."""
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    acc: dict[str, dict[int, list[int]]] = {}
    for r in requests:
        if r.offset_s < 0:
            continue
        idx = int(math.floor(r.offset_s / window_s))
        row = acc.setdefault(r.model, {}).setdefault(idx, [0, 0, 0, 0, 0])
        row[0] += r.in_tokens or 0
        row[1] += r.max_tokens or 0
        row[2] += 1
        row[3] += r.in_tokens is None
        row[4] += r.max_tokens is None
    return {m: {i: WindowTotals(*v) for i, v in w.items()} for m, w in acc.items()}


def window_index(now_ms: int, replay: ReplayInfo, window_s: float, lead_s: float = 0.0) -> int:
    """Tier-1 window that is (about to be, with ``lead_s``) served at ``now_ms`` (Redis
    clock, like ``t0_ms``). Negative before the replay starts."""
    elapsed_s = (int(now_ms) - int(replay.t0_ms)) / 1000.0 + float(lead_s)
    return int(math.floor(elapsed_s / float(window_s)))


def path_tail(path: str, parts: int) -> tuple[str, ...]:
    pieces = [p for p in str(path).replace("\\", "/").split("/") if p and p != "."]
    return tuple(pieces[-parts:]) if parts > 0 else tuple(pieces)


@dataclass(frozen=True)
class TraceOracle:
    path: str
    fmt: str
    window_s: float
    duration_s: float
    totals: Mapping[str, Mapping[int, WindowTotals]]
    #: Largest max_tokens per model (the default ``max_output_len`` of the look-ahead map).
    max_tokens_max: Mapping[str, int]
    n_requests: int
    #: Windows the trace spans (indices ``0 .. n_windows - 1``).
    n_windows: int
    skipped_records: int = 0

    def window(self, model: str, idx: int) -> WindowTotals:
        return self.totals.get(model, {}).get(int(idx), WindowTotals())

    def window_len_s(self, idx: int) -> float:
        """Effective length of window ``idx`` (the last one may be partial)."""
        start = idx * self.window_s
        floor = min(self.window_s, MIN_EFFECTIVE_WINDOW_S)
        return max(floor, min(self.window_s, self.duration_s - start))

    def matches(self, replay: ReplayInfo, parts: int = 2) -> bool:
        """Whether the replay marker names this trace: equal paths, or equal last
        ``parts`` path components (the campaign host and the shell pod mount the trace at
        different roots; replayer traces are all called ``trace.json``). ``parts`` 0
        disables the check."""
        if parts <= 0:
            return True
        if str(replay.trace_path) == self.path:
            return True
        return path_tail(replay.trace_path, parts) == path_tail(self.path, parts)


def build_oracle(requests: Sequence[TraceRequest], *, path: str, fmt: str, window_s: float,
                 duration_s: float, skipped: int = 0) -> TraceOracle:
    max_tok: dict[str, int] = {}
    for r in requests:
        if r.max_tokens is not None:
            max_tok[r.model] = max(max_tok.get(r.model, 0), int(r.max_tokens))
    if fmt == FORMAT_SEGMENTS:  # end_time is exclusive
        n_windows = int(math.ceil(duration_s / window_s)) if duration_s > 0 else 0
    else:  # the last arrival itself belongs to a window
        n_windows = int(math.floor(duration_s / window_s)) + 1 if requests else 0
    return TraceOracle(
        path=str(path), fmt=fmt, window_s=float(window_s), duration_s=float(duration_s),
        totals=bucket(requests, window_s), max_tokens_max=max_tok, n_requests=len(requests),
        n_windows=n_windows, skipped_records=skipped,
    )


def load_oracle(path: str | Path, *, window_s: float = 600.0, seed: int = 0,
                schedule: str = SCHEDULE_POISSON) -> TraceOracle:
    """Read ``path`` once and bucket it (format auto-detected)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if _is_segment_doc(data):
        reqs, duration = requests_from_segments(p, seed=seed, schedule=schedule)
        return build_oracle(reqs, path=str(path), fmt=FORMAT_SEGMENTS, window_s=window_s,
                            duration_s=duration)
    records = data.get("requests") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"trace {path}: neither a segment trace nor a request list")
    reqs, duration, skipped = requests_from_records(records)
    return build_oracle(reqs, path=str(path), fmt=FORMAT_REQUESTS, window_s=window_s,
                        duration_s=duration, skipped=skipped)
