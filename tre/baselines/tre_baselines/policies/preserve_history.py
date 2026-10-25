"""PreServe Tier-1 history before t0 (v1 redesign 2026-10-24 #8, ported 2026-10-26).

Paper section 5.3: the forecaster is trained on the past and tested on the future, so at the
replay start ``t0`` its input already holds the windows *before* the test trace. The shell
reads that past from a **history file** next to the replayed trace (policy param
``history_path``; default: ``<trace dir>/pre_t0.json`` if it exists, else
``<trace dir>/preserve_history.json``); it never builds the past from the test windows
themselves (v1's 10-22 cyclic prefill did, D18). The file must name the same trace (and seed)
as the trace path ``.../<trace>/seed<k>/<file>``.

Two formats, one meaning (per model, the token sums of the ``window_s`` windows before t0,
oldest first, the last one ``[t0 - W, t0)``):

* ``tre-pre-t0-v1`` (``pre_t0.json``), written by the trace generator for every synthetic
  trace from its spec's ``pre_t0`` rule (``tre/docs/design/20261026-traces-720.md`` section 4)::

    {"format": "tre-pre-t0-v1", "trace", "seed", "rule", "t0_s", "window_s",
     "windows": {"<model>": [{"start_s", "end_s", "requests", "prompt_tokens", "output_tokens"}]},
     "requests": [...]}                       # per request (not read here)

  ``P`` = ``prompt_tokens``, ``D`` = ``output_tokens`` (the sum of ``max_output_tokens``);
  the windows must be contiguous and end at ``t0_s``.
* ``tre-preserve-history-v1`` (``preserve_history.json``), written for ``Real_*`` by
  :mod:`tre_baselines.tools.preserve_history` ``real`` from the real Azure windows right before
  the slice (v1 ``tools/real_history.py`` counts file) times the trace's scale and mean lengths::

    {"format": "tre-preserve-history-v1", "trace", "seed", "window_s",
     "models": {"<model>": [[P, D], ...]}, "source": {...}}

``P`` is the trace's ``prompt_length`` (the observed windows after t0 sum the gateway's counted
input, chat template included: a small offset, disclosed). The shell uses the last ``seq_len``
rows of a model; fewer rows, another ``window_s``, another trace / seed or a negative /
non-finite value refuse the policy (fail closed).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

FORMAT = "tre-preserve-history-v1"
PRE_T0_FORMAT = "tre-pre-t0-v1"
DEFAULT_NAME = "preserve_history.json"
PRE_T0_NAME = "pre_t0.json"


class HistoryError(ValueError):
    pass


def default_path(trace_path: str) -> str:
    """``pre_t0.json`` next to the trace if present (synthetic), else ``preserve_history.json``."""
    d = Path(trace_path).parent
    return str(d / PRE_T0_NAME) if (d / PRE_T0_NAME).exists() else str(d / DEFAULT_NAME)


def _pre_t0_rows(doc: Mapping[str, Any], window_s: float) -> dict[str, list]:
    """``{model: [[P, D], ...]}`` of a ``tre-pre-t0-v1`` document (contiguous, ending at t0)."""
    if not isinstance(doc.get("t0_s"), (int, float)):
        raise HistoryError("history: t0_s missing")
    t0 = float(doc["t0_s"])
    wins = doc.get("windows")
    if not isinstance(wins, Mapping):
        raise HistoryError("history: windows missing")
    out: dict[str, list] = {}
    for m, rows in wins.items():
        if not isinstance(rows, list):
            raise HistoryError(f"history: model {m}: windows must be a list")
        end = t0
        for w in reversed(rows):
            try:
                a, b = float(w["start_s"]), float(w["end_s"])
            except (KeyError, TypeError, ValueError):
                raise HistoryError(f"history: model {m}: bad window {w!r}") from None
            if abs(b - end) > 1e-6 or abs((b - a) - window_s) > 1e-6:
                raise HistoryError(f"history: model {m}: windows not contiguous {window_s} s ending at t0 "
                                   f"(at {w!r})")
            end = a
        out[m] = [[w.get("prompt_tokens"), w.get("output_tokens")] for w in rows]
    return out


def parse(doc: Any, *, window_s: float, models: Sequence[str], seq_len: int,
          trace: Optional[str] = None, seed: Optional[int] = None) -> dict[str, list[tuple[float, float]]]:
    """``{model: [(P, D)] * seq_len}`` (oldest first) of a history document; raises HistoryError.
    ``trace`` / ``seed``: the replayed trace's; a document naming another refuses."""
    fmt = doc.get("format") if isinstance(doc, Mapping) else None
    if fmt not in (FORMAT, PRE_T0_FORMAT):
        raise HistoryError(f"history: format must be {FORMAT!r} or {PRE_T0_FORMAT!r}")
    try:
        w = float(doc["window_s"])
    except (KeyError, TypeError, ValueError):
        raise HistoryError("history: window_s missing") from None
    if abs(w - float(window_s)) > 1e-9:
        raise HistoryError(f"history: window_s {w} != the policy's {window_s}")
    # fail closed: the file must say which trace (and, for a .../seed<k>/ path, which seed) it is for
    if trace is not None and doc.get("trace") != trace:
        raise HistoryError(f"history: for trace {doc.get('trace')!r}, the replayed trace is {trace!r}")
    if seed is not None and (doc.get("seed") is None or int(doc["seed"]) != int(seed)):
        raise HistoryError(f"history: for seed {doc.get('seed')}, the replayed trace is seed {seed}")
    rows_all = _pre_t0_rows(doc, w) if fmt == PRE_T0_FORMAT else doc.get("models")
    if not isinstance(rows_all, Mapping):
        raise HistoryError("history: models missing")
    out: dict[str, list[tuple[float, float]]] = {}
    for m in models:
        rows = rows_all.get(m)
        if not isinstance(rows, list) or len(rows) < seq_len:
            raise HistoryError(f"history: model {m} has {len(rows) if isinstance(rows, list) else 0} windows "
                               f"< seq_len {seq_len}")
        vals: list[tuple[float, float]] = []
        for r in rows[-seq_len:]:
            try:
                p, d = float(r[0]), float(r[1])
            except (TypeError, ValueError, IndexError):
                raise HistoryError(f"history: model {m}: bad row {r!r}") from None
            if not (math.isfinite(p) and math.isfinite(d) and p >= 0 and d >= 0):
                raise HistoryError(f"history: model {m}: bad row {r!r}")
            vals.append((p, d))
        out[m] = vals
    return out


def load(path: str, *, window_s: float, models: Sequence[str], seq_len: int, trace: Optional[str] = None,
         seed: Optional[int] = None) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
    """(history per model, info: path, format, trace, seed, sha256, source / rule) of a history
    file; raises HistoryError."""
    try:
        raw = Path(path).read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise HistoryError(f"history: cannot read {path}: {exc}") from None
    hist = parse(doc, window_s=window_s, models=models, seq_len=seq_len, trace=trace, seed=seed)
    return hist, {"path": str(path), "format": doc.get("format"), "trace": doc.get("trace"),
                  "seed": doc.get("seed"), "sha256": hashlib.sha256(raw).hexdigest(),
                  "source": doc.get("source") or doc.get("rule")}


def seed_of(trace_path: str) -> Optional[int]:
    """``k`` of a ``.../seed<k>/<file>`` trace path, else None."""
    m = re.fullmatch(r"seed(\d+)", Path(trace_path).parent.name)
    return int(m.group(1)) if m else None


def real_counts(counts_doc: Mapping[str, Any], reqs: Sequence[tuple[float, str, float, float]], *,
                window_s: float, duration_s: float, n: Optional[int] = None) -> tuple[dict[str, list[list[float]]], dict]:
    """Real_* history (v1 ``real_counts`` mode): per model the counted requests per window of the
    Azure seconds right before its slice (``counts_doc``, v1 ``tools/real_history.py``, already
    mapped to the slice's units) times ``scale`` x the trace's mean prompt / output length of that
    model. ``scale`` = the trace's mean requests/s of the model / the slice's mean count/s
    (``slice_mean_count_s``): the factor the trace applied to the slice (v2 Real_* = v1 slice x k x
    v1 factor), measured, so the history has the trace's load level. ``reqs`` = (t, model, P, D)."""
    if float(counts_doc.get("window_s", -1)) != float(window_s):
        raise HistoryError(f"counts file window {counts_doc.get('window_s')} != {window_s}")
    out: dict[str, list[list[float]]] = {}
    info: dict[str, Any] = {}
    models = sorted({m for _t, m, _p, _d in reqs})
    for m in models:
        rec = (counts_doc.get("models") or {}).get(m)
        if not rec:
            raise HistoryError(f"counts file has no model {m}")
        counts = [float(c) for c in rec["counts"]]
        k = len(counts) if n is None else int(n)
        if len(counts) < k:
            raise HistoryError(f"{m}: {len(counts)} counted windows < {k}")
        mine = [(p, d) for _t, mm, p, d in reqs if mm == m]
        mp, md = sum(p for p, _ in mine) / len(mine), sum(d for _, d in mine) / len(mine)
        rate = len(mine) / float(duration_s)
        scale = rate / float(rec["slice_mean_count_s"])
        out[m] = [[round(c * scale * mp, 1), round(c * scale * md, 1)] for c in counts[-k:]]
        info[m] = {"scale": round(scale, 5), "trace_rps": round(rate, 4), "slice_mean_count_s": rec["slice_mean_count_s"],
                   "mean_in": round(mp, 1), "mean_out": round(md, 1), "windows": k}
    return out, info
