"""TokenScale baseline (Token-Velocity autoscaler), collapsed to colocated prefill+decode.

Source: TokenScale SS III-B (token velocity), SS IV-B2 (offline profiling), SS IV-C
(scaler), Table II (3x3 request buckets). The paper scales separate prefiller and decoder
pools; TRE serves colocated replicas, so both terms are computed and the larger one wins::

    desired = ceil(max( sum_b lambda_b / V_b ,  lambda_in / V_P ))

* ``lambda_b``: (input + output) token arrival rate of bucket ``b`` over a sliding window;
* ``V_b``: profiled (in+out) token velocity of bucket ``b`` (paper: decode velocity);
* ``lambda_in`` / ``V_P``: input token arrival rate over all buckets / profiled prefill
  velocity (paper: prefiller autoscaler, ``I^P = lambda / min(V_P, V_BW)``; V_BW is dropped
  because nothing is transferred between pods here).

Scale-down is immediate (paper: no hysteresis); the shell clamps ``desired``.

Input: ``ModelSnapshot.events`` of kind ``arr`` (fields ``in_tokens``, ``max_tokens``,
``in_src``, ``reissue``, ``ts_ms``); other kinds are ignored.

Parameters (``config.policy_params``); "paper" = given by the paper, "ours" = our choice
(``# not in paper``):

===========================  ==========  =================================================
key                          default     meaning
===========================  ==========  =================================================
``velocity``                 required    ``{model: {buckets: 3x3 (in+out) tok/s, v_prefill:
                                         input tok/s}}``; missing / null / <= 0 -> the
                                         factory raises (fail closed). paper (SS IV-B2).
``bucket_edges``             required    ``{model | "*": {in: [e1, e2], out: [e1, e2]}}``;
                                         value x -> index 0 if x <= e1, 1 if x <= e2, else
                                         2. Tertiles from ``tools/tokenscale_buckets.py``
                                         (paper Table II uses fixed edges). ours
``models``                   None        restrict the managed models (others: no Decision);
                                         default = every model in ``config.models``.
``out_len_source``           max_tokens  only ``max_tokens`` is implemented. ours (paper
                                         uses a length predictor)
``default_out_tokens``       256         out_len when an event lacks ``max_tokens`` (counted
                                         as ``missing_out``). ours
``misbucket_rate``           0.15        P(output bucket replaced by a uniformly random
                                         different one); paper evaluates with an 85 %
                                         accurate length predictor.
``seed``                     config.seed RNG seed (``random.Random``). ours
``skip_reissue``             true        ignore events with ``reissue != "none"``. ours
``window_s``                 10          lambda window over ``ts_ms``. ours (not in paper)
``stale_s``                  30          stream considered stalled when the newest ``arr``
                                         is older than this while the pods are busy. ours
``max_estimate_frac``        0.05        degraded when the window's share of
                                         ``in_src == "estimate"`` exceeds it. ours
===========================  ==========  =================================================

Degraded (``desired = awake``, reason ``degraded_<why>``):

* ``stale_events``: newest ``arr`` older than ``stale_s``, the model has awake pods, events
  were seen before, and its pods still report running/waiting requests (traffic exists but
  the event stream stopped). Without busy pods it is plain idleness: ``desired`` 0 (the
  shell clamps to min), reason ``idle``. Judgement call: events alone cannot tell "stream
  broken" from "trace ended".
* ``estimate_frac``: too many token counts are estimates (inaccurate for Chinese text).
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping, Optional

from tre_baselines.policies.base import Decision
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot

N_IN = 3
N_OUT = 3


@dataclass(frozen=True)
class _Ev:
    ts_ms: int
    in_tokens: int
    out_len: int
    b_in: int
    b_out: int
    estimate: bool


def _bucket_index(x: float, edges: tuple[float, float]) -> int:
    if x <= edges[0]:
        return 0
    if x <= edges[1]:
        return 1
    return 2


def _pos(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _parse_velocity(model: str, raw: Any, problems: list[str]) -> Optional[tuple[tuple[tuple[float, ...], ...], float]]:
    if not isinstance(raw, Mapping):
        problems.append(f"velocity[{model}] missing")
        return None
    ok = True
    grid = raw.get("buckets")
    rows: list[tuple[float, ...]] = []
    if not isinstance(grid, (list, tuple)) or len(grid) != N_IN:
        problems.append(f"velocity[{model}].buckets must be a {N_IN}x{N_OUT} list")
        ok = False
    else:
        for i in range(N_IN):
            row = grid[i]
            if not isinstance(row, (list, tuple)) or len(row) != N_OUT:
                problems.append(f"velocity[{model}].buckets[{i}] must have {N_OUT} entries")
                ok = False
                continue
            for j in range(N_OUT):
                if not _pos(row[j]):
                    problems.append(f"velocity[{model}].buckets[{i}][{j}]={row[j]!r} must be > 0")
                    ok = False
            rows.append(tuple(float(v) if _pos(v) else 0.0 for v in row))
    vp = raw.get("v_prefill")
    if not _pos(vp):
        problems.append(f"velocity[{model}].v_prefill={vp!r} must be > 0")
        ok = False
    if not ok:
        return None
    return tuple(rows), float(vp)


def _parse_edges(model: str, raw: Any, problems: list[str]) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if not isinstance(raw, Mapping):
        problems.append(f"bucket_edges[{model}] (or bucket_edges['*']) missing")
        return None
    out = []
    for key in ("in", "out"):
        e = raw.get(key)
        if (
            not isinstance(e, (list, tuple)) or len(e) != 2
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in e)
            or e[0] > e[1]
        ):
            problems.append(f"bucket_edges[{model}].{key} must be two ascending numbers, got {e!r}")
            return None
        out.append((float(e[0]), float(e[1])))
    return out[0], out[1]


class TokenScalePolicy:
    name = "tokenscale"

    def __init__(self, config: Any) -> None:
        p: Mapping[str, Any] = dict(getattr(config, "policy_params", None) or {})
        self.window_s = float(p.get("window_s", 10.0))  # not in paper
        self.stale_s = float(p.get("stale_s", 30.0))  # not in paper
        self.max_estimate_frac = float(p.get("max_estimate_frac", 0.05))  # not in paper
        self.misbucket_rate = float(p.get("misbucket_rate", 0.15))  # paper: 85 % predictor
        self.skip_reissue = bool(p.get("skip_reissue", True))
        self.default_out_tokens = int(p.get("default_out_tokens", 256))  # not in paper
        self.out_len_source = str(p.get("out_len_source", "max_tokens"))
        problems: list[str] = []
        if self.out_len_source != "max_tokens":
            problems.append(f"out_len_source={self.out_len_source!r}: only 'max_tokens' is implemented")
        if self.window_s <= 0:
            problems.append("window_s must be > 0")
        if not 0.0 <= self.misbucket_rate <= 1.0:
            problems.append("misbucket_rate must be in [0, 1]")
        seed = p.get("seed")
        self._rng = random.Random(int(seed) if seed is not None else int(getattr(config, "seed", 0)))

        restrict = p.get("models")
        cfg_models = getattr(config, "models", None) or {}
        velocity = p.get("velocity") or {}
        if restrict:
            managed = [str(m) for m in restrict]
        elif cfg_models:
            managed = list(cfg_models)
        else:
            managed = list(velocity)
        if not managed:
            problems.append("no managed models (set 'models' or provide velocity)")
        edges_all = p.get("bucket_edges") or {}
        self._vel: dict[str, tuple[tuple[tuple[float, ...], ...], float]] = {}
        self._edges: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        for m in managed:
            v = _parse_velocity(m, velocity.get(m) if isinstance(velocity, Mapping) else None, problems)
            e = _parse_edges(m, edges_all.get(m, edges_all.get("*")) if isinstance(edges_all, Mapping) else None, problems)
            if v is not None:
                self._vel[m] = v
            if e is not None:
                self._edges[m] = e
        if problems:
            raise ValueError("tokenscale policy cannot start:\n  " + "\n  ".join(problems))
        self._managed = frozenset(managed)
        self._win: dict[str, Deque[_Ev]] = {m: deque() for m in managed}
        self._last_arr_ms: dict[str, int] = {}
        self._skipped_reissue: dict[str, int] = {m: 0 for m in managed}
        self._skipped_missing_in: dict[str, int] = {m: 0 for m in managed}

    def _ingest(self, model: str, ms: ModelSnapshot) -> tuple[int, int, int]:
        """Add new arrivals; returns (new_events, missing_out, misbucketed) of this tick."""
        edges_in, edges_out = self._edges[model]
        win = self._win[model]
        n_new = n_missing = n_mis = 0
        for ev in ms.events:
            if ev.kind != "arr":
                continue
            if ev.ts_ms > self._last_arr_ms.get(model, -1):
                self._last_arr_ms[model] = ev.ts_ms
            if self.skip_reissue and ev.reissue != "none":
                self._skipped_reissue[model] += 1
                continue
            if ev.in_tokens is None:
                self._skipped_missing_in[model] += 1
                continue
            if ev.max_tokens is None:
                out_len = self.default_out_tokens
                n_missing += 1
            else:
                out_len = int(ev.max_tokens)
            b_in = _bucket_index(ev.in_tokens, edges_in)
            b_out = _bucket_index(out_len, edges_out)
            if self._rng.random() < self.misbucket_rate:
                b_out = self._rng.choice([k for k in range(N_OUT) if k != b_out])
                n_mis += 1
            win.append(_Ev(ev.ts_ms, int(ev.in_tokens), out_len, b_in, b_out, ev.in_src == "estimate"))
            n_new += 1
        return n_new, n_missing, n_mis

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        out: dict[str, Decision] = {}
        for model, ms in snap.models.items():
            if model not in self._managed:
                # Restricted via ``models`` (or absent from the registry): not ours.
                continue
            out[model] = self._decide_model(snap.now_ms, model, ms)
        return out

    def _decide_model(self, now_ms: int, model: str, ms: ModelSnapshot) -> Decision:
        n_new, n_missing, n_mis = self._ingest(model, ms)
        win = self._win[model]
        cutoff = now_ms - self.window_s * 1000.0
        while win and win[0].ts_ms <= cutoff:
            win.popleft()
        grid, v_p = self._vel[model]
        lam = [[0.0] * N_OUT for _ in range(N_IN)]
        in_sum = 0
        n_est = 0
        for e in win:
            lam[e.b_in][e.b_out] += e.in_tokens + e.out_len
            in_sum += e.in_tokens
            n_est += e.estimate
        w = self.window_s
        lam = [[x / w for x in row] for row in lam]
        lam_in = in_sum / w
        term_b = sum(lam[i][j] / grid[i][j] for i in range(N_IN) for j in range(N_OUT))
        term_p = lam_in / v_p
        est_frac = (n_est / len(win)) if win else 0.0
        inputs: dict[str, Any] = {
            "lambda_b": {f"{i}{j}": round(lam[i][j], 3) for i in range(N_IN) for j in range(N_OUT)},
            "lambda_in": round(lam_in, 3),
            "term_buckets": round(term_b, 4),
            "term_prefill": round(term_p, 4),
            "window_s": w,
            "events": len(win),
            "new_events": n_new,
            "skipped_reissue": self._skipped_reissue[model],
            "skipped_missing_in": self._skipped_missing_in[model],
            "missing_out": n_missing,
            "misbucketed": n_mis,
            "estimate_frac": round(est_frac, 4),
            "awake": ms.awake,
        }
        last = self._last_arr_ms.get(model)
        if last is not None and ms.awake > 0 and now_ms - last > self.stale_s * 1000.0:
            if any((p.running + p.waiting) > 0 for p in ms.pods):
                inputs["newest_event_age_s"] = round((now_ms - last) / 1000.0, 1)
                return Decision(ms.awake, "degraded_stale_events", inputs)
        if est_frac > self.max_estimate_frac:
            return Decision(ms.awake, "degraded_estimate_frac", inputs)
        if not win:
            return Decision(0, "idle", inputs)
        desired = math.ceil(max(term_b, term_p) - 1e-9)
        return Decision(max(int(desired), 0), "velocity", inputs)
