"""TokenScale-colocated baseline (Token-Velocity autoscaler on colocated prefill+decode).

Source: TokenScale SS III-B (token velocity), SS IV-B2 (offline profiling), SS IV-C
(scaler), Table II (3x3 request buckets). The paper scales separate prefiller and decoder
pools; TRE serves colocated replicas, so one velocity per bucket counts both phases::

    desired = ceil( sum_b lambda'_b / V_b )

* buckets: the paper's boundaries (TokenScale Table II / DynamoLLM Table IV), input
  ``< 256`` / ``< 1024`` / else (an input above 8192 is long too), output ``< 100`` / ``< 350``
  / ``>= 350`` (``bucket_edges`` as inclusive integer upper bounds ``in [255, 1023]``,
  ``out [99, 349]``; decision 2026-10-26);
* ``lambda'_b``: (input + output) token arrival rate of bucket ``b`` over a sliding window,
  where the output part is the **represented output** of the bucket the request lands in
  after the simulated length-prediction error (the output of that bucket's calibration point
  ``bucket_centers``), not the request's true length (v1 D47 / D66: the output is a
  prediction at arrival in the paper); the input part is the true counted input;
* ``V_b``: the profiled (in+out) token velocity of bucket ``b`` at its calibration point,
  the peak of full steps (v1 D53, ``tools/velocity_ladder``). There is no prefill term:
  ``lambda_in / V_P`` won none of ~3100 v1 decisions and was dropped (v1 D51).
* misbucketing: with ``misbucket_rate`` (0.15, the paper's 85 %-accurate predictor) a
  request's output bucket is replaced by a uniformly chosen other one; the draw is a hash of
  ``(seed, model, client request id)`` (:mod:`tre_baselines.draws`), the same in every arm
  and on every reissue (v1 ``buckets.misbucket``).

Scale-down is immediate (paper: no hysteresis; a sleep is cheap here and the crowd-out
cost is reported as a result); the shell clamps ``desired``. Unknown is not idle:

* a scale-down needs complete evidence (:func:`~tre_baselines.snapshot.evidence_gaps`
  with events, including ``window_s`` of gap-free event history, so a window that started
  before the shell or a stream gap is never read as low load); otherwise ``desired`` is
  held at ``awake`` (reason ``incomplete``). A scale-up uses the window as it is.
* an empty window is not "0 replicas needed" by itself: ``desired`` 0 (reason ``idle``;
  the shell clamps to the floor) only when the evidence is complete **and** every pod
  reports nothing running or waiting; otherwise it holds (``incomplete`` /
  ``empty_window_busy``).

Input: ``ModelSnapshot.events`` of kind ``arr`` (fields ``in_tokens``, ``max_tokens``,
``in_src``, ``reissue``, ``client_id``, ``ts_ms``); other kinds are ignored.

Parameters (``config.policy_params``); "paper" = given by the paper, "ours" = our choice
(``# not in paper``):

===========================  ==========  =================================================
key                          default     meaning
===========================  ==========  =================================================
``velocity``                 required    ``{model: {buckets: 3x3 (in+out) tok/s}}``; missing
                                         / null / <= 0 -> the factory raises (fail closed).
                                         paper (SS IV-B2). ``v_prefill`` is refused (D51).
``bucket_edges``             required    ``{model | "*": {in: [e1, e2], out: [e1, e2]}}``;
                                         value x -> index 0 if x <= e1, 1 if x <= e2, else
                                         2. Paper boundaries: in [255, 1023], out [99, 349].
``bucket_centers``           required    ``{model | "*": 3x3 of [in, out]}``: each bucket's
                                         calibration point (mean (in, out) of the campaign's
                                         requests in it, ``tools/tokenscale_buckets``); must
                                         lie in its own bucket. Its output is what
                                         ``lambda'`` counts. ours (v1 D49)
``models``                   None        restrict the managed models (others: no Decision);
                                         default = every model in ``config.models``.
``out_len_source``           max_tokens  the true output that picks the bucket before the
                                         simulated error; only ``max_tokens``. ours
``default_out_tokens``       256         out_len when an event lacks ``max_tokens`` (counted
                                         as ``missing_out``). ours
``misbucket_rate``           0.15        P(output bucket replaced by a uniformly random
                                         different one); paper evaluates with an 85 %
                                         accurate length predictor.
``seed``                     config.seed seed of the per-request draws. ours
``skip_reissue``             true        ignore events with ``reissue != "none"``. ours
``window_s``                 30          lambda window over ``ts_ms`` (= TRE TSS window).
                                         ours (not in paper)
``max_estimate_frac``        0.05        degraded when the window's share of
                                         ``in_src == "estimate"`` exceeds it. ours
``calibration``              (none)      ``pending``: the factory refuses (values not frozen)
===========================  ==========  =================================================

Degraded (``desired = awake``, reason ``degraded_estimate_frac``): too many token counts
are estimates (inaccurate for Chinese text). A stalled event stream needs no timer of its
own: requests the engines run but the stream does not show are an ``event_gap``.

Logged per decision (``inputs``): ``lambda_b`` (lambda'), ``term_buckets`` (the sum),
``term_true_out`` (the same sum with each request's true output and true bucket, for
disclosure of the predictor's effect), ``misbucketed`` (this tick), ``no_client_id`` (arrivals
whose draw fell back to the gateway id).
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping, Optional

from tre_baselines.draws import unit
from tre_baselines.policies.base import INCOMPLETE, Decision, hold_if_incomplete, refuse_pending
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, evidence_gaps

N_IN = 3
N_OUT = 3

Grid = tuple[tuple[float, ...], ...]
Centers = tuple[tuple[tuple[float, float], ...], ...]
Edges = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class _Ev:
    ts_ms: int
    in_tokens: int
    #: true output (max_tokens) and its bucket, before the simulated error
    out_len: int
    b_out_true: int
    b_in: int
    #: output bucket after the simulated error (the one lambda' counts in)
    b_out: int
    estimate: bool


def _bucket_index(x: float, edges: tuple[float, float]) -> int:
    if x <= edges[0]:
        return 0
    if x <= edges[1]:
        return 1
    return 2


def misbucket(out_idx: int, rate: float, seed: Any, model: str, key: str) -> tuple[int, bool]:
    """(output bucket after the simulated prediction error, whether it moved); a pure
    function of ``(seed, model, key)`` (v1 ``tre_tokenscale.buckets.misbucket``)."""
    if rate <= 0 or unit(seed, "mis", model, key) >= rate:
        return out_idx, False
    others = [k for k in range(N_OUT) if k != out_idx]
    return others[min(len(others) - 1, int(unit(seed, "pick", model, key) * len(others)))], True


def _pos(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _parse_velocity(model: str, raw: Any, problems: list[str]) -> Optional[Grid]:
    if not isinstance(raw, Mapping):
        problems.append(f"velocity[{model}] missing")
        return None
    ok = True
    if "v_prefill" in raw:
        problems.append(f"velocity[{model}].v_prefill: the prefill term is not used (v1 D51); remove it")
        ok = False
    grid = raw.get("buckets")
    rows: list[tuple[float, ...]] = []
    if not isinstance(grid, (list, tuple)) or len(grid) != N_IN:
        problems.append(f"velocity[{model}].buckets must be a {N_IN}x{N_OUT} list")
        return None
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
    return tuple(rows) if ok else None


def _parse_edges(model: str, raw: Any, problems: list[str]) -> Optional[Edges]:
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


def _parse_centers(model: str, raw: Any, edges: Optional[Edges], problems: list[str]) -> Optional[Centers]:
    """``bucket_centers[model]``: 3x3 of ``[in, out]`` > 0, each inside its own bucket."""
    if not isinstance(raw, (list, tuple)) or len(raw) != N_IN:
        problems.append(f"bucket_centers[{model}] (or bucket_centers['*']) must be a {N_IN}x{N_OUT} grid of [in, out]")
        return None
    rows = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != N_OUT:
            problems.append(f"bucket_centers[{model}][{i}] must have {N_OUT} entries")
            return None
        cells = []
        for j, c in enumerate(row):
            if not isinstance(c, (list, tuple)) or len(c) != 2 or not all(_pos(x) for x in c):
                problems.append(f"bucket_centers[{model}][{i}][{j}] must be [in, out] > 0, got {c!r}")
                return None
            if edges is not None and (_bucket_index(c[0], edges[0]), _bucket_index(c[1], edges[1])) != (i, j):
                problems.append(f"bucket_centers[{model}][{i}][{j}]={list(c)!r} is not in bucket {i}{j} of "
                                f"edges in {list(edges[0])} out {list(edges[1])}")
                return None
            cells.append((float(c[0]), float(c[1])))
        rows.append(tuple(cells))
    return tuple(rows)


def _per_model(raw: Any, model: str) -> Any:
    return raw.get(model, raw.get("*")) if isinstance(raw, Mapping) else None


class TokenScalePolicy:
    name = "tokenscale"
    #: Arm name in decision records and docs (PD-disaggregated policy on colocated replicas).
    label = "TokenScale-colocated"
    needs_events = True

    def __init__(self, config: Any) -> None:
        p: Mapping[str, Any] = dict(getattr(config, "policy_params", None) or {})
        refuse_pending(p, self.name)
        self.window_s = float(p.get("window_s", 30.0))  # not in paper
        #: A scale-down needs this much gap-free event history (the whole window).
        self.event_history_s = self.window_s
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
        self.seed = int(seed) if seed is not None else int(getattr(config, "seed", 0))

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
        centers_all = p.get("bucket_centers") or {}
        self._vel: dict[str, Grid] = {}
        self._edges: dict[str, Edges] = {}
        self._centers: dict[str, Centers] = {}
        for m in managed:
            v = _parse_velocity(m, velocity.get(m) if isinstance(velocity, Mapping) else None, problems)
            e = _parse_edges(m, _per_model(edges_all, m), problems)
            c = _parse_centers(m, _per_model(centers_all, m), e, problems)
            if v is not None:
                self._vel[m] = v
            if e is not None:
                self._edges[m] = e
            if c is not None:
                self._centers[m] = c
        if problems:
            raise ValueError("tokenscale policy cannot start:\n  " + "\n  ".join(problems))
        self._managed = frozenset(managed)
        self._win: dict[str, Deque[_Ev]] = {m: deque() for m in managed}
        self._last_arr_ms: dict[str, int] = {}
        self._skipped_reissue: dict[str, int] = {m: 0 for m in managed}
        self._skipped_missing_in: dict[str, int] = {m: 0 for m in managed}
        #: Arrival times (window) of events without in_tokens: no lambda contribution, but
        #: they count like estimates toward ``max_estimate_frac``.
        self._missing_in: dict[str, Deque[int]] = {m: deque() for m in managed}
        #: Cumulative per-model counters reported by the shell (``counters()``).
        self._counts: dict[str, Counter] = {m: Counter() for m in managed}

    def counters(self) -> Mapping[str, Mapping[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for m in self._managed:
            c = dict(self._counts[m])
            c.update(skipped_reissue=self._skipped_reissue[m], skipped_missing_in=self._skipped_missing_in[m])
            out[m] = c
        return out

    def represented_out(self, model: str, b_in: int, b_out: int) -> float:
        """Output tokens lambda' counts for a request in bucket (b_in, b_out)."""
        return self._centers[model][b_in][b_out][1]

    def _ingest(self, model: str, ms: ModelSnapshot) -> tuple[int, int, int, int]:
        """Add new arrivals; returns (new_events, missing_out, misbucketed, no_client_id) of this tick."""
        edges_in, edges_out = self._edges[model]
        win = self._win[model]
        n_new = n_missing = n_mis = n_no_id = 0
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
                self._missing_in[model].append(int(ev.ts_ms))
                continue
            if ev.max_tokens is None:
                out_len = self.default_out_tokens
                n_missing += 1
            else:
                out_len = int(ev.max_tokens)
            if not ev.client_id:
                n_no_id += 1
                self._counts[model]["no_client_id"] += 1
            b_in = _bucket_index(ev.in_tokens, edges_in)
            b_true = _bucket_index(out_len, edges_out)
            b_out, moved = misbucket(b_true, self.misbucket_rate, self.seed, model, ev.draw_key)
            n_mis += moved
            win.append(_Ev(ev.ts_ms, int(ev.in_tokens), out_len, b_true, b_in, b_out, ev.in_src == "estimate"))
            n_new += 1
        return n_new, n_missing, n_mis, n_no_id

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        out: dict[str, Decision] = {}
        for model, ms in snap.models.items():
            if model not in self._managed:
                # Restricted via ``models`` (or absent from the registry): not ours.
                continue
            out[model] = self._decide_model(snap.now_ms, snap.tick_s, model, ms)
        return out

    def _decide_model(self, now_ms: int, tick_s: float, model: str, ms: ModelSnapshot) -> Decision:
        n_new, n_missing, n_mis, n_no_id = self._ingest(model, ms)
        win = self._win[model]
        cutoff = now_ms - self.window_s * 1000.0
        while win and win[0].ts_ms <= cutoff:
            win.popleft()
        missing_in = self._missing_in[model]
        while missing_in and missing_in[0] <= cutoff:
            missing_in.popleft()
        grid = self._vel[model]
        lam = [[0.0] * N_OUT for _ in range(N_IN)]
        lam_true = [[0.0] * N_OUT for _ in range(N_IN)]
        n_est = n_mis_win = 0
        for e in win:
            lam[e.b_in][e.b_out] += e.in_tokens + self.represented_out(model, e.b_in, e.b_out)
            lam_true[e.b_in][e.b_out_true] += e.in_tokens + e.out_len
            n_est += e.estimate
            n_mis_win += e.b_out != e.b_out_true
        w = self.window_s
        lam = [[x / w for x in row] for row in lam]
        term_b = sum(lam[i][j] / grid[i][j] for i in range(N_IN) for j in range(N_OUT))
        term_true = sum(lam_true[i][j] / w / grid[i][j] for i in range(N_IN) for j in range(N_OUT))
        # an arrival without an input count is no better than an estimate
        n_all = len(win) + len(missing_in)
        est_frac = ((n_est + len(missing_in)) / n_all) if n_all else 0.0
        inputs: dict[str, Any] = {
            "lambda_b": {f"{i}{j}": round(lam[i][j], 3) for i in range(N_IN) for j in range(N_OUT)},
            "term_buckets": round(term_b, 4),
            "term_true_out": round(term_true, 4),
            "window_s": w,
            "events": len(win),
            "events_b": {f"{i}{j}": sum(1 for e in win if (e.b_in, e.b_out) == (i, j))
                         for i in range(N_IN) for j in range(N_OUT)},
            "new_events": n_new,
            "skipped_reissue": self._skipped_reissue[model],
            "skipped_missing_in": self._skipped_missing_in[model],
            "missing_out": n_missing,
            "misbucketed": n_mis,
            "misbucketed_window": n_mis_win,
            "no_client_id": n_no_id,
            "estimate_frac": round(est_frac, 4),
            "awake": ms.awake,
        }
        last = self._last_arr_ms.get(model)
        if last is not None:
            inputs["newest_event_age_s"] = round((now_ms - last) / 1000.0, 1)
        gaps = evidence_gaps(ms, now_ms, tick_s, events=True, history_s=self.window_s)
        if est_frac > self.max_estimate_frac:
            self._counts[model]["degraded_estimate_frac"] += 1
            return Decision(ms.awake, "degraded_estimate_frac", inputs)
        if not win:
            # An empty window alone is unknown, not idle: 0 only on complete evidence with
            # every engine reporting nothing in flight.
            if gaps:
                return Decision(ms.awake, INCOMPLETE, {**inputs, "gaps": list(gaps)})
            if not ms.pods or any(p.queued for p in ms.pods):
                # engines still busy: the state equivalent of the paper's drain before a
                # pod is deleted (decision 2026-10-06); counted and reported
                self._counts[model]["empty_window_busy"] += 1
                return Decision(ms.awake, "empty_window_busy", inputs)
            return Decision(0, "idle", inputs)
        desired = math.ceil(term_b - 1e-9)
        return hold_if_incomplete(Decision(max(int(desired), 0), "velocity", inputs), ms.awake, gaps)
