"""PreServe-oracle baseline, scaling part only (§4.1 Alg.2, §4.3.1, §4.3.2).

Not reproduced: the mLSTM workload forecaster (Tier-1 uses the replayed trace, see
:mod:`preserve_tier1`), the DistilBERT response-length predictor (Tier-2 uses the request's
``max_tokens`` as its predicted output length) and the load-aware router.

Tier-1, once per window of ``window_s`` (paper: 10 min), on the first tick whose
``(replay t0, window index)`` differs from the last one handled (window index
``floor(((now - t0) / 1000 + lead_s) / window_s)``; a new replay marker restarts at window 0
without a shell restart):
``N = ceil(max(P/(mu_p W), D/(mu_d W), (P+D)/(mu_t W)))`` -> target ``N``.

Tier-2, every tick, per awake pod: the load-look-ahead map (:mod:`preserve_anticipator`)
fed by the gateway request events. Of it the scaler reads ``U[0:l]`` only:

* a pod is *overloaded* when more than ``overload_frac`` of ``U[0:l]`` is above ``kv_high``
  -> one extra instance per overloaded pod ("one potentially overloaded instance, one
  additional instance"). A pod is credited once per overload episode (it is credited again
  only after it left overload), so a persisting overload does not add an instance every
  tick; ours, the paper does not say how often the rule is evaluated.
* scale-down, at most once per Tier-1 window (paper), when every pod's
  ``max U[0:l] < T_f``: the paper isolates ``N_c - sum_p max(U'_p) / T_f`` instances, i.e.
  keeps ``sum_p max(U'_p) / T_f`` of them: the load packed onto instances each at ``T_f``.
  We keep ``ceil`` of that (isolate the floor), at least 1, and only act when it is below
  ``N_c = awake``. The map must be complete and agree with the engines (ours; no timer):
  no evidence gap (:func:`~tre_baselines.snapshot.evidence_gaps` with events: every pod
  scraped, gauges present, every request in flight known from the stream since the last
  gap) and, per pod, ``|U[0] - vllm:kv_cache_usage_perc| <= kv_agree_tol``; otherwise the
  scale-down waits (``down: incomplete``) without using up the window's one scale-down.
  The isolated instances are put to sleep through the shared transparent-sleep path
  (abort + sidecar continuation), not drained: that cost is reported, not hidden.

Composition: on a window start Tier-1 sets the target and Tier-2 may only add to it in the
same tick; inside a window only Tier-2 changes it; otherwise the target is held. Any target
below ``awake`` is held at ``awake`` while the evidence is incomplete (reason
``incomplete``; the target itself is kept and applies once the evidence is complete).

Iterations vs wall clock (ours; the paper is iteration-based): a pod's head advances
``dt / TPOT_pod`` iterations, fractional part carried, with ``TPOT_pod`` the mean
inter-token latency between the pod's last two scrapes (``d itl_sum / d itl_count``) or the
model's TPOT SLO when there is no sample. Each event first advances its pod to the event's
Redis time, so a prefill that finished early in the tick lands where it belongs; the tick
then walks each head to ``max(scrape time, last event time)`` (an event written between the
scrape and the stream read is not "out of order").

Output length (ours): the predicted output is the request's ``max_tokens``, which vLLM
enforces, so by default (``out_len_is_upper_bound: true``) there is no virtual extension.
A request leaves the map only on its ``done`` (or when its pod is gone, or when the engine
reports nothing in flight at all, which is a ``done`` for everything there): when the
estimated head passes ``D`` first, the request keeps its full size ``(P + D) / M`` until
then (anomaly ``overdue``). The paper's 0.2 * D extension (``ext_frac``) exists because its
length predictor can under-estimate; it is used only with ``out_len_is_upper_bound: false``.
Non-streaming requests have no ``ft`` before their output is complete, so they are not in
the map (the experiments stream).

Params (``config.policy_params``, example in ``examples/preserve.yaml``):

====================  ====================  ===================================================
key                   default               origin
====================  ====================  ===================================================
trace_path            (required)            ours: the trace the campaign replays (read once)
trace_seed            0                     ours: fallback of the replay's ``--seed``; the marker's
                                            ``seed`` (``snap.replay.seed``) wins (segment traces)
trace_schedule        poisson               ours: what ``tre_replayer.run_trace`` sends
trace_match_parts     2                     ours: replay marker must end in the same N parts
window_s              600                   paper (10 min window)
lead_s                0                     ours: fire Tier-1 this early (s); wake is seconds
tier1                 oracle_noisy          ours (D8): oracle_noisy | oracle | last_window
noise_sigma           ~0.0772               ours: lognormal sigma, mean APE 6.17 % (Table 1)
noise_seed            config.seed           ours
mu                    (required)            paper Alg.1 profile; ``tools/preserve_mu.py``
max_output_len        trace max max_tokens  paper: "maximum output tokens" (e.g. 4096);
                      per model, else 4096  int or {model: int}
map_factor            1.2                   ours: ring length = ceil(max_output_len * 1.2)
lookahead_iters       100                   paper (l)
kv_high               0.95                  paper
overload_frac         0.10                  paper
t_f                   0.30                  paper (T_f)
ext_frac              0.2                   paper (virtual extension); only with
                                            ``out_len_is_upper_bound: false``
out_len_is_upper_bound true                 ours: max_tokens is a hard cap -> no extension
kv_capacity_tokens    {}                    ours: M per model when the pod lacks cache_config
kv_agree_tol          0.15                  ours: scale-down needs |U[0] - engine KV| <= this
hold_mode             target                ours: target (keep the last target) | awake
req_ttl_s             1800                  ours: forget arr/done bookkeeping older than this
                                            (never a request in the map)
====================  ====================  ===================================================
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional

from tre_baselines import trace_oracle
from tre_baselines.policies import preserve_tier1 as tier1
from tre_baselines.policies.base import Decision, hold_if_incomplete
from tre_baselines.policies.preserve_anticipator import LookaheadMap
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, RequestEvent, evidence_gaps

LOG = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_LEN = 4096
HOLD_MODES = ("target", "awake")


@dataclass
class _PodState:
    amap: LookaheadMap
    M: float
    last_ms: Optional[int] = None
    carry: float = 0.0
    tpot_s: float = 0.0
    tpot_src: str = "slo"
    prev_itl: Optional[tuple[float, float]] = None
    overloaded: bool = False


@dataclass
class _ArrInfo:
    in_tokens: Optional[int]
    max_tokens: Optional[int]
    ts_ms: int


@dataclass
class _ModelState:
    pods: dict[str, _PodState] = field(default_factory=dict)
    arr: dict[tuple[str, Optional[str]], _ArrInfo] = field(default_factory=dict)
    #: (req_id, pod) whose ``done`` came before their ``ft`` -> ts_ms.
    tombstones: dict[tuple[str, Optional[str]], int] = field(default_factory=dict)
    #: active request key -> pod holding it.
    active: dict[tuple[str, Optional[str]], str] = field(default_factory=dict)
    target: Optional[int] = None
    #: (replay t0_ms, window index) of the last Tier-1 firing.
    t1_window: Optional[tuple[int, int]] = None
    credited: set[str] = field(default_factory=set)
    last_down_key: Optional[Hashable] = None


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _per_model(raw: Any, cast=float) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k): cast(v) for k, v in raw.items()}
    return {"*": cast(raw)}


class PreServePolicy:
    name = "preserve"
    #: Arm name in decision records and docs (Tier-1 = trace oracle + noise, not mLSTM).
    label = "PreServe-oracle"
    needs_events = True

    def __init__(self, config: Any = None, *, oracle: Optional[trace_oracle.TraceOracle] = None) -> None:
        params = dict(getattr(config, "policy_params", None) or {})
        models = dict(getattr(config, "models", None) or {})
        self.seed = int(params.get("noise_seed", getattr(config, "seed", 0) or 0))
        self.window_s = float(params.get("window_s", 600.0))
        if self.window_s <= 0:
            raise ValueError("preserve: window_s must be > 0")
        self.lead_s = float(params.get("lead_s", 0.0))
        self.mode = str(params.get("tier1", "oracle_noisy"))
        if self.mode not in tier1.TIER1_MODES:
            raise ValueError(f"preserve: tier1 must be one of {tier1.TIER1_MODES}")
        self.sigma = float(params.get("noise_sigma", tier1.DEFAULT_NOISE_SIGMA))
        if self.sigma < 0:
            raise ValueError("preserve: noise_sigma must be >= 0")
        self.mu = tier1.parse_mu(params.get("mu"), models)
        self.match_parts = int(params.get("trace_match_parts", 2))
        self._oracle_injected = oracle is not None
        self._trace_schedule = str(params.get("trace_schedule", trace_oracle.SCHEDULE_POISSON))
        self._oracle_seed = int(params.get("trace_seed", 0))
        if oracle is None:
            path = params.get("trace_path")
            if not path:
                raise ValueError("preserve: params.trace_path is required (Tier-1 reads the replayed trace)")
            oracle = trace_oracle.load_oracle(
                str(path), window_s=self.window_s, seed=self._oracle_seed, schedule=self._trace_schedule,
            )
        elif oracle.window_s != self.window_s:
            raise ValueError("preserve: oracle window_s differs from params.window_s")
        self.oracle = oracle
        self.max_output_len = _per_model(params.get("max_output_len"), int)
        self.map_factor = float(params.get("map_factor", 1.2))
        self.l = int(params.get("lookahead_iters", 100))
        self.kv_high = float(params.get("kv_high", 0.95))
        self.overload_frac = float(params.get("overload_frac", 0.10))
        self.t_f = float(params.get("t_f", 0.30))
        self.ext_frac = float(params.get("ext_frac", 0.2))
        self.out_len_is_upper_bound = bool(params.get("out_len_is_upper_bound", True))
        self.kv_agree_tol = float(params.get("kv_agree_tol", 0.15))
        if not 0.0 < self.kv_agree_tol <= 1.0:
            raise ValueError("preserve: kv_agree_tol must be in (0, 1]")
        if self.l < 1 or self.map_factor < 1.0 or not 0 < self.t_f <= 1 or self.ext_frac <= 0:
            raise ValueError("preserve: need lookahead_iters >= 1, map_factor >= 1, 0 < t_f <= 1, ext_frac > 0")
        self.kv_capacity = _per_model(params.get("kv_capacity_tokens"), float)
        self.hold_mode = str(params.get("hold_mode", "target"))
        if self.hold_mode not in HOLD_MODES:
            raise ValueError(f"preserve: hold_mode must be one of {HOLD_MODES}")
        self.req_ttl_ms = int(float(params.get("req_ttl_s", 1800.0)) * 1000)
        self._models: dict[str, _ModelState] = {}
        #: Cumulative anomaly counters (per model), for tests and the run summary.
        self.anomalies: dict[str, Counter] = {}

    # --------------------------------------------------------------- helpers

    def map_length(self, model: str) -> int:
        mol = self.max_output_len.get(model, self.max_output_len.get("*"))
        if mol is None:
            mol = self.oracle.max_tokens_max.get(model) or DEFAULT_MAX_OUTPUT_LEN
        return max(1, int(math.ceil(int(mol) * self.map_factor)))

    def _default_D(self, model: str) -> int:
        mol = self.max_output_len.get(model, self.max_output_len.get("*"))
        return int(mol if mol is not None else (self.oracle.max_tokens_max.get(model) or DEFAULT_MAX_OUTPUT_LEN))

    def _capacity(self, model: str, pod: PodSnapshot) -> Optional[float]:
        if pod.num_gpu_blocks and pod.block_size and pod.num_gpu_blocks > 0 and pod.block_size > 0:
            return float(pod.num_gpu_blocks) * float(pod.block_size)
        cap = self.kv_capacity.get(model, self.kv_capacity.get("*"))
        return float(cap) if cap and cap > 0 else None

    @staticmethod
    def _advance_to(st: _ModelState, ps: _PodState, t_ms: int, anom: Counter) -> None:
        if ps.last_ms is None:
            ps.last_ms = int(t_ms)
            return
        if t_ms < ps.last_ms:
            anom["out_of_order"] += 1
            return
        k_float = (t_ms - ps.last_ms) / 1000.0 / ps.tpot_s + ps.carry
        k = int(math.floor(k_float))
        ps.carry = k_float - k
        ps.last_ms = int(t_ms)
        overdue = ps.amap.advance(k)
        if overdue:  # the estimate ran ahead of the request: it stays (floor) until done
            anom["overdue"] += len(overdue)

    # ------------------------------------------------------------- Tier-2 feed

    def _sync_pods(self, ms: ModelSnapshot, st: _ModelState, anom: Counter, skipped: dict) -> None:
        present = {p.pod: p for p in ms.pods}
        keep = set(present) | set(ms.unscraped)
        for name in [n for n in st.pods if n not in keep]:
            dropped = st.pods.pop(name)
            anom["pod_dropped"] += 1
            for key in list(dropped.amap.requests):
                st.active.pop(key, None)
            st.credited.discard(name)
        fallback = ms.tpot_slo_ms / 1000.0
        for name, pod in present.items():
            ps = st.pods.get(name)
            if ps is None:
                M = self._capacity(ms.model, pod)
                if M is None:
                    skipped[name] = "no_kv_capacity"
                    continue
                amap = LookaheadMap(self.map_length(ms.model), ext_frac=self.ext_frac,
                                    out_len_is_upper_bound=self.out_len_is_upper_bound)
                ps = _PodState(amap=amap, M=M)
                st.pods[name] = ps
            s, c = pod.counters.get("itl_sum"), pod.counters.get("itl_count")
            tpot = None
            if s is not None and c is not None and ps.prev_itl is not None:
                ds, dc = s - ps.prev_itl[0], c - ps.prev_itl[1]
                if dc > 0 and ds > 0:
                    tpot = ds / dc
                elif dc < 0 or ds < 0:
                    anom["counter_reset"] += 1
            ps.prev_itl = (s, c) if s is not None and c is not None else None
            ps.tpot_s, ps.tpot_src = (tpot, "itl") if tpot else (fallback, "slo")
        for name in ms.unscraped:
            ps = st.pods.get(name)
            if ps is not None:  # keep its map, walk it at the fallback rate
                ps.tpot_s, ps.tpot_src = fallback, "slo"

    def _find_active(self, st: _ModelState, req_id: str, pod: Optional[str]):
        key = (req_id, pod)
        if key in st.active:
            return key
        hits = [k for k in st.active if k[0] == req_id and (pod is None or k[1] is None)]
        return hits[0] if len(hits) == 1 else None

    def _on_event(self, ms: ModelSnapshot, st: _ModelState, e: RequestEvent, anom: Counter) -> None:
        key = (e.req_id, e.pod)
        if e.kind == "arr":
            st.arr[key] = _ArrInfo(e.in_tokens, e.max_tokens, int(e.ts_ms))
            return
        if e.kind == "ft":
            if key in st.tombstones or (e.req_id, None) in st.tombstones:
                st.tombstones.pop(key, None)
                st.tombstones.pop((e.req_id, None), None)
                anom["ft_after_done"] += 1
                return
            info = st.arr.pop(key, None) or st.arr.pop((e.req_id, None), None)
            if info is None:
                anom["ft_without_arr"] += 1
                info = _ArrInfo(e.in_tokens, e.max_tokens, int(e.ts_ms))
            pod = e.pod
            ps = st.pods.get(pod) if pod else None
            if ps is None:
                anom["unknown_pod"] += 1
                return
            P = info.in_tokens
            if P is None:
                anom["missing_in_tokens"] += 1
                P = 0
            D = info.max_tokens
            if D is None or D <= 0:
                anom["missing_max_tokens"] += 1
                D = self._default_D(ms.model)
            if key in st.active:
                anom["dup_ft"] += 1
                old_pod = st.active.pop(key)
                if old_pod in st.pods:
                    st.pods[old_pod].amap.remove(key)
            self._advance_to(st, ps, int(e.ts_ms), anom)
            ps.amap.add(key, int(P), int(D), ps.M, ts_ms=int(e.ts_ms))
            st.active[key] = pod
            return
        if e.kind == "done":
            akey = self._find_active(st, e.req_id, e.pod)
            if akey is not None:
                pod = st.active.pop(akey)
                ps = st.pods.get(pod)
                if ps is not None:
                    self._advance_to(st, ps, int(e.ts_ms), anom)
                    ps.amap.remove(akey)
                return
            info_key = key if key in st.arr else ((e.req_id, None) if (e.req_id, None) in st.arr else None)
            if info_key is not None:
                st.arr.pop(info_key)
                st.tombstones[key] = int(e.ts_ms)  # a late ft must not be counted
                anom["done_before_ft"] += 1
                return
            anom["unknown_req"] += 1
            return
        anom["bad_event"] += 1

    def _expire(self, st: _ModelState, now_ms: int, anom: Counter) -> None:
        """Bookkeeping only: arr records whose ft never came and tombstones. A request in
        the map is never expired by age (it leaves on done)."""
        horizon = now_ms - self.req_ttl_ms
        for key in [k for k, v in st.arr.items() if v.ts_ms < horizon]:
            del st.arr[key]
            anom["expired_arr"] += 1
        for key in [k for k, ts in st.tombstones.items() if ts < horizon]:
            del st.tombstones[key]

    @staticmethod
    def _engine_idle(st: _ModelState, ps: _PodState, pod: PodSnapshot, anom: Counter) -> None:
        """The engine reports nothing running or waiting: every request in this pod's map
        that started before the scrape is done (its done event was lost)."""
        if pod.queued != 0:
            return
        for key, req in list(ps.amap.requests.items()):
            if req.ts_ms <= pod.scraped_at_ms:
                ps.amap.remove(key)
                st.active.pop(key, None)
                anom["engine_idle_done"] += 1

    # ---------------------------------------------------------------- Tier-1

    def _tier1(self, snap: ClusterSnapshot, ms: ModelSnapshot, st: _ModelState):
        """(window index or None, inactive reason or None)."""
        if snap.replay is None:
            return None, "tier1_no_replay"
        seed = snap.replay.seed
        if (seed is not None and not self._oracle_injected and seed != self._oracle_seed
                and self.oracle.fmt == trace_oracle.FORMAT_SEGMENTS):
            # The campaign's marker carries the replay's --seed: a segment trace's arrival
            # schedule depends on it, so rebuild the oracle with that seed (once per seed).
            try:
                self.oracle = trace_oracle.load_oracle(
                    self.oracle.path, window_s=self.window_s, seed=int(seed), schedule=self._trace_schedule)
            except (OSError, ValueError) as exc:
                LOG.warning("preserve: cannot reload the trace with seed %s: %s", seed, exc)
                return None, "tier1_oracle_reload_failed"
            self._oracle_seed = int(seed)
        if not self.oracle.matches(snap.replay, self.match_parts):
            return None, "tier1_trace_mismatch"
        idx = trace_oracle.window_index(snap.now_ms, snap.replay, self.window_s, self.lead_s)
        if idx < 0:
            return None, "tier1_before_t0"
        if idx >= self.oracle.n_windows:
            return None, "tier1_after_trace"
        return idx, None

    def _tier1_n(self, model: str, idx: int) -> tuple[Optional[int], dict]:
        true = self.oracle.window(model, idx)
        prev = None
        if idx > 0:
            p = self.oracle.window(model, idx - 1)
            prev = (p.P, p.D)
        est = tier1.estimate(self.mode, true_P=true.P, true_D=true.D, prev=prev, seed=self.seed,
                             model=model, window=idx, sigma=self.sigma)
        info: dict = {"window": idx, "mode": self.mode}
        if est is None:
            return None, info
        W = self.oracle.window_len_s(idx)
        N = tier1.required_replicas(est[0], est[1], self.mu[model], W)
        info.update({"P_hat": round(est[0], 1), "D_hat": round(est[1], 1), "W": W, "N": N})
        return N, info

    # ---------------------------------------------------------------- decide

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        out: dict[str, Decision] = {}
        for model, ms in snap.models.items():
            out[model] = self._decide_model(snap, ms)
        return out

    def _decide_model(self, snap: ClusterSnapshot, ms: ModelSnapshot) -> Decision:
        st = self._models.setdefault(ms.model, _ModelState())
        cum = self.anomalies.setdefault(ms.model, Counter())
        anom: Counter = Counter()
        awake = int(ms.awake)
        if ms.model not in self.mu:
            return Decision(desired=awake, reason="no_mu", inputs={"awake": awake})

        # Tier-2 feed: pods, events, then walk every head to its scrape instant.
        skipped: dict[str, str] = {}
        self._sync_pods(ms, st, anom, skipped)
        for e in ms.events:
            self._on_event(ms, st, e, anom)
        for pod in ms.pods:
            ps = st.pods.get(pod.pod)
            if ps is not None:
                # An event may carry a Redis time a little after the scrape (written between
                # the scrape and the stream read): the head is already there, not out of order.
                self._advance_to(st, ps, max(int(pod.scraped_at_ms), ps.last_ms or 0), anom)
                self._engine_idle(st, ps, pod, anom)
        for name in ms.unscraped:
            ps = st.pods.get(name)
            if ps is not None:
                self._advance_to(st, ps, max(int(snap.now_ms), ps.last_ms or 0), anom)
        self._expire(st, int(snap.now_ms), anom)

        # Tier-2 read: U[0:l] of each evaluated pod.
        t2: dict[str, dict] = {}
        overloaded: list[str] = []
        max_us: list[float] = []
        kv_disagree: dict[str, Any] = {}
        for pod in ms.pods:
            ps = st.pods.get(pod.pod)
            if ps is None:
                continue
            w = ps.amap.window(self.l)
            frac = sum(1 for u in w if u > self.kv_high) / float(self.l)
            max_u = max(w) if w else 0.0
            ps.overloaded = frac > self.overload_frac
            if ps.overloaded:
                overloaded.append(pod.pod)
            max_us.append(max_u)
            u0 = ps.amap.at(0)
            t2[pod.pod] = {"overload_frac": _r(frac, 3), "maxU": _r(max_u), "U0": _r(u0),
                           "kv": _r(pod.kv_usage), "tpot": ps.tpot_src}
            # The map must agree with what the engine holds before it may justify a
            # scale-down (a map missing requests reads low).
            if pod.kv_usage is None or abs(u0 - pod.kv_usage) > self.kv_agree_tol:
                kv_disagree[pod.pod] = [_r(u0), _r(pod.kv_usage)]
        gaps = evidence_gaps(ms, int(snap.now_ms), snap.tick_s, events=True)
        for name in list(st.credited):
            ps = st.pods.get(name)
            if ps is None or not ps.overloaded:
                st.credited.discard(name)  # the episode ended: may be credited again

        idx, t1_off = self._tier1(snap, ms, st)
        lo, hi = int(ms.min_replicas), int(ms.max_replicas)

        def clamp(n: int) -> int:
            return max(lo, min(hi, n)) if hi >= lo else n

        base = st.target if (self.hold_mode == "target" and st.target is not None) else awake
        reason = "hold"
        t1_info: dict = {"inactive": t1_off} if t1_off else {"window": idx}
        down_info: Optional[str] = None

        t1_key = None if idx is None else (int(snap.replay.t0_ms), idx)
        if t1_key is not None and t1_key != st.t1_window:
            st.t1_window = t1_key
            N, t1_info = self._tier1_n(ms.model, idx)
            st.credited = set(overloaded)
            if N is None:
                reason = "tier1_no_history"
                new = base + len(overloaded)
            else:
                reason = "tier1_window"
                new = N + len(overloaded)
            if overloaded:
                reason += "+tier2_overload"
            st.target = clamp(new)
        else:
            fresh = [p for p in overloaded if p not in st.credited]
            if fresh:
                st.credited.update(fresh)
                st.target = clamp(base + len(fresh))
                reason = "tier2_overload"
            else:
                down_key: Hashable = (t1_key if t1_key is not None
                                      else ("wall", int(snap.now_ms) // int(self.window_s * 1000)))
                if not max_us:
                    down_info = "no_pods"
                elif skipped or gaps or kv_disagree:
                    down_info = "incomplete"
                elif any(u >= self.t_f for u in max_us):
                    down_info = "above_t_f"
                elif down_key == st.last_down_key:
                    down_info = "done_this_window"
                else:
                    keep = max(1, int(math.ceil(sum(max_us) / self.t_f - 1e-9)))
                    if keep < awake:
                        st.last_down_key = down_key
                        st.target = clamp(keep)
                        reason = "tier2_underload"
                    else:
                        down_info = "no_gain"
                if reason == "hold":
                    st.target = clamp(base)

        for k, v in anom.items():
            cum[k] += v
        inputs: dict[str, Any] = {
            "awake": awake,
            "target": st.target,
            "tier1": t1_info,
            "tier2": t2,
        }
        if down_info:
            inputs["down"] = down_info
        if skipped:
            inputs["skipped"] = skipped
        if kv_disagree:
            inputs["kv_disagree"] = kv_disagree
        if anom:
            inputs["anom"] = dict(sorted(anom.items()))
        # Any target below awake (Tier-1 or Tier-2) waits for complete evidence; the
        # target itself is kept.
        return hold_if_incomplete(Decision(desired=int(st.target), reason=reason, inputs=inputs), awake, gaps)
