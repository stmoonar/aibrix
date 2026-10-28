"""PreServe baseline, scaling part only (§4.1 Alg.2, §4.3.1, §4.3.2), v1 "faithful" composition.

Decision 2026-10-26 (``docs/campaign-v2-20261026-plan.md`` #5): Tier-1 is the paper's mLSTM
forecaster, the rest follows TRE v1's faithful PreServe (``tre_code`` dev ``ceb88b4``,
``tre_preserve/policy.py`` ``_compose_faithful``; v1 D54-D58, D64, D65). Not reproduced: the
DistilBERT response-length predictor (Tier-2 uses the request's ``max_tokens`` plus the paper's
Tab. 2 error, "PreServe w/ load errors", :mod:`preserve_load_errors`) and the load-aware router.

Composition (v1 faithful)::

    desired = clamp(max(N_T1, N_T2), min_replicas, max_replicas)

* **N_T1** (Tier-1), set at the first tick of each window ``i`` of ``window_s`` (30 s, v1 D18)
  from the replay start ``t0``: ``N = ceil(max(P/(mu_p W), D/(mu_d W), (P+D)/(mu_t W)))`` with
  the forecast ``P_hat``, ``D_hat`` (:class:`preserve_tier1.MlstmTier1`; the history before
  ``t0`` is the trace's history file, :mod:`preserve_history`). None before ``t0`` or without a
  replay; after the trace and on a forecast error the last value is kept. Tier-1 reads no
  Tier-2 signal, so **an evidence gap never holds it** (v1 D56 / D64); the shell's
  unknown-is-not-idle backstop is off for this policy (``evidence_backstop = False``), the
  policy gates Tier-2 itself.
* **N_T2** (Tier-2), its own count across windows (upstream ``num_inst_lookahead``), starting
  at the awake count. A window start resets it to ``min_replicas`` and forgets the credited
  overload episodes (``t2_reset_at_window``, v1 D54 #2: the window starts at ``N_{i+1}``). This
  reset happens even while Tier-2 is frozen (as v1 ``_compose_faithful``): ``frozen`` means
  Tier-2 made no evidence-driven change, not that ``N_T2`` stayed constant.
  Every tick, per awake pod, the load-look-ahead map ``U`` (:mod:`preserve_anticipator`):

  - **overload** (v1 D55, sliding look-ahead): a pod is overloaded when, for any start offset
    ``t`` in ``[0, H]``, ``H = ceil(tick_s / TPOT_pod)``, more than ``overload_frac`` of the
    paper window ``U[t, t + l)`` is above ``kv_high`` (a checker that runs every iteration
    between two ticks). A new overload episode on ``k`` pods: ``N_T2 = max(N_T1, N_T2) + k``
    (one extra instance per potentially overloaded instance, on the current count);
  - **underload**, at most once per window: when every pod's ``max U[0:l] < T_f``,
    ``N_T2 = max(1, ceil(sum max U / T_f))`` if that is below ``N_T2`` (the load packed onto
    instances each at ``T_f``; the isolated ones sleep through the shared abort path);
  - **incomplete evidence holds Tier-2 in both directions** (``frozen``): any gap of
    :func:`~tre_baselines.snapshot.evidence_gaps` (event loss, unscraped pod, missing gauges,
    stale scrape) or a pod without a KV capacity. ``frozen`` (every tick, also under
    ``inputs``) and ``t2_hold`` (the reasons) are logged; the run's frozen-tick share is a
    validity check. The map / engine KV disagreement is logged only (v1 #7, no gate).

Iterations vs wall clock (ours, as v1 and v2 before): a pod's head advances ``dt / TPOT_pod``
iterations, fraction carried; ``TPOT_pod`` = mean inter-token latency between the pod's last two
scrapes, else the model's TPOT SLO. Each event first walks its pod to the event's Redis time.

A request is pending at ``arr`` and enters the decode look-ahead map at ``ft``
(prefill completed). Waiting requests do not accrue imaginary decoded tokens and prevent
evidence-free Tier-2 shrink. Attempt identity is (gateway req_id, pod); client_id links
sidecar reissues. A reissue supersedes its former execution immediately; a late parent
ft/done cannot resurrect it or remove its successor. No routing rule changes.

Events may precede the snapshot of a newly woken pod: keep arr/ft until that pod appears.
Only known attempt identities are discarded on a pod's departure. No tick-derived cutoff
discards new requests. Lost done expiry and engine-idle reconciliation remain bounded.
Before the replay marker arrives, retain bounded new-demand arrivals for Tier-1 backfill.

Params (``config.policy_params``, example ``examples/preserve.yaml``):

======================  =====================  ================================================
key                     default                origin
======================  =====================  ================================================
calibration             (none)                 ``pending``: refuse (mu not frozen)
trace_path              (required)             ours: the replayed trace (window count, max lengths)
trace_seed              0                      ours: fallback of the replay's ``--seed``
trace_schedule          poisson                ours: what ``tre_replayer.run_trace`` sends
trace_match_parts       2                      ours: replay marker must end in the same N parts
trace_name              path part -3           the trace's name (``<name>/seed<k>/<file>``):
                                               picks the predictor set
window_s                30                     v1 D18 (paper 600 s; our traces are 12 min)
lead_s                  0                      ours: fire Tier-1 this early (s)
tier1                   mlstm                  paper; ``oracle_noisy`` / ``oracle`` /
                                               ``last_window`` = sensitivity (D8)
predictors              (required, mlstm)      ``{set: {path, sha256}}`` (v1 paper_past weights)
predictor_rules         Real_code->code,       ``[[trace name prefix, set]]``; else ``pooled``
                        Real_conv->conv
seq_len                 25                     paper (the weights' own seq_len must match)
history_path            <trace dir>/           the history before t0 (:mod:`preserve_history`):
                        pre_t0.json, else      the generator's (synthetic) / the Azure windows
                        preserve_history.json  before the slice (Real_*)
noise_sigma, noise_seed                        oracle_noisy only (sensitivity)
mu                      (required)             paper Alg.1 (``tools/velocity_ladder mu``)
load_error_mode         paper_tab2             paper RQ2 "w/ load errors" (Tab. 2) | none
load_error_seed         config.seed            ours
max_model_len           131072                 engine context (upper clip of D_hat)
max_output_len          trace max max_tokens   map ring base length; int or {model: int}
map_factor              1.2                    ours: ring = ceil(max_output_len x 1.2), >= H + l
lookahead_iters         100                    paper (l)
kv_high                 0.95                   paper
overload_frac           0.10                   paper
t_f                     0.30                   paper (T_f)
out_len_is_upper_bound  not load errors        ours: true = ``max_tokens`` is a hard cap (no
                                               extension); must be false with load errors
ext_frac                0.2                    paper (virtual extension, with load errors)
t2_reset_at_window      true                   v1 D54 #2
kv_capacity_tokens      {}                     ours: M per model when the pod lacks cache_config
kv_agree_tol            0.15                   ours: logged only (``kv_disagree``)
req_ttl_s               1800                   ours: forget buffered events older than this
done_tpot_bound_s       0.2                    v1: a request without done expires after
done_margin_s           180                    max(D_hat, max_tokens) x bound + margin (v1 D65)
======================  =====================  ================================================
"""
from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Optional

from tre_baselines import trace_oracle
from tre_baselines.policies import preserve_history
from tre_baselines.policies import preserve_load_errors as load_errors
from tre_baselines.policies import preserve_tier1 as tier1
from tre_baselines.policies.base import Decision, refuse_pending
from tre_baselines.policies.preserve_anticipator import LookaheadMap
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, RequestEvent, evidence_gaps

LOG = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_LEN = 4096
DEFAULT_PREDICTOR_RULES = (("Real_code", "code"), ("Real_conv", "conv"))
DEFAULT_PREDICTOR_SET = "pooled"
#: An idle engine clears only map entries routed at least this long before the scrape (v1
#: ``ENGINE_IDLE_MARGIN_MS``): a request routed just before it may not have reached the engine.
ENGINE_IDLE_MARGIN_MS = 1000
#: tre/baselines (in the image /app/tre/baselines): relative predictor paths resolve here.
BASELINES_ROOT = Path(__file__).resolve().parents[2]


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
    draw_key: str = ""
    reissue: bool = False
    ft_ms: Optional[int] = None
    client_id: Optional[str] = None


@dataclass
class _ModelState:
    pods: dict[str, _PodState] = field(default_factory=dict)
    #: pod -> request key -> arr info: ``arr`` events for a pod the snapshot does not show
    #: yet (the stream can run ahead of /v2/state); applied when the pod appears.
    pending: dict[str, dict[tuple[str, Optional[str]], "_ArrInfo"]] = field(default_factory=dict)
    #: Latest execution attempt of a client request; an index, not a second load ledger.
    latest: dict[str, tuple[tuple[str, Optional[str]], bool]] = field(default_factory=dict)
    clients: dict[tuple[str, Optional[str]], str] = field(default_factory=dict)
    #: New-demand arrivals consumed before the replay marker arrives; bounded below.
    pre_replay: deque = field(default_factory=deque)
    pre_replay_lost_ms: Optional[int] = None
    #: request key -> now_ms: requests of a slept pod (map, buffer or discarded), for ``done_pod_gone``.
    gone_keys: dict[tuple[str, Optional[str]], int] = field(default_factory=dict)
    #: now_ms of the last request expired without done (``done_lost`` gap for one window, v1).
    last_done_lost_ms: Optional[int] = None
    #: active request key -> pod holding it.
    active: dict[tuple[str, Optional[str]], str] = field(default_factory=dict)
    #: (replay t0_ms, window index) of the last Tier-1 firing.
    t1_window: Optional[tuple[int, int]] = None
    n_t1: Optional[int] = None
    n_t2: Optional[int] = None
    credited: set[str] = field(default_factory=set)
    #: window key of the last Tier-2 underload change (once per window)
    t2_down_key: Optional[Hashable] = None
    t2_hold_ticks: int = 0
    ticks: int = 0
    #: mLSTM Tier-1 of the current replay (t0_ms, object)
    mlstm: Optional[tuple[int, tier1.MlstmTier1]] = None
    err: load_errors.ErrorStats = field(default_factory=load_errors.ErrorStats)
    err_rows: list = field(default_factory=list)


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _per_model(raw: Any, cast=float) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k): cast(v) for k, v in raw.items()}
    return {"*": cast(raw)}


def predictor_set_for(trace: str, rules: Any, default: str = DEFAULT_PREDICTOR_SET) -> str:
    """Real_code* -> code, Real_conv* -> conv, anything else -> pooled (v1 ``tier1.trace_sets``)."""
    for prefix, name in rules:
        if str(trace).startswith(str(prefix)):
            return str(name)
    return default


def trace_name_of(trace_path: str, part: int = 3) -> str:
    """The trace name in ``.../<name>/seed<k>/<file>`` (``part`` = position from the end)."""
    pieces = [p for p in str(trace_path).replace("\\", "/").split("/") if p]
    return pieces[-part] if len(pieces) >= part else (pieces[0] if pieces else "")


class PreServePolicy:
    name = "preserve"
    #: Arm name in decision records and docs.
    label = "PreServe"
    needs_events = True
    #: The policy gates Tier-2 itself and never holds Tier-1 (v1 D56 / D64): no shell backstop.
    evidence_backstop = False

    def __init__(self, config: Any = None, *, oracle: Optional[trace_oracle.TraceOracle] = None,
                 predictors: Optional[Mapping[str, Any]] = None,
                 history: Optional[Mapping[str, Any]] = None) -> None:
        """``predictors`` ({"prompt", "response"}) and ``history`` ({model: rows}) are for tests:
        they replace the files of the mLSTM Tier-1."""
        params = dict(getattr(config, "policy_params", None) or {})
        refuse_pending(params, self.name)
        models = dict(getattr(config, "models", None) or {})
        cfg_seed = int(getattr(config, "seed", 0) or 0)
        self.seed = int(params.get("noise_seed", cfg_seed))
        self.window_s = float(params.get("window_s", 30.0))
        if self.window_s <= 0:
            raise ValueError("preserve: window_s must be > 0")
        self.lead_s = float(params.get("lead_s", 0.0))
        if self.lead_s != 0.0 and str(params.get("tier1", "mlstm")) == "mlstm":
            raise ValueError("preserve: lead_s must be 0 with tier1 mlstm (an early window start would close "
                             "the observed window before it ends)")
        self.mode = str(params.get("tier1", "mlstm"))
        if self.mode not in tier1.TIER1_MODES:
            raise ValueError(f"preserve: tier1 must be one of {tier1.TIER1_MODES}")
        if self.mode != "mlstm":
            self.label = "PreServe-oracle"
        self.sigma = float(params.get("noise_sigma", tier1.DEFAULT_NOISE_SIGMA))
        if self.sigma < 0:
            raise ValueError("preserve: noise_sigma must be >= 0")
        self.mu = tier1.parse_mu(params.get("mu"), models)
        self.match_parts = int(params.get("trace_match_parts", 2))
        self._oracle_injected = oracle is not None
        self._trace_schedule = str(params.get("trace_schedule", trace_oracle.SCHEDULE_POISSON))
        self._oracle_seed = int(params.get("trace_seed", 0))
        trace_path = params.get("trace_path")
        if oracle is None:
            if not trace_path:
                raise ValueError("preserve: params.trace_path is required (the replayed trace)")
            oracle = trace_oracle.load_oracle(
                str(trace_path), window_s=self.window_s, seed=self._oracle_seed, schedule=self._trace_schedule,
            )
        elif oracle.window_s != self.window_s:
            raise ValueError("preserve: oracle window_s differs from params.window_s")
        self.oracle = oracle
        self.trace_name = str(params.get("trace_name") or trace_name_of(str(trace_path or oracle.path)))

        # Tier-1 mLSTM: weights (sha256 checked) and the history before t0, at start (fail closed)
        self.seq_len = int(params.get("seq_len", 25))
        self.predictor_set: Optional[str] = None
        self._predictors: Optional[Mapping[str, Any]] = None
        self._history: dict[str, list] = {}
        self.history_info: dict[str, Any] = {}
        if self.mode == "mlstm":
            rules = params.get("predictor_rules") or DEFAULT_PREDICTOR_RULES
            self.predictor_set = predictor_set_for(self.trace_name, rules)
            if predictors is not None:
                self._predictors = predictors
            else:
                from tre_baselines.policies import preserve_predictor  # numpy: only for this mode

                spec = (params.get("predictors") or {}).get(self.predictor_set)
                if not isinstance(spec, Mapping) or not spec.get("path") or not spec.get("sha256"):
                    raise ValueError(f"preserve: predictors.{self.predictor_set} needs path and sha256")
                wpath = Path(str(spec["path"]))
                if not wpath.is_absolute():  # shipped weights: relative to tre/baselines
                    wpath = BASELINES_ROOT / wpath
                self._predictors = preserve_predictor.load_predictors(str(wpath), str(spec["sha256"]))
                for k, pr in self._predictors.items():
                    if pr.seq_len != self.seq_len:
                        raise ValueError(f"preserve: predictor {k} seq_len {pr.seq_len} != seq_len {self.seq_len}")
            if history is not None:
                self._history = {m: list(v) for m, v in history.items()}
                self.history_info = {"path": None, "injected": True}
            else:
                tpath = str(trace_path or oracle.path)
                hpath = str(params.get("history_path") or preserve_history.default_path(tpath))
                self._history, self.history_info = preserve_history.load(
                    hpath, window_s=self.window_s, models=sorted(self.mu), seq_len=self.seq_len,
                    trace=self.trace_name, seed=preserve_history.seed_of(tpath))
            missing = sorted(m for m in self.mu if len(self._history.get(m, ())) < self.seq_len)
            if missing:
                raise ValueError(f"preserve: history lacks {self.seq_len} windows for {missing}")

        # load errors (paper RQ2 "PreServe w/ load errors")
        self.load_error_mode = str(params.get("load_error_mode", "paper_tab2"))
        self.errors = load_errors.make(self.load_error_mode, int(params.get("load_error_seed", cfg_seed)))
        self.max_model_len = int(params.get("max_model_len", 131072))
        self.out_len_is_upper_bound = bool(params.get("out_len_is_upper_bound", self.errors is None))
        if self.errors is not None and self.out_len_is_upper_bound:
            raise ValueError("preserve: with load errors D_hat is a prediction, not an upper bound "
                             "(out_len_is_upper_bound must be false)")

        self.max_output_len = _per_model(params.get("max_output_len"), int)
        self.map_factor = float(params.get("map_factor", 1.2))
        self.l = int(params.get("lookahead_iters", 100))
        self.kv_high = float(params.get("kv_high", 0.95))
        self.overload_frac = float(params.get("overload_frac", 0.10))
        self.t_f = float(params.get("t_f", 0.30))
        self.ext_frac = float(params.get("ext_frac", 0.2))
        self.t2_reset_at_window = bool(params.get("t2_reset_at_window", True))
        self.kv_agree_tol = float(params.get("kv_agree_tol", 0.15))
        if not 0.0 < self.kv_agree_tol <= 1.0:
            raise ValueError("preserve: kv_agree_tol must be in (0, 1]")
        if self.l < 1 or self.map_factor < 1.0 or not 0 < self.t_f <= 1 or self.ext_frac <= 0:
            raise ValueError("preserve: need lookahead_iters >= 1, map_factor >= 1, 0 < t_f <= 1, ext_frac > 0")
        self.kv_capacity = _per_model(params.get("kv_capacity_tokens"), float)
        self.req_ttl_ms = int(float(params.get("req_ttl_s", 1800.0)) * 1000)
        self.done_tpot_bound_s = float(params.get("done_tpot_bound_s", 0.2))
        self.done_margin_s = float(params.get("done_margin_s", 180.0))
        if self.done_tpot_bound_s < 0 or self.done_margin_s < 0:
            raise ValueError("preserve: done_tpot_bound_s and done_margin_s must be >= 0")
        self._models: dict[str, _ModelState] = {}
        #: Cumulative anomaly counters (per model), for tests and the run summary.
        self.anomalies: dict[str, Counter] = {}

    def counters(self) -> Mapping[str, Mapping[str, int]]:
        """Cumulative counters per model (anomalies, tier2 holds, ...), reported by the shell."""
        out = {m: dict(c) for m, c in self.anomalies.items()}
        for m, st in self._models.items():
            out.setdefault(m, {}).update(t2_hold_ticks=st.t2_hold_ticks, ticks=st.ticks)
        return out

    # --------------------------------------------------------------- helpers

    def _mol(self, model: str) -> int:
        mol = self.max_output_len.get(model, self.max_output_len.get("*"))
        return int(mol if mol is not None else (self.oracle.max_tokens_max.get(model) or DEFAULT_MAX_OUTPUT_LEN))

    def horizon(self, tpot_s: float, tick_s: float) -> int:
        """H: iterations a pod runs until the next tick (the continuous checker's start offsets)."""
        return int(math.ceil(float(tick_s) / float(tpot_s) - 1e-9))

    def map_length(self, model: str, tpot_s: float = 0.0, tick_s: float = 0.0) -> int:
        n = max(self.l, int(math.ceil(self._mol(model) * self.map_factor)))
        if tpot_s > 0 and tick_s > 0:
            n = max(n, self.horizon(tpot_s, tick_s) + self.l)
        return n

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

    def overload_scan(self, ps: _PodState, tick_s: float) -> tuple[float, int, Optional[int], float, list[float]]:
        """v1 D55: emulate a checker that runs every iteration between two ticks. For every start
        offset t in [0, H] the paper window U[t, t + l) is overloaded when more than
        ``overload_frac`` of it is above ``kv_high``; the pod is when any t is. The map holds only
        requests already routed, so a window ahead is a lower bound of the load then (without
        load errors). Returns (fraction at t = 0, H, first overloaded t or None, the largest
        fraction, U[0, l))."""
        l = self.l
        H = self.horizon(ps.tpot_s, tick_s)
        need = H + l
        if need > ps.amap.length:
            ps.amap.grow(need)
        w = ps.amap.window(need)
        acc = [0]
        for u in w:
            acc.append(acc[-1] + (1 if u > self.kv_high else 0))
        frac0 = acc[l] / float(l)
        t_star: Optional[int] = None
        best = 0.0
        for t in range(H + 1):
            f = (acc[t + l] - acc[t]) / float(l)
            best = max(best, f)
            if t_star is None and f > self.overload_frac:
                t_star = t
        return frac0, H, t_star, best, w[:l]

    # ------------------------------------------------------------- Tier-2 feed

    def _sync_pods(self, ms: ModelSnapshot, st: _ModelState, anom: Counter, skipped: dict, tick_s: float,
                   now_ms: int) -> None:
        present = {p.pod: p for p in ms.pods}
        keep = set(present) | set(ms.unscraped)
        for name in [n for n in st.pods if n not in keep]:
            for key in list(st.pods[name].amap.requests):
                self._forget(st, key, int(now_ms))
            for key in list(st.pending.get(name, {})):
                self._forget(st, key, int(now_ms))
            st.pods.pop(name)
            st.credited.discard(name)
            anom["pod_dropped"] += 1
        # Do not infer an execution epoch from a snapshot's absence. New arr/ft
        # events can legitimately precede the first snapshot of a newly woken pod.
        fallback = ms.tpot_slo_ms / 1000.0
        for name, pod in present.items():
            ps = st.pods.get(name)
            if ps is None:
                M = self._capacity(ms.model, pod)
                if M is None:
                    skipped[name] = "no_kv_capacity"
                    continue
                amap = LookaheadMap(self.map_length(ms.model, fallback, tick_s), ext_frac=self.ext_frac,
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

    @staticmethod
    def _forget(st: _ModelState, key, now_ms: int) -> None:
        """End exactly one physical attempt; late parent events cannot end its successor."""
        pod = st.active.pop(key, None)
        if pod in st.pods:
            st.pods[pod].amap.remove(key)
        pending = st.pending.get(key[1])
        if pending is not None:
            pending.pop(key, None)
            if not pending:
                st.pending.pop(key[1], None)
        client = st.clients.pop(key, None)
        if client and st.latest.get(client, (None,))[0] == key:
            st.latest.pop(client, None)
        st.gone_keys[key] = int(now_ms)

    def _on_event(self, ms: ModelSnapshot, st: _ModelState, e: RequestEvent, anom: Counter,
                  t1m: Optional[tier1.MlstmTier1]) -> None:
        key = (e.req_id, e.pod)
        if e.kind == "arr":
            if key in st.gone_keys or key in st.active or key in st.pending.get(e.pod, {}):
                anom["duplicate_or_ended_arr"] += 1
                return
            if not e.client_id:
                anom["no_client_id"] += 1
            if e.reissue == "none":
                if t1m is not None:
                    t1m.observe_arrival(int(e.ts_ms), e.in_tokens, e.max_tokens)
                elif self.mode == "mlstm":
                    # A late marker must not lose the first window. Retain only arrivals,
                    # never ft/done, and bound both age and count (same scale as the stream).
                    st.pre_replay.append(e)
                    while st.pre_replay and (len(st.pre_replay) > 200000 or
                                            e.ts_ms - st.pre_replay[0].ts_ms > self.req_ttl_ms):
                        st.pre_replay_lost_ms = st.pre_replay.popleft().ts_ms
            if not e.pod:
                anom["unknown_pod"] += 1
                return
            old = st.latest.get(e.client_id) if e.client_id else None
            if old and old[0] != key:
                if e.reissue != "none":
                    # Sidecar reissue begins only after the former engine execution
                    # stopped; the parent HTTP stream can remain open forwarding it.
                    self._forget(st, old[0], int(e.ts_ms))
                    anom["attempt_superseded"] += 1
                elif old[1]:
                    st.gone_keys[key] = int(e.ts_ms)  # delayed parent arr
                    anom["late_parent_arr"] += 1
                    return
            if e.client_id:
                st.latest[e.client_id] = (key, e.reissue != "none")
                st.clients[key] = e.client_id
            info = _ArrInfo(e.in_tokens, e.max_tokens, int(e.ts_ms), e.draw_key,
                            e.reissue != "none", client_id=e.client_id)
            st.pending.setdefault(e.pod, {})[key] = info
            return
        if e.kind == "ft":
            if key in st.gone_keys or key in st.active:
                return
            info = st.pending.get(e.pod, {}).get(key)
            if info is None:
                anom["ft_without_arr"] += 1
                return
            if info.ft_ms is None:
                info.ft_ms = int(e.ts_ms)
            ps = st.pods.get(e.pod)
            if ps is not None:
                self._add(ms, st, key, info, e.pod, ps, anom)
                st.pending[e.pod].pop(key)
                if not st.pending[e.pod]:
                    st.pending.pop(e.pod)
            return
        if e.kind == "done":
            if key in st.gone_keys:
                return
            if key in st.active or key in st.pending.get(e.pod, {}):
                self._forget(st, key, int(e.ts_ms))
            else:
                st.gone_keys[key] = int(e.ts_ms)
                anom["unknown_req"] += 1
            return
        anom["bad_event"] += 1

    def _add(self, ms: ModelSnapshot, st: _ModelState, key, info: "_ArrInfo", pod: str, ps: _PodState,
             anom: Counter) -> None:
        assert info.ft_ms is not None
        ts_ms = info.ft_ms
        P = info.in_tokens
        if P is None:
            anom["missing_in_tokens"] += 1
            P = 0
        D = info.max_tokens
        if D is None or D <= 0:
            anom["missing_max_tokens"] += 1
            D = self._mol(ms.model)
        true_d = int(D)
        if self.errors is not None and info.max_tokens:
            D = self.errors.predict(ms.model, info.draw_key or key[0], true_d, int(P), self.max_model_len)
            if not info.reissue:  # the realised error of new demand (a reissue repeats its original's)
                st.err.add(true_d, int(D))
                st.err_rows.append([info.draw_key or key[0], true_d, int(D)])
        if key in st.active:
            anom["dup_arr"] += 1
            old_pod = st.active.pop(key)
            if old_pod in st.pods:
                st.pods[old_pod].amap.remove(key)
        self._advance_to(st, ps, int(ts_ms), anom)
        ps.amap.add(key, int(P), int(D), ps.M, ts_ms=int(ts_ms), max_tokens=true_d)
        st.active[key] = pod

    def _apply_buffered(self, ms: ModelSnapshot, st: _ModelState, anom: Counter) -> None:
        """Start known attempts whose first token preceded the pod's first snapshot."""
        for pod in list(st.pending):
            ps = st.pods.get(pod)
            if ps is None:
                continue
            for key, info in sorted(list(st.pending[pod].items()), key=lambda kv: kv[1].ts_ms):
                if info.ft_ms is not None:
                    self._add(ms, st, key, info, pod, ps, anom)
                    st.pending[pod].pop(key)
                    anom["ft_buffered_applied"] += 1
            if not st.pending[pod]:
                st.pending.pop(pod)

    def _expire(self, st: _ModelState, now_ms: int, anom: Counter) -> None:
        """Bound lost-event bookkeeping; expiry is missing evidence, not proof of idle."""
        horizon = now_ms - self.req_ttl_ms
        for ps in st.pods.values():
            for key, req in list(ps.amap.requests.items()):
                bound_s = max(req.D_pred, req.max_tokens or 0) * self.done_tpot_bound_s + self.done_margin_s
                if now_ms - req.ts_ms > bound_s * 1000:
                    self._forget(st, key, now_ms)
                    st.last_done_lost_ms = int(now_ms)
                    anom["expired_routed"] += 1
        for pod in list(st.pending):
            for key, info in list(st.pending[pod].items()):
                if info.ts_ms < horizon:
                    self._forget(st, key, now_ms)
                    st.last_done_lost_ms = int(now_ms)
                    anom["expired_buffered_arr"] += 1
        for key in [k for k, ts in st.gone_keys.items() if ts < horizon]:
            del st.gone_keys[key]

    def _engine_idle(self, st: _ModelState, ps: _PodState, pod: PodSnapshot, anom: Counter) -> None:
        """An idle scrape clears old known attempts, never a newer route decision."""
        if pod.queued != 0:
            return
        entries = [(k, r.ts_ms) for k, r in ps.amap.requests.items()]
        entries += [(k, r.ts_ms) for k, r in st.pending.get(pod.pod, {}).items()]
        for key, ts in entries:
            if ts <= pod.scraped_at_ms - ENGINE_IDLE_MARGIN_MS:
                self._forget(st, key, int(pod.scraped_at_ms))
                anom["engine_idle_done"] += 1

    # ---------------------------------------------------------------- Tier-1

    def _tier1(self, snap: ClusterSnapshot):
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

    def _mlstm_for(self, snap: ClusterSnapshot, model: str, st: _ModelState) -> Optional[tier1.MlstmTier1]:
        """The mLSTM Tier-1 of the current replay (a new ``t0`` starts a fresh history)."""
        if self.mode != "mlstm" or snap.replay is None or model not in self.mu:
            return None
        if not self.oracle.matches(snap.replay, self.match_parts):
            return None
        t0 = int(snap.replay.t0_ms)
        if st.mlstm is None or st.mlstm[0] != t0:
            if st.pre_replay_lost_ms is not None and st.pre_replay_lost_ms >= t0:
                raise ValueError("replay marker arrived after its arrival history was lost")
            st.mlstm = (t0, tier1.MlstmTier1(model=model, t0_ms=t0, window_s=self.window_s,
                                             n_windows=self.oracle.n_windows, predictors=self._predictors,
                                             mu=self.mu[model], seq_len=self.seq_len,
                                             history=self._history[model]))
            t1m = st.mlstm[1]
            for event in st.pre_replay:
                if event.ts_ms >= t0:
                    t1m.observe_arrival(event.ts_ms, event.in_tokens, event.max_tokens)
            st.pre_replay.clear()
            st.pre_replay_lost_ms = None
        return st.mlstm[1]

    def _tier1_n(self, model: str, idx: int, t1m: Optional[tier1.MlstmTier1]) -> tuple[Optional[int], dict]:
        if self.mode == "mlstm":
            if t1m is None:
                return None, {"window": idx, "mode": "mlstm", "error": "no_mlstm"}
            return t1m.decide(idx)
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
            return Decision(desired=awake, reason="no_mu", inputs={"awake": awake, "frozen": False})
        st.ticks += 1
        st.err_rows = []
        tick_s = float(snap.tick_s)
        t1m = self._mlstm_for(snap, ms.model, st)

        # Tier-2 feed: pods, events, then walk every head to its scrape instant.
        skipped: dict[str, str] = {}
        self._sync_pods(ms, st, anom, skipped, tick_s, int(snap.now_ms))
        self._apply_buffered(ms, st, anom)
        for e in ms.events:
            self._on_event(ms, st, e, anom, t1m)
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
        gaps = list(evidence_gaps(ms, int(snap.now_ms), tick_s, events=True))
        if st.last_done_lost_ms is not None and int(snap.now_ms) - st.last_done_lost_ms < self.window_s * 1000:
            gaps.append("done_lost")  # v1: a request may still run unseen
        lossy = sorted(g for g in gaps if g in ("no_event_history", "event_gap"))
        if t1m is not None and lossy:
            t1m.flag(t1m.index(int(snap.now_ms)), "+".join(lossy))

        # Tier-2 read
        t2: dict[str, dict] = {}
        overloaded: list[str] = []
        max_us: list[float] = []
        kv_disagree: dict[str, Any] = {}
        for pod in ms.pods:
            ps = st.pods.get(pod.pod)
            if ps is None:
                continue
            frac, H, t_star, frac_max, w = self.overload_scan(ps, tick_s)
            ps.overloaded = t_star is not None
            max_u = max(w) if w else 0.0  # underload: the window at now (ahead lacks future arrivals)
            if ps.overloaded:
                overloaded.append(pod.pod)
            max_us.append(max_u)
            u0 = ps.amap.at(0)
            t2[pod.pod] = {"overload_frac": _r(frac, 3), "maxU": _r(max_u), "U0": _r(u0), "kv": _r(pod.kv_usage),
                           "reqs": len(ps.amap.requests), "pending": len(st.pending.get(pod.pod, {})), "tpot": _r(ps.tpot_s, 4), "tpot_src": ps.tpot_src,
                           "H": H, "t_star": t_star, "frac_max": _r(frac_max, 3), "ring": ps.amap.length}
            if pod.kv_usage is None or abs(u0 - pod.kv_usage) > self.kv_agree_tol:
                kv_disagree[pod.pod] = [_r(u0), _r(pod.kv_usage)]  # logged only (v1 #7)
        for name in list(st.credited):
            ps = st.pods.get(name)
            if ps is None or not ps.overloaded:
                st.credited.discard(name)  # the episode ended: may be credited again

        lo, hi = int(ms.min_replicas), int(ms.max_replicas)

        def clamp(n: int) -> int:
            return max(lo, min(hi, int(n))) if hi >= lo else int(n)

        # Tier-1
        idx, t1_off = self._tier1(snap)
        t1_key = None if idx is None else (int(snap.replay.t0_ms), idx)
        t1_info: dict = {"inactive": t1_off} if idx is None else {"window": idx}
        if st.n_t2 is None:
            st.n_t2 = clamp(awake)
        reasons: list[str] = []
        if idx is None:
            if t1_off != "tier1_after_trace":
                st.n_t1 = None  # before t0 / no replay: Tier-2 alone
            # after the trace: the last forecast stays the floor (no evidence-free scale-down)
        elif t1_key != st.t1_window:
            st.t1_window = t1_key
            N, t1_info = self._tier1_n(ms.model, idx, t1m)
            if N is None:
                reasons.append("tier1_error")  # N_T1 keeps its last value
            else:
                st.n_t1 = clamp(N)
                reasons.append("tier1_window")
                if self.t2_reset_at_window:
                    st.n_t2 = lo
                    st.credited.clear()  # a pod still overloaded counts again on the new base

        # Tier-2: incomplete inputs hold it in both directions
        hold = sorted(set(gaps) | ({"no_kv_capacity"} if skipped else set()))
        down_info: Optional[str] = None
        if hold:
            st.t2_hold_ticks += 1
            down_info = "tier2_hold"
        else:
            fresh = [p for p in overloaded if p not in st.credited]
            if fresh:
                st.credited.update(fresh)
                cur = max(st.n_t1 if st.n_t1 is not None else lo, st.n_t2)
                st.n_t2 = clamp(cur + len(fresh))
                reasons.append("tier2_overload")
            else:
                down_key: Hashable = (t1_key if t1_key is not None
                                      else ("wall", int(snap.now_ms) // int(self.window_s * 1000)))
                if not max_us:
                    down_info = "no_pods"
                elif st.pending:
                    down_info = "pending_prefill"  # queued/prefilling is not idle decode capacity
                elif any(u >= self.t_f for u in max_us):
                    down_info = "above_t_f"
                elif down_key == st.t2_down_key:
                    down_info = "done_this_window"
                else:
                    keep = max(1, int(math.ceil(sum(max_us) / self.t_f - 1e-9)))
                    if keep < st.n_t2:
                        st.t2_down_key = down_key
                        st.n_t2 = clamp(keep)
                        reasons.append("tier2_underload")
                    else:
                        down_info = "no_gain"
        desired = clamp(max(st.n_t1 if st.n_t1 is not None else lo, st.n_t2))
        frozen = bool(hold)
        if frozen:
            reasons.append("tier2_frozen")

        for k, v in anom.items():
            cum[k] += v
        inputs: dict[str, Any] = {
            "awake": awake, "target": desired, "n_t1": st.n_t1, "n_t2": st.n_t2,
            "frozen": frozen, "tier1": t1_info, "tier2": t2, "t2_hold_ticks": st.t2_hold_ticks,
        }
        if gaps:
            inputs["gaps"] = list(gaps)
        if hold:
            inputs["t2_hold"] = hold
        if down_info:
            inputs["down"] = down_info
        if skipped:
            inputs["skipped"] = skipped
        if kv_disagree:
            inputs["kv_disagree"] = kv_disagree
        if anom:
            inputs["anom"] = dict(sorted(anom.items()))
        if self.errors is not None:
            inputs["load_err"] = {"new": st.err_rows, "cum": st.err.summary()}
        return Decision(desired=int(desired), reason="+".join(reasons) or "hold", inputs=inputs)
