"""Chiron baseline (interactive-only): per-instance batch size B + global IBP loop.

Source: Chiron section 4.1 (Algorithm 1) and sections 5.1-5.2. ``B`` only feeds the busy
test of the global loop; it is never pushed to the engine.

``batch_mode`` (decision 2026-10-06, methodology "10-06 sanity 第二轮"):

* ``static`` (main runs): B = ``static_b`` for every pod, fixed (set it to the engine's
  max_num_seqs, 256). This is the paper's own ablation that replaces the local autoscaler
  with static batch sizes. Reason: Alg.1 acts on the engine's real batch cap, and TBP only
  means something when a throughput change was caused by the previous B change. The
  substrate cannot change max_num_seqs at run time, so a virtual B leaves the engine
  unchanged and TBP tracks load noise ("halve vs ~x1.1 grow" is a random walk towards
  B = 1; sanity S4: 8b B 153 -> 1, target stuck at the cap). The Alg.1 loop does not run.
* ``alg1`` (sensitivity row ``sensitivity/chiron-alg1.yaml``): the virtual Alg.1 loop below.

Local loop (alg1 only; per pod, every tick), from deltas of the pod's cumulative counters:
  ITL = d(itl_sum)/d(itl_count); LBP = ITL/ITL_SLO; thr = d(gen_tokens)/dt;
  TBP = thr_prev/thr only when the cap was binding on the previous tick, else
  neutral (dropped from the max; a literal 1 would forbid growth);
  LocalBP = max(LBP, TBP); LocalBP < 1: B <- a*B/LocalBP + (1-a)*B, else B <- max(1, B/2).
Global loop (Chiron-global): IBP = busy / N; desired = max(1, ceil(busy / theta)) (reason
  ``ibp_target``): the instance count at which IBP would sit at theta, i.e. the paper's
  over-provisioning level (section 5.2: keep enough idle instances that a burst of
  1/theta x fits). It depends only on ``busy``, so a constant load gives a constant target;
  a +-1 step on ``IBP > theta`` / ``IBP < theta`` instead flip-flops whenever theta is not
  exactly 1/k (e.g. theta 0.37, busy 1: N=2 -> .5 > .37 up, N=3 -> .33 < .37 down).

``busy`` (ours, 4th disclosed adaptation, decision 2026-10-06): the paper counts the
instances running interactive requests under *packing* routing. Our gateway spreads
requests over every awake pod, so "pods with running > 0" grows with N itself and the
target ratchets to the cap. The default ``busy_def: effective`` counts the instances the
load would fill if packed: ``busy = ceil(sum_pods(running + waiting) / mean B)``, with B
the pods' batch sizes (``static_b``, or the virtual caps of the alg1 loop). ``at_cap`` / ``nonidle`` remain for
sensitivity runs.

State band on the scale-down edge (ours, part of adaptation 4, decision 2026-10-06). With
x = sum_q / mean B, busy rises to k as soon as x > k - 1 (the ceil above, unchanged), but
falls from k to k - 1 only when x <= k - 1 - h (``busy_band_h``); from 1 to 0 only when
sum_q = 0. Under packing the paper's count is ~ceil(x) and its hysteresis is implicit: a
request that spilled to instance k runs there to completion (~20 s on our trace), so k stays
"running" while x dips. We re-pack every tick and lose that state; the band restores it on
the down edge only. It is a state gate, not a timer. The previous busy is per-model state;
it restarts from the ceil on a shell restart (new policy object), when the model has no
pods, and when the awake pod set changes in a way this policy did not ask for (a pod
replaced, another actuator, an owner / run-mode change that moved pods: :func:`_carried`).
The policy's own scale step keeps it (a reset there would re-open the flap at every step).
With an evidence gap it may rise but not fall (the gap cannot prove the load is low).
``h = 0`` is exactly the old ceil (sensitivity row). Only ``busy_def: effective`` is banded.

Params (``config.policy_params``; see ``examples/chiron.yaml``):

==============  ===========  ==============================================================
key             default      origin
==============  ===========  ==============================================================
batch_mode      static       static (paper ablation: static batch size; main) | alg1
static_b        (required    B of every pod in static mode, an int > 0 (main: the engine
                in static)   max_num_seqs, 256); no default; ignored in alg1
alpha           0.5          paper (Alg.1 smoothing factor); alg1 only
b_init          None         ours: initial B = b_init, else model max_num_seqs, else 256;
                             alg1 only
b_max           None         ours (part of adaptation 4; alg1 only): cap on B = b_max, else
                             min(max_num_seqs, floor(num_gpu_blocks x block_size /
                             kv_request_tokens)) - the requests of the trace shape the
                             engine's KV cache holds at once (num_gpu_blocks / block_size
                             from the pods' vllm:cache_config_info) - else max_num_seqs,
                             else 256; the value and its source are in the decision inputs
kv_request_tokens None       ours: in + out tokens of one request of the trace shape (the
                             profile shape, e.g. 492 + 400 = 892)
busy_def        effective    ours (see above): effective (packed busy count) | at_cap
                             (running+waiting >= B) | nonidle (running > 0); the last two
                             only as sensitivity runs
busy_band_h     0.25         ours (see above): ``{model | "*": h}`` or one number >= 0; main
                             runs h = max(0.25, 3 sigma(sum_q) / mean B) from the sanity 3x
                             plateau (30 s windows); 0 = no band (sensitivity row)
theta         (required)   ``{model | "*": theta}`` or one number; no silent default: the
                             main runs use theta_trace (``tools/chiron_theta``), the 3x
                             example (1/3) is a sensitivity row; the policy refuses to start
                             without it for a managed model
==============  ===========  ==============================================================

Unknown is not idle: a pod with a missing gauge is never counted busy (scale-up uses the
evidence there is) and any gap (:func:`~tre_baselines.snapshot.evidence_gaps`) holds a
scale-down at ``awake`` (reason ``incomplete``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tre_baselines.policies.base import Decision, hold_if_incomplete
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, evidence_gaps

DEFAULT_ALPHA = 0.5
DEFAULT_B = 256  # not in paper; chosen: order of the engine default max-num-seqs
#: busy / theta within this of an integer is that integer (theta written as 0.3333333333).
_CEIL_TOL = 1e-6
BUSY_DEFS = ("effective", "at_cap", "nonidle")
BATCH_MODES = ("static", "alg1")
_NEEDED = ("gen_tokens", "itl_sum", "itl_count")
#: Scale-down band in units of mean B (spec floor of max(0.25, 3 sigma / B)).
DEFAULT_BUSY_BAND_H = 0.25


def banded_busy(prev: Optional[int], x: float, h: float) -> int:
    """Packed busy count for load ``x`` = sum_q / mean B, given the previous count.

    Up edge: ceil(x) (unchanged). Down edge: from k to k - 1 only when x <= k - 1 - h
    (k = 1 -> 0 only when x = 0). ``prev`` None or ``h`` = 0 gives ceil(x) exactly."""
    c = int(math.ceil(x - _CEIL_TOL)) if x > 0 else 0
    if prev is None or c >= prev:
        return c
    k = prev
    while k > c and (k == 1 or x <= k - 1 - h + _CEIL_TOL):
        k -= 1
    return k


@dataclass
class _Band:
    busy: int
    pods: frozenset  # the awake pod set the count was made on
    desired: int     # what this policy asked for on that tick


def _carried(band: Optional[_Band], pods: frozenset) -> Optional[int]:
    """The previous busy if it still applies to ``pods``, else None (restart from the ceil).

    It applies on the same pod set and after a move this policy asked for (pods only added
    up to its desired, or only removed down to it). Any other change - a pod replaced,
    another actuator, an owner or run-mode change that moved pods - restarts the band."""
    if band is None:
        return None
    if pods == band.pods:
        return band.busy
    old, n = band.pods, len(pods)
    if (pods > old and n <= band.desired) or (pods < old and n >= band.desired):
        return band.busy
    return None


@dataclass
class _PodState:
    B: float
    prev: Optional[dict] = None  # counters at the previous tick
    prev_at_ms: int = 0
    thr_prev: Optional[float] = None
    capped_prev: bool = False  # running >= B (cap in force) at the previous tick


def _num(x: Any, nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


class ChironPolicy:
    name = "chiron"
    #: Arm name in decision records and docs (B is virtual; only the global loop scales).
    label = "Chiron-global"
    needs_events = False

    def __init__(self, config: Any = None) -> None:
        params = dict(getattr(config, "policy_params", None) or {})
        self.batch_mode = str(params.get("batch_mode", "static"))
        if self.batch_mode not in BATCH_MODES:
            raise ValueError(f"chiron: batch_mode must be one of {BATCH_MODES}")
        self.static_b: Optional[int] = None
        if self.batch_mode == "static":
            sb = params.get("static_b")
            if isinstance(sb, bool) or not isinstance(sb, int) or sb <= 0:
                raise ValueError("chiron: batch_mode static needs static_b, an int > 0 "
                                 "(the engine max_num_seqs; no default)")
            self.static_b = sb
        self.alpha = float(params.get("alpha", DEFAULT_ALPHA))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("chiron: alpha must be in (0, 1]")
        self.b_init = params.get("b_init")
        self.b_max = params.get("b_max")
        kv_req = params.get("kv_request_tokens")
        self.kv_request_tokens = None if kv_req is None else int(kv_req)
        if self.kv_request_tokens is not None and self.kv_request_tokens <= 0:
            raise ValueError("chiron: kv_request_tokens must be > 0")
        self.busy_def = str(params.get("busy_def", "effective"))
        if self.busy_def not in BUSY_DEFS:
            raise ValueError(f"chiron: busy_def must be one of {BUSY_DEFS}")
        theta = params.get("theta")
        if theta is None:
            raise ValueError("chiron: params.theta is required (theta_trace from tools/chiron_theta; "
                             "1/3 only as a sensitivity run)")
        if not isinstance(theta, Mapping):
            theta = {"*": theta}
        unset = sorted(str(k) for k, v in theta.items() if v is None)
        if unset:
            raise ValueError(f"chiron: theta is null for {unset} (fill theta_trace from tools/chiron_theta)")
        self.theta = {str(k): float(v) for k, v in theta.items()}
        if any(not 0.0 < v <= 1.0 for v in self.theta.values()):
            raise ValueError("chiron: theta must be in (0, 1]")
        missing = [m for m in (getattr(config, "models", None) or {}) if m not in self.theta and "*" not in self.theta]
        if missing:
            raise ValueError(f"chiron: no theta for {sorted(missing)} (and no '*')")
        band = params.get("busy_band_h", DEFAULT_BUSY_BAND_H)
        if not isinstance(band, Mapping):
            band = {"*": band}
        self.busy_band_h = {str(k): float(v) for k, v in band.items()}
        if any(not (v >= 0.0 and math.isfinite(v)) for v in self.busy_band_h.values()):
            raise ValueError("chiron: busy_band_h must be a finite number >= 0")
        self._state: dict[str, dict[str, _PodState]] = {}
        self._band: dict[str, _Band] = {}

    def _h_for(self, model: str) -> float:
        h = self.busy_band_h
        return h[model] if model in h else h.get("*", DEFAULT_BUSY_BAND_H)

    def _theta_for(self, model: str) -> float:
        try:
            return self.theta[model] if model in self.theta else self.theta["*"]
        except KeyError:
            raise ValueError(f"chiron: no theta for {model!r}") from None

    def _b_max(self, ms: ModelSnapshot) -> tuple[float, str]:
        """(cap on the virtual B, where it came from)."""
        if self.b_max is not None:
            return float(self.b_max), "param"
        seqs = ms.max_num_seqs or DEFAULT_B
        if self.kv_request_tokens:
            caps = [p.num_gpu_blocks * p.block_size // self.kv_request_tokens for p in ms.pods
                    if p.num_gpu_blocks and p.block_size]
            if caps:  # the requests of the trace shape the KV cache holds at once
                kv = max(1, min(caps))
                return float(min(seqs, kv)), "kv_cache" if kv < seqs else "max_num_seqs"
        return float(seqs), "max_num_seqs"

    def _b_bounds(self, ms: ModelSnapshot) -> tuple[float, float]:
        if self.static_b is not None:  # static mode: B is static_b, recorded as such
            self._b_max_src = "static_b"
            return float(self.static_b), float(self.static_b)
        b_init = self.b_init if self.b_init is not None else (ms.max_num_seqs or DEFAULT_B)
        b_max, self._b_max_src = self._b_max(ms)
        return float(b_init), float(b_max)

    def _local(self, st: _PodState, pod: PodSnapshot, ms: ModelSnapshot, b_max: float) -> dict:
        """One Alg.1 step for one pod; returns the per-pod inputs entry."""
        info: dict[str, Any] = {}
        c = pod.counters
        # cap in force during the window that just ended (unknown gauge: not capped)
        capped_now = pod.running is not None and pod.running >= st.B
        prev, prev_at = st.prev, st.prev_at_ms
        st.prev, st.prev_at_ms = dict(c), pod.scraped_at_ms
        if prev is None or any(k not in c or k not in prev for k in _NEEDED):
            st.capped_prev = capped_now
            info["skip"] = "baseline" if prev is None else "missing_counters"
            return info
        d = {k: c[k] - prev[k] for k in _NEEDED}
        dt = (pod.scraped_at_ms - prev_at) / 1000.0
        if any(v < 0 for v in d.values()):  # counter reset (engine restart): re-baseline
            st.thr_prev, st.capped_prev = None, capped_now
            info["skip"] = "counter_reset"
            return info
        if dt <= 0 or d["itl_count"] <= 0 or d["gen_tokens"] <= 0:
            st.capped_prev = capped_now
            if d["gen_tokens"] <= 0 and dt > 0:
                st.thr_prev = 0.0
            info["skip"] = "empty_window"
            return info
        itl = d["itl_sum"] / d["itl_count"]
        thr = d["gen_tokens"] / dt
        lbp = itl / (ms.tpot_slo_ms / 1000.0)
        # not in paper; chosen: TBP only counts when the cap was binding on the previous
        # tick, so an arrival-rate drop does not halve B.
        # With the gate off TBP is neutral (left out of the max): a literal TBP = 1 would
        # make LocalBP >= 1 always and B could never grow.
        tbp = st.thr_prev / thr if (st.thr_prev is not None and st.capped_prev) else None
        bp = max(lbp, tbp) if tbp is not None else lbp
        if bp < 1.0:
            st.B = self.alpha * st.B / max(bp, 1e-9) + (1.0 - self.alpha) * st.B
        else:
            st.B = max(1.0, st.B / 2.0)
        st.B = min(st.B, b_max)  # not in paper; chosen: cap B at b_max
        st.thr_prev, st.capped_prev = thr, capped_now
        info.update(LBP=_num(lbp), TBP=_num(tbp), ITL=_num(itl, 5), thr=_num(thr, 2))
        return info

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        out: dict[str, Decision] = {}
        for model, ms in snap.models.items():
            states = self._state.setdefault(model, {})
            b_init, b_max = self._b_bounds(ms)
            live = {p.pod for p in ms.pods} | set(ms.unscraped)
            for gone in [k for k in states if k not in live]:
                del states[gone]
            n = len(ms.pods)
            theta = self._theta_for(model)
            if n == 0:
                self._band.pop(model, None)
                out[model] = Decision(ms.awake, "no_pods", {"N": 0, "theta": _num(theta)})
                continue
            gaps = evidence_gaps(ms, snap.now_ms, snap.tick_s)
            per_pod: dict[str, Any] = {}
            busy = 0
            queued = 0.0
            caps: list[float] = []
            for pod in ms.pods:
                st = states.get(pod.pod)
                if st is None:
                    st = states[pod.pod] = _PodState(B=min(b_init, b_max))
                # static mode: B stays static_b, the Alg.1 loop does not run
                info = self._local(st, pod, ms, b_max) if self.static_b is None else {}
                # Unknown gauges are not busy here (scale-up uses the evidence there is);
                # they are an evidence gap, which blocks the scale-down below.
                caps.append(st.B)
                queued += pod.queued or 0.0
                if self.busy_def == "at_cap":
                    is_busy = pod.queued is not None and pod.queued >= st.B
                else:
                    is_busy = pod.running is not None and pod.running > 0
                busy += int(is_busy)
                info.update(B=_num(st.B, 2), busy=bool(is_busy), q=_num(pod.queued, 2))
                per_pod[pod.pod] = info
            extra: dict[str, Any] = {}
            if self.busy_def == "effective":
                b_mean = sum(caps) / len(caps)
                x = queued / b_mean
                h = self._h_for(model)
                pods_now = frozenset(live)
                prev = _carried(self._band.get(model), pods_now)
                busy_ceil = banded_busy(None, x, 0.0)
                busy = banded_busy(prev, x, h)
                if gaps and prev is not None:  # unknown pods: may rise, cannot fall
                    busy = max(busy, prev)
                extra = {"queued": _num(queued, 2), "B_mean": _num(b_mean, 2), "busy_ceil": busy_ceil,
                         "band_h": _num(h), "busy_prev": prev}
            ibp = busy / n
            desired = max(1, int(math.ceil(busy / theta - _CEIL_TOL)))
            decision = Decision(desired, "ibp_target", {
                "IBP": _num(ibp), "theta": _num(theta), "N": n, "busy": busy, **extra,
                "target": desired, "busy_def": self.busy_def, "batch_mode": self.batch_mode, "b_max": _num(b_max, 1), "b_max_src": self._b_max_src,
                "pods": per_pod,
            })
            final = hold_if_incomplete(decision, ms.awake, gaps)
            if self.busy_def == "effective":
                self._band[model] = _Band(busy, pods_now, int(final.desired))
            out[model] = final
        return out
