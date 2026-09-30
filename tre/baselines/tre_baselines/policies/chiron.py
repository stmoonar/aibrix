"""Chiron baseline (interactive-only, virtual batch cap): Alg.1 local loop + global IBP loop.

Source: Chiron section 4.1 (Algorithm 1) and sections 5.1-5.2. The batch size ``B`` is
*virtual*: it is never pushed to the engine, it only feeds the busy test of the global loop.

Local loop (per pod, every tick), from deltas of the pod's cumulative counters:
  ITL = d(itl_sum)/d(itl_count); LBP = ITL/ITL_SLO; thr = d(gen_tokens)/dt;
  TBP = thr_prev/thr only when the cap was binding on the previous tick, else
  neutral (dropped from the max; a literal 1 would forbid growth);
  LocalBP = max(LBP, TBP); LocalBP < 1: B <- a*B/LocalBP + (1-a)*B, else B <- max(1, B/2).
Global loop: IBP = busy pods / N; desired = max(1, ceil(busy / theta)) (reason
  ``ibp_target``): the instance count at which IBP would sit at theta, i.e. the paper's
  over-provisioning level (section 5.2: keep enough idle instances that a burst of
  1/theta x fits). It depends only on ``busy``, so a constant load gives a constant target;
  a +-1 step on ``IBP > theta`` / ``IBP < theta`` instead flip-flops whenever theta is not
  exactly 1/k (e.g. theta 0.37, busy 1: N=2 -> .5 > .37 up, N=3 -> .33 < .37 down).

Params (``config.policy_params``; see ``examples/chiron.yaml``):

==============  ===========  ==============================================================
key             default      origin
==============  ===========  ==============================================================
alpha           0.5          paper (Alg.1 smoothing factor)
b_init          None         ours: initial B = b_init, else model max_num_seqs, else 256
b_max           None         ours: cap on B = b_max, else max_num_seqs, else 256
busy_def        at_cap       ours (D11): at_cap (running+waiting >= B) | nonidle (running > 0)
theta           {"*": 1/3}   paper's 3x example as the default; per model via chiron_theta
==============  ===========  ==============================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tre_baselines.policies.base import Decision
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot

DEFAULT_ALPHA = 0.5
DEFAULT_THETA = 1.0 / 3.0  # the paper's 3x-burst example
DEFAULT_B = 256  # not in paper; chosen: order of the engine default max-num-seqs
#: busy / theta within this of an integer is that integer (theta written as 0.3333333333).
_CEIL_TOL = 1e-6
BUSY_DEFS = ("at_cap", "nonidle")
_NEEDED = ("gen_tokens", "itl_sum", "itl_count")


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

    def __init__(self, config: Any = None) -> None:
        params = dict(getattr(config, "policy_params", None) or {})
        self.alpha = float(params.get("alpha", DEFAULT_ALPHA))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("chiron: alpha must be in (0, 1]")
        self.b_init = params.get("b_init")
        self.b_max = params.get("b_max")
        self.busy_def = str(params.get("busy_def", "at_cap"))
        if self.busy_def not in BUSY_DEFS:
            raise ValueError(f"chiron: busy_def must be one of {BUSY_DEFS}")
        theta = params.get("theta", {})
        if not isinstance(theta, Mapping):
            theta = {"*": theta}
        self.theta = {str(k): float(v) for k, v in theta.items()}
        if any(not 0.0 < v <= 1.0 for v in self.theta.values()):
            raise ValueError("chiron: theta must be in (0, 1]")
        self._state: dict[str, dict[str, _PodState]] = {}

    def _theta_for(self, model: str) -> float:
        return self.theta.get(model, self.theta.get("*", DEFAULT_THETA))

    def _b_bounds(self, ms: ModelSnapshot) -> tuple[float, float]:
        b_init = self.b_init if self.b_init is not None else (ms.max_num_seqs or DEFAULT_B)
        b_max = self.b_max if self.b_max is not None else (ms.max_num_seqs or DEFAULT_B)
        return float(b_init), float(b_max)

    def _local(self, st: _PodState, pod: PodSnapshot, ms: ModelSnapshot, b_max: float) -> dict:
        """One Alg.1 step for one pod; returns the per-pod inputs entry."""
        info: dict[str, Any] = {}
        c = pod.counters
        capped_now = pod.running >= st.B  # cap in force during the window that just ended
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
                out[model] = Decision(ms.awake, "no_pods", {"N": 0, "theta": _num(theta)})
                continue
            per_pod: dict[str, Any] = {}
            busy = 0
            for pod in ms.pods:
                st = states.get(pod.pod)
                if st is None:
                    st = states[pod.pod] = _PodState(B=min(b_init, b_max))
                info = self._local(st, pod, ms, b_max)
                if self.busy_def == "at_cap":
                    is_busy = pod.running + pod.waiting >= st.B
                else:
                    is_busy = pod.running > 0
                busy += int(is_busy)
                info.update(B=_num(st.B, 2), busy=bool(is_busy))
                per_pod[pod.pod] = info
            ibp = busy / n
            desired = max(1, int(math.ceil(busy / theta - _CEIL_TOL)))
            out[model] = Decision(desired, "ibp_target", {
                "IBP": _num(ibp), "theta": _num(theta), "N": n, "busy": busy,
                "target": desired, "busy_def": self.busy_def, "pods": per_pod,
            })
        return out
