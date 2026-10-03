"""B' - the severity-aligned CRITICAL criterion (user 2026-09-24; the acceptance gate since
2026-10-03).

B' replaces the old criterion B (CRITICAL recall of both / TPOT-only violations) as the
gate of ``dline_refit accept``: the CRITICAL recall of the violating windows whose
*severity* is at least the model's TRAINING .65 quantile of violating-window severity,
and the CRITICAL false alarm on healthy windows, both with their cell-bootstrap CI95.

* severity of a window = ``0.8 * p95 ratio + 0.2 * average ratio`` - the grading of
  ``tre_calibration.fit.fit_delta_margins`` (its own critical cut is the same .65
  quantile); the average falls back to the p95 ratio (always so under the D6' slowdown
  label, whose windows carry no average ratio);
* the cut comes from the training windows only, and is sealed before any test window is
  read: ``dline_refit freeze`` computes it from the freeze's training CSV and writes it
  into the freeze (format revision 3); a freeze that predates that takes it from an
  explicit thresholds file (``--b-prime-thresholds``, e.g. the 2026-09-24
  ``b_prime_thresholds.json``: ``severity_cut_train`` + ``gate``);
* CRITICAL = ``Z < tau_crit`` dwell-confirmed over ``dwell_windows`` new windows
  (``theta_verdict.critical_dwell_flags`` = the controller's ``tre_common.dwell``). The
  gate is judged at the dwell the controller runs with (``TRE_DWELL_WINDOWS``, 1 since the
  v1 alignment); dwell 2 and no dwell are disclosed next to it.

Used by ``scripts.dline_refit`` (accept: the gate; freeze: the cut) and by
``scripts.v1_lambda_compare`` (disclosure tables, through ``scripts.v1_lambda_fit``).
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

#: The training quantile of violating-window severity B' recalls above.
SEVERITY_QUANTILE = 0.65
SEVERITY_RULE = ("0.8 * p95 ratio + 0.2 * average ratio (tre_calibration.fit.fit_delta_margins "
                 "grading; the average falls back to the p95 ratio)")
#: The gate (b_prime_thresholds.json of 2026-09-24, decided before any test number of B').
GATE_KEYS = ("recall_severe_min", "recall_severe_ci95_low_min", "false_alarm_max", "false_alarm_ci95_high_max")
DEFAULT_GATE = {"recall_severe_min": 0.80, "recall_severe_ci95_low_min": 0.70,
                "false_alarm_max": 0.05, "false_alarm_ci95_high_max": 0.08}
#: Dwell always disclosed next to the gate's own (the D22 deployed form).
DISCLOSED_DWELL_WINDOWS = (1, 2)


def severity(w) -> float:
    """Severity of one v2 window (:data:`SEVERITY_RULE`)."""
    p95 = w.latency_ratio_p95 if w.latency_ratio_p95 is not None else (1.0 / w.health_score) - 1.0
    avg = w.latency_ratio_avg if w.latency_ratio_avg is not None else p95
    return 0.8 * p95 + 0.2 * avg


def violating_severities(train_windows: Sequence[Any]) -> list[float]:
    return [severity(w) for w in train_windows if math.isfinite(w.signal) and not w.slo_met]


def severity_cut(train_windows: Sequence[Any], q: float = SEVERITY_QUANTILE) -> float:
    """The training set's severity quantile of the violating windows - the cut
    ``fit_delta_margins`` labels its critical windows with (same ``_quantile``)."""
    from tre_calibration.fit import _quantile

    cut = _quantile(violating_severities(train_windows), q)
    if cut is None:
        raise ValueError("no violating training window: the severity cut is undefined")
    return float(cut)


def _rate(sel: Sequence[int], flags: Sequence[bool]) -> Optional[float]:
    return (sum(1 for i in sel if flags[i]) / len(sel)) if sel else None


def _sets(windows: Sequence[Any], cut: float):
    idx = [i for i, w in enumerate(windows) if math.isfinite(w.signal)]
    viol = [i for i in idx if not windows[i].slo_met]
    ok = [i for i in idx if windows[i].slo_met]
    sev = [i for i in viol if severity(windows[i]) >= cut]
    return idx, viol, ok, sev


def series_point(windows: Sequence[Any], *, theta: float, cut: float, crit: Sequence[bool]) -> dict:
    """One CRITICAL flag series: recall of the severe violations, healthy false alarm,
    all-violation recall, and the share of missed violations the slow loop sees (Z < 1)."""
    idx, viol, ok, sev = _sets(windows, cut)
    missed = [i for i in viol if not crit[i]]
    return {"recall_severe": _rate(sev, crit), "false_alarm": _rate(ok, crit), "recall_all": _rate(viol, crit),
            "missed_caught_by_slow_loop": (sum(1 for i in missed if windows[i].signal / theta < 1.0) / len(missed))
            if missed else None}


def band_shares(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float) -> dict:
    """Window counts and where the violations fall (CRITICAL / LOW band tau_crit <= Z < 1 /
    Z >= 1); independent of the dwell."""
    idx, viol, ok, sev = _sets(windows, cut)
    z = {i: windows[i].signal / theta for i in idx}
    share = (lambda n: (n / len(viol)) if viol else None)
    return {"violating": len(viol), "healthy": len(ok), "severe_windows": len(sev), "severity_cut": cut,
            "violations_by_band": {"critical": share(sum(1 for i in viol if z[i] < tau_crit)),
                                   "low": share(sum(1 for i in viol if tau_crit <= z[i] < 1.0)),
                                   "healthy_side_z_ge_1": share(sum(1 for i in viol if z[i] >= 1.0))}}


def b_prime_point(windows: Sequence[Any], *, theta: float, tau_crit: float, cut: float,
                  crit0: Sequence[bool], crit2: Sequence[bool]) -> dict:
    """B' on one window list, no dwell and dwell 2 (the disclosure shape of
    ``v1_lambda_compare``)."""
    p0 = series_point(windows, theta=theta, cut=cut, crit=crit0)
    p2 = series_point(windows, theta=theta, cut=cut, crit=crit2)
    keys = ("recall_severe", "false_alarm", "recall_all")
    return {**{k: v for k, v in band_shares(windows, theta=theta, tau_crit=tau_crit, cut=cut).items()},
            "nodwell": {k: p0[k] for k in keys}, "dwell2": {k: p2[k] for k in keys},
            "missed_caught_by_slow_loop": p0["missed_caught_by_slow_loop"]}


def b_prime_boot(windows: Sequence[Any], *, theta: float = 0.0, tau_crit: float = 0.0, cut: float,
                 crit: Sequence[bool], n: int = 1000, seed: int = 20260922) -> dict:
    """Cell bootstrap (scenario ids with replacement) CI95 of the B' recall / false alarm
    for one flag series (``theta`` / ``tau_crit`` unused, kept for the old call sites)."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        if math.isfinite(w.signal):
            by[w.scenario_id].append(i)
    cells = sorted(by)
    cnt = {}
    for c in cells:
        sev = [i for i in by[c] if not windows[i].slo_met and severity(windows[i]) >= cut]
        ok = [i for i in by[c] if windows[i].slo_met]
        cnt[c] = (sum(crit[i] for i in sev), len(sev), sum(crit[i] for i in ok), len(ok))
    rec, fa = [], []
    rng = random.Random(seed)
    for _ in range(n if cells else 0):
        pick = [rng.choice(cells) for _ in cells]
        a = [sum(cnt[c][k] for c in pick) for k in range(4)]
        if a[1]:
            rec.append(a[0] / a[1])
        if a[3]:
            fa.append(a[2] / a[3])

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [None, None]

    return {"recall_severe_ci95": ci(rec), "false_alarm_ci95": ci(fa), "n": n, "seed": seed}


def check_gate(gate: Mapping[str, Any]) -> dict:
    """``gate`` with exactly the :data:`GATE_KEYS`, each a finite number in [0, 1]; raises
    ``ValueError`` otherwise."""
    out = {}
    for k in GATE_KEYS:
        v = gate.get(k) if isinstance(gate, Mapping) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"B' gate {k!r} is {v!r}: needs a number in [0, 1]")
        out[k] = float(v)
    return out


def load_thresholds(path: Path) -> dict:
    """An explicit B' thresholds file (``severity_cut_train`` {model: cut} + ``gate``),
    for a freeze that predates the sealed cut. Raises ``ValueError`` on anything else."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable ({exc})")
    cuts = doc.get("severity_cut_train") if isinstance(doc, dict) else None
    if not isinstance(cuts, dict) or not cuts:
        raise ValueError(f"{path}: no severity_cut_train {{model: cut}}")
    for m, c in cuts.items():
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
            raise ValueError(f"{path}: severity_cut_train[{m!r}] = {c!r} is not a positive number")
    return {"severity_cut_train": {m: float(c) for m, c in cuts.items()}, "gate": check_gate(doc.get("gate") or {}),
            "doc": doc}


def criteria(point: Mapping[str, Any], ci: Mapping[str, Any], gate: Mapping[str, float]) -> list[dict]:
    """The four B' criteria of one flag series (``point`` from :func:`series_point`,
    ``ci`` from :func:`b_prime_boot`)."""
    def crit(name, value, op, threshold):
        v = value if isinstance(value, (int, float)) and math.isfinite(value) else None
        met = v is not None and (v >= threshold if op == ">=" else v <= threshold)
        return {"name": name, "value": v, "op": op, "threshold": threshold, "met": met}

    return [crit("CRITICAL recall of severe violations", point["recall_severe"], ">=", gate["recall_severe_min"]),
            crit("its CI95 lower bound", ci["recall_severe_ci95"][0], ">=", gate["recall_severe_ci95_low_min"]),
            crit("CRITICAL false alarm on healthy windows", point["false_alarm"], "<=", gate["false_alarm_max"]),
            crit("its CI95 upper bound", ci["false_alarm_ci95"][1], "<=", gate["false_alarm_ci95_high_max"])]
