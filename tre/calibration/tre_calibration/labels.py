"""The one SLO label every calibration fit uses (plan 2026-09-21 §6.3 B2/B4).

A window is **violated** when

* its p95 TTFT is over ``ttft_p95_ms`` or its p95 TPOT is over ``tpot_p95_ms``, or
* it holds a request that went unserved - a model error, a dropped connection or a
  client timeout - which the window CSV records as ``slo_violated`` (set by
  ``openloop.mark_unserved_request_windows``, online and in ``rewindow_from_raw``).

End-to-end latency is **not** part of the label: it scales with the output length the
load shape chose, not with how loaded the engine was, and the TSS theta was always fitted
on TTFT/TPOT only - fitting the alternative signals with e2e made the ablation compare two
different labels.

A window with a missing p95 (too few samples for the percentile guard) carries no latency
evidence and is dropped - unless it is marked unserved, in which case it is kept as
violated: a window where requests timed out is exactly the one that must not vanish.

Every fit writes :meth:`LabelDefinition.as_dict` into its output as ``label_def`` so two
artifacts can be checked for label equality instead of trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

LABEL_DEF_NAME = "p95_ttft_tpot_plus_unserved_v1"

#: Latency columns of the window CSV, keyed by the SLO name. e2e is listed only so the
#: generic loader keeps reading ad-hoc CSVs; LabelDefinition never asks for it.
LATENCY_COLUMNS = {"ttft_p95": "p95_ttft", "tpot_p95": "p95_tpot", "e2e_p95": "p95_e2e"}

#: Severity ratio assigned to an unserved window with no latency sample at all. It only
#: feeds the delta-margin severity ordering; any value > 1 keeps it on the violated side,
#: and 2.0 ranks it as "clearly over SLO" without dominating measured ratios.
UNSERVED_MIN_RATIO = 2.0


@dataclass(frozen=True)
class LabelDefinition:
    ttft_p95_ms: float
    tpot_p95_ms: float

    def __post_init__(self) -> None:
        for name in ("ttft_p95_ms", "tpot_p95_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

    def latency_slo_ms(self) -> dict[str, float]:
        return {"ttft_p95": float(self.ttft_p95_ms), "tpot_p95": float(self.tpot_p95_ms)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": LABEL_DEF_NAME,
            "ttft_p95_ms": float(self.ttft_p95_ms),
            "tpot_p95_ms": float(self.tpot_p95_ms),
            "e2e": "excluded",
            "unserved": "violated (model_error, proxy_transient, client_timeout -> slo_violated column)",
            "missing_latency": "dropped unless unserved",
            "violated_if": "p95_ttft > ttft_p95_ms or p95_tpot > tpot_p95_ms or slo_violated",
        }


@dataclass(frozen=True)
class WindowLabel:
    slo_met: bool
    #: max(p95 / SLO) over the active SLOs (>= UNSERVED_MIN_RATIO for an unserved window
    #: without any latency sample).
    ratio_max: float
    unserved: bool


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def row_unserved(row: Mapping[str, Any]) -> bool:
    value = row.get("slo_violated")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def label_window(row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]) -> Optional[WindowLabel]:
    """Label one window-CSV row, or None when it carries no usable evidence.

    ``latency_slo_ms`` is keyed by SLO name (``ttft_p95``/``tpot_p95``, or any CSV column
    name for ad-hoc tests); production callers pass :meth:`LabelDefinition.latency_slo_ms`.
    """
    unserved = row_unserved(row)
    ratios: list[float] = []
    missing = False
    for key, slo in latency_slo_ms.items():
        value = _as_float(row.get(LATENCY_COLUMNS.get(key, key)))
        if value is None:
            missing = True
            continue
        ratios.append(value / float(slo))
    if missing or not ratios:
        if not unserved:
            return None
        ratio_max = max(ratios + [UNSERVED_MIN_RATIO])
        return WindowLabel(slo_met=False, ratio_max=ratio_max, unserved=True)
    ratio_max = max(ratios)
    return WindowLabel(
        slo_met=(not unserved) and all(r <= 1.0 for r in ratios),
        ratio_max=ratio_max,
        unserved=unserved,
    )

