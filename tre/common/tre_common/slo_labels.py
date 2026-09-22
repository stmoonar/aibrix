"""The one definition of "did this window meet its SLO".

Why this module exists
----------------------
Until 2026-09-22 the calibration had two answers to that question. The boundary search
judged a probe on the online window CSV, whose p95 TPOT came from the *server's* vLLM
histogram (bucketed, and counting every single-token stall); the fit judged the same
operating point on windows re-built from the *client's* per-request log. On the same
capture the two disagreed on one window in four (7b S5 dwell: 5/10 online violations,
0/56 after re-windowing), so the search placed the boundary with one ruler and the fit
measured the evidence there with another.

Every consumer of a window verdict now calls :func:`window_slo_label`: the probe verdict
in ``scripts.adaptive_boundary``, the window CSVs written by ``scripts.r3_grid`` and
``scripts.rewindow_from_raw``, the fit loader in ``tre_calibration.dataset`` and its
mirror in ``scripts.refit_trs_params``, the capacity fit in ``scripts.r3_capacity``, and
the standard dataset builder. A second implementation is the bug this module replaces.

The label
---------
Latency is **client per-request** (the same basis E1 scores ``V_req`` on):

* TTFT - first streamed token minus the instant the request went on the wire;
* TPOT - ``(e2e - ttft) / (completion_tokens - 1)``, the mean inter-token gap of one
  request (``scripts.r3_grid.build_raw_record``);
* p95 - over the requests that *completed* inside the window, with the live N1 guard:
  fewer than ``min_latency_samples`` completions means "no p95" (``None``), never 0.

A window is, in this order:

1. ``violated`` when any request *sent* inside it went unserved - a model error, a
   transient proxy failure, or a client timeout. A request that never came back has no
   latency sample, so without this rule a window whose slowest work all failed shows a
   comfortable p95 and is scored as healthy.
2. ``unlabeled`` when a p95 the SLO needs is missing (too few completions). No evidence
   is not evidence of health: it is neither side, and every consumer must treat it as
   such rather than as ``False``.
3. ``violated`` when any active p95 is above its SLO.
4. ``healthy`` otherwise.

Column names carry their source and unit. A bare ``p95_tpot`` (no source) is exactly
the ambiguity that let the two rulers coexist, so it no longer appears in any table.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

#: Window verdicts. Strings, not bools, so a missing verdict cannot be read as False.
LABEL_VIOLATED = "violated"
LABEL_HEALTHY = "healthy"
LABEL_UNLABELED = "unlabeled"
LABELS = (LABEL_VIOLATED, LABEL_HEALTHY, LABEL_UNLABELED)

#: Where the latency the label is computed on comes from. Recorded in every manifest.
LATENCY_SOURCE = "client per-request"

#: Client-side p95 columns - the ones every label is computed on.
P95_TTFT_CLIENT = "p95_ttft_client_ms"
P95_TPOT_CLIENT = "p95_tpot_client_ms"
P95_E2E_CLIENT = "p95_e2e_client_ms"
#: Server-side (vLLM histogram, read back from redis) p95 columns. Diagnostic only: no
#: label, fit or verdict reads them.
P95_TTFT_SERVER = "p95_ttft_server_ms"
P95_TPOT_SERVER = "p95_tpot_server_ms"
P95_E2E_SERVER = "p95_e2e_server_ms"

#: SLO key (as the fit CLI and the registry spell it) -> the client column it is judged on.
SLO_COLUMNS: dict[str, str] = {
    "ttft_p95": P95_TTFT_CLIENT,
    "tpot_p95": P95_TPOT_CLIENT,
    "e2e_p95": P95_E2E_CLIENT,
}

#: Per-window counts of requests sent in the window that went unserved. Each one makes
#: the window a violation (rule 1 above).
UNSERVED_COLUMNS = ("model_errors", "proxy_transient_errors", "client_timeouts")

#: The column a window's label is written to.
LABEL_COLUMN = "slo_label"
#: Boolean view of the label: True when violated, False when healthy, empty when
#: unlabeled. Written only by :func:`apply_label`, together with :data:`LABEL_COLUMN`.
VIOLATED_COLUMN = "slo_violated"


def slo_targets(*, ttft_slo_ms: float, tpot_slo_ms: float, e2e_slo_ms: Optional[float] = None) -> dict:
    """The ``latency_slo_ms`` mapping :func:`window_slo_label` takes, from the three SLOs."""
    out = {"ttft_p95": float(ttft_slo_ms), "tpot_p95": float(tpot_slo_ms)}
    if e2e_slo_ms is not None:
        out["e2e_p95"] = float(e2e_slo_ms)
    return out


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "" or value == "None":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_count(value: Any) -> int:
    number = _as_float(value)
    return 0 if number is None else int(number)


def unserved_requests(row: Mapping[str, Any]) -> int:
    """Requests sent inside this window that went unserved (all three classes)."""
    return sum(_as_count(row.get(column)) for column in UNSERVED_COLUMNS)


def latency_ratios(row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]) -> Optional[list[float]]:
    """``p95 / SLO`` for every active SLO, or None when any of those p95s is missing."""
    if not latency_slo_ms:
        raise ValueError("latency_slo_ms must name at least one SLO")
    ratios: list[float] = []
    for key, slo in latency_slo_ms.items():
        column = SLO_COLUMNS.get(key)
        if column is None:
            raise ValueError(f"unknown SLO key {key!r}; expected one of {sorted(SLO_COLUMNS)}")
        value = _as_float(row.get(column))
        if value is None:
            return None
        ratios.append(value / float(slo))
    return ratios


def window_slo_label(row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]) -> str:
    """``violated`` / ``healthy`` / ``unlabeled`` for one window row (see module docstring).

    ``row`` is a window CSV row (strings are fine) or the dict ``scripts.r3_grid`` builds;
    ``latency_slo_ms`` is keyed ``ttft_p95`` / ``tpot_p95`` / ``e2e_p95``.
    """
    if unserved_requests(row) > 0:
        return LABEL_VIOLATED
    ratios = latency_ratios(row, latency_slo_ms)
    if ratios is None:
        return LABEL_UNLABELED
    return LABEL_VIOLATED if any(r > 1.0 for r in ratios) else LABEL_HEALTHY


def apply_label(row: dict, latency_slo_ms: Mapping[str, float]) -> str:
    """Write :data:`LABEL_COLUMN` and :data:`VIOLATED_COLUMN` onto ``row``; return the label.

    The one place a window row gets its verdict columns, so the two can never disagree.
    ``slo_violated`` is None (an empty CSV cell) for an unlabeled window: "no evidence" has
    no truthful boolean, and False would be read as healthy.
    """
    label = window_slo_label(row, latency_slo_ms)
    row[LABEL_COLUMN] = label
    row[VIOLATED_COLUMN] = None if label == LABEL_UNLABELED else (label == LABEL_VIOLATED)
    return label


def label_definition(latency_slo_ms: Mapping[str, float], *, min_latency_samples: int) -> dict:
    """A self-describing record of the label, for manifests and cell artifacts."""
    return {
        "latency_source": LATENCY_SOURCE,
        "ttft": "first streamed token minus the instant the request went on the wire",
        "tpot": "(e2e_ms - ttft_ms) / (completion_tokens - 1), per request",
        "window_membership": {
            "latency": "requests whose completion (done_ts_ms) is in [window_start, window_end)",
            "unserved": "requests whose send instant (send_ts_ms) is in [window_start, window_end)",
        },
        "p95": "tre_common.percentile.histogram_percentile over exact samples, bucket_upper",
        "min_latency_samples": int(min_latency_samples),
        "unserved_classes": list(UNSERVED_COLUMNS),
        "slo_ms": {SLO_COLUMNS[k]: float(v) for k, v in latency_slo_ms.items()},
        "rules": [
            "violated if any request sent in the window went unserved",
            "unlabeled if any active client p95 is missing (fewer than min_latency_samples completions)",
            "violated if any active client p95 exceeds its SLO",
            "healthy otherwise",
        ],
        "implementation": "tre_common.slo_labels.window_slo_label",
    }
