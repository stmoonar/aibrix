"""The one definition of "did this window meet its SLO".

Why this module exists
----------------------
Until 2026-09-22 the calibration had two answers to that question. The boundary search
judged a probe on the online window CSV, whose p95 TPOT came from the *server's* vLLM
histogram (bucketed, and counting every single-token stall); the fit judged the same
operating point on windows re-built from the *client's* per-request log. On the same
capture the two disagreed on one window in four (7b S5 dwell: 5/10 online violations,
0/56 after re-windowing), so the search placed the boundary with one ruler and the fit
measured the evidence there with another. Two lines of work then each grew a label of
their own (``tre_calibration.labels`` on the 09-22 line, this module on the 09-23 line);
the merge of 2026-09-23 folded both into this file, and it is the only one.

Every consumer of a window verdict goes through here: the probe verdict in
``scripts.adaptive_boundary`` and ``scripts.calibration_design``, the window CSVs written
by ``scripts.r3_grid`` and ``scripts.rewindow_from_raw`` (``label_cell``), the fit loader
``tre_calibration.dataset.calibration_window_from_row`` and its users (``cli``,
``refit_trs_params``, ``theta_verdict``, ``alpha_fit``, ``dline_refit``, the alt-signal
fits), the capacity fit in ``scripts.r3_capacity`` and the standard dataset builder.
A second implementation is the bug this module replaces.

The label
---------
Latency is **client per-request** (the same basis E1 scores ``V_req`` on):

* TTFT - first streamed token minus the instant the request went on the wire;
* TPOT - ``(e2e - ttft) / (completion_tokens - 1)``, the mean inter-token gap of one
  request (``scripts.r3_grid.build_raw_record``);
* both over the requests that were *served* and completed inside the window; the
  re-windower's p95 guard (``min_latency_samples``, 10) leaves a p95 empty below it.

A window is labelled by a :class:`LabelDefinition`, in this order:

1. ``violated`` when any request *sent* inside it went unserved - a model error, a
   transient proxy failure or a client timeout, i.e. any of the three count columns
   :data:`UNSERVED_COLUMNS` is non-zero. The counts are the only unserved evidence read:
   ``slo_violated`` is this module's *output* column and never an input (reading it as
   "unserved" is how a 09-22 reader of a 09-23 CSV turned every latency violation into
   an unserved one).
2. ``unlabeled`` when the window holds fewer than ``min_completed_requests`` completed
   requests (default 20: a p95 over fewer is essentially the window maximum), or a p95
   the label needs is missing. No evidence is not evidence of health: every consumer
   treats it as neither side, never as ``False``.
3. ``violated`` when the window TTFT term or the p95 TPOT is above its SLO.
4. ``healthy`` otherwise.

The thresholds come from ``tre_common.slo`` (the one SLO definition); the label's
``ttft_slo_mode`` is one of its modes (default: the registry's top-level ``slo_mode``):

``idle_multiple`` - the 2026-11-04 definition
    like ``slowdown`` with the model's own TTFT multiple and floor
    (``models[].slo.idle_multiple.ttft_k`` / ``ttft_floor_ms``) and the TPOT
    threshold ``tpot_k * tpot_idle_ms`` (the label's ``tpot_p95_ms`` is that product).

``slowdown`` - the primary label (plan 2026-09-21 §6.11 D6')
    per request ``TTFT_slo(L) = max(floor, k * (c_m + b_m * L))`` with the model's idle
    TTFT fit ``c_m + b_m * L``, k and floor from the registry ``slo`` block; the
    window term is ``p95_i(TTFT_i / TTFT_slo(L_i))`` over the window's requests, read from
    the per-request ``ttft_len_samples`` column. A CSV without it is refused rather than
    silently labelled with the fixed rule.
``fixed`` - the comparison column
    window p95 TTFT against one ``ttft_p95_ms`` for every prompt length.

Every threshold parameter is the registry's (``models[].slo``, through ``tre_common.slo``)
or an explicit argument: this module has no default SLO value (2026-11-04).

Every published window table carries three arms (:func:`label_arms`): ``slo_label`` /
``slo_violated`` are the **primary** (D6') label, ``slo_label_fixed`` the fixed-mode
comparison (registry ``ttft_p95_ms`` / ``tpot_p95_ms``) and ``slo_label_k3`` the k = 3 /
150 ms-floor ablation (D6, plan 2026-09-21 6.9h: a label column of that design, not an SLO). End-to-end
latency is not part of any arm: it scales with the output length the load shape chose,
not with how loaded the engine was.

Column names carry their source and unit (``p95_*_client_ms``). The bare 09-22 names
``p95_ttft`` / ``p95_tpot`` are still *read* when the client column is absent, so the
CSVs re-windowed before the merge keep loading; nothing writes them any more.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Union

from tre_common import slo as _slo
from tre_common.percentile import histogram_percentile

# --------------------------------------------------------------------------- columns

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
#: The 09-22 names of the same client columns (``rewindow_from_raw`` wrote client p95s
#: under them). Read-only fallback: consulted only when the ``*_client_ms`` column is
#: absent from a row; never written.
LEGACY_LATENCY_COLUMNS: dict[str, str] = {
    "ttft_p95": "p95_ttft",
    "tpot_p95": "p95_tpot",
    "e2e_p95": "p95_e2e",
}

#: Per-window counts of requests sent in the window that went unserved. Each one makes
#: the window a violation (rule 1 above). The ONLY unserved evidence a label reads.
UNSERVED_COLUMNS = ("model_errors", "proxy_transient_errors", "client_timeouts")

#: Requests served and completed inside the window (the min-n guard).
COMPLETED_REQUESTS_COLUMN = "completed_requests"
#: ``ttft_ms:prompt_tokens`` of every served request completed inside the window, ``;``
#: separated - the per-request evidence the slowdown TTFT term needs.
TTFT_LEN_SAMPLES_COLUMN = "ttft_len_samples"

#: The column a window's primary label is written to.
LABEL_COLUMN = "slo_label"
#: Boolean view of the primary label: True when violated, False when healthy, empty when
#: unlabeled. Written only by :func:`apply_label`, together with :data:`LABEL_COLUMN`.
VIOLATED_COLUMN = "slo_violated"
#: The two comparison arms (plan §6.11 D6'), written next to the primary label.
LABEL_COLUMN_FIXED = "slo_label_fixed"
LABEL_COLUMN_K3 = "slo_label_k3"

#: Arm names (``label_arms``) and the column each is written to.
ARM_PRIMARY = "primary"
ARM_FIXED = "fixed_comparison"
ARM_K3 = "ablation_k3_floor150"
ARM_COLUMNS: dict[str, str] = {
    ARM_PRIMARY: LABEL_COLUMN,
    ARM_FIXED: LABEL_COLUMN_FIXED,
    ARM_K3: LABEL_COLUMN_K3,
}
#: Every label column a window table carries, primary first.
LABEL_COLUMNS = (LABEL_COLUMN, VIOLATED_COLUMN, LABEL_COLUMN_FIXED, LABEL_COLUMN_K3)

# ---------------------------------------------------------------------- definitions

LABEL_DEF_NAME = "p95_ttft_tpot_plus_unserved_v1"
LABEL_DEF_NAME_SLOWDOWN = "p95_ttft_slowdown_tpot_plus_unserved_v1"
#: Label definition v2 (2026-10-05, next calibration round): the same verdict rule, with
#: the request-to-window attribution an explicit field (:data:`ATTRIBUTION_HYBRID`). A v1
#: definition (completion attribution) keeps its v1 name and its byte-identical record.
LABEL_DEF_NAME_V2 = "p95_ttft_tpot_plus_unserved_v2"
LABEL_DEF_NAME_SLOWDOWN_V2 = "p95_ttft_slowdown_tpot_plus_unserved_v2"
#: The 2026-11-04 definition (``tre_common.slo`` mode ``idle_multiple``).
LABEL_DEF_NAME_IDLE_MULTIPLE = "p95_ttft_idle_multiple_tpot_idle_multiple_plus_unserved_v1"
LABEL_DEF_NAME_IDLE_MULTIPLE_V2 = "p95_ttft_idle_multiple_tpot_idle_multiple_plus_unserved_v2"

#: Which window a served request's latency samples count in.
#: ``completion`` (v1, every label before 2026-10-05): TTFT, TPOT and the min-n count all go
#: to the window holding the request's completion (``done_ts_ms``).
#: ``hybrid`` (v2): the TTFT sample (and the min-n count, ``completed_requests``) goes to the
#: window holding the request's FIRST TOKEN (``recv_first_token_ts_ms``); TPOT / e2e stay in
#: the completion window. An unserved request counts by its send instant under both. Why:
#: by completion, the requests queued during a burst complete in the ~30 s drain tail after
#: it and keep labelling recovered windows as severe violations; by send time, the label
#: leads every window signal by 10-20 s (docs calib-next-round-design-20261005).
ATTRIBUTION_COMPLETION = "completion"
ATTRIBUTION_HYBRID = "hybrid"
ATTRIBUTIONS = (ATTRIBUTION_COMPLETION, ATTRIBUTION_HYBRID)
ATTRIBUTION_RULES = {
    ATTRIBUTION_COMPLETION: "TTFT, TPOT and the min-n count by completion (done_ts_ms)",
    ATTRIBUTION_HYBRID: ("TTFT samples and the min-n count (completed_requests) by first token "
                         "(recv_first_token_ts_ms); TPOT / e2e samples by completion (done_ts_ms)"),
}

TTFT_SLO_MODE_FIXED = _slo.SLO_MODE_FIXED
TTFT_SLO_MODE_SLOWDOWN = _slo.SLO_MODE_SLOWDOWN
TTFT_SLO_MODE_IDLE_MULTIPLE = _slo.SLO_MODE_IDLE_MULTIPLE
TTFT_SLO_MODES = _slo.SLO_MODES

#: Primary label (plan §6.11 D6', supersedes D6): ``max(floor, k * idle TTFT(L))``, k and
#: floor from the registry (DynamoLLM / SLOs-Serve loose 5x; the floor keeps short prompts
#: on the fixed rule - a purely length-normalised SLO turns head-of-line blocking of short
#: prompts into violations TSS cannot see).
#: The mode of the D6' primary label (plan 2026-09-21 6.11): the label the 10-03 calibration
#: chain (calibration_campaign / _supplement / _dataset, dline_refit, their freezes and
#: sealed label hashes) is defined on. Those tools name it explicitly (a frozen record's
#: definition, not the one in force); every other consumer defaults to the registry's
#: ``slo_mode``.
D6_PRIME_MODE = TTFT_SLO_MODE_SLOWDOWN
#: The D6 arm (plan §6.9h: k = 3, 150 ms floor), kept as the "why not pure slowdown" ablation
#: column of the D6' design (not an SLO mode: no consumer judges a request with it).
ABLATION_TTFT_SLOWDOWN_K = 3.0
ABLATION_TTFT_FLOOR_MS = 150.0
#: Plan §6.9g/§6.11 note 6: a p95 over fewer requests is essentially the window maximum.
DEFAULT_MIN_COMPLETED_REQUESTS = 20
#: The re-windower's p95 guard (N1, identical to MetricsStore): fewer samples -> no p95.
DEFAULT_MIN_LATENCY_SAMPLES = 10
#: Same percentile mode ``rewindow_from_raw`` uses for the p95 columns.
DEFAULT_PERCENTILE_MODE = "bucket_upper"

#: Severity ratio assigned to an unserved window with no latency sample at all. It only
#: feeds the delta-margin severity ordering; any value > 1 keeps it on the violated side,
#: and 2.0 ranks it as "clearly over SLO" without dominating measured ratios.
UNSERVED_MIN_RATIO = 2.0

#: Why :meth:`LabelDefinition.classify` returned no label (see :func:`count_label_exclusions`).
EXCLUDED_LOW_N = "low_n"
EXCLUDED_MISSING_LATENCY = "missing_latency"


class LabelInputError(ValueError):
    """The window row lacks a column the label definition needs."""


#: The per-request TTFT threshold ``max(floor, k * (c + b * L))`` (``tre_common.slo``).
request_ttft_slo_ms = _slo.request_ttft_slo_ms


# -------------------------------------------------------------------- row accessors


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


def latency_value(row: Mapping[str, Any], key: str) -> Optional[float]:
    """The client p95 of SLO ``key`` (``ttft_p95`` / ``tpot_p95`` / ``e2e_p95``) in ``row``.

    The ``*_client_ms`` column; only when that column is *absent* from the row, the 09-22
    bare name (read-only compatibility)."""
    column = SLO_COLUMNS.get(key)
    if column is None:
        raise ValueError(f"unknown SLO key {key!r}; expected one of {sorted(SLO_COLUMNS)}")
    if column in row:
        return _as_float(row.get(column))
    return _as_float(row.get(LEGACY_LATENCY_COLUMNS[key]))


def unserved_requests(row: Mapping[str, Any]) -> int:
    """Requests sent inside this window that went unserved (all three classes)."""
    return sum(_as_count(row.get(column)) for column in UNSERVED_COLUMNS)


def row_unserved(row: Mapping[str, Any]) -> bool:
    """True when the window holds an unserved request - read from the three count
    columns only. ``slo_violated`` is deliberately NOT consulted: it is the label."""
    return unserved_requests(row) > 0


def completed_requests(row: Mapping[str, Any]) -> Optional[int]:
    value = _as_float(row.get(COMPLETED_REQUESTS_COLUMN))
    return None if value is None else int(value)


def format_ttft_len_samples(pairs: Iterable[tuple[Optional[float], Optional[float]]]) -> str:
    """``ttft_ms:prompt_tokens`` pairs joined by ``;`` (requests without a length are
    skipped: the slowdown SLO is undefined for them)."""
    return ";".join(
        f"{float(ttft):.2f}:{int(length)}" for ttft, length in pairs
        if ttft is not None and length is not None
    )


def parse_ttft_len_samples(value: Any) -> list[tuple[float, float]]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    out: list[tuple[float, float]] = []
    for part in text.split(";"):
        ttft, _, length = part.partition(":")
        out.append((float(ttft), float(length)))
    return out


def _percentile(samples: list[float], quantile: float, mode: str) -> Optional[float]:
    """Same cumulative-from-samples percentile ``rewindow_from_raw`` uses for p95 columns."""
    if not samples:
        return None
    counts: dict[float, int] = {}
    for s in samples:
        counts[float(s)] = counts.get(float(s), 0) + 1
    cumulative: list[tuple[float, float]] = []
    running = 0.0
    for value in sorted(counts):
        running += counts[value]
        cumulative.append((value, running))
    return histogram_percentile(cumulative, quantile, mode=mode)


# ----------------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class WindowLabel:
    slo_met: bool
    #: max(p95 / SLO) over the active SLOs (>= UNSERVED_MIN_RATIO for an unserved window
    #: without any latency sample).
    ratio_max: float
    unserved: bool
    #: Per-SLO ratios (None when that SLO had no sample); used for the violation-class
    #: split (TTFT-only / TPOT-only / both / unserved) and the average severity.
    ttft_ratio: Optional[float] = None
    tpot_ratio: Optional[float] = None

    @property
    def label(self) -> str:
        return LABEL_HEALTHY if self.slo_met else LABEL_VIOLATED

    @property
    def violation_class(self) -> Optional[str]:
        """``None`` for a healthy window, else unserved / both / ttft_only / tpot_only."""
        if self.slo_met:
            return None
        if self.unserved:
            return "unserved"
        ttft = self.ttft_ratio is not None and self.ttft_ratio > 1.0
        tpot = self.tpot_ratio is not None and self.tpot_ratio > 1.0
        if ttft and tpot:
            return "both"
        return "ttft_only" if ttft else "tpot_only"

    @property
    def ratio_avg(self) -> Optional[float]:
        ratios = [r for r in (self.ttft_ratio, self.tpot_ratio) if r is not None]
        return sum(ratios) / len(ratios) if ratios else None


def _decide(
    *, unserved: bool, ratios: list[float], complete: bool,
    ttft_ratio: Optional[float] = None, tpot_ratio: Optional[float] = None,
) -> Optional[WindowLabel]:
    """The decision rule, shared by every label form: unserved -> violated; a missing
    ratio (``complete`` False) -> no label; otherwise violated iff any ratio > 1."""
    if not complete or not ratios:
        if not unserved:
            return None
        return WindowLabel(
            slo_met=False, ratio_max=max(ratios + [UNSERVED_MIN_RATIO]), unserved=True,
            ttft_ratio=ttft_ratio, tpot_ratio=tpot_ratio,
        )
    return WindowLabel(
        slo_met=(not unserved) and all(r <= 1.0 for r in ratios),
        ratio_max=max(ratios),
        unserved=unserved,
        ttft_ratio=ttft_ratio,
        tpot_ratio=tpot_ratio,
    )


@dataclass(frozen=True)
class LabelDefinition:
    #: The thresholds have no default (the registry's, through :func:`label_def_for_model`).
    #: ``ttft_p95_ms``: the fixed-mode TTFT (recorded, unused, in a per-request mode).
    ttft_p95_ms: float
    tpot_p95_ms: float
    ttft_slo_mode: str = TTFT_SLO_MODE_FIXED
    #: Per-request modes: TTFT multiple and floor (required there; unused in ``fixed``).
    ttft_slowdown_k: Optional[float] = None
    ttft_floor_ms: Optional[float] = None
    #: Idle TTFT intercept (ms) and slope (ms/prompt token) of the model; slowdown only.
    ttft_idle_c_ms: Optional[float] = None
    ttft_idle_b_ms_per_token: Optional[float] = None
    min_completed_requests: int = DEFAULT_MIN_COMPLETED_REQUESTS
    percentile_mode: str = DEFAULT_PERCENTILE_MODE
    #: Request-to-window attribution of the window's latency evidence (:data:`ATTRIBUTIONS`).
    #: The verdict rule reads the window row as it is; this field says how the row's
    #: ``ttft_len_samples`` / ``completed_requests`` / p95 columns were attributed
    #: (``scripts.rewindow_from_raw``), and it is part of the definition's hash.
    attribution: str = ATTRIBUTION_COMPLETION
    #: ``idle_multiple`` only: TPOT multiple and idle TPOT (ms); ``tpot_p95_ms`` must be
    #: ``tre_common.slo.tpot_idle_multiple_ms`` of them.
    tpot_k: Optional[float] = None
    tpot_idle_ms: Optional[float] = None
    #: ``idle_multiple`` only: the registry's fixed-mode TPOT (``slo.tpot_p95_ms``), for the
    #: fixed comparison arm (:func:`label_arms`); recorded.
    fixed_tpot_p95_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.attribution not in ATTRIBUTIONS:
            raise ValueError(f"attribution must be one of {ATTRIBUTIONS}, got {self.attribution!r}")
        for name in ("ttft_p95_ms", "tpot_p95_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if self.ttft_slo_mode not in TTFT_SLO_MODES:
            raise ValueError(f"ttft_slo_mode must be one of {TTFT_SLO_MODES}")
        if int(self.min_completed_requests) < 0:
            raise ValueError("min_completed_requests must be >= 0")
        if self.per_request_ttft:
            for name in ("ttft_slowdown_k", "ttft_floor_ms"):
                raw_value = getattr(self, name)
                if raw_value is None:
                    raise ValueError(f"{self.ttft_slo_mode} label needs {name} (registry slo block or CLI)")
                value = float(raw_value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive number")
            if self.ttft_idle_c_ms is None or self.ttft_idle_b_ms_per_token is None:
                raise ValueError(
                    "slowdown TTFT SLO needs the model's idle TTFT fit (ttft_idle_c_ms, "
                    "ttft_idle_b_ms_per_token): registry slo block or CLI override"
                )
            c, b = float(self.ttft_idle_c_ms), float(self.ttft_idle_b_ms_per_token)
            if not (math.isfinite(c) and math.isfinite(b)) or c < 0 or b < 0 or c + b <= 0:
                raise ValueError("idle TTFT fit must be non-negative and not all zero")
        if self.ttft_slo_mode == TTFT_SLO_MODE_IDLE_MULTIPLE:
            if self.tpot_k is None or self.tpot_idle_ms is None:
                raise ValueError("idle_multiple SLO needs tpot_k and tpot_idle_ms (registry slo.idle_multiple)")
            try:
                expected = self.thresholds().tpot_ms
            except _slo.SloConfigError as exc:
                raise ValueError(str(exc)) from None
            if abs(float(self.tpot_p95_ms) - expected) > 1e-9 * max(1.0, expected):
                raise ValueError(f"idle_multiple tpot_p95_ms {self.tpot_p95_ms} != tpot_k * tpot_idle_ms "
                                 f"{expected} (build the label with label_def_for_model)")

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_targets(
        cls, latency_slo_ms: Mapping[str, float], *, min_completed_requests: int = 0,
    ) -> "LabelDefinition":
        """The fixed label of a ``{"ttft_p95": .., "tpot_p95": ..}`` mapping (the 09-23
        interface). ``min_completed_requests`` defaults to 0 - that interface had no
        min-n guard, only the re-windower's p95 guard."""
        unknown = set(latency_slo_ms) - {"ttft_p95", "tpot_p95"}
        if unknown or not {"ttft_p95", "tpot_p95"} <= set(latency_slo_ms):
            raise ValueError(
                f"a LabelDefinition needs exactly ttft_p95 and tpot_p95, got {sorted(latency_slo_ms)}"
            )
        return cls(
            ttft_p95_ms=float(latency_slo_ms["ttft_p95"]),
            tpot_p95_ms=float(latency_slo_ms["tpot_p95"]),
            ttft_slo_mode=TTFT_SLO_MODE_FIXED,
            min_completed_requests=int(min_completed_requests),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelDefinition":
        """Rebuild a definition from :meth:`as_dict` - or from the 09-23 record
        (:func:`label_definition` of a mapping: ``slo_ms`` keyed by client column, no
        ``mode``, no ``min_n``) and pre-D6 artifacts (fixed, no min_n -> 0, i.e. the
        behaviour they were made with)."""
        mode = str(raw.get("mode") or TTFT_SLO_MODE_FIXED)
        if "ttft_p95_ms" in raw:
            ttft, tpot = float(raw["ttft_p95_ms"]), float(raw["tpot_p95_ms"])
        else:
            slo = raw.get("slo_ms") or {}
            ttft, tpot = float(slo[P95_TTFT_CLIENT]), float(slo[P95_TPOT_CLIENT])
        kwargs: dict[str, Any] = {
            "ttft_p95_ms": ttft,
            "tpot_p95_ms": tpot,
            "ttft_slo_mode": mode,
            "min_completed_requests": int(raw.get("min_n", 0)),
            "attribution": str(raw.get("attribution") or ATTRIBUTION_COMPLETION),
        }
        if mode in _slo.PER_REQUEST_TTFT_MODES:
            kwargs.update(
                ttft_slowdown_k=float(raw["k"]),
                ttft_floor_ms=float(raw["floor_ms"]),
                ttft_idle_c_ms=float(raw["c_ms"]),
                ttft_idle_b_ms_per_token=float(raw["b_ms_per_token"]),
                percentile_mode=str(raw.get("percentile_mode", DEFAULT_PERCENTILE_MODE)),
            )
        if mode == TTFT_SLO_MODE_IDLE_MULTIPLE:
            kwargs.update(tpot_k=float(raw["tpot_k"]), tpot_idle_ms=float(raw["tpot_idle_ms"]))
            if raw.get("fixed_tpot_p95_ms") is not None:
                kwargs["fixed_tpot_p95_ms"] = float(raw["fixed_tpot_p95_ms"])
        return cls(**kwargs)

    def with_mode(self, mode: str, **overrides: Any) -> "LabelDefinition":
        return replace(self, ttft_slo_mode=mode, **overrides)

    # -- description -------------------------------------------------------------------

    @property
    def slowdown(self) -> bool:
        """The 09-21 slowdown mode (record names, the k = 3 arm). For "the TTFT term needs
        each request's prompt length" use :attr:`per_request_ttft`."""
        return self.ttft_slo_mode == TTFT_SLO_MODE_SLOWDOWN

    @property
    def idle_multiple(self) -> bool:
        return self.ttft_slo_mode == TTFT_SLO_MODE_IDLE_MULTIPLE

    @property
    def per_request_ttft(self) -> bool:
        return self.ttft_slo_mode in _slo.PER_REQUEST_TTFT_MODES

    def thresholds(self) -> "_slo.SloThresholds":
        """This label's thresholds, built by the one SLO implementation (``tre_common.slo``)."""
        block: dict[str, Any] = {
            "ttft_p95_ms": self.ttft_p95_ms, "tpot_p95_ms": self.tpot_p95_ms,
            "ttft_idle_c_ms": self.ttft_idle_c_ms, "ttft_idle_b_ms_per_token": self.ttft_idle_b_ms_per_token,
            "ttft_slowdown_k": self.ttft_slowdown_k, "ttft_floor_ms": self.ttft_floor_ms,
        }
        if self.idle_multiple:
            block[_slo.IDLE_MULTIPLE_KEY] = {
                "ttft_k": self.ttft_slowdown_k, "ttft_floor_ms": self.ttft_floor_ms,
                "tpot_k": self.tpot_k, "tpot_idle_ms": self.tpot_idle_ms,
            }
        return _slo.slo_thresholds(block, self.ttft_slo_mode)

    def ttft_slo_ms(self, prompt_tokens: Optional[float] = None) -> float:
        """TTFT SLO of one request: fixed, or ``max(floor, k*(c + b*L))``."""
        if not self.per_request_ttft:
            return float(self.ttft_p95_ms)
        if prompt_tokens is None:
            raise ValueError(f"{self.ttft_slo_mode} TTFT SLO needs the prompt length")
        return self._thresholds.ttft_ms(prompt_tokens)

    @property
    def _thresholds(self) -> "_slo.SloThresholds":
        cached = self.__dict__.get("_thresholds_cache")
        if cached is None:
            cached = self.thresholds()
            object.__setattr__(self, "_thresholds_cache", cached)
        return cached

    def latency_slo_ms(self) -> dict[str, float]:
        """Fixed-mode thresholds keyed by SLO name (the fixed arm; legacy callers)."""
        return {"ttft_p95": float(self.ttft_p95_ms), "tpot_p95": float(self.tpot_p95_ms)}

    def required_columns(self) -> dict[str, str]:
        cols = {"tpot_p95": SLO_COLUMNS["tpot_p95"]}
        cols["ttft_p95"] = TTFT_LEN_SAMPLES_COLUMN if self.per_request_ttft else SLO_COLUMNS["ttft_p95"]
        return cols

    @property
    def hybrid(self) -> bool:
        return self.attribution == ATTRIBUTION_HYBRID

    def as_dict(self) -> dict[str, Any]:
        """The definition's record (its canonical sha256 is the label's identity). A
        completion (v1) definition's record is byte-identical to the one written before the
        ``attribution`` field existed; a hybrid one is v2: own name, ``attribution`` and its
        rule."""
        if self.idle_multiple:
            name = LABEL_DEF_NAME_IDLE_MULTIPLE_V2 if self.hybrid else LABEL_DEF_NAME_IDLE_MULTIPLE
        elif self.hybrid:
            name = LABEL_DEF_NAME_SLOWDOWN_V2 if self.slowdown else LABEL_DEF_NAME_V2
        else:
            name = LABEL_DEF_NAME_SLOWDOWN if self.slowdown else LABEL_DEF_NAME
        base: dict[str, Any] = {
            "name": name,
            "mode": self.ttft_slo_mode,
            "ttft_p95_ms": float(self.ttft_p95_ms),
            "tpot_p95_ms": float(self.tpot_p95_ms),
            "min_n": int(self.min_completed_requests),
            "e2e": "excluded",
            "unserved": (
                "violated when any of " + ", ".join(UNSERVED_COLUMNS) + " > 0 "
                "(requests sent in the window); slo_violated is never read"
            ),
            "missing_latency": "unlabeled unless unserved (also when completed_requests < min_n)",
            "latency_columns": [SLO_COLUMNS["ttft_p95"], SLO_COLUMNS["tpot_p95"]],
        }
        if self.per_request_ttft:
            base.update(
                {
                    "k": float(self.ttft_slowdown_k),
                    "floor_ms": float(self.ttft_floor_ms),
                    "c_ms": float(self.ttft_idle_c_ms),  # type: ignore[arg-type]
                    "b_ms_per_token": float(self.ttft_idle_b_ms_per_token),  # type: ignore[arg-type]
                    "percentile_mode": self.percentile_mode,
                    "ttft_slo_ms": "max(floor_ms, k * (c_ms + b_ms_per_token * prompt_tokens))",
                    "ttft_p95_ms_unused": True,
                    "violated_if": (
                        "p95_i(ttft_i / ttft_slo_ms(L_i)) > 1 or p95_tpot_client_ms > tpot_p95_ms "
                        "or unserved"
                    ),
                }
            )
            if self.idle_multiple:
                base.update({"tpot_k": float(self.tpot_k), "tpot_idle_ms": float(self.tpot_idle_ms),  # type: ignore[arg-type]
                             "tpot_p95_ms_rule": "tpot_k * tpot_idle_ms"})
                if self.fixed_tpot_p95_ms is not None:
                    base["fixed_tpot_p95_ms"] = float(self.fixed_tpot_p95_ms)
        else:
            base["violated_if"] = (
                "p95_ttft_client_ms > ttft_p95_ms or p95_tpot_client_ms > tpot_p95_ms or unserved"
            )
        if self.hybrid:
            base["attribution"] = self.attribution
            base["attribution_rule"] = ATTRIBUTION_RULES[self.attribution]
            base["min_n_counts"] = "the TTFT set (served requests whose first token is in the window)"
        return base

    # -- labelling ---------------------------------------------------------------------

    def classify(self, row: Mapping[str, Any]) -> tuple[Optional[WindowLabel], Optional[str]]:
        """(label, None), or (None, reason) with reason one of EXCLUDED_*."""
        unserved = row_unserved(row)
        n = completed_requests(row)
        low_n = n is not None and self.min_completed_requests > 0 and n < self.min_completed_requests
        tpot_ratio: Optional[float] = None
        ttft_ratio: Optional[float] = None
        if not low_n:
            tpot = latency_value(row, "tpot_p95")
            tpot_ratio = None if tpot is None else tpot / float(self.tpot_p95_ms)
            if self.per_request_ttft:
                ttft_ratio = self.window_ttft_ratio(row)
            else:
                ttft = latency_value(row, "ttft_p95")
                ttft_ratio = None if ttft is None else ttft / float(self.ttft_p95_ms)
        ratios = [r for r in (ttft_ratio, tpot_ratio) if r is not None]
        label = _decide(
            unserved=unserved, ratios=ratios, complete=len(ratios) == 2,
            ttft_ratio=ttft_ratio, tpot_ratio=tpot_ratio,
        )
        if label is None:
            return None, (EXCLUDED_LOW_N if low_n else EXCLUDED_MISSING_LATENCY)
        return label, None

    def label(self, row: Mapping[str, Any]) -> Optional[WindowLabel]:
        return self.classify(row)[0]

    def window_label(self, row: Mapping[str, Any]) -> str:
        """``violated`` / ``healthy`` / ``unlabeled``."""
        lab = self.label(row)
        return LABEL_UNLABELED if lab is None else lab.label

    def window_ttft_ratio(self, row: Mapping[str, Any]) -> Optional[float]:
        """p95 over the window's requests of TTFT_i / TTFT_slo(L_i) (per-request modes)."""
        if TTFT_LEN_SAMPLES_COLUMN not in row:
            raise LabelInputError(
                f"{self.ttft_slo_mode} TTFT label needs the {TTFT_LEN_SAMPLES_COLUMN!r} column; "
                "re-window the raw with this version of rewindow_from_raw"
            )
        pairs = parse_ttft_len_samples(row.get(TTFT_LEN_SAMPLES_COLUMN))
        ratios = [ttft / self.ttft_slo_ms(length) for ttft, length in pairs]
        return _percentile(ratios, 0.95, self.percentile_mode)


#: What a label can be given as: the definition itself (production), or the 09-23
#: ``{"ttft_p95": .., "tpot_p95": ..}`` mapping of fixed thresholds (no min-n guard).
LabelSpec = Union[LabelDefinition, Mapping[str, float]]


def _mapping_label(row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]) -> Optional[WindowLabel]:
    """A plain mapping of fixed thresholds: every listed SLO is active, no min-n guard."""
    if not latency_slo_ms:
        raise ValueError("latency_slo_ms must name at least one SLO")
    ratios: list[float] = []
    by_key: dict[str, Optional[float]] = {}
    complete = True
    for key, slo in latency_slo_ms.items():
        value = latency_value(row, key)
        if value is None:
            complete = False
            by_key[key] = None
            continue
        by_key[key] = value / float(slo)
        ratios.append(by_key[key])  # type: ignore[arg-type]
    return _decide(
        unserved=row_unserved(row), ratios=ratios, complete=complete,
        ttft_ratio=by_key.get("ttft_p95"), tpot_ratio=by_key.get("tpot_p95"),
    )


def label_window(row: Mapping[str, Any], spec: LabelSpec) -> Optional[WindowLabel]:
    """Label one window row, or None when it carries no usable evidence (``unlabeled``)."""
    if isinstance(spec, LabelDefinition):
        return spec.label(row)
    return _mapping_label(row, spec)


def window_slo_label(row: Mapping[str, Any], spec: LabelSpec) -> str:
    """``violated`` / ``healthy`` / ``unlabeled`` for one window row (see module docstring).

    ``row`` is a window CSV row (strings are fine) or the dict ``scripts.r3_grid`` builds;
    ``spec`` is a :class:`LabelDefinition` or a mapping keyed ``ttft_p95`` / ``tpot_p95``
    (/ ``e2e_p95``).
    """
    lab = label_window(row, spec)
    return LABEL_UNLABELED if lab is None else lab.label


def latency_ratios(row: Mapping[str, Any], latency_slo_ms: Mapping[str, float]) -> Optional[list[float]]:
    """``p95 / SLO`` for every SLO of a fixed-threshold mapping, or None when any is missing."""
    if not latency_slo_ms:
        raise ValueError("latency_slo_ms must name at least one SLO")
    ratios: list[float] = []
    for key, slo in latency_slo_ms.items():
        value = latency_value(row, key)
        if value is None:
            return None
        ratios.append(value / float(slo))
    return ratios


def count_label_exclusions(rows: Iterable[Mapping[str, Any]], label: LabelDefinition) -> dict[str, int]:
    """How many rows the label keeps / drops, and why (reported with the fits)."""
    out = {"labelled": 0, EXCLUDED_LOW_N: 0, EXCLUDED_MISSING_LATENCY: 0}
    for row in rows:
        lab, reason = label.classify(row)
        out["labelled" if lab is not None else str(reason)] += 1
    return out


# ----------------------------------------------------------------- the three arms


def label_arms(primary: LabelDefinition) -> dict[str, LabelDefinition]:
    """The labels every published fit reports (plan §6.11 D6'): the primary slowdown
    label, the fixed-mode comparison column (the primary's ``ttft_p95_ms`` / ``tpot_p95_ms``,
    the registry's fixed values) and the k = 3 / 150 ms-floor ablation. An
    ``idle_multiple`` primary gets the fixed comparison (its ``ttft_p95_ms`` and
    ``fixed_tpot_p95_ms``, the registry's fixed values) and no k = 3 arm."""
    if not primary.per_request_ttft:
        raise ValueError("label arms are built from a per-request (slowdown / idle_multiple) primary label")
    fixed = replace(primary, ttft_slo_mode=TTFT_SLO_MODE_FIXED, ttft_idle_c_ms=None, ttft_idle_b_ms_per_token=None,
                    tpot_k=None, tpot_idle_ms=None, fixed_tpot_p95_ms=None)
    if primary.idle_multiple:
        if primary.fixed_tpot_p95_ms is None:
            raise ValueError("idle_multiple label without fixed_tpot_p95_ms: no fixed comparison arm "
                             "(build the label with label_def_for_model from a registry)")
        return {ARM_PRIMARY: primary,
                ARM_FIXED: replace(fixed, ttft_slowdown_k=None, ttft_floor_ms=None,
                                   tpot_p95_ms=primary.fixed_tpot_p95_ms)}
    return {
        ARM_PRIMARY: primary,
        ARM_FIXED: fixed,
        ARM_K3: replace(primary, ttft_slowdown_k=ABLATION_TTFT_SLOWDOWN_K,
                        ttft_floor_ms=ABLATION_TTFT_FLOOR_MS),
    }


LabelArms = Mapping[str, LabelSpec]


def resolve_arms(spec: Union[LabelSpec, LabelArms]) -> dict[str, LabelSpec]:
    """The arms a window table is labelled with, from whatever the caller holds.

    * a slowdown :class:`LabelDefinition` -> :func:`label_arms` (all three);
    * a fixed one, or a threshold mapping -> the primary IS the fixed comparison, and
      there is no k = 3 arm (its column stays empty);
    * a mapping of arm name -> spec -> taken as given (must name ``primary``).
    """
    if isinstance(spec, LabelDefinition):
        if spec.per_request_ttft:
            return dict(label_arms(spec))
        return {ARM_PRIMARY: spec, ARM_FIXED: spec}
    if spec and all(isinstance(v, (LabelDefinition, Mapping)) for v in spec.values()):
        arms = dict(spec)  # type: ignore[arg-type]
        if ARM_PRIMARY not in arms:
            raise ValueError(f"label arms must include {ARM_PRIMARY!r}, got {sorted(arms)}")
        unknown = set(arms) - set(ARM_COLUMNS)
        if unknown:
            raise ValueError(f"unknown label arm(s) {sorted(unknown)}; expected {sorted(ARM_COLUMNS)}")
        return arms
    return {ARM_PRIMARY: spec, ARM_FIXED: spec}  # type: ignore[dict-item]


def apply_label(row: dict, spec: LabelSpec) -> str:
    """Write :data:`LABEL_COLUMN` and :data:`VIOLATED_COLUMN` onto ``row``; return the label.

    The one place a window row gets its primary verdict columns, so the two can never
    disagree. ``slo_violated`` is None (an empty CSV cell) for an unlabeled window: "no
    evidence" has no truthful boolean, and False would be read as healthy.
    """
    label = window_slo_label(row, spec)
    row[LABEL_COLUMN] = label
    row[VIOLATED_COLUMN] = None if label == LABEL_UNLABELED else (label == LABEL_VIOLATED)
    return label


def apply_label_arms(row: dict, arms: Union[LabelSpec, LabelArms]) -> str:
    """Write every arm's column onto ``row`` (primary through :func:`apply_label`); an arm
    the caller has not got (the k = 3 ablation of a fixed primary) is left empty. Returns
    the primary label."""
    resolved = resolve_arms(arms)
    label = apply_label(row, resolved[ARM_PRIMARY])
    for arm, column in ARM_COLUMNS.items():
        if arm == ARM_PRIMARY:
            continue
        spec = resolved.get(arm)
        row[column] = None if spec is None else window_slo_label(row, spec)
    return label


# ------------------------------------------------------------------------ manifests


def slo_targets(*, ttft_slo_ms: float, tpot_slo_ms: float, e2e_slo_ms: Optional[float] = None) -> dict:
    """The ``{"ttft_p95": .., "tpot_p95": ..}`` threshold mapping (the fixed arm)."""
    out = {"ttft_p95": float(ttft_slo_ms), "tpot_p95": float(tpot_slo_ms)}
    if e2e_slo_ms is not None:
        out["e2e_p95"] = float(e2e_slo_ms)
    return out


def label_definition(
    spec: Union[LabelSpec, LabelArms],
    *,
    min_latency_samples: int = DEFAULT_MIN_LATENCY_SAMPLES,
    window_membership: str = "[start, end)",
) -> dict:
    """A self-describing record of the label(s), for manifests and cell artifacts.

    ``spec`` may be one label or the arms (:func:`resolve_arms`); the record of the
    primary is at the top level (``LabelDefinition.from_dict`` rebuilds it) and every
    arm is under ``arms``. ``window_membership`` is the interval the windowing used
    (``(start, end]`` for the grid-aligned re-window).
    """
    arms = resolve_arms(spec)
    primary = arms[ARM_PRIMARY]

    def record(one: LabelSpec) -> dict:
        if isinstance(one, LabelDefinition):
            return one.as_dict()
        return {
            "name": LABEL_DEF_NAME,
            "mode": TTFT_SLO_MODE_FIXED,
            "slo_ms": {SLO_COLUMNS[k]: float(v) for k, v in one.items()},
            "min_n": 0,
            "violated_if": "any listed client p95 > its SLO, or unserved",
        }

    out = record(primary)
    if not isinstance(primary, LabelDefinition):
        # The 09-23 record shape, kept for readers of it.
        out["slo_ms"] = {SLO_COLUMNS[k]: float(v) for k, v in primary.items()}
    else:
        out["slo_ms"] = {SLO_COLUMNS[k]: v for k, v in primary.latency_slo_ms().items()}
    membership = {
        "latency": f"served requests whose completion (done_ts_ms) is in {window_membership}",
        "unserved": f"requests whose send instant (send_ts_ms) is in {window_membership}",
    }
    if isinstance(primary, LabelDefinition) and primary.hybrid:
        membership["latency"] = (
            f"TTFT and the min-n count: served requests whose first token (recv_first_token_ts_ms) "
            f"is in {window_membership}; TPOT / e2e: served requests whose completion (done_ts_ms) "
            f"is in {window_membership}")
    out.update({
        "latency_source": LATENCY_SOURCE,
        "ttft": "first streamed token minus the instant the request went on the wire",
        "tpot": "(e2e_ms - ttft_ms) / (completion_tokens - 1), per request",
        "window_membership": membership,
        "p95": "tre_common.percentile.histogram_percentile over exact samples, bucket_upper",
        "min_latency_samples": int(min_latency_samples),
        "unserved_classes": list(UNSERVED_COLUMNS),
        "rules": [
            "violated if any request sent in the window went unserved (count columns)",
            "unlabeled if completed_requests < min_n or a p95 the label needs is missing",
            "violated if the TTFT term or the client p95 TPOT exceeds its SLO",
            "healthy otherwise",
        ],
        "columns": {arm: ARM_COLUMNS[arm] for arm in arms},
        "arms": {arm: record(one) for arm, one in arms.items()},
        "implementation": "tre_common.slo_labels.LabelDefinition",
    })
    return out


# ------------------------------------------------------------ CLI / registry plumbing


def registry_fixed_slo_ms(registry: Any = None, models: Optional[Iterable[str]] = None) -> tuple[float, float]:
    """The fixed-mode (TTFT, TPOT) thresholds (``slo.ttft_p95_ms`` / ``tpot_p95_ms``, through
    ``tre_common.slo``) shared by ``models`` (default: every model) of ``registry`` (a loaded
    registry, a path, or None: the shared one). Refuses (SystemExit) models that differ: one
    value pair is what the callers (a CLI's ``--ttft-slo-ms`` / ``--tpot-slo-ms``) carry."""
    if registry is None or isinstance(registry, (str, bytes)) or hasattr(registry, "__fspath__"):
        from tre_common.registry import load_registry

        registry = load_registry(None if registry is None else str(registry))
    names = list(models) if models else [m.name for m in registry.models]
    pairs = {}
    for name in names:
        try:
            t = _slo.slo_thresholds(registry.model(name).slo, TTFT_SLO_MODE_FIXED)
        except (KeyError, _slo.SloConfigError) as exc:
            raise SystemExit(f"registry: model {name}: no fixed-mode SLO ({exc})") from None
        pairs[name] = (float(t.ttft_ms(None)), float(t.tpot_ms))
    distinct = set(pairs.values())
    if len(distinct) != 1:
        raise SystemExit(f"registry fixed-mode SLO differs between models {pairs}: pass the values explicitly")
    return distinct.pop()


def fill_fixed_slo_args(args: argparse.Namespace, models: Optional[Iterable[str]] = None, *, registry: Any = None,
                        ttft_attr: str = "ttft_slo_ms", tpot_attr: str = "tpot_slo_ms") -> None:
    """Set a CLI's fixed thresholds (default None) from the registry when the caller did not
    pass them (:func:`registry_fixed_slo_ms`): no SLO value is a CLI default."""
    if getattr(args, ttft_attr, None) is not None and getattr(args, tpot_attr, None) is not None:
        return
    ttft, tpot = registry_fixed_slo_ms(registry, models)
    if getattr(args, ttft_attr, None) is None:
        setattr(args, ttft_attr, ttft)
    if getattr(args, tpot_attr, None) is None:
        setattr(args, tpot_attr, tpot)


def add_label_arguments(
    parser: argparse.ArgumentParser, *, require_fixed: bool = False, include_fixed: bool = True,
    require_mode: bool = False,
) -> None:
    """The label flags every fit entry point shares. ``include_fixed=False`` leaves out
    ``--ttft-p95-ms`` / ``--tpot-p95-ms`` for a tool that already has its own
    ``--ttft-slo-ms`` / ``--tpot-slo-ms`` (``label_def_from_args`` reads either). The fixed
    thresholds default to the registry's (``slo.ttft_p95_ms`` / ``tpot_p95_ms``).
    ``require_mode`` (tools whose output is compared with a frozen record, e.g. a D6'
    reproduction): ``--ttft-slo-mode`` is required - the registry's switch may have moved
    since the record was made (2026-11-04)."""
    if include_fixed:
        parser.add_argument("--ttft-p95-ms", type=float, required=require_fixed,
                            help="fixed-mode TTFT p95 SLO (ms; default: the registry's slo.ttft_p95_ms); "
                                 "recorded but unused in a per-request mode")
        parser.add_argument("--tpot-p95-ms", type=float, required=require_fixed,
                            help="fixed-mode TPOT p95 SLO (ms; default: the registry's slo.tpot_p95_ms)")
    parser.add_argument("--ttft-slo-mode", choices=TTFT_SLO_MODES, default=None, required=require_mode,
                        help="fixed: one TTFT p95 threshold (comparison column); slowdown: "
                             "max(floor, k*(c_m+b_m*L)) per request (plan 6.11 D6-prime); idle_multiple: "
                             "the 2026-11-04 SLO (registry slo.idle_multiple). "
                             + ("Required (the output is compared with a frozen record)" if require_mode else
                                "Default: the registry's top-level slo_mode (tre_common.slo)"))
    parser.add_argument("--ttft-slowdown-k", type=float, default=None,
                        help="TTFT multiple k of a per-request mode. Default: registry slo.ttft_slowdown_k "
                             "(slowdown) / slo.idle_multiple.ttft_k")
    parser.add_argument("--ttft-floor-ms", type=float, default=None,
                        help="TTFT floor of a per-request mode. Default: registry slo.ttft_floor_ms "
                             "(slowdown) / slo.idle_multiple.ttft_floor_ms")
    parser.add_argument("--tpot-k", type=float, default=None,
                        help="idle_multiple: TPOT multiple (default: registry slo.idle_multiple.tpot_k)")
    parser.add_argument("--tpot-idle-ms", type=float, default=None,
                        help="idle_multiple: idle TPOT ms (default: registry slo.idle_multiple.tpot_idle_ms)")
    parser.add_argument("--ttft-idle-c-ms", type=float, default=None,
                        help="override the registry slo.ttft_idle_c_ms of the model")
    parser.add_argument("--ttft-idle-b-ms-per-token", type=float, default=None,
                        help="override the registry slo.ttft_idle_b_ms_per_token of the model")
    parser.add_argument("--min-completed-requests", type=int, default=DEFAULT_MIN_COMPLETED_REQUESTS,
                        help="windows with fewer completed requests carry no latency evidence")
    parser.add_argument("--label-registry", default=None,
                        help="registry the idle TTFT fit is read from (default: the shared one)")
    parser.add_argument("--label-attribution", choices=ATTRIBUTIONS, default=None,
                        help="request-to-window attribution of the latency evidence (default "
                             f"{ATTRIBUTION_COMPLETION}, label v1; {ATTRIBUTION_HYBRID} = label v2: "
                             "TTFT by first token, TPOT by completion)")


def label_def_for_model(
    model: Optional[str],
    *,
    ttft_p95_ms: Optional[float] = None,
    tpot_p95_ms: Optional[float] = None,
    mode: Optional[str] = None,
    k: Optional[float] = None,
    floor: Optional[float] = None,
    c: Optional[float] = None,
    b: Optional[float] = None,
    tpot_k: Optional[float] = None,
    tpot_idle_ms: Optional[float] = None,
    min_completed_requests: int = DEFAULT_MIN_COMPLETED_REQUESTS,
    registry: Any = None,
    attribution: Optional[str] = None,
) -> LabelDefinition:
    """Build the label for one model. ``mode`` defaults to the registry's top-level
    ``slo_mode`` (the one switch, ``tre_common.slo``). Precedence per parameter: explicit
    value > the model's registry ``slo`` block (read through ``tre_common.slo``); there is no
    module default (a value neither gives is refused, SystemExit). ``idle_multiple``'s TPOT
    threshold is ``tpot_k * tpot_idle_ms``; ``tpot_p95_ms`` (explicit, else the registry's)
    is its ``fixed_tpot_p95_ms`` (the fixed comparison arm). ``registry`` is a loaded
    registry or a path (None: the shared one)."""
    slo = None
    registry_mode = None
    needs_registry = bool(model) and (
        mode is None
        or None in (ttft_p95_ms, tpot_p95_ms)
        or (mode != TTFT_SLO_MODE_FIXED and None in (k, floor, c, b))
        or (mode == TTFT_SLO_MODE_IDLE_MULTIPLE and None in (tpot_k, tpot_idle_ms))
    )
    if needs_registry:
        if registry is None or isinstance(registry, str):
            from tre_common.registry import load_registry

            registry = load_registry(registry)
        try:
            slo = registry.model(model).slo
        except KeyError:
            slo = None
        mode_of = getattr(registry, "slo_mode", None)
        registry_mode = mode_of() if callable(mode_of) else None
    if mode is None:
        # the one switch (2026-11-04): no built-in default mode
        if registry_mode is None:
            raise SystemExit(f"model {model}: no SLO mode - pass mode (--ttft-slo-mode) or a registry with slo_mode")
        mode = registry_mode
    if mode not in TTFT_SLO_MODES:
        raise ValueError(f"ttft_slo_mode must be one of {TTFT_SLO_MODES}, got {mode!r}")
    base = fixed_base = None
    if slo is not None:
        try:
            base = _slo.slo_thresholds(slo, mode)
        except _slo.SloConfigError:
            base = None
        try:
            fixed_base = _slo.slo_thresholds(slo, TTFT_SLO_MODE_FIXED)
        except _slo.SloConfigError:
            fixed_base = None

    def pick(explicit: Optional[float], registry_value: Optional[float], default: Optional[float]) -> Optional[float]:
        if explicit is not None:
            return float(explicit)
        if registry_value is not None:
            return float(registry_value)
        return None if default is None else float(default)

    idle = mode == TTFT_SLO_MODE_IDLE_MULTIPLE
    per_request = mode in _slo.PER_REQUEST_TTFT_MODES
    ttft_fixed = pick(ttft_p95_ms, fixed_base.ttft_fixed_ms if fixed_base is not None else None, None)
    tpot_fixed = pick(tpot_p95_ms, fixed_base.tpot_ms if fixed_base is not None else None, None)
    if ttft_fixed is None or tpot_fixed is None:
        raise SystemExit(f"model {model}: no fixed-mode TTFT / TPOT (registry slo.ttft_p95_ms / tpot_p95_ms, "
                         "or --ttft-p95-ms / --tpot-p95-ms)")
    k = pick(k, base.ttft_k if base is not None else None, None) if per_request else None
    floor = pick(floor, base.ttft_floor_ms if base is not None else None, None) if per_request else None
    if per_request and None in (k, floor):
        raise SystemExit(f"model {model}: {mode} needs its TTFT multiple and floor (registry slo block, or "
                         "--ttft-slowdown-k / --ttft-floor-ms)")
    if per_request and (c is None or b is None):
        if not model:
            raise SystemExit(
                f"--ttft-slo-mode {mode} needs a model name or --ttft-idle-c-ms/"
                "--ttft-idle-b-ms-per-token; pass --ttft-slo-mode fixed for the fixed-mode label"
            )
        if slo is None:
            raise SystemExit(f"registry has no model {model}: pass its idle TTFT fit or --ttft-slo-mode fixed")
        c = slo.ttft_idle_c_ms if c is None else c
        b = slo.ttft_idle_b_ms_per_token if b is None else b
        if c is None or b is None:
            raise SystemExit(f"registry model {model} has no slo.ttft_idle_c_ms / ttft_idle_b_ms_per_token")
    if idle:
        tpot_k = pick(tpot_k, base.tpot_k if base is not None else None, None)
        tpot_idle_ms = pick(tpot_idle_ms, base.tpot_idle_ms if base is not None else None, None)
        if None in (k, floor, tpot_k, tpot_idle_ms):
            raise SystemExit(
                f"model {model}: idle_multiple needs slo.idle_multiple (ttft_k, ttft_floor_ms, tpot_k, "
                "tpot_idle_ms) in the registry or --ttft-slowdown-k / --ttft-floor-ms / --tpot-k / --tpot-idle-ms"
            )
        tpot = _slo.tpot_idle_multiple_ms(tpot_k=tpot_k, tpot_idle_ms=tpot_idle_ms)  # type: ignore[arg-type]
        fixed_tpot = tpot_fixed
    else:
        tpot = tpot_fixed
        tpot_k = tpot_idle_ms = fixed_tpot = None
    return LabelDefinition(
        ttft_p95_ms=float(ttft_fixed),  # type: ignore[arg-type]
        tpot_p95_ms=float(tpot),  # type: ignore[arg-type]
        ttft_slo_mode=mode,
        ttft_slowdown_k=None if k is None else float(k),
        ttft_floor_ms=None if floor is None else float(floor),
        ttft_idle_c_ms=c if per_request else None,
        ttft_idle_b_ms_per_token=b if per_request else None,
        min_completed_requests=int(min_completed_requests),
        attribution=attribution or ATTRIBUTION_COMPLETION,
        tpot_k=tpot_k,
        tpot_idle_ms=tpot_idle_ms,
        fixed_tpot_p95_ms=fixed_tpot,
    )


def label_def_from_args(args: argparse.Namespace, model: Optional[str]) -> LabelDefinition:
    """Build the label for one model from the shared CLI arguments + registry profile
    (:func:`label_def_for_model`). The fixed thresholds are read from ``--ttft-p95-ms`` /
    ``--tpot-p95-ms``, or from a tool's own ``--ttft-slo-ms`` / ``--tpot-slo-ms``."""
    def first(*names: str) -> Any:
        for name in names:
            value = getattr(args, name, None)
            if value is not None:
                return value
        return None

    return label_def_for_model(
        model,
        ttft_p95_ms=first("ttft_p95_ms", "ttft_slo_ms"),
        tpot_p95_ms=first("tpot_p95_ms", "tpot_slo_ms"),
        mode=getattr(args, "ttft_slo_mode", None),
        k=getattr(args, "ttft_slowdown_k", None),
        floor=getattr(args, "ttft_floor_ms", None),
        c=getattr(args, "ttft_idle_c_ms", None),
        b=getattr(args, "ttft_idle_b_ms_per_token", None),
        tpot_k=getattr(args, "tpot_k", None),
        tpot_idle_ms=getattr(args, "tpot_idle_ms", None),
        min_completed_requests=first("min_completed_requests")
        if first("min_completed_requests") is not None else DEFAULT_MIN_COMPLETED_REQUESTS,
        registry=first("label_registry", "registry"),
        attribution=getattr(args, "label_attribution", None),
    )


def label_cli_args(label: LabelDefinition) -> list[str]:
    """The CLI arguments that rebuild ``label`` in another process (fit plans)."""
    out = [
        "--ttft-p95-ms", str(label.ttft_p95_ms), "--tpot-p95-ms", str(label.tpot_p95_ms),
        "--ttft-slo-mode", label.ttft_slo_mode,
        "--min-completed-requests", str(int(label.min_completed_requests)),
    ]
    if label.per_request_ttft:
        out += [
            "--ttft-slowdown-k", str(label.ttft_slowdown_k),
            "--ttft-floor-ms", str(label.ttft_floor_ms),
            "--ttft-idle-c-ms", str(label.ttft_idle_c_ms),
            "--ttft-idle-b-ms-per-token", str(label.ttft_idle_b_ms_per_token),
        ]
    if label.idle_multiple:
        out += ["--tpot-k", repr(float(label.tpot_k)), "--tpot-idle-ms", repr(float(label.tpot_idle_ms))]  # type: ignore[arg-type]
    if label.hybrid:
        # only when not the default, so a v1 label's command line is unchanged
        out += ["--label-attribution", label.attribution]
    return out


def label_mode_cli_args(label: LabelDefinition) -> list[str]:
    """:func:`label_cli_args` without the fixed thresholds, for a tool whose own
    ``--ttft-slo-ms`` / ``--tpot-slo-ms`` carry them (``r3_grid``, ``rewindow_from_raw``)."""
    full = label_cli_args(label)
    return full[4:]
