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

The TTFT term has two modes (``ttft_slo_mode``):

``slowdown`` - the primary label (plan 2026-09-21 §6.11 D6')
    per request ``TTFT_slo(L) = max(floor, k * (c_m + b_m * L))`` with the model's idle
    TTFT fit ``c_m + b_m * L`` (registry ``slo`` block), k = 5 and a 500 ms floor; the
    window term is ``p95_i(TTFT_i / TTFT_slo(L_i))`` over the window's requests, read from
    the per-request ``ttft_len_samples`` column. A CSV without it is refused rather than
    silently labelled with the fixed rule.
``fixed`` - the comparison column
    window p95 TTFT against one ``ttft_p95_ms`` (500 ms) for every prompt length.

Every published window table carries three arms (:func:`label_arms`): ``slo_label`` /
``slo_violated`` are the **primary** (D6') label, ``slo_label_fixed`` the fixed 500/75 ms
comparison and ``slo_label_k3`` the k = 3 / 150 ms-floor ablation (D6). End-to-end
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

TTFT_SLO_MODE_FIXED = "fixed"
TTFT_SLO_MODE_SLOWDOWN = "slowdown"
TTFT_SLO_MODES = (TTFT_SLO_MODE_FIXED, TTFT_SLO_MODE_SLOWDOWN)

#: Primary label (plan §6.11 D6', supersedes D6): ``max(500 ms, 5 * idle TTFT(L))``
#: (DynamoLLM / SLOs-Serve loose 5x; the 500 ms floor keeps short prompts on the fixed
#: rule - a purely length-normalised SLO turns head-of-line blocking of short prompts
#: into violations TSS cannot see).
DEFAULT_TTFT_SLO_MODE = TTFT_SLO_MODE_SLOWDOWN
DEFAULT_TTFT_SLOWDOWN_K = 5.0
DEFAULT_TTFT_FLOOR_MS = 500.0
DEFAULT_TTFT_P95_MS = 500.0
DEFAULT_TPOT_P95_MS = 75.0
#: The D6 arm (plan §6.9h: k = 3, 150 ms floor), kept as the "why not pure slowdown" ablation.
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
    ttft_p95_ms: float = DEFAULT_TTFT_P95_MS
    tpot_p95_ms: float = DEFAULT_TPOT_P95_MS
    ttft_slo_mode: str = TTFT_SLO_MODE_FIXED
    ttft_slowdown_k: float = DEFAULT_TTFT_SLOWDOWN_K
    ttft_floor_ms: float = DEFAULT_TTFT_FLOOR_MS
    #: Idle TTFT intercept (ms) and slope (ms/prompt token) of the model; slowdown only.
    ttft_idle_c_ms: Optional[float] = None
    ttft_idle_b_ms_per_token: Optional[float] = None
    min_completed_requests: int = DEFAULT_MIN_COMPLETED_REQUESTS
    percentile_mode: str = DEFAULT_PERCENTILE_MODE

    def __post_init__(self) -> None:
        for name in ("ttft_p95_ms", "tpot_p95_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if self.ttft_slo_mode not in TTFT_SLO_MODES:
            raise ValueError(f"ttft_slo_mode must be one of {TTFT_SLO_MODES}")
        if int(self.min_completed_requests) < 0:
            raise ValueError("min_completed_requests must be >= 0")
        if self.ttft_slo_mode == TTFT_SLO_MODE_SLOWDOWN:
            for name in ("ttft_slowdown_k", "ttft_floor_ms"):
                value = float(getattr(self, name))
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
        }
        if mode == TTFT_SLO_MODE_SLOWDOWN:
            kwargs.update(
                ttft_slowdown_k=float(raw["k"]),
                ttft_floor_ms=float(raw["floor_ms"]),
                ttft_idle_c_ms=float(raw["c_ms"]),
                ttft_idle_b_ms_per_token=float(raw["b_ms_per_token"]),
                percentile_mode=str(raw.get("percentile_mode", DEFAULT_PERCENTILE_MODE)),
            )
        return cls(**kwargs)

    def with_mode(self, mode: str, **overrides: Any) -> "LabelDefinition":
        return replace(self, ttft_slo_mode=mode, **overrides)

    # -- description -------------------------------------------------------------------

    @property
    def slowdown(self) -> bool:
        return self.ttft_slo_mode == TTFT_SLO_MODE_SLOWDOWN

    def ttft_slo_ms(self, prompt_tokens: Optional[float] = None) -> float:
        """TTFT SLO of one request: fixed, or ``max(floor, k*(c + b*L))``."""
        if not self.slowdown:
            return float(self.ttft_p95_ms)
        if prompt_tokens is None:
            raise ValueError("slowdown TTFT SLO needs the prompt length")
        idle = float(self.ttft_idle_c_ms) + float(self.ttft_idle_b_ms_per_token) * float(prompt_tokens)
        return max(float(self.ttft_floor_ms), float(self.ttft_slowdown_k) * idle)

    def latency_slo_ms(self) -> dict[str, float]:
        """Fixed-mode thresholds keyed by SLO name (the fixed arm; legacy callers)."""
        return {"ttft_p95": float(self.ttft_p95_ms), "tpot_p95": float(self.tpot_p95_ms)}

    def required_columns(self) -> dict[str, str]:
        cols = {"tpot_p95": SLO_COLUMNS["tpot_p95"]}
        cols["ttft_p95"] = TTFT_LEN_SAMPLES_COLUMN if self.slowdown else SLO_COLUMNS["ttft_p95"]
        return cols

    def as_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "name": LABEL_DEF_NAME_SLOWDOWN if self.slowdown else LABEL_DEF_NAME,
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
        if self.slowdown:
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
        else:
            base["violated_if"] = (
                "p95_ttft_client_ms > ttft_p95_ms or p95_tpot_client_ms > tpot_p95_ms or unserved"
            )
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
            if self.slowdown:
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
        """p95 over the window's requests of TTFT_i / TTFT_slo(L_i) (slowdown mode)."""
        if TTFT_LEN_SAMPLES_COLUMN not in row:
            raise LabelInputError(
                f"slowdown TTFT label needs the {TTFT_LEN_SAMPLES_COLUMN!r} column; "
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
    label, the fixed 500/75 ms comparison column and the k = 3 / 150 ms-floor ablation."""
    if not primary.slowdown:
        raise ValueError("label arms are built from the slowdown primary label")
    return {
        ARM_PRIMARY: primary,
        ARM_FIXED: replace(primary, ttft_slo_mode=TTFT_SLO_MODE_FIXED,
                           ttft_idle_c_ms=None, ttft_idle_b_ms_per_token=None),
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
        if spec.slowdown:
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
    out.update({
        "latency_source": LATENCY_SOURCE,
        "ttft": "first streamed token minus the instant the request went on the wire",
        "tpot": "(e2e_ms - ttft_ms) / (completion_tokens - 1), per request",
        "window_membership": {
            "latency": f"served requests whose completion (done_ts_ms) is in {window_membership}",
            "unserved": f"requests whose send instant (send_ts_ms) is in {window_membership}",
        },
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


def add_label_arguments(
    parser: argparse.ArgumentParser, *, require_fixed: bool = True, include_fixed: bool = True,
) -> None:
    """The label flags every fit entry point shares. ``include_fixed=False`` leaves out
    ``--ttft-p95-ms`` / ``--tpot-p95-ms`` for a tool that already has its own
    ``--ttft-slo-ms`` / ``--tpot-slo-ms`` (``label_def_from_args`` reads either)."""
    if include_fixed:
        parser.add_argument("--ttft-p95-ms", type=float, required=require_fixed,
                            help="fixed-mode TTFT p95 SLO (ms); recorded but unused in slowdown mode")
        parser.add_argument("--tpot-p95-ms", type=float, required=require_fixed)
    parser.add_argument("--ttft-slo-mode", choices=TTFT_SLO_MODES, default=None,
                        help="fixed: one TTFT p95 threshold (comparison column); slowdown: "
                             "max(floor, k*(c_m+b_m*L)) per request (primary, plan 6.11 D6-prime). "
                             "Default: the registry slo.ttft_slo_mode of the model, else slowdown")
    parser.add_argument("--ttft-slowdown-k", type=float, default=None,
                        help=f"default: registry slo.ttft_slowdown_k, else {DEFAULT_TTFT_SLOWDOWN_K}")
    parser.add_argument("--ttft-floor-ms", type=float, default=None,
                        help=f"default: registry slo.ttft_floor_ms, else {DEFAULT_TTFT_FLOOR_MS}")
    parser.add_argument("--ttft-idle-c-ms", type=float, default=None,
                        help="override the registry slo.ttft_idle_c_ms of the model")
    parser.add_argument("--ttft-idle-b-ms-per-token", type=float, default=None,
                        help="override the registry slo.ttft_idle_b_ms_per_token of the model")
    parser.add_argument("--min-completed-requests", type=int, default=DEFAULT_MIN_COMPLETED_REQUESTS,
                        help="windows with fewer completed requests carry no latency evidence")
    parser.add_argument("--label-registry", default=None,
                        help="registry the idle TTFT fit is read from (default: the shared one)")


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
    min_completed_requests: int = DEFAULT_MIN_COMPLETED_REQUESTS,
    registry: Any = None,
) -> LabelDefinition:
    """Build the label for one model. Precedence per field: explicit value > the model's
    registry ``slo`` block > the module default (the D6' primary label: slowdown, k = 5,
    floor 500 ms, TPOT 75 ms). ``registry`` is a loaded registry or a path (None: the
    shared one)."""
    slo = None
    if mode != TTFT_SLO_MODE_FIXED and model and None in (mode, k, floor, c, b):
        if registry is None or isinstance(registry, str):
            from tre_common.registry import load_registry

            registry = load_registry(registry)
        try:
            slo = registry.model(model).slo
        except KeyError:
            slo = None
    if mode is None:
        mode = (slo.ttft_slo_mode if slo is not None else None) or DEFAULT_TTFT_SLO_MODE
    if k is None:
        k = slo.ttft_slowdown_k if slo is not None and slo.ttft_slowdown_k is not None else DEFAULT_TTFT_SLOWDOWN_K
    if floor is None:
        floor = slo.ttft_floor_ms if slo is not None and slo.ttft_floor_ms is not None else DEFAULT_TTFT_FLOOR_MS
    if mode == TTFT_SLO_MODE_SLOWDOWN and (c is None or b is None):
        if not model:
            raise SystemExit(
                "--ttft-slo-mode slowdown (the default) needs a model name or --ttft-idle-c-ms/"
                "--ttft-idle-b-ms-per-token; pass --ttft-slo-mode fixed for the 500 ms label"
            )
        if slo is None:
            raise SystemExit(f"registry has no model {model}: pass its idle TTFT fit or --ttft-slo-mode fixed")
        c = slo.ttft_idle_c_ms if c is None else c
        b = slo.ttft_idle_b_ms_per_token if b is None else b
        if c is None or b is None:
            raise SystemExit(f"registry model {model} has no slo.ttft_idle_c_ms / ttft_idle_b_ms_per_token")
    return LabelDefinition(
        ttft_p95_ms=DEFAULT_TTFT_P95_MS if ttft_p95_ms is None else float(ttft_p95_ms),
        tpot_p95_ms=DEFAULT_TPOT_P95_MS if tpot_p95_ms is None else float(tpot_p95_ms),
        ttft_slo_mode=mode,
        ttft_slowdown_k=float(k),
        ttft_floor_ms=float(floor),
        ttft_idle_c_ms=c if mode == TTFT_SLO_MODE_SLOWDOWN else None,
        ttft_idle_b_ms_per_token=b if mode == TTFT_SLO_MODE_SLOWDOWN else None,
        min_completed_requests=int(min_completed_requests),
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
        min_completed_requests=first("min_completed_requests")
        if first("min_completed_requests") is not None else DEFAULT_MIN_COMPLETED_REQUESTS,
        registry=first("label_registry", "registry"),
    )


def label_cli_args(label: LabelDefinition) -> list[str]:
    """The CLI arguments that rebuild ``label`` in another process (fit plans)."""
    out = [
        "--ttft-p95-ms", str(label.ttft_p95_ms), "--tpot-p95-ms", str(label.tpot_p95_ms),
        "--ttft-slo-mode", label.ttft_slo_mode,
        "--min-completed-requests", str(int(label.min_completed_requests)),
    ]
    if label.slowdown:
        out += [
            "--ttft-slowdown-k", str(label.ttft_slowdown_k),
            "--ttft-floor-ms", str(label.ttft_floor_ms),
            "--ttft-idle-c-ms", str(label.ttft_idle_c_ms),
            "--ttft-idle-b-ms-per-token", str(label.ttft_idle_b_ms_per_token),
        ]
    return out


def label_mode_cli_args(label: LabelDefinition) -> list[str]:
    """:func:`label_cli_args` without the fixed thresholds, for a tool whose own
    ``--ttft-slo-ms`` / ``--tpot-slo-ms`` carry them (``r3_grid``, ``rewindow_from_raw``)."""
    full = label_cli_args(label)
    return full[4:]
