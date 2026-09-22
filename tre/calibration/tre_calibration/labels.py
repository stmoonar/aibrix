"""The one SLO label every calibration fit uses (plan 2026-09-21 §6.3 B2/B4, §6.9h/§6.11 D6).

A window is **violated** when

* its TTFT is over the TTFT SLO or its p95 TPOT is over ``tpot_p95_ms``, or
* it holds a request that went unserved - a model error, a dropped connection or a
  client timeout - which the window CSV records as ``slo_violated`` (set by
  ``openloop.mark_unserved_request_windows``, online and in ``rewindow_from_raw``).

Two TTFT modes (``ttft_slo_mode``):

``fixed``
    the historical label: window p95 TTFT against one ``ttft_p95_ms`` for every prompt
    length. Kept as the paper's comparison column.
``slowdown`` (plan §6.9h, decision D6)
    a per-request, length-normalised SLO: ``TTFT_slo,m(L) = max(floor, k * (c_m + b_m*L))``
    where ``c_m + b_m*L`` is the model's idle (isolated-request) TTFT at prompt length
    ``L`` (vLLM ``usage.prompt_tokens``, the raw ``input_tokens``). The window TTFT term is
    ``p95_i(TTFT_i / TTFT_slo(L_i))`` over the requests completed in the window, i.e.
    "at least 95 % of the window's requests met their own SLO". It needs the per-request
    ``ttft_len_samples`` column ``rewindow_from_raw`` writes; a CSV without it is refused
    rather than silently labelled with the fixed rule.

In both modes a window must hold at least ``min_completed_requests`` completed requests
(``completed_requests`` column) for its percentiles to count; below that the window
carries no latency evidence (a p95 over < 20 samples is essentially the maximum). Such a
window is dropped - unless it is marked unserved, in which case it is kept as violated: a
window where requests timed out is exactly the one that must not vanish. CSVs written
before the column existed carry no count, and then only the p95 guard of the re-windower
applies (the pre-D6 behaviour).

End-to-end latency is **not** part of the label: it scales with the output length the
load shape chose, not with how loaded the engine was.

Every fit writes :meth:`LabelDefinition.as_dict` into its output as ``label_def`` so two
artifacts can be checked for label equality instead of trusted.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Union

from tre_common.percentile import histogram_percentile

LABEL_DEF_NAME = "p95_ttft_tpot_plus_unserved_v1"
LABEL_DEF_NAME_SLOWDOWN = "p95_ttft_slowdown_tpot_plus_unserved_v1"

TTFT_SLO_MODE_FIXED = "fixed"
TTFT_SLO_MODE_SLOWDOWN = "slowdown"
TTFT_SLO_MODES = (TTFT_SLO_MODE_FIXED, TTFT_SLO_MODE_SLOWDOWN)

#: Plan §6.9h defaults: k = 3 (Splitwise P90 / SLOs-Serve tight), 150 ms floor.
DEFAULT_TTFT_SLOWDOWN_K = 3.0
DEFAULT_TTFT_FLOOR_MS = 150.0
#: Plan §6.9g/§6.11 note 6: a p95 over fewer requests is essentially the window maximum.
DEFAULT_MIN_COMPLETED_REQUESTS = 20
#: Same percentile mode ``rewindow_from_raw`` uses for the p95 columns.
DEFAULT_PERCENTILE_MODE = "bucket_upper"

#: Window-CSV columns the slowdown label reads (written by ``rewindow_from_raw``).
COMPLETED_REQUESTS_COLUMN = "completed_requests"
TTFT_LEN_SAMPLES_COLUMN = "ttft_len_samples"

#: Latency columns of the window CSV, keyed by the SLO name. e2e is listed only so the
#: generic loader keeps reading ad-hoc CSVs; LabelDefinition never asks for it.
LATENCY_COLUMNS = {"ttft_p95": "p95_ttft", "tpot_p95": "p95_tpot", "e2e_p95": "p95_e2e"}

#: Severity ratio assigned to an unserved window with no latency sample at all. It only
#: feeds the delta-margin severity ordering; any value > 1 keeps it on the violated side,
#: and 2.0 ranks it as "clearly over SLO" without dominating measured ratios.
UNSERVED_MIN_RATIO = 2.0

#: Why :meth:`LabelDefinition.classify` returned no label (see :func:`count_label_exclusions`).
EXCLUDED_LOW_N = "low_n"
EXCLUDED_MISSING_LATENCY = "missing_latency"


class LabelInputError(ValueError):
    """The window CSV lacks a column the label definition needs."""


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


@dataclass(frozen=True)
class LabelDefinition:
    ttft_p95_ms: float
    tpot_p95_ms: float
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

    def with_mode(self, mode: str, **overrides: Any) -> "LabelDefinition":
        return replace(self, ttft_slo_mode=mode, **overrides)

    def latency_slo_ms(self) -> dict[str, float]:
        """Fixed-mode thresholds keyed by SLO name (legacy callers and the fixed label)."""
        return {"ttft_p95": float(self.ttft_p95_ms), "tpot_p95": float(self.tpot_p95_ms)}

    def required_columns(self) -> dict[str, str]:
        cols = {"tpot_p95": LATENCY_COLUMNS["tpot_p95"]}
        cols["ttft_p95"] = TTFT_LEN_SAMPLES_COLUMN if self.slowdown else LATENCY_COLUMNS["ttft_p95"]
        return cols

    def as_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "name": LABEL_DEF_NAME_SLOWDOWN if self.slowdown else LABEL_DEF_NAME,
            "mode": self.ttft_slo_mode,
            "ttft_p95_ms": float(self.ttft_p95_ms),
            "tpot_p95_ms": float(self.tpot_p95_ms),
            "min_n": int(self.min_completed_requests),
            "e2e": "excluded",
            "unserved": "violated (model_error, proxy_transient, client_timeout -> slo_violated column)",
            "missing_latency": "dropped unless unserved (also when completed_requests < min_n)",
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
                        "p95_i(ttft_i / ttft_slo_ms(L_i)) > 1 or p95_tpot > tpot_p95_ms or slo_violated"
                    ),
                }
            )
        else:
            base["violated_if"] = "p95_ttft > ttft_p95_ms or p95_tpot > tpot_p95_ms or slo_violated"
        return base

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelDefinition":
        """Rebuild a definition from :meth:`as_dict` (also pre-D6 artifacts: fixed, and
        no min_n recorded -> 0, i.e. the old behaviour)."""
        mode = str(raw.get("mode") or TTFT_SLO_MODE_FIXED)
        kwargs: dict[str, Any] = {
            "ttft_p95_ms": float(raw["ttft_p95_ms"]),
            "tpot_p95_ms": float(raw["tpot_p95_ms"]),
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

    # -- labelling -------------------------------------------------------------------

    def classify(self, row: Mapping[str, Any]) -> tuple[Optional[WindowLabel], Optional[str]]:
        """(label, None), or (None, reason) with reason one of EXCLUDED_*."""
        unserved = row_unserved(row)
        n = completed_requests(row)
        low_n = n is not None and self.min_completed_requests > 0 and n < self.min_completed_requests
        tpot_ratio: Optional[float] = None
        ttft_ratio: Optional[float] = None
        if not low_n:
            tpot = _as_float(row.get(LATENCY_COLUMNS["tpot_p95"]))
            tpot_ratio = None if tpot is None else tpot / float(self.tpot_p95_ms)
            if self.slowdown:
                ttft_ratio = self.window_ttft_ratio(row)
            else:
                ttft = _as_float(row.get(LATENCY_COLUMNS["ttft_p95"]))
                ttft_ratio = None if ttft is None else ttft / float(self.ttft_p95_ms)
        ratios = [r for r in (ttft_ratio, tpot_ratio) if r is not None]
        if len(ratios) < 2:
            if not unserved:
                return None, (EXCLUDED_LOW_N if low_n else EXCLUDED_MISSING_LATENCY)
            return (
                WindowLabel(
                    slo_met=False, ratio_max=max(ratios + [UNSERVED_MIN_RATIO]), unserved=True,
                    ttft_ratio=ttft_ratio, tpot_ratio=tpot_ratio,
                ),
                None,
            )
        return (
            WindowLabel(
                slo_met=(not unserved) and all(r <= 1.0 for r in ratios),
                ratio_max=max(ratios),
                unserved=unserved,
                ttft_ratio=ttft_ratio,
                tpot_ratio=tpot_ratio,
            ),
            None,
        )

    def label(self, row: Mapping[str, Any]) -> Optional[WindowLabel]:
        return self.classify(row)[0]

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


LabelSpec = Union[LabelDefinition, Mapping[str, float]]


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def row_unserved(row: Mapping[str, Any]) -> bool:
    value = row.get("slo_violated")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def label_window(row: Mapping[str, Any], latency_slo_ms: LabelSpec) -> Optional[WindowLabel]:
    """Label one window-CSV row, or None when it carries no usable evidence.

    Production callers pass the :class:`LabelDefinition` itself. A plain mapping keyed by
    SLO name (``ttft_p95``/``tpot_p95``, or any CSV column name for ad-hoc tests) keeps
    the historical fixed-threshold behaviour without the min-n guard.
    """
    if isinstance(latency_slo_ms, LabelDefinition):
        return latency_slo_ms.label(row)
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


def count_label_exclusions(rows: Iterable[Mapping[str, Any]], label: LabelDefinition) -> dict[str, int]:
    """How many rows the label keeps / drops, and why (reported with the fits)."""
    out = {"labelled": 0, EXCLUDED_LOW_N: 0, EXCLUDED_MISSING_LATENCY: 0}
    for row in rows:
        lab, reason = label.classify(row)
        out["labelled" if lab is not None else str(reason)] += 1
    return out


# -- CLI plumbing shared by every fit entry point ---------------------------------------


def add_label_arguments(parser: argparse.ArgumentParser, *, require_fixed: bool = True) -> None:
    parser.add_argument("--ttft-p95-ms", type=float, required=require_fixed,
                        help="fixed-mode TTFT p95 SLO (ms); recorded but unused in slowdown mode")
    parser.add_argument("--tpot-p95-ms", type=float, required=require_fixed)
    parser.add_argument("--ttft-slo-mode", choices=TTFT_SLO_MODES, default=TTFT_SLO_MODE_FIXED,
                        help="fixed: one TTFT p95 threshold; slowdown: max(floor, k*(c_m+b_m*L)) "
                             "per request (plan 6.9h)")
    parser.add_argument("--ttft-slowdown-k", type=float, default=DEFAULT_TTFT_SLOWDOWN_K)
    parser.add_argument("--ttft-floor-ms", type=float, default=DEFAULT_TTFT_FLOOR_MS)
    parser.add_argument("--ttft-idle-c-ms", type=float, default=None,
                        help="override the registry slo.ttft_idle_c_ms of the model")
    parser.add_argument("--ttft-idle-b-ms-per-token", type=float, default=None,
                        help="override the registry slo.ttft_idle_b_ms_per_token of the model")
    parser.add_argument("--min-completed-requests", type=int, default=DEFAULT_MIN_COMPLETED_REQUESTS,
                        help="windows with fewer completed requests carry no latency evidence")
    parser.add_argument("--label-registry", default=None,
                        help="registry the idle TTFT fit is read from (default: the shared one)")


def label_def_from_args(args: argparse.Namespace, model: Optional[str]) -> LabelDefinition:
    """Build the label for one model from the shared CLI arguments + registry profile."""
    mode = getattr(args, "ttft_slo_mode", TTFT_SLO_MODE_FIXED)
    c = getattr(args, "ttft_idle_c_ms", None)
    b = getattr(args, "ttft_idle_b_ms_per_token", None)
    if mode == TTFT_SLO_MODE_SLOWDOWN and (c is None or b is None):
        if not model:
            raise SystemExit(
                "--ttft-slo-mode slowdown needs a model name or --ttft-idle-c-ms/--ttft-idle-b-ms-per-token"
            )
        from tre_common.registry import load_registry

        slo = load_registry(getattr(args, "label_registry", None)).model(model).slo
        c = slo.ttft_idle_c_ms if c is None else c
        b = slo.ttft_idle_b_ms_per_token if b is None else b
        if c is None or b is None:
            raise SystemExit(f"registry model {model} has no slo.ttft_idle_c_ms / ttft_idle_b_ms_per_token")
    ttft = getattr(args, "ttft_p95_ms", None)
    tpot = getattr(args, "tpot_p95_ms", None)
    return LabelDefinition(
        ttft_p95_ms=500.0 if ttft is None else ttft,
        tpot_p95_ms=75.0 if tpot is None else tpot,
        ttft_slo_mode=mode,
        ttft_slowdown_k=getattr(args, "ttft_slowdown_k", DEFAULT_TTFT_SLOWDOWN_K),
        ttft_floor_ms=getattr(args, "ttft_floor_ms", DEFAULT_TTFT_FLOOR_MS),
        ttft_idle_c_ms=c if mode == TTFT_SLO_MODE_SLOWDOWN else None,
        ttft_idle_b_ms_per_token=b if mode == TTFT_SLO_MODE_SLOWDOWN else None,
        min_completed_requests=getattr(args, "min_completed_requests", DEFAULT_MIN_COMPLETED_REQUESTS),
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
