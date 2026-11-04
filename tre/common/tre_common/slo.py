"""The one SLO definition: per-request TTFT / TPOT thresholds and the request verdict.

Every consumer of "what is this request's SLO" and "did it meet it" calls this module:
the calibration window label (``tre_common.slo_labels``), the scorer
(``tre_eval.metrics``), the SafeScale probe thresholds
(``tre_controller.planning.safescale_evidence``), the baseline tools
(``tre_baselines.tools.slo_judge`` / ``preserve_nl_slo`` / ``tokenscale_profile`` through
the label) and the baseline-scaler limits (``tre_baselines.config``). A second
implementation is a bug (guard: ``common/tests/test_slo_single_definition.py``).

One switch, one parameter set per mode
--------------------------------------
The registry's top-level ``slo_mode`` names the definition in force for EVERY consumer
(user rule 2026-10-29: one switch acts on all consumers, each mode has its own
parameters in ``models[].slo``):

``fixed`` (user 2026-10-29; live since 2026-10-31)
    TTFT_slo = ``ttft_p95_ms`` (500 ms), TPOT_slo = ``tpot_p95_ms`` (75 ms), any L.
``slowdown`` (plan 2026-09-21 6.11 D6'; the theta label of the 10-03 calibration and the
    scorer of every run before 2026-11-04 - kept to reproduce that data)
    TTFT_slo(L) = ``max(ttft_floor_ms, ttft_slowdown_k * (c + b * L))`` (k 5, floor 500 ms),
    TPOT_slo = ``tpot_p95_ms`` (75 ms).
``idle_multiple`` (user 2026-11-04, docs slo-idle-multiples-20261104)
    TTFT_slo(L) = ``max(idle_multiple.ttft_floor_ms, idle_multiple.ttft_k * (c + b * L))``,
    TPOT_slo = ``idle_multiple.tpot_k * idle_multiple.tpot_idle_ms``.

``c`` / ``b`` = ``ttft_idle_c_ms`` / ``ttft_idle_b_ms_per_token`` (the idle TTFT fit),
shared by both per-request modes. ``L`` = the request's prompt tokens.

The request verdict (:func:`judge_request`): a request that did not complete is a
violation (``failed``); else TTFT > TTFT_slo(L) (``ttft``); else TPOT =
``(e2e - ttft) / (output_tokens - 1)`` > TPOT_slo (``tpot``); one output token: TPOT is
undefined and TTFT alone is judged.

Registries written before 2026-11-04 have no ``slo_mode``; :func:`resolve_slo_mode` reads
their two legacy keys (``models[].slo.ttft_slo_mode``, which the scorer and the labels
obeyed, and ``safescale.slo_mode``, which the probe obeyed) for reproduction only, and
reports when they disagree (the controller refuses such a registry).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

SLO_MODE_FIXED = "fixed"
SLO_MODE_SLOWDOWN = "slowdown"
SLO_MODE_IDLE_MULTIPLE = "idle_multiple"
SLO_MODES = (SLO_MODE_FIXED, SLO_MODE_SLOWDOWN, SLO_MODE_IDLE_MULTIPLE)
#: Modes whose TTFT threshold depends on the prompt length.
PER_REQUEST_TTFT_MODES = (SLO_MODE_SLOWDOWN, SLO_MODE_IDLE_MULTIPLE)
#: A registry with no switch at all (neither ``slo_mode`` nor a legacy key): the built-in
#: default both legacy consumers had (labels / slowdown).
DEFAULT_SLO_MODE = SLO_MODE_SLOWDOWN

#: Parameter defaults of the legacy modes when a ``models[].slo`` block leaves them out
#: (the values every registry since 2026-09-21 carries). ``idle_multiple`` has no defaults:
#: its block must be complete.
FIXED_TTFT_MS = 500.0
FIXED_TPOT_MS = 75.0
SLOWDOWN_K = 5.0
SLOWDOWN_FLOOR_MS = 500.0

#: ``models[].slo.idle_multiple`` keys (all required in that block).
IDLE_MULTIPLE_KEY = "idle_multiple"
IDLE_MULTIPLE_FIELDS = ("ttft_k", "tpot_k", "ttft_floor_ms", "tpot_idle_ms")

#: Where the switch lives, and the two legacy keys it replaces.
SLO_MODE_KEY = "slo_mode"
LEGACY_MODEL_KEY = "ttft_slo_mode"
LEGACY_SAFESCALE_KEY = "slo_mode"
#: ``safescale.slo_mode`` names -> the mode they meant.
LEGACY_SAFESCALE_MODES = {"labels": SLO_MODE_SLOWDOWN, "fixed": SLO_MODE_FIXED}

CAUSE_FAILED = "failed"
CAUSE_TTFT = "ttft"
CAUSE_TPOT = "tpot"
CAUSES = (CAUSE_FAILED, CAUSE_TTFT, CAUSE_TPOT)


class SloConfigError(ValueError):
    """The ``models[].slo`` block lacks a parameter the mode needs (or has a bad one)."""


def request_ttft_slo_ms(
    prompt_tokens: float, *, c_ms: float, b_ms_per_token: float, k: float, floor_ms: float,
) -> float:
    """The per-request TTFT threshold of both per-request modes: ``max(floor, k * (c + b * L))``."""
    idle = float(c_ms) + float(b_ms_per_token) * float(prompt_tokens)
    return max(float(floor_ms), float(k) * idle)


def tpot_idle_multiple_ms(*, tpot_k: float, tpot_idle_ms: float) -> float:
    """The ``idle_multiple`` TPOT threshold: ``tpot_k * tpot_idle_ms``."""
    return float(tpot_k) * float(tpot_idle_ms)


# ------------------------------------------------------------------ the thresholds


@dataclass(frozen=True)
class SloThresholds:
    """The thresholds of one model under one mode (build with :func:`slo_thresholds`)."""

    mode: str
    tpot_ms: float
    #: ``fixed``: the TTFT threshold. Per-request modes: None.
    ttft_fixed_ms: Optional[float] = None
    #: Per-request modes: idle TTFT fit, multiple and floor.
    c_ms: Optional[float] = None
    b_ms_per_token: Optional[float] = None
    ttft_k: Optional[float] = None
    ttft_floor_ms: Optional[float] = None
    #: ``idle_multiple``: what ``tpot_ms`` is made of (recorded).
    tpot_k: Optional[float] = None
    tpot_idle_ms: Optional[float] = None

    @property
    def per_request_ttft(self) -> bool:
        return self.mode in PER_REQUEST_TTFT_MODES

    def ttft_ms(self, prompt_tokens: Optional[float]) -> float:
        """TTFT threshold of a request of ``prompt_tokens``. A per-request mode without a
        length (``None``) gives the floor (the threshold of the shortest prompts)."""
        if not self.per_request_ttft:
            return float(self.ttft_fixed_ms)  # type: ignore[arg-type]
        if prompt_tokens is None:
            return float(self.ttft_floor_ms)  # type: ignore[arg-type]
        return request_ttft_slo_ms(
            prompt_tokens, c_ms=self.c_ms, b_ms_per_token=self.b_ms_per_token,  # type: ignore[arg-type]
            k=self.ttft_k, floor_ms=self.ttft_floor_ms,  # type: ignore[arg-type]
        )

    def as_record(self) -> dict[str, Any]:
        """What a result file records of the SLO it judged with."""
        out: dict[str, Any] = {"mode": self.mode, "tpot_slo_ms": self.tpot_ms}
        if self.per_request_ttft:
            out.update({
                "ttft_slo_ms": "max(ttft_floor_ms, ttft_k * (c_ms + b_ms_per_token * prompt_tokens))",
                "c_ms": self.c_ms, "b_ms_per_token": self.b_ms_per_token,
                "ttft_k": self.ttft_k, "ttft_floor_ms": self.ttft_floor_ms,
            })
        else:
            out["ttft_slo_ms"] = self.ttft_fixed_ms
        if self.mode == SLO_MODE_IDLE_MULTIPLE:
            out.update({"tpot_slo_rule": "tpot_k * tpot_idle_ms", "tpot_k": self.tpot_k,
                        "tpot_idle_ms": self.tpot_idle_ms})
        return out


def _field(block: Any, key: str) -> Any:
    """``key`` of a ``models[].slo`` block given as a mapping (raw registry YAML) or as the
    loaded ``tre_common.registry.SloSpec``."""
    if block is None:
        return None
    if isinstance(block, Mapping):
        return block.get(key)
    return getattr(block, key, None)


def _positive(value: Any, where: str, *, allow_zero: bool = False) -> float:
    if value is None or isinstance(value, bool):
        raise SloConfigError(f"{where} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SloConfigError(f"{where} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise SloConfigError(f"{where} must be a {'non-negative' if allow_zero else 'positive'} number, got {value!r}")
    return number


def _idle_fit(block: Any) -> tuple[float, float]:
    c = _positive(_field(block, "ttft_idle_c_ms"), "slo.ttft_idle_c_ms", allow_zero=True)
    b = _positive(_field(block, "ttft_idle_b_ms_per_token"), "slo.ttft_idle_b_ms_per_token", allow_zero=True)
    if c + b <= 0:
        raise SloConfigError("slo idle TTFT fit (ttft_idle_c_ms, ttft_idle_b_ms_per_token) is all zero")
    return c, b


def slo_thresholds(block: Any, mode: str) -> SloThresholds:
    """The thresholds of a ``models[].slo`` block (mapping or ``SloSpec``) under ``mode``.

    Raises :class:`SloConfigError` when the block lacks a parameter the mode needs."""
    if mode not in SLO_MODES:
        raise SloConfigError(f"slo_mode must be one of {SLO_MODES}, got {mode!r}")
    if mode == SLO_MODE_FIXED:
        ttft = _field(block, "ttft_p95_ms")
        tpot = _field(block, "tpot_p95_ms")
        return SloThresholds(
            mode=mode,
            ttft_fixed_ms=_positive(FIXED_TTFT_MS if ttft is None else ttft, "slo.ttft_p95_ms"),
            tpot_ms=_positive(FIXED_TPOT_MS if tpot is None else tpot, "slo.tpot_p95_ms"),
        )
    c, b = _idle_fit(block)
    if mode == SLO_MODE_SLOWDOWN:
        k = _field(block, "ttft_slowdown_k")
        floor = _field(block, "ttft_floor_ms")
        tpot = _field(block, "tpot_p95_ms")
        return SloThresholds(
            mode=mode, c_ms=c, b_ms_per_token=b,
            ttft_k=_positive(SLOWDOWN_K if k is None else k, "slo.ttft_slowdown_k"),
            ttft_floor_ms=_positive(SLOWDOWN_FLOOR_MS if floor is None else floor, "slo.ttft_floor_ms"),
            tpot_ms=_positive(FIXED_TPOT_MS if tpot is None else tpot, "slo.tpot_p95_ms"),
        )
    sub = _field(block, IDLE_MULTIPLE_KEY)
    if sub is None:
        raise SloConfigError(f"slo_mode {SLO_MODE_IDLE_MULTIPLE} needs models[].slo.{IDLE_MULTIPLE_KEY}")
    values = {f: _positive(_field(sub, f), f"slo.{IDLE_MULTIPLE_KEY}.{f}") for f in IDLE_MULTIPLE_FIELDS}
    return SloThresholds(
        mode=mode, c_ms=c, b_ms_per_token=b,
        ttft_k=values["ttft_k"], ttft_floor_ms=values["ttft_floor_ms"],
        tpot_k=values["tpot_k"], tpot_idle_ms=values["tpot_idle_ms"],
        tpot_ms=tpot_idle_multiple_ms(tpot_k=values["tpot_k"], tpot_idle_ms=values["tpot_idle_ms"]),
    )


# ------------------------------------------------------------------ the verdict


@dataclass(frozen=True)
class RequestVerdict:
    """One request against its thresholds (times in ms)."""

    violated: bool
    #: ``failed`` / ``ttft`` / ``tpot`` (first that applies) or None.
    cause: Optional[str]
    failed: bool
    #: Each limit on its own (a failed request that delivered a first token can still be
    #: over the TTFT limit; the scorer reports the counts separately).
    v_ttft: bool
    v_tpot: bool
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    ttft_slo_ms: float
    tpot_slo_ms: float


def request_tpot_ms(*, ttft_ms: Optional[float], e2e_ms: Optional[float], output_tokens: Optional[int]) -> Optional[float]:
    """``(e2e - ttft) / (output_tokens - 1)``; None with one output token or a missing time."""
    if ttft_ms is None or e2e_ms is None or output_tokens is None or int(output_tokens) <= 1:
        return None
    return (float(e2e_ms) - float(ttft_ms)) / (int(output_tokens) - 1)


def judge_request(
    thresholds: SloThresholds,
    *,
    completed: bool,
    prompt_tokens: Optional[float],
    ttft_ms: Optional[float],
    e2e_ms: Optional[float] = None,
    output_tokens: Optional[int] = None,
    tpot_ms: Optional[float] = None,
) -> RequestVerdict:
    """The verdict of one request. ``completed`` False (failed, timed out, truncated) is a
    violation; TPOT is judged only for a completed request with more than one output token;
    TTFT is judged whenever there is one. ``tpot_ms`` = a TPOT the caller's log already holds
    (computed the same way, :func:`request_tpot_ms`) instead of e2e / output tokens."""
    ttft_thr = thresholds.ttft_ms(prompt_tokens)
    if tpot_ms is None:
        tpot = request_tpot_ms(ttft_ms=ttft_ms, e2e_ms=e2e_ms, output_tokens=output_tokens) if completed else None
    else:
        tpot = float(tpot_ms) if completed else None
    v_ttft = ttft_ms is not None and float(ttft_ms) > ttft_thr
    v_tpot = tpot is not None and tpot > thresholds.tpot_ms
    failed = not completed
    cause = CAUSE_FAILED if failed else (CAUSE_TTFT if v_ttft else (CAUSE_TPOT if v_tpot else None))
    return RequestVerdict(
        violated=failed or v_ttft or v_tpot, cause=cause, failed=failed, v_ttft=v_ttft, v_tpot=v_tpot,
        ttft_ms=None if ttft_ms is None else float(ttft_ms), tpot_ms=tpot,
        ttft_slo_ms=ttft_thr, tpot_slo_ms=thresholds.tpot_ms,
    )


# ------------------------------------------------------------------ the switch


@dataclass(frozen=True)
class SloModeResolution:
    mode: str
    #: ``slo_mode`` (the switch), ``legacy:models.slo.ttft_slo_mode``,
    #: ``legacy:safescale.slo_mode`` or ``default``.
    source: str
    #: Legacy registries only: why the probe and the scorer used different modes (None
    #: when they agree). The controller refuses to start on such a registry.
    legacy_conflict: Optional[str] = None


def resolve_slo_mode(raw: Mapping[str, Any]) -> SloModeResolution:
    """The mode a registry (raw YAML mapping) puts in force. Raises ValueError for an
    unknown mode, or for a registry that has the switch AND a legacy key (two switches)."""
    raw = raw or {}
    model_modes: dict[str, str] = {}
    for model in raw.get("models") or []:
        value = ((model or {}).get("slo") or {}).get(LEGACY_MODEL_KEY)
        if value is not None:
            model_modes[str((model or {}).get("name"))] = str(value).strip().lower()
    safescale = raw.get("safescale") or {}
    legacy_safescale = safescale.get(LEGACY_SAFESCALE_KEY) if isinstance(safescale, Mapping) else None
    switch = raw.get(SLO_MODE_KEY)
    if switch is not None:
        mode = str(switch).strip().lower()
        if mode not in SLO_MODES:
            raise ValueError(f"{SLO_MODE_KEY} must be one of {SLO_MODES}, got {switch!r}")
        legacy = [f"models[{m}].slo.{LEGACY_MODEL_KEY}" for m in sorted(model_modes)]
        if legacy_safescale is not None:
            legacy.append(f"safescale.{LEGACY_SAFESCALE_KEY}")
        if legacy:
            raise ValueError(
                f"registry has {SLO_MODE_KEY} and the retired key(s) {legacy}: one switch only "
                f"(remove them; {SLO_MODE_KEY} acts on every consumer)"
            )
        return SloModeResolution(mode=mode, source=SLO_MODE_KEY)
    safescale_mode: Optional[str] = None
    if legacy_safescale is not None:
        name = str(legacy_safescale).strip().lower()
        if name not in LEGACY_SAFESCALE_MODES:
            raise ValueError(f"safescale.slo_mode must be one of {tuple(LEGACY_SAFESCALE_MODES)}, got {legacy_safescale!r}")
        safescale_mode = LEGACY_SAFESCALE_MODES[name]
    if model_modes:
        distinct = sorted(set(model_modes.values()))
        bad = [m for m in distinct if m not in (SLO_MODE_FIXED, SLO_MODE_SLOWDOWN)]
        if bad:
            raise ValueError(f"models[].slo.{LEGACY_MODEL_KEY} must be fixed or slowdown, got {bad}")
        if len(distinct) > 1:
            raise ValueError(f"models[].slo.{LEGACY_MODEL_KEY} differs between models {model_modes}; set {SLO_MODE_KEY}")
        mode = distinct[0]
        conflict = None
        if safescale_mode is not None and safescale_mode != mode:
            conflict = (f"legacy registry: the scorer / labels used models[].slo.{LEGACY_MODEL_KEY} {mode}, "
                        f"the SafeScale probe safescale.slo_mode {legacy_safescale}; set {SLO_MODE_KEY}")
        return SloModeResolution(mode=mode, source=f"legacy:models.slo.{LEGACY_MODEL_KEY}", legacy_conflict=conflict)
    if safescale_mode is not None:
        return SloModeResolution(mode=safescale_mode, source=f"legacy:safescale.{LEGACY_SAFESCALE_KEY}")
    # No switch at all: slowdown (the built-in default of both legacy consumers) when every
    # model has the idle fit it needs, else fixed (what the probe fell back to without it).
    blocks = [(m or {}).get("slo") or {} for m in raw.get("models") or []]
    has_fit = all(b.get("ttft_idle_c_ms") is not None and b.get("ttft_idle_b_ms_per_token") is not None
                  for b in blocks)
    return SloModeResolution(mode=DEFAULT_SLO_MODE if has_fit else SLO_MODE_FIXED, source="default")


def registry_slo_thresholds(raw: Mapping[str, Any], *, mode: Optional[str] = None) -> dict[str, SloThresholds]:
    """model -> thresholds of a raw registry mapping (a run's registry snapshot), under the
    registry's own mode unless ``mode`` is given."""
    use = mode or resolve_slo_mode(raw).mode
    return {str(m["name"]): slo_thresholds(m.get("slo") or {}, use) for m in raw.get("models") or [] if m.get("slo")}
