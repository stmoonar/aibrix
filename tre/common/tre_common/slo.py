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
    TTFT_slo = ``ttft_p95_ms``, TPOT_slo = ``tpot_p95_ms``, any L.
``slowdown`` (plan 2026-09-21 6.11 D6'; the theta label of the 10-03 calibration and the
    scorer of every run before 2026-11-04 - kept to reproduce that data)
    TTFT_slo(L) = ``max(ttft_floor_ms, ttft_slowdown_k * (c + b * L))``,
    TPOT_slo = ``tpot_p95_ms``.
``idle_multiple`` (user 2026-11-04, docs slo-idle-multiples-20261104)
    TTFT_slo(L) = ``max(idle_multiple.ttft_floor_ms, idle_multiple.ttft_k * (c + b * L))``,
    TPOT_slo = ``idle_multiple.tpot_k * idle_multiple.tpot_idle_ms``.

``c`` / ``b`` = ``ttft_idle_c_ms`` / ``ttft_idle_b_ms_per_token`` (the idle TTFT fit),
shared by both per-request modes. ``L`` = the request's prompt tokens.

Every parameter comes from the ``models[].slo`` block (the registry, or a run's registry
snapshot); this module has no default value of any of them (2026-11-04): a block that lacks
a parameter its mode needs is refused (:class:`SloConfigError`). The values: ``deploy/registry.yaml``
(and ``docs/slo-idle-multiples-20261104.md`` of the local workspace for idle_multiple).

Offline tools that judge many requests at once use :func:`violated_array` (the vectorised
:func:`judge_request`, same rule) and read the registry with :func:`load_registry_slo`.

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

    Raises :class:`SloConfigError` when the block lacks a parameter the mode needs (there
    are no defaults: every value is the registry's)."""
    if mode not in SLO_MODES:
        raise SloConfigError(f"slo_mode must be one of {SLO_MODES}, got {mode!r}")
    if mode == SLO_MODE_FIXED:
        return SloThresholds(
            mode=mode,
            ttft_fixed_ms=_positive(_field(block, "ttft_p95_ms"), "slo.ttft_p95_ms"),
            tpot_ms=_positive(_field(block, "tpot_p95_ms"), "slo.tpot_p95_ms"),
        )
    c, b = _idle_fit(block)
    if mode == SLO_MODE_SLOWDOWN:
        return SloThresholds(
            mode=mode, c_ms=c, b_ms_per_token=b,
            ttft_k=_positive(_field(block, "ttft_slowdown_k"), "slo.ttft_slowdown_k"),
            ttft_floor_ms=_positive(_field(block, "ttft_floor_ms"), "slo.ttft_floor_ms"),
            tpot_ms=_positive(_field(block, "tpot_p95_ms"), "slo.tpot_p95_ms"),
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


def violated_array(
    thresholds: SloThresholds,
    *,
    completed: Any,
    prompt_tokens: Any = None,
    ttft_ms: Any,
    tpot_ms: Any,
) -> Any:
    """:func:`judge_request`'s ``violated`` for many requests of one model at once (numpy
    arrays or pandas columns of equal length; numpy is imported here, not by the module).

    Same rule: not ``completed`` -> violated; else TTFT > TTFT_slo(L); else TPOT > TPOT_slo.
    A missing (NaN) TTFT or TPOT is not judged (one output token: TPOT is NaN); a missing
    prompt length in a per-request mode gives the floor. ``prompt_tokens`` may be None in
    ``fixed`` (no length needed). ``tpot_ms`` = ``(e2e - ttft) / (output_tokens - 1)`` as the
    caller's log has it (:func:`request_tpot_ms`). Returns a bool array."""
    import numpy as np

    done = np.asarray(completed, dtype=bool)
    ttft = np.asarray(ttft_ms, dtype=float)
    tpot = np.asarray(tpot_ms, dtype=float)
    if thresholds.per_request_ttft:
        if prompt_tokens is None:
            raise ValueError(f"{thresholds.mode} TTFT threshold needs prompt_tokens")
        length = np.asarray(prompt_tokens, dtype=float)
        idle = float(thresholds.c_ms) + float(thresholds.b_ms_per_token) * length  # type: ignore[arg-type]
        thr = np.maximum(float(thresholds.ttft_floor_ms), float(thresholds.ttft_k) * idle)  # type: ignore[arg-type]
        thr = np.where(np.isfinite(length), thr, float(thresholds.ttft_floor_ms))  # type: ignore[arg-type]
    else:
        thr = float(thresholds.ttft_fixed_ms)  # type: ignore[arg-type]
    v_ttft = np.nan_to_num(ttft, nan=0.0) > thr
    v_tpot = done & (np.nan_to_num(tpot, nan=0.0) > float(thresholds.tpot_ms))
    return (~done) | v_ttft | v_tpot


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


def _legacy_meaning(raw: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """What the two retired keys meant to the consumers of an image from before 2026-11-04,
    per model: (old scorer / labels mode, old SafeScale probe mode, the retired keys present).

    * scorer / labels: ``models[].slo.ttft_slo_mode`` (fixed | slowdown), absent = slowdown;
    * probe: ``safescale.slo_mode: fixed`` = fixed; ``labels`` (and absent, its old default) =
      the model's label rule, i.e. that model's scorer mode."""
    models = [m or {} for m in raw.get("models") or []]
    safescale = raw.get("safescale") or {}
    legacy_safescale = safescale.get(LEGACY_SAFESCALE_KEY) if isinstance(safescale, Mapping) else None
    present: list[str] = []
    probe_fixed = False
    if legacy_safescale is not None:
        name = str(legacy_safescale).strip().lower()
        if name not in LEGACY_SAFESCALE_MODES:
            raise ValueError(f"safescale.slo_mode must be one of {tuple(LEGACY_SAFESCALE_MODES)}, got {legacy_safescale!r}")
        probe_fixed = name == "fixed"
        present.append(f"safescale.{LEGACY_SAFESCALE_KEY}")
    scorer: dict[str, str] = {}
    probe: dict[str, str] = {}
    for model in models:
        name = str(model.get("name"))
        value = (model.get("slo") or {}).get(LEGACY_MODEL_KEY)
        if value is not None:
            value = str(value).strip().lower()
            if value not in (SLO_MODE_FIXED, SLO_MODE_SLOWDOWN):
                raise ValueError(f"models[].slo.{LEGACY_MODEL_KEY} must be fixed or slowdown, got {value!r}")
            present.append(f"models[{name}].slo.{LEGACY_MODEL_KEY}")
        scorer[name] = value or SLO_MODE_SLOWDOWN
        probe[name] = SLO_MODE_FIXED if probe_fixed else scorer[name]
    return scorer, probe, present


def resolve_slo_mode(raw: Mapping[str, Any]) -> SloModeResolution:
    """The mode a registry (raw YAML mapping) puts in force. Raises ValueError for an
    unknown mode, or for a registry whose switch and retired keys disagree.

    A registry may carry the switch AND the retired keys when they mean the same mode to
    an image from before 2026-11-04 (:func:`_legacy_meaning`: its scorer / labels and its
    probe) - a transition registry that the old and the new images read alike, so a
    rollback of the images alone keeps the SLO. ``idle_multiple`` has no legacy
    spelling: next to a retired key it is refused."""
    raw = raw or {}
    scorer, probe, present = _legacy_meaning(raw)
    switch = raw.get(SLO_MODE_KEY)
    if switch is not None:
        mode = str(switch).strip().lower()
        if mode not in SLO_MODES:
            raise ValueError(f"{SLO_MODE_KEY} must be one of {SLO_MODES}, got {switch!r}")
        if present:
            differ = sorted({f"{m} scorer {scorer[m]} / probe {probe[m]}" for m in scorer
                             if scorer[m] != mode or probe[m] != mode})
            if differ:
                raise ValueError(
                    f"registry has {SLO_MODE_KEY} {mode} and the retired key(s) {present} meaning otherwise to an "
                    f"image before 2026-11-04 ({'; '.join(differ)}): one switch (remove them, or make them mean "
                    f"{mode}; {SLO_MODE_KEY} acts on every consumer)"
                )
        return SloModeResolution(mode=mode, source=SLO_MODE_KEY)
    if present:
        safescale = raw.get("safescale") or {}
        legacy_safescale = safescale.get(LEGACY_SAFESCALE_KEY) if isinstance(safescale, Mapping) else None
        keyed = {str((m or {}).get("name")): str(((m or {}).get("slo") or {})[LEGACY_MODEL_KEY]).strip().lower()
                 for m in raw.get("models") or [] if ((m or {}).get("slo") or {}).get(LEGACY_MODEL_KEY) is not None}
        if not keyed:   # only the probe key (as before 2026-11-04: its mode)
            name = str(legacy_safescale).strip().lower()
            return SloModeResolution(mode=LEGACY_SAFESCALE_MODES[name], source=f"legacy:safescale.{LEGACY_SAFESCALE_KEY}")
        modes = sorted(set(keyed.values()))
        if len(modes) > 1:
            raise ValueError(f"models[].slo.{LEGACY_MODEL_KEY} differs between models {keyed}; set {SLO_MODE_KEY}")
        mode = modes[0]
        conflict = None
        # the probe: ``fixed`` = fixed; ``labels`` = the label rule, i.e. this same mode
        if legacy_safescale is not None and str(legacy_safescale).strip().lower() == "fixed" and mode != SLO_MODE_FIXED:
            conflict = (f"legacy registry: the scorer / labels used models[].slo.{LEGACY_MODEL_KEY} {mode}, "
                        f"the SafeScale probe safescale.slo_mode {legacy_safescale}; set {SLO_MODE_KEY}")
        return SloModeResolution(mode=mode, source=f"legacy:models.slo.{LEGACY_MODEL_KEY}", legacy_conflict=conflict)
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


@dataclass(frozen=True)
class RegistrySlo:
    """The SLO of every model of one registry file (:func:`load_registry_slo`)."""

    path: str
    sha256: str
    mode: str
    #: ``slo_mode`` / ``legacy:...`` / ``default`` (the registry's), or ``explicit``.
    source: str
    thresholds: Mapping[str, SloThresholds]

    def of(self, model: str) -> SloThresholds:
        if model not in self.thresholds:
            raise SloConfigError(f"{self.path}: no models[{model}].slo")
        return self.thresholds[model]

    def as_record(self) -> dict[str, Any]:
        """What a result file records: the file, its sha256, the mode and each model's thresholds."""
        return {"registry": self.path, "registry_sha256": self.sha256, "mode": self.mode, "mode_source": self.source,
                "models": {m: t.as_record() for m, t in sorted(self.thresholds.items())}}


def load_registry_slo(path: str, *, mode: Optional[str] = None) -> RegistrySlo:
    """Every model's thresholds of a registry YAML (the deployed registry, a run's snapshot or
    the repo bootstrap; plain YAML read, so a snapshot without the cluster sections works too).

    ``mode`` None: the registry's own (:func:`resolve_slo_mode`); a legacy registry whose two
    retired keys disagree is refused then (name the mode). Raises :class:`SloConfigError`."""
    import hashlib

    import yaml

    with open(path, "rb") as fh:
        data = fh.read()
    raw = yaml.safe_load(data) or {}
    if mode is None:
        try:
            res = resolve_slo_mode(raw)
        except ValueError as exc:
            raise SloConfigError(f"{path}: {exc}") from None
        if res.legacy_conflict:
            raise SloConfigError(f"{path}: {res.legacy_conflict} (or name the mode explicitly)")
        use, source = res.mode, res.source
    else:
        use, source = str(mode).strip().lower(), "explicit"
    try:
        thresholds = registry_slo_thresholds(raw, mode=use)
    except SloConfigError as exc:
        raise SloConfigError(f"{path}: slo_mode {use}: {exc}") from None
    if not thresholds:
        raise SloConfigError(f"{path}: no models[].slo block")
    return RegistrySlo(path=str(path), sha256=hashlib.sha256(data).hexdigest(), mode=use, source=source,
                       thresholds=thresholds)
