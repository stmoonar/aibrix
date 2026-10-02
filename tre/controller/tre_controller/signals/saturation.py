"""Onset saturation rescue (2026-10-02, design docs/design/20261002-saturation-onset-rescue.md).

The TSS numerator counts tokens when a request COMPLETES. At a load onset nothing has
completed yet: the window numerator is zero (Z undefined, the model takes part in no
decision), and once the first requests complete the O1 evidence gate still holds the
model for about two gateway grids. A model that is visibly full meanwhile - requests
waiting in vLLM, or the KV cache nearly full - is not scaled for 40-50 s.

This module decides, per model and per metrics window, whether such a model is a
CRITICAL receiver anyway:

* **eligible** only while the TSS cannot decide: numerator zero (``numerator_zero``) or
  the receiver gate not warm - the O1 evidence gate, or the ADR-0013 onset guard when O1
  is off (``o1_hold``). A warm TSS decides alone (TSS / Z / C1 rules unchanged);
* **engine full**: ``num_requests_waiting`` summed over the routable pods > 0, or their
  mean KV-cache fill >= ``kv_threshold``, from each pod's newest gateway sample in the
  window (not the 30 s average);
* **confirmed** on ``consecutive_ticks`` consecutive metrics windows (counted once per
  distinct window end: the rescue / fairness re-reads of one snapshot never count twice);
* after a rescue step the count restarts and only windows observed after the model's
  routable count changed count again (bounded doubling 1 -> 2 -> 4, never straight to the
  cap). A step that never lands (refused, observe mode) releases the wait after
  ``await_timeout_ms``.

Not full - a single long request running, a stuck request - stays with the TSS verdict:
the v2 idle rule (no completed token is no evidence of starvation) is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from tre_common.metrics_schema import ModelWindowMetrics

REASON_NUMERATOR_ZERO = "numerator_zero"
REASON_O1_HOLD = "o1_hold"


@dataclass(frozen=True)
class SaturationRescueConfig:
    """Registry ``scaling.saturation_*`` (see :class:`tre_common.registry.ScalingRegistryConfig`)."""

    enabled: bool = True
    kv_threshold: float = 0.9
    consecutive_ticks: int = 2
    max_step_factor: float = 2.0
    #: Gateway instant-sample period: a pod's newest sample older than one grid before
    #: the window end is stale and ignored.
    grid_ms: int = 10_000
    #: A rescue step whose routable change is not seen within this long releases the
    #: wait (the SM refused it, observe mode): counting restarts from that window.
    await_timeout_ms: int = 30_000

    @classmethod
    def from_registry(cls, registry: Any, *, grid_ms: int) -> "SaturationRescueConfig":
        scaling = getattr(registry, "scaling", None)
        config = scaling() if callable(scaling) else None
        defaults = cls()
        grid = max(1, int(grid_ms))
        return cls(
            enabled=bool(getattr(config, "saturation_rescue", defaults.enabled)),
            kv_threshold=float(getattr(config, "saturation_kv_threshold", defaults.kv_threshold)),
            consecutive_ticks=max(1, int(getattr(config, "saturation_consecutive_ticks", defaults.consecutive_ticks))),
            max_step_factor=max(1.0, float(getattr(config, "saturation_max_step_factor", defaults.max_step_factor))),
            grid_ms=grid,
            await_timeout_ms=3 * grid,
        )


@dataclass(frozen=True)
class SaturationSample:
    """The routable pods' newest gateway samples of one window."""

    waiting: float
    kv: float | None
    pods: int
    sample_ms: int


@dataclass(frozen=True)
class SaturationVerdict:
    model: str
    window_end_ms: int
    #: ``numerator_zero`` / ``o1_hold`` (None: the TSS decides - not eligible).
    reason: str | None
    sample: SaturationSample | None
    full: bool
    #: Consecutive confirmed windows up to this one.
    ticks: int
    fire: bool
    #: Waiting for the routable change of the last rescue step (no counting).
    awaiting_step: bool = False


def saturation_sample(
    metrics: ModelWindowMetrics,
    *,
    hidden_pods: Iterable[str] = (),
    fresh_after_ms: int | None = None,
) -> SaturationSample | None:
    """Sum of ``num_requests_waiting`` and mean KV-cache fill over the routable pods
    (``metrics.per_pod`` of the serving window minus hidden probe pods), each pod's
    newest instant sample. Samples at or before ``fresh_after_ms`` are stale and
    skipped. None: no pod with a fresh sample."""
    hidden = set(hidden_pods)
    waiting = 0.0
    kv_values: list[float] = []
    pods = 0
    newest = None
    for name, pod in (metrics.per_pod or {}).items():
        if name in hidden or getattr(pod, "pod", name) in hidden:
            continue
        stamp = getattr(pod, "latest_instant_ms", None)
        if stamp is None or (fresh_after_ms is not None and int(stamp) <= int(fresh_after_ms)):
            continue
        pods += 1
        waiting += max(0.0, float(getattr(pod, "latest_waiting", 0.0) or 0.0))
        kv = getattr(pod, "latest_gpu_cache", None)
        if kv is not None:
            kv_values.append(float(kv))
        newest = int(stamp) if newest is None else max(newest, int(stamp))
    if pods == 0 or newest is None:
        return None
    return SaturationSample(
        waiting=waiting,
        kv=(sum(kv_values) / len(kv_values)) if kv_values else None,
        pods=pods,
        sample_ms=newest,
    )


def eligibility_reason(context: Mapping[str, Any] | None) -> str | None:
    """Why the model's TSS cannot decide this window (the saturation path applies), or
    None (the TSS decides). TSS (``zm``) signal only; never on missing metrics."""
    if not context or context.get("signal_source", "zm") != "zm":
        return None
    if context.get("signal_unavailable_reason") == "tokens_missing" or "tss_defined" not in context:
        return None
    y_total = context.get("Y_m")
    if y_total is not None and float(y_total) <= 1e-9:
        return REASON_NUMERATOR_ZERO
    if context.get("signal_warm") is False:
        return REASON_O1_HOLD
    return None


def saturation_target(n: int, factor: float, cap: int) -> int:
    """Bounded doubling: ``min(max(n + 1, floor(factor * n)), cap)``."""
    n = max(0, int(n))
    return min(max(n + 1, int(float(factor) * n + 1e-9)), int(cap))


@dataclass
class _ModelState:
    last_end: int | None = None
    streak: int = 0
    verdict: SaturationVerdict | None = None
    #: Routable count at the last rescue step while its change is not yet seen.
    await_n: int | None = None
    await_end: int | None = None
    #: Only windows ending after this count (the window the routable change was seen on).
    count_after: int | None = None


class SaturationTracker:
    """Per-model consecutive-window state of the onset saturation rescue (in-process,
    like the EMA: a restart starts every count from zero)."""

    def __init__(self, config: SaturationRescueConfig | None = None) -> None:
        self.config = config or SaturationRescueConfig()
        self._state: dict[str, _ModelState] = {}

    def observe(
        self,
        model: str,
        *,
        window_end_ms: int,
        routable: int,
        reason: str | None,
        sample: SaturationSample | None,
    ) -> SaturationVerdict:
        """The model's verdict for the window ending at ``window_end_ms``; a re-read of
        the same window returns the same verdict without counting again."""
        state = self._state.setdefault(model, _ModelState())
        end = int(window_end_ms)
        if state.last_end == end and state.verdict is not None:
            return state.verdict
        if state.last_end is not None and end < state.last_end:
            # An older window than one already counted: report, never count.
            return SaturationVerdict(model, end, reason, sample, False, 0, False)
        state.last_end = end
        cfg = self.config
        if state.await_n is not None:
            if int(routable) != state.await_n:
                # The step's routable change is seen on this tick: this window's sample
                # may predate it; count from the next window on.
                state.count_after = end
                state.await_n = state.await_end = None
            elif end - int(state.await_end or end) >= cfg.await_timeout_ms:
                state.count_after = end
                state.await_n = state.await_end = None
        eligible = cfg.enabled and reason is not None and sample is not None and int(routable) > 0
        full = bool(
            eligible
            and (sample.waiting > 0.0 or (sample.kv is not None and sample.kv >= cfg.kv_threshold))
        )
        counted = state.await_n is None and (state.count_after is None or end > state.count_after)
        state.streak = state.streak + 1 if (full and counted) else 0
        fire = state.streak >= cfg.consecutive_ticks
        state.verdict = SaturationVerdict(
            model=model,
            window_end_ms=end,
            reason=reason if eligible else None,
            sample=sample,
            full=full,
            ticks=state.streak,
            fire=fire,
            awaiting_step=state.await_n is not None,
        )
        return state.verdict

    def note_step(self, model: str, *, window_end_ms: int, routable: int) -> None:
        """A rescue scale-up was planned from this window: restart the count and wait for
        the routable count to move off ``routable``."""
        state = self._state.setdefault(model, _ModelState())
        state.streak = 0
        state.await_n = int(routable)
        state.await_end = int(window_end_ms)
        if state.verdict is not None and state.verdict.window_end_ms == int(window_end_ms):
            state.verdict = replace(state.verdict, fire=False, ticks=0, awaiting_step=True)

    def reset(self, model: str) -> None:
        self._state.pop(model, None)
