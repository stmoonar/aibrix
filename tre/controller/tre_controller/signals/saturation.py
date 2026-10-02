"""Onset saturation rescue (2026-10-02, design docs/design/20261002-saturation-onset-rescue.md).

The TSS numerator counts tokens when a request COMPLETES. At a load onset nothing has
completed yet: the window numerator is zero (Z undefined, the model takes part in no
decision), and once the first requests complete the O1 evidence gate still holds the
model for about two gateway grids. A model that is visibly full meanwhile - requests
waiting in vLLM, or the KV cache nearly full - is not scaled for 40-50 s.

This module decides, per model and per metrics window, whether such a model is a
CRITICAL receiver anyway:

* **eligible** only while the TSS cannot decide:

  - ``numerator_zero``: nothing completed in the window (tokens present);
  - ``o1_hold``: the receiver gate is not warm (O1 evidence gate, or the ADR-0013 onset
    guard with O1 off) AND the breakpoint is the traffic onset or the routable change of
    this tracker's own last step: once a window of the current traffic period was warm,
    an O1 hold caused by anything else (a C1 scale-up, a donor release, a SafeScale hide
    or unhide) is not eligible until the next idle reset / traffic onset;

  a warm TSS decides alone (TSS / Z / C1 rules unchanged);
* **engine full**, from each routable pod's newest gateway sample in the window (not the
  30 s average): ``num_requests_waiting`` summed over the routable pods > 0, or their mean
  KV-cache fill >= ``kv_threshold`` with at least ``kv_min_running`` requests running (a
  single very long prompt can fill the KV cache alone). After a rescue step, the pods that
  step added must be full themselves (vLLM's waiting queue is per pod: a backlog left on
  the old pods is not a reason for the next step);
* **confirmed** on ``consecutive_ticks`` consecutive metrics windows (counted once per
  distinct window end: the rescue / fairness re-reads of one snapshot never count twice);
* after a rescue step the count restarts and only windows after the routable count rose
  above the decision's count count again (bounded doubling 1 -> 2 -> 4). A step that never
  lands (refused, observe mode) releases the wait after ``await_timeout_ms``
  (``saturation_step_unconfirmed``); a routable change the tracker did not cause restarts
  the count too.

Not full - a single long request running, a stuck request - stays with the TSS verdict:
the v2 idle rule (no completed token is no evidence of starvation) is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from tre_common.metrics_schema import ModelWindowMetrics

REASON_NUMERATOR_ZERO = "numerator_zero"
REASON_O1_HOLD = "o1_hold"

#: Upper bound of ``scaling.saturation_max_step_factor`` (the registry enforces it too).
MAX_STEP_FACTOR_LIMIT = 4.0


@dataclass(frozen=True)
class SaturationRescueConfig:
    """Registry ``scaling.saturation_*`` (see :class:`tre_common.registry.ScalingRegistryConfig`)."""

    enabled: bool = True
    kv_threshold: float = 0.9
    consecutive_ticks: int = 2
    max_step_factor: float = 2.0
    #: The KV condition alone also needs this many running requests (summed over the
    #: routable pods; per pod for the added-pods check).
    kv_min_running: int = 2
    #: Gateway instant-sample period: a pod's newest sample older than one grid before
    #: the window end is stale and ignored.
    grid_ms: int = 10_000
    #: A rescue step whose routable change is not seen within this long releases the
    #: wait (the SM refused it, observe mode): counting restarts from that window.
    await_timeout_ms: int = 30_000

    @classmethod
    def from_registry(cls, registry: Any, *, grid_ms: int) -> "SaturationRescueConfig":
        grid = max(1, int(grid_ms))
        return cls(grid_ms=grid, await_timeout_ms=3 * grid).with_registry(registry)

    def with_registry(self, registry: Any) -> "SaturationRescueConfig":
        """This config with the registry's ``scaling.saturation_*`` values (a registry
        without a scaling section keeps it unchanged)."""
        scaling = getattr(registry, "scaling", None)
        if not callable(scaling):
            return self
        config = scaling()
        return replace(
            self,
            enabled=bool(getattr(config, "saturation_rescue", self.enabled)),
            kv_threshold=float(getattr(config, "saturation_kv_threshold", self.kv_threshold)),
            consecutive_ticks=max(1, int(getattr(config, "saturation_consecutive_ticks", self.consecutive_ticks))),
            max_step_factor=min(
                MAX_STEP_FACTOR_LIMIT,
                max(1.0, float(getattr(config, "saturation_max_step_factor", self.max_step_factor))),
            ),
        )


@dataclass(frozen=True)
class PodSample:
    """One routable pod's newest gateway sample."""

    pod: str
    waiting: float
    kv: float | None
    running: float


@dataclass(frozen=True)
class SaturationSample:
    """The routable pods' newest gateway samples of one window."""

    waiting: float
    kv: float | None
    pods: int
    sample_ms: int
    running: float = 0.0
    per_pod: tuple[PodSample, ...] = ()


@dataclass(frozen=True)
class SaturationVerdict:
    model: str
    window_end_ms: int
    #: ``numerator_zero`` / ``o1_hold`` (None: not eligible - the TSS decides).
    reason: str | None
    sample: SaturationSample | None
    full: bool
    #: Consecutive confirmed windows up to this one.
    ticks: int
    fire: bool
    #: Waiting for the routable change of the last rescue step (no counting).
    awaiting_step: bool = False
    #: Only windows ending after this count (a routable change was seen on it).
    count_after: int | None = None
    #: ``saturation_step_landed`` / ``saturation_step_unconfirmed`` /
    #: ``saturation_reset_external`` events of this window.
    events: tuple[str, ...] = ()


def saturation_sample(
    metrics: ModelWindowMetrics,
    *,
    hidden_pods: Iterable[str] = (),
    fresh_after_ms: int | None = None,
) -> SaturationSample | None:
    """Sum of ``num_requests_waiting`` / running and mean KV-cache fill over the routable
    pods (``metrics.per_pod`` of the serving window minus hidden probe pods), each pod's
    newest instant sample, plus the per-pod samples. Samples at or before
    ``fresh_after_ms`` are stale and skipped. None: no pod with a fresh sample."""
    hidden = set(hidden_pods)
    per_pod: list[PodSample] = []
    newest = None
    for name, pod in sorted((metrics.per_pod or {}).items()):
        pod_name = getattr(pod, "pod", name) or name
        if name in hidden or pod_name in hidden:
            continue
        stamp = getattr(pod, "latest_instant_ms", None)
        if stamp is None or (fresh_after_ms is not None and int(stamp) <= int(fresh_after_ms)):
            continue
        kv = getattr(pod, "latest_gpu_cache", None)
        per_pod.append(
            PodSample(
                pod=str(pod_name),
                waiting=max(0.0, float(getattr(pod, "latest_waiting", 0.0) or 0.0)),
                kv=None if kv is None else float(kv),
                running=max(0.0, float(getattr(pod, "latest_running", 0.0) or 0.0)),
            )
        )
        newest = int(stamp) if newest is None else max(newest, int(stamp))
    if not per_pod or newest is None:
        return None
    kv_values = [item.kv for item in per_pod if item.kv is not None]
    return SaturationSample(
        waiting=sum(item.waiting for item in per_pod),
        kv=(sum(kv_values) / len(kv_values)) if kv_values else None,
        pods=len(per_pod),
        sample_ms=newest,
        running=sum(item.running for item in per_pod),
        per_pod=tuple(per_pod),
    )


def eligibility_reason(context: Mapping[str, Any] | None) -> str | None:
    """Why the model's TSS cannot decide this window, before the tracker's breakpoint
    origin check (``o1_hold`` is only eligible for the onset / its own steps), or None
    (the TSS decides). TSS (``zm``) signal only; never on missing metrics."""
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
    last_routable: int | None = None
    #: Routable count at the last rescue step while its change is not yet seen.
    await_n: int | None = None
    await_end: int | None = None
    #: Only windows ending after this count (a routable change was seen on it).
    count_after: int | None = None
    #: A window of the current traffic period was warm (``warm_onset`` = its onset):
    #: an O1 hold is then only eligible inside this tracker's own step chain.
    warm: bool = False
    warm_onset: int | None = None
    #: The breakpoint is (or follows) this tracker's own step.
    chain: bool = False
    #: Pods routable when the last landed step was decided (the added pods must be full).
    step_pods: frozenset[str] | None = None


class SaturationTracker:
    """Per-model consecutive-window state of the onset saturation rescue (in-process,
    like the EMA: a restart starts every count from zero)."""

    def __init__(self, config: SaturationRescueConfig | None = None) -> None:
        self.config = config or SaturationRescueConfig()
        self._state: dict[str, _ModelState] = {}

    def configure(self, registry: Any) -> None:
        """Re-read ``scaling.saturation_*`` (every tick, like PlanConfig)."""
        self.config = self.config.with_registry(registry)

    def _pod_full(self, pod: PodSample) -> bool:
        cfg = self.config
        return pod.waiting > 0.0 or (
            pod.kv is not None and pod.kv >= cfg.kv_threshold and pod.running >= cfg.kv_min_running
        )

    def engine_full(self, sample: SaturationSample | None) -> bool:
        cfg = self.config
        if sample is None:
            return False
        return sample.waiting > 0.0 or (
            sample.kv is not None and sample.kv >= cfg.kv_threshold and sample.running >= cfg.kv_min_running
        )

    def observe(
        self,
        model: str,
        *,
        window_end_ms: int,
        routable: int,
        reason: str | None,
        sample: SaturationSample | None,
        tss_warm: bool = False,
        onset_ms: int | None = None,
    ) -> SaturationVerdict:
        """The model's verdict for the window ending at ``window_end_ms``; a re-read of
        the same window returns the same verdict without counting again.

        ``reason``: :func:`eligibility_reason` of the window; ``tss_warm``: the TSS
        decides this window (tokens present, not eligible); ``onset_ms``: the model's
        current traffic onset (a new onset re-opens the O1-hold eligibility)."""
        state = self._state.setdefault(model, _ModelState())
        end = int(window_end_ms)
        routable = int(routable)
        if state.last_end == end and state.verdict is not None:
            return state.verdict
        if state.last_end is not None and end < state.last_end:
            # An older window than one already counted: report, never count.
            return SaturationVerdict(model, end, None, sample, False, 0, False)
        state.last_end = end
        cfg = self.config
        events: list[str] = []
        if state.warm and onset_ms != state.warm_onset:
            # New traffic period (an idle reset cleared the onset, or a new one began).
            state.warm = False
            state.chain = False
            state.step_pods = None
        changed = state.last_routable is not None and routable != state.last_routable
        if state.await_n is not None:
            if routable > state.await_n:
                events.append(f"saturation_step_landed:{model}:{state.await_n}->{routable}")
                state.count_after = end  # this window's sample may predate the change
                state.await_n = state.await_end = None
            elif routable < state.await_n:
                events.append(f"saturation_reset_external:{model}:{state.await_n}->{routable}")
                self._external(state, end)
            elif end - int(state.await_end or end) >= cfg.await_timeout_ms:
                events.append(f"saturation_step_unconfirmed:{model}:n={routable}:waited_ms={end - int(state.await_end)}")
                state.count_after = end
                state.await_n = state.await_end = None
                state.step_pods = None  # nothing landed: the next step is a first step again
        elif changed:
            events.append(f"saturation_reset_external:{model}:{state.last_routable}->{routable}")
            self._external(state, end)
        state.last_routable = routable
        if tss_warm:
            state.warm = True
            state.warm_onset = onset_ms
            state.chain = False
            state.step_pods = None
        effective = reason
        if reason == REASON_O1_HOLD and state.warm and not state.chain:
            effective = None  # the hold comes from a breakpoint the tracker did not cause
        eligible = cfg.enabled and effective is not None and sample is not None and routable > 0
        full = bool(eligible and self.engine_full(sample) and self._added_pods_full(state, sample))
        counted = state.await_n is None and (state.count_after is None or end > state.count_after)
        state.streak = state.streak + 1 if (full and counted) else 0
        fire = state.streak >= cfg.consecutive_ticks
        state.verdict = SaturationVerdict(
            model=model,
            window_end_ms=end,
            reason=effective if eligible else None,
            sample=sample,
            full=full,
            ticks=state.streak,
            fire=fire,
            awaiting_step=state.await_n is not None,
            count_after=state.count_after,
            events=tuple(events),
        )
        return state.verdict

    def _added_pods_full(self, state: _ModelState, sample: SaturationSample | None) -> bool:
        """After a landed step: every pod that step added is full itself."""
        if state.step_pods is None or state.await_n is not None or sample is None:
            return True
        added = [pod for pod in sample.per_pod if pod.pod not in state.step_pods]
        return bool(added) and all(self._pod_full(pod) for pod in added)

    @staticmethod
    def _external(state: _ModelState, end: int) -> None:
        state.streak = 0
        state.count_after = end
        state.await_n = state.await_end = None
        state.chain = False
        state.step_pods = None

    def note_step(
        self, model: str, *, window_end_ms: int, routable: int, pods: Iterable[str] = ()
    ) -> None:
        """A rescue scale-up was planned from this window (``pods`` = the routable pods
        then): restart the count, wait for the routable count to rise above
        ``routable``, and require the added pods to be full for the next step."""
        state = self._state.setdefault(model, _ModelState())
        state.streak = 0
        state.await_n = int(routable)
        state.await_end = int(window_end_ms)
        state.last_routable = int(routable)
        state.chain = True
        state.step_pods = frozenset(str(pod) for pod in pods)
        if state.verdict is not None and state.verdict.window_end_ms == int(window_end_ms):
            state.verdict = replace(state.verdict, fire=False, ticks=0, awaiting_step=True)

    def reset(self, model: str) -> None:
        self._state.pop(model, None)
