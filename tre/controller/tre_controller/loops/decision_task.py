"""The controller's single decision loop (design docs/design/20261028-single-loop.md).

The paper's "one main polling loop with two logical cadences" (v1
``_build_paper_scaling_plan``, one call per tick): one task wakes on every published
snapshot (:meth:`SnapshotBox.wait_newer`, wakes coalesce, the newest snapshot is read)
and makes ONE planner call (:func:`run_planner_tick`) per snapshot with

* the rescue section (CRITICAL receivers, IDLE release, the receiver-less HIGH shrink)
  when ``rescue_interval_s`` of observation time passed since its last run;
* the rebalance section (LOW receivers; ``fairness`` in the planner) when
  ``fairness_interval_s`` passed since its last run - never with
  TRE_ABLATION_DISABLE_SLOW_LOOP - and always together with the rescue section (v1:
  the slow beat is always a fast beat too), so a LOW receiver is never planned on a
  call that does not plan the CRITICAL ones first.

The periods are counted on the snapshot grid: the observation span since a section's
last run is snapped to the nearest whole number of snapshot (producer) periods - that
absorbs a scrape jitter of anything below half a period, and nothing else - and the
section is due once that reaches its period. The cadence is deterministic and rounds UP
to the grid (5 s rescue with 2 s snapshots: every 6 s; a rebalance call, every 10 s,
also runs the rescue section).

Both sections of one call share one ledger (planned deltas, claimed GPUs, released
donor pods) and one :class:`PaperStateCache`; the plan goes to the ActionQueue (the loop
never waits for the SM). Observation time = the newest ``scraped_ms`` of the snapshot's
windows (:func:`observation_ms`), never the controller clock.

A stale snapshot (the sampler marks it after its stale hold) decides nothing; no publish
at all for ``silence_alert_s`` raises the same alert (``decision_snapshot_stale``, once
per stale run; ``decision_snapshot_recovered`` on the next fresh snapshot).
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Callable, Protocol

from tre_common.metrics_schema import MetricsSnapshot, observed_of
from tre_common.registry import Registry
from tre_controller.loops.tick import (
    ActiveProbeModels,
    LoopTickResult,
    PaperStateCache,
    PlannerQueue,
    SafeScaleController,
    resolve_active_probe_models,
    run_planner_tick,
)
from tre_controller.planning.planner import ClusterView
from tre_controller.loops.model_state_box import ModelStateBox
from tre_controller.signals.trs import SignalState

if TYPE_CHECKING:
    from tre_controller.loops.metrics_task import SnapshotBox
    from tre_controller.maintenance import MaintenanceWatch
    from tre_controller.profiling import TickProfiler

LOG = logging.getLogger("tre_controller.decision_loop")


class ClusterViewReader(Protocol):
    def get(self) -> ClusterView | None: ...


class DecisionWriter(Protocol):
    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None: ...


def observation_ms(snapshot: MetricsSnapshot) -> int:
    """The newest observation (gateway ``scraped_ms``) of the snapshot's windows; the
    snapshot's ``ts_ms`` when it has none."""
    newest = [
        int(last)
        for window in snapshot.models.values()
        if (last := observed_of(window).t_last_ms) is not None
    ]
    return max(newest) if newest else int(snapshot.ts_ms)


def grid_span_ms(span_ms: int, producer_ms: int) -> int:
    """``span_ms`` snapped to the nearest whole number of producer (snapshot) periods: two
    observations k periods apart give k periods for any scrape jitter below half a period,
    so a section never runs one snapshot early or late."""
    period = max(1, int(producer_ms))
    return (int(span_ms) + period // 2) // period * period


class SectionSchedule:
    """Which planner sections are due on a snapshot observed at ``obs_ms`` (module
    docstring): a section runs once its period of observation time passed since its last
    run, counted in whole producer periods (:func:`grid_span_ms`); the rebalance section
    only together with the rescue section. ``rebalance_ms`` None: the rebalance section
    never runs (TRE_ABLATION_DISABLE_SLOW_LOOP). An observation time older than a
    section's last run (the gateway clock stepped back) runs it. ``producer_ms``: the
    snapshot producer's period, bound by the caller (the app: the sampler's); 1 = no
    grid (exact spans)."""

    def __init__(
        self, *, rescue_ms: int, rebalance_ms: int | None, producer_ms: int = 1
    ) -> None:
        self.rescue_ms = int(rescue_ms)
        self.rebalance_ms = None if rebalance_ms is None else int(rebalance_ms)
        self.producer_ms = max(1, int(producer_ms))
        self._last_rescue_ms: int | None = None
        self._last_rebalance_ms: int | None = None

    def peek(self, obs_ms: int) -> tuple[bool, bool]:
        """(rescue due, rebalance due) without counting them as run."""
        rebalance = self.rebalance_ms is not None and self._due(self._last_rebalance_ms, obs_ms, self.rebalance_ms)
        return rebalance or self._due(self._last_rescue_ms, obs_ms, self.rescue_ms), rebalance

    def mark(self, obs_ms: int, rescue: bool, rebalance: bool) -> None:
        """The sections that ran at ``obs_ms``."""
        if rescue:
            self._last_rescue_ms = int(obs_ms)
        if rebalance:
            self._last_rebalance_ms = int(obs_ms)

    def due(self, obs_ms: int) -> tuple[bool, bool]:
        """:meth:`peek`, the due sections counted as run at ``obs_ms``."""
        rescue, rebalance = self.peek(obs_ms)
        self.mark(obs_ms, rescue, rebalance)
        return rescue, rebalance

    def _due(self, last_ms: int | None, obs_ms: int, need_ms: int) -> bool:
        if last_ms is None or int(obs_ms) < last_ms:
            return True
        return grid_span_ms(int(obs_ms) - last_ms, self.producer_ms) >= need_ms


class DecisionLoop:
    """One decision per published snapshot (module docstring). :meth:`on_snapshot` is the
    synchronous step; :func:`decision_task` feeds it."""

    def __init__(
        self,
        *,
        queue: PlannerQueue,
        registry: Registry,
        cfg,
        cluster_view: ClusterView | None = None,
        cluster_view_box: ClusterViewReader | None = None,
        active_probe_models: ActiveProbeModels = None,
        decision_writer: DecisionWriter | None = None,
        safescale: SafeScaleController | None = None,
        signal_state: SignalState | None = None,
        prof: "TickProfiler | None" = None,
        model_state_box: ModelStateBox | None = None,
        is_observe: Callable[[], bool] | None = None,
        maintenance: "MaintenanceWatch | None" = None,
        leftover_probes: SafeScaleController | None = None,
        producer_ms: int = 1,
        tick: Callable[..., LoopTickResult] = run_planner_tick,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._cfg = cfg
        self._cluster_view = cluster_view
        self._cluster_view_box = cluster_view_box
        self._active_probe_models = active_probe_models
        self._writer = decision_writer
        self._safescale = safescale
        self._signal_state = signal_state
        self._prof = prof
        self._model_state_box = model_state_box
        self._is_observe = is_observe
        self._maintenance = maintenance
        self._leftover_probes = leftover_probes
        self._tick = tick
        self.paper_state_cache = PaperStateCache(max_stale_ms=getattr(cfg, "paper_stale_max_ms", 30_000))
        self.schedule = SectionSchedule(
            rescue_ms=int(float(cfg.rescue_interval_s) * 1000),
            # TRE_ABLATION_DISABLE_SLOW_LOOP: LOW receivers are never served.
            rebalance_ms=(
                None
                if bool(getattr(cfg, "ablation_disable_slow_loop", False))
                else int(float(cfg.fairness_interval_s) * 1000)
            ),
            producer_ms=producer_ms,
        )
        #: True while the newest snapshot is stale or none was published for the
        #: silence limit (the alert is logged once per stale run).
        self.stale = False
        self._last_obs: tuple[int, int] | None = None

    def on_snapshot(self, snapshot: MetricsSnapshot) -> LoopTickResult | None:
        """Plan on ``snapshot``; None = nothing due (a re-publish of the same
        observations, or no section's period has passed)."""
        if snapshot.stale:
            self._alert_stale(f"snapshot_stale:{snapshot.stale_reason or 'stale_hold'}", snapshot.ts_ms)
            result = LoopTickResult(submitted=0, events=("snapshot_stale",))
            self._write("rescue", snapshot, result)
            return result
        self._recovered(snapshot.ts_ms)
        obs_ms = observation_ms(snapshot)
        key = (int(snapshot.ts_ms), obs_ms)
        if key == self._last_obs:
            return None  # the same window again: nothing new to decide on
        self._last_obs = key
        rescue_due, rebalance_due = self.schedule.peek(obs_ms)
        if not (rescue_due or rebalance_due):
            return None
        loop_name = "rescue"  # the rebalance section never runs without the rescue section
        current_view = self._current_view()
        if self._cluster_view is None and self._cluster_view_box is not None and current_view is None:
            # Nothing decided: the sections stay due for the next snapshot.
            result = LoopTickResult(submitted=0, events=("cluster_view_unavailable",))
        else:
            self.schedule.mark(obs_ms, rescue_due, rebalance_due)
            cfg = self._cfg
            result = self._tick(
                snapshot,
                queue=self._queue,
                registry=self._registry,
                rescue_due=rescue_due,
                fairness_due=rebalance_due,
                cluster_view=current_view,
                active_probe_models=resolve_active_probe_models(self._active_probe_models),
                signal_source=getattr(cfg, "signal_source", "zm"),
                signal_idle_rps_eps=getattr(cfg, "signal_idle_rps_eps", 0.05),
                safescale=self._safescale,
                paper_state_cache=self.paper_state_cache,
                incomplete_policy=getattr(cfg, "incomplete_policy", "drop_model"),
                signal_state=self._signal_state,
                suppress_hot_proactive_probe=getattr(cfg, "safescale_suppress_hot_proactive", False),
                disable_eta_gate=getattr(cfg, "disable_eta_gate", False),
                prof=self._prof,
                loop=loop_name,
                action_cooldown=getattr(cfg, "action_cooldown", True),
                # B8: controller mode, read per decision (ObserveModeGate, cached).
                observe_mode=bool(self._is_observe()) if self._is_observe is not None else False,
                # P2-3: no probe starts while the SM maintenance lock is held.
                probe_block_reason=(
                    self._maintenance.probe_block_reason() if self._maintenance is not None else None
                ),
                leftover_probes=self._leftover_probes,
                safescale_scope=getattr(cfg, "safescale_scope", "all"),
            )
        if self._model_state_box is not None and result.classifications:
            # Review 3: the latest signal state, for the queue's commit revalidation.
            self._model_state_box.update(result.classifications, result.model_contexts, ts_ms=snapshot.ts_ms)
        self._write(loop_name, snapshot, result)
        return result

    def on_silence(self, silent_s: float) -> None:
        """No snapshot was published for ``silent_s``: no decision, the stale alert."""
        self._alert_stale(f"no_snapshot_published_for_s={silent_s:g}", None)

    def _alert_stale(self, reason: str, ts_ms: int | None) -> None:
        if self.stale:
            return
        self.stale = True
        LOG.warning(json.dumps({"event": "decision_snapshot_stale", "reason": reason, "ts_ms": ts_ms}, sort_keys=True))

    def _recovered(self, ts_ms: int) -> None:
        if self.stale:
            self.stale = False
            LOG.info(json.dumps({"event": "decision_snapshot_recovered", "ts_ms": ts_ms}, sort_keys=True))

    def _current_view(self) -> ClusterView | None:
        if self._cluster_view is not None:
            return self._cluster_view
        if self._cluster_view_box is None:
            return None
        return self._cluster_view_box.get()

    def _write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None:
        if self._writer is None:
            return
        if self._prof is None:
            self._writer.write(loop_name, snapshot, result)
            return
        t0 = time.perf_counter_ns()
        self._writer.write(loop_name, snapshot, result)
        self._prof.record(
            {
                "kind": "decision",
                "loop": loop_name,
                "ts_ms": self._prof.now_ms(),
                "decision_write_ns": time.perf_counter_ns() - t0,
            }
        )


async def decision_task(
    snapshot_box: "SnapshotBox",
    loop: DecisionLoop,
    *,
    silence_alert_s: float | None = None,
) -> None:
    """Wait for each publish and decide on the newest snapshot (module docstring).
    ``silence_alert_s``: no publish for this long raises the stale alert (None = never)."""
    seen = -1
    while True:
        if not await snapshot_box.wait_newer(seen, silence_alert_s):
            loop.on_silence(float(silence_alert_s or 0.0))
            continue
        seen = snapshot_box.version
        snapshot = snapshot_box.get()
        if snapshot is not None:
            loop.on_snapshot(snapshot)


def run_rescue_tick(snapshot: MetricsSnapshot, **kwargs) -> LoopTickResult:
    """The rescue section alone on one snapshot (offline replays, tests)."""
    return run_planner_tick(snapshot, **{"rescue_due": True, "fairness_due": False, "loop": "rescue", **kwargs})


def run_fairness_tick(snapshot: MetricsSnapshot, **kwargs) -> LoopTickResult:
    """The rebalance section alone on one snapshot (offline replays, tests)."""
    return run_planner_tick(snapshot, **{"rescue_due": False, "fairness_due": True, "loop": "fairness", **kwargs})
