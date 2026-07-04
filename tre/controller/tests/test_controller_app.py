from __future__ import annotations

from types import SimpleNamespace

from tre_controller.app import ControllerDependencies, build_controller_task_specs
from tre_controller.loops.metrics_task import SnapshotBox


class FakeQueue:
    async def run(self) -> None:
        return None


def _cfg(
    *,
    enable_tre_scaling: bool = True,
    ablation_disable_fast_loop: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_tre_scaling=enable_tre_scaling,
        ablation_disable_fast_loop=ablation_disable_fast_loop,
        metrics_window_ms=60_000,
        monitor_interval_s=20.0,
        rescue_interval_s=5.0,
        fairness_interval_s=10.0,
    )


def _deps() -> ControllerDependencies:
    return ControllerDependencies(
        store=object(),
        snapshot_box=SnapshotBox(),
        queue=FakeQueue(),
        registry=object(),
    )


def test_build_controller_task_specs_includes_all_runtime_tasks_by_default() -> None:
    specs = build_controller_task_specs(_deps(), _cfg())

    assert tuple(spec.name for spec in specs) == (
        "metrics",
        "rescue",
        "fairness",
        "action_queue",
    )


def test_build_controller_task_specs_honors_fast_loop_ablation() -> None:
    specs = build_controller_task_specs(_deps(), _cfg(ablation_disable_fast_loop=True))

    assert tuple(spec.name for spec in specs) == ("metrics", "fairness", "action_queue")


def test_build_controller_task_specs_disables_scaling_tasks_but_keeps_metrics() -> None:
    specs = build_controller_task_specs(_deps(), _cfg(enable_tre_scaling=False))

    assert tuple(spec.name for spec in specs) == ("metrics",)
