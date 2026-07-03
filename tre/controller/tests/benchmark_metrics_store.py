
from __future__ import annotations

import time
from pathlib import Path

from make_redis_fixture import FakeRedis, populate_large_fixture
from tre_common.registry import load_registry
from tre_controller.store.metrics_store import MetricsStore

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"


def main() -> None:
    redis = FakeRedis()
    window = populate_large_fixture(redis)
    registry = load_registry(str(REGISTRY_PATH))
    store = MetricsStore(redis, registry, instant_sample_interval_ms=5_000)

    started = time.perf_counter()
    snapshot = store.read_snapshot(window.start_ms, window.end_ms)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(
        "metrics_store_benchmark",
        f"models={len(snapshot.models)}",
        "pods=24",
        "duration_minutes=30",
        f"elapsed_ms={elapsed_ms:.3f}",
    )
    if elapsed_ms >= 100.0:
        raise SystemExit(f"benchmark exceeded 100ms target: {elapsed_ms:.3f}ms")


if __name__ == "__main__":
    main()
