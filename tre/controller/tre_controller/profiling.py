from __future__ import annotations

import asyncio
import json
import resource
import time
from typing import Any, Awaitable, Callable

PROFILE_STREAM_KEY = "tre:v2:profile:events"


def _now_ms() -> int:
    return int(time.time() * 1000)


class TickProfiler:
    """Opt-in control-loop profiler.

    The hot path only appends plain dicts to an in-memory buffer (``record``);
    a background ``flush_loop`` drains the buffer to a bounded Redis stream via a
    pipeline every ``flush_interval_s`` so no redis round-trip lands on a tick.
    A ``proc_sampler_loop`` samples process CPU/RSS between ticks. Everything the
    controller pod's single process burns is TRE's control-plane cost.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        stream_key: str = PROFILE_STREAM_KEY,
        maxlen: int = 200_000,
        flush_interval_s: float = 1.0,
    ) -> None:
        self._redis = redis_client
        self._stream_key = stream_key
        self._maxlen = int(maxlen)
        self._flush_interval_s = float(flush_interval_s)
        self._buffer: list[dict] = []
        self._seq: dict[str, int] = {}

    # --- hot path (must stay cheap: no I/O, no formatting) ---
    def record(self, event: dict) -> None:
        self._buffer.append(event)

    def next_seq(self, loop: str) -> int:
        value = self._seq.get(loop, 0) + 1
        self._seq[loop] = value
        return value

    @staticmethod
    def now_ms() -> int:
        return _now_ms()

    @staticmethod
    def rusage_ms() -> tuple[float, float]:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime * 1000.0, r.ru_stime * 1000.0

    # --- background sinks ---
    def flush(self) -> int:
        if not self._buffer:
            return 0
        batch = self._buffer
        self._buffer = []
        payloads = [{"data": json.dumps(event, separators=(",", ":"))} for event in batch]
        try:
            pipeline_factory = getattr(self._redis, "pipeline", None)
            if callable(pipeline_factory):
                pipe = pipeline_factory()
                for fields in payloads:
                    pipe.xadd(self._stream_key, fields, maxlen=self._maxlen, approximate=True)
                pipe.execute()
            else:
                for fields in payloads:
                    self._redis.xadd(self._stream_key, fields, maxlen=self._maxlen, approximate=True)
        except Exception:  # noqa: BLE001 - profiling is best-effort; never crash the loop.
            return 0
        return len(batch)

    async def flush_loop(self, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        while True:
            await sleep(self._flush_interval_s)
            self.flush()

    async def proc_sampler_loop(
        self,
        *,
        interval_s: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        import psutil  # local import: psutil is only needed when profiling is enabled.

        proc = psutil.Process()
        # psutil.cpu_percent() needs a priming call; the first reading is meaningless.
        proc.cpu_percent(None)
        while True:
            await sleep(interval_s)
            try:
                cpu_percent = proc.cpu_percent(None)
                rss_mib = proc.memory_info().rss / 1_048_576
            except Exception:  # noqa: BLE001 - never crash the sampler.
                continue
            self.record(
                {
                    "kind": "proc",
                    "ts_ms": _now_ms(),
                    "cpu_percent": cpu_percent,
                    "rss_mib": rss_mib,
                }
            )


def build_profiler(cfg: Any, redis_client: Any) -> "TickProfiler | None":
    if not getattr(cfg, "profile_enabled", False):
        return None
    return TickProfiler(
        redis_client,
        maxlen=getattr(cfg, "profile_stream_maxlen", 200_000),
        flush_interval_s=getattr(cfg, "profile_flush_interval_s", 1.0),
    )
