"""N processes x one asyncio loop each: the paper's client's concurrency model, kept;
its hand-off, removed.

v1 (``CustomTraceGenerator.client_dispatcher``) ran ``process_count`` worker processes,
each an ``asyncio.run`` loop with one ``openai.AsyncOpenAI`` client, and a dispatcher
that every 0.5 s put the next 5 s of requests onto the least-loaded worker's
``multiprocessing.Queue``. The worker read that queue with a *blocking*
``Queue.get(timeout=0.1)`` inside its event loop, which froze the loop - every pending
send, every open stream - for up to 100 ms whenever the queue was empty (most of the
time), so both send times and TTFT stamps carried up to ~0.1 s of client-made jitter.

:class:`ProcessPoolRunner` keeps the model (processes x asyncio x pooled httpx) and
drops the hand-off: the schedule is known before the run, so it is **pre-sharded**
(request *k* of the time-ordered schedule goes to worker ``k mod N``), each worker is
forked with its shard already in memory, and after a single "go" carrying the shared
start instant every worker fires its own requests at their absolute times
(:func:`tre_replayer.engine.dispatcher.dispatch_open_loop` with ``start_at`` - the
monotonic clock is one clock for all processes of a host). Nothing blocks a worker's
loop. Records stream back to the parent over a pipe as requests complete, so the
parent can act on them while the run is live (:class:`StopGate`).

:class:`StopGate` is the cross-process form of the calibration drivers' two stop rules
(``scripts.openloop.TruncateOnProxyShed`` / ``StopOnBacklog``): the parent trips the
admission-overflow truncation when it sees the first overflowed record; the backlog
ceiling is checked against an in-flight count kept in shared memory.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import multiprocessing.connection as mp_connection
import os
import signal
import time
import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from tre_replayer.engine.dispatcher import DispatchRecord, DispatchReport, dispatch_open_loop

#: Seconds between "every worker is ready" and the schedule's offset 0.
DEFAULT_START_DELAY_S = 0.25


def fork_context():
    """``fork``: the workers inherit their shard, the prompts and the sender factory
    without pickling (the prompt store of a cell is megabytes)."""
    return multiprocessing.get_context("fork")


class SharedCounter:
    """An in-flight count shared by every worker (``in_flight_at_send`` is the run's)."""

    def __init__(self, ctx) -> None:
        self._value = ctx.Value("i", 0)

    def inc(self) -> int:
        with self._value.get_lock():
            self._value.value += 1
            return self._value.value

    def dec(self) -> None:
        with self._value.get_lock():
            self._value.value -= 1

    @property
    def value(self) -> int:
        return self._value.value


class StopGate:
    """Cross-process stop rules, checked by each worker immediately before a send.

    * ``truncate`` - once :meth:`trip_truncation` ran (the parent saw an admission
      overflow), requests scheduled before ``drain_start_s`` are dropped and counted as
      censored; the drain still fires (``keep_drain``), exactly as
      ``TruncateOnProxyShed``.
    * ``max_backlog`` - when the run's outstanding requests reach the ceiling, sending
      stops for good and the rest is censored, exactly as ``StopOnBacklog``.

    Built in the parent *before* the workers fork (it holds shared memory).
    """

    def __init__(self, *, truncate: bool = False, drain_start_s: Optional[float] = None, keep_drain: bool = True,
                 max_backlog: Optional[int] = None, ctx=None) -> None:
        ctx = ctx or fork_context()
        if max_backlog is not None and int(max_backlog) <= 0:
            raise ValueError(f"max_backlog must be positive, got {max_backlog}")
        self.truncate = bool(truncate)
        self.keep_drain = bool(keep_drain)
        self.drain_start_s = drain_start_s if self.keep_drain else None
        self.max_backlog = int(max_backlog) if max_backlog else None
        self._trunc = ctx.Value("i", 0)
        self._trunc_offset = ctx.Value("d", 0.0)
        self._trunc_ts = ctx.Value("q", 0)
        self._backlog = ctx.Value("i", 0)
        self._backlog_offset = ctx.Value("d", 0.0)
        self._backlog_ts = ctx.Value("q", 0)
        self._outstanding = ctx.Value("i", 0)
        self._peak = ctx.Value("i", 0)
        self.first_overflow_record: Optional[dict] = None

    # ---------------------------------------------------------------- parent side

    def trip_truncation(self, offset_s: float, ts_ms: Optional[int], record: Optional[dict] = None) -> None:
        with self._trunc.get_lock():
            if self._trunc.value:
                return
            self._trunc_offset.value = float(offset_s)
            self._trunc_ts.value = int(ts_ms or 0)
            self._trunc.value = 1
        self.first_overflow_record = record

    @property
    def truncated(self) -> bool:
        return bool(self._trunc.value)

    def summary(self, workers: Sequence[dict]) -> tuple[SimpleNamespace, SimpleNamespace]:
        """(truncation, backlog) with the attributes of the in-process wrappers."""
        trunc = SimpleNamespace(
            truncated=bool(self._trunc.value),
            truncated_at_offset_s=self._trunc_offset.value if self._trunc.value else None,
            truncated_at_ts_ms=(self._trunc_ts.value or None) if self._trunc.value else None,
            censored=sum(int(w.get("censored_truncation", 0)) for w in workers),
            first_overflow_record=self.first_overflow_record,
            keep_drain=self.keep_drain,
        )
        backlog = SimpleNamespace(
            truncated=bool(self._backlog.value),
            truncated_at_offset_s=self._backlog_offset.value if self._backlog.value else None,
            truncated_at_ts_ms=(self._backlog_ts.value or None) if self._backlog.value else None,
            censored=sum(int(w.get("censored_backlog", 0)) for w in workers),
            max_backlog=self.max_backlog,
            peak_outstanding=self._peak.value,
        )
        return trunc, backlog

    # ----------------------------------------------------------------- child side

    def wrap(self, sender, counters: dict) -> Callable:
        async def gated(request, scheduled_ts: float, actual_ts: float) -> None:
            offset = float(getattr(request, "scheduled_offset_s", 0.0))
            if self.truncate and self._trunc.value and (self.drain_start_s is None or offset < self.drain_start_s):
                counters["censored_truncation"] += 1
                return
            if not self.max_backlog:
                await sender(request, scheduled_ts, actual_ts)
                return
            with self._outstanding.get_lock():
                stop = bool(self._backlog.value)
                if not stop and self._outstanding.value >= self.max_backlog:
                    self._backlog_offset.value = offset
                    self._backlog_ts.value = int(time.time() * 1000)
                    self._backlog.value = 1
                    stop = True
                if not stop:
                    self._outstanding.value += 1
                    if self._outstanding.value > self._peak.value:
                        self._peak.value = self._outstanding.value
            if stop:
                counters["censored_backlog"] += 1
                return
            try:
                await sender(request, scheduled_ts, actual_ts)
            finally:
                with self._outstanding.get_lock():
                    self._outstanding.value -= 1

        return gated


@dataclass
class ShardedRun:
    records: list[dict]
    report: DispatchReport
    workers: list[dict] = field(default_factory=list)

    @property
    def prompt_store_misses(self) -> int:
        return sum(int(w.get("prompt_store_misses") or 0) for w in self.workers)


#: ``make_sender(worker_index, in_flight_counter, on_record) -> sender``; runs in the worker.
SenderFactory = Callable[[int, Any, Callable[[dict], None]], Any]


def _worker_main(index: int, shard: list, make_sender: SenderFactory, conn, gate: Optional[StopGate],
                 in_flight: SharedCounter) -> None:
    # Ctrl-C goes to the parent, which terminates the workers; a worker must not die
    # half-way through writing a record to the pipe.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        sender = make_sender(index, in_flight, lambda record: conn.send(("rec", record)))
        counters = {"censored_truncation": 0, "censored_backlog": 0}
        call = gate.wrap(sender, counters) if gate is not None else sender

        async def main():
            try:
                # Build the client on this loop before the start: its construction (and the
                # SDK's lazy imports) must not land on the first request's lateness.
                prepare = getattr(sender, "prepare", None)
                if prepare is not None:
                    await prepare()
                conn.send(("ready", os.getpid()))
                message = conn.recv()  # nothing else is scheduled on this loop yet
                if message[0] != "go":
                    return None
                return await dispatch_open_loop(shard, call, start_at=float(message[1]))
            finally:
                await sender.aclose()

        report = asyncio.run(main())
        if report is None:
            return
        close = getattr(sender, "close", None)
        if close is not None:
            close()
        usage = os.times()
        conn.send(("end", {
            "worker": index, "pid": os.getpid(), "requests": len(shard),
            "dispatch": [(r.request_id, r.model, r.scheduled_ts, r.actual_ts) for r in report.records],
            "prompt_store_misses": getattr(sender, "prompt_store_misses", 0),
            "cpu_user_s": usage.user, "cpu_sys_s": usage.system, **counters,
        }))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which fails the run
        try:
            conn.send(("error", f"worker {index}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class ProcessPoolRunner:
    """Fork ``processes`` workers, each holding its shard of ``events``; :meth:`run`
    starts them on one instant and gathers every record. See the module docstring.

    ``observer(record)`` runs in the parent for each record as it arrives (in arrival
    order, across workers); ``gate`` is checked in the workers before each send.
    """

    def __init__(self, events: Sequence, make_sender: SenderFactory, *, processes: int,
                 gate: Optional[StopGate] = None, observer: Optional[Callable[[dict], None]] = None,
                 start_delay_s: float = DEFAULT_START_DELAY_S, ctx=None) -> None:
        self.ctx = ctx or fork_context()
        ordered = sorted(events, key=lambda event: event.scheduled_offset_s)
        self.events = ordered
        count = max(1, min(int(processes), len(ordered))) if ordered else 1
        self.processes = count
        self.gate = gate
        self.observer = observer
        self.start_delay_s = float(start_delay_s)
        self.in_flight = SharedCounter(self.ctx)
        self._procs: list = []
        self._conns: list = []
        for index in range(count):
            shard = ordered[index::count]
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            proc = self.ctx.Process(target=_worker_main, name=f"tre-send-{index}", daemon=True,
                                    args=(index, shard, make_sender, child_conn, gate, self.in_flight))
            proc.start()
            child_conn.close()
            self._procs.append(proc)
            self._conns.append(parent_conn)

    def _terminate(self) -> None:
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(timeout=5)

    def run(self) -> ShardedRun:
        records: list[dict] = []
        dispatch: list[tuple] = []
        workers: list[dict] = []
        try:
            for conn in self._conns:  # every worker has built its sender
                kind, payload = conn.recv()
                if kind == "error":
                    raise RuntimeError(payload)
            start_at = time.monotonic() + self.start_delay_s
            for conn in self._conns:
                conn.send(("go", start_at))
            live = set(self._conns)
            while live:
                for conn in mp_connection.wait(list(live)):
                    try:
                        kind, payload = conn.recv()
                    except EOFError:
                        live.discard(conn)
                        raise RuntimeError("a sender worker exited without reporting") from None
                    if kind == "rec":
                        records.append(payload)
                        if self.observer is not None:
                            self.observer(payload)
                    elif kind == "end":
                        dispatch.extend(payload.pop("dispatch"))
                        workers.append(payload)
                        live.discard(conn)
                    elif kind == "error":
                        raise RuntimeError(payload)
        except BaseException:
            self._terminate()
            raise
        for proc in self._procs:
            proc.join(timeout=30)
        dispatch_records = sorted((DispatchRecord(*row) for row in dispatch), key=lambda r: r.scheduled_ts)
        planned = (self.events[-1].scheduled_offset_s - self.events[0].scheduled_offset_s) if self.events else 0.0
        actual = (max(r.actual_ts for r in dispatch_records) - min(r.actual_ts for r in dispatch_records)
                  if len(dispatch_records) >= 2 else 0.0)
        report = DispatchReport(records=dispatch_records, planned_duration_s=planned, actual_duration_s=actual,
                                base_ts=start_at)
        workers.sort(key=lambda w: w["worker"])
        return ShardedRun(records=records, report=report, workers=workers)


def run_open_loop(events: Sequence, make_sender: SenderFactory, *, processes: int,
                  gate: Optional[StopGate] = None, observer: Optional[Callable[[dict], None]] = None,
                  start_delay_s: float = DEFAULT_START_DELAY_S) -> ShardedRun:
    """Send ``events`` open-loop from ``processes`` workers (>= 1; 1 is still a worker
    process, so the parent stays free for the gate and the sidecars)."""
    runner = ProcessPoolRunner(events, make_sender, processes=processes, gate=gate, observer=observer,
                               start_delay_s=start_delay_s)
    return runner.run()
