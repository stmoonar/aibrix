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
(``scripts.openloop.TruncateOnProxyShed`` / ``StopOnBacklog``): the worker that sees the
first overflowed record trips the admission-overflow truncation for all of them at
once (the parent's observer is a second chance); the backlog ceiling is checked
against an in-flight count kept in shared memory.

A worker never outlives its parent: on Linux it asks the kernel for ``SIGKILL`` when the
parent dies (``prctl(PR_SET_PDEATHSIG)``); everywhere it also watches ``getppid()`` and
stops sending the moment its pipe to the parent breaks. The parent terminates its
workers on ``SIGTERM`` and on any error, including one before :meth:`run`.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import multiprocessing.connection as mp_connection
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from tre_replayer.engine.dispatcher import DispatchRecord, DispatchReport, dispatch_open_loop

#: Seconds between "every worker is ready" and the schedule's offset 0.
DEFAULT_START_DELAY_S = 0.25

#: How often a worker checks that its parent is still there (the fallback where
#: ``PR_SET_PDEATHSIG`` is not available, and the backstop where it is).
PARENT_POLL_S = 0.2


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

    * ``truncate`` - once :meth:`trip_truncation` ran (a worker or the parent saw an
      admission overflow, ``trip_if(record)``), requests scheduled before
      ``drain_start_s`` are dropped and counted as censored; the drain still fires
      (``keep_drain``), exactly as ``TruncateOnProxyShed``.
    * ``max_backlog`` - when the run's outstanding requests reach the ceiling, sending
      stops for good and the rest is censored, exactly as ``StopOnBacklog``.

    Built in the parent *before* the workers fork (it holds shared memory).
    """

    def __init__(self, *, truncate: bool = False, drain_start_s: Optional[float] = None, keep_drain: bool = True,
                 max_backlog: Optional[int] = None, trip_if: Optional[Callable[[dict], bool]] = None,
                 ctx=None) -> None:
        ctx = ctx or fork_context()
        if max_backlog is not None and int(max_backlog) <= 0:
            raise ValueError(f"max_backlog must be positive, got {max_backlog}")
        self.truncate = bool(truncate)
        self.keep_drain = bool(keep_drain)
        self.drain_start_s = drain_start_s if self.keep_drain else None
        self.max_backlog = int(max_backlog) if max_backlog else None
        self.trip_if = trip_if
        self._trunc = ctx.Value("i", 0)
        self._trunc_offset = ctx.Value("d", 0.0)
        self._trunc_ts = ctx.Value("q", 0)
        self._backlog = ctx.Value("i", 0)
        self._backlog_offset = ctx.Value("d", 0.0)
        self._backlog_ts = ctx.Value("q", 0)
        self._outstanding = ctx.Value("i", 0)
        self._peak = ctx.Value("i", 0)
        self.first_overflow_record: Optional[dict] = None

    def trip_truncation(self, offset_s: float, ts_ms: Optional[int], record: Optional[dict] = None) -> bool:
        """Trip the truncation (once); True when this call tripped it."""
        with self._trunc.get_lock():
            if self._trunc.value:
                return False
            self._trunc_offset.value = float(offset_s)
            self._trunc_ts.value = int(ts_ms or 0)
            self._trunc.value = 1
        self.first_overflow_record = record
        return True

    def observe(self, record: dict) -> None:
        """Trip on ``record`` when it is an overflow (``trip_if``). Called by the worker
        that made the record before it goes to the parent - so the other workers stop at
        once, not a pipe hop later - and again by the parent, which keeps the record.
        The truncation instant and offset are always the overflowed record's own."""
        if not self.truncate or self.trip_if is None or not self.trip_if(record):
            return
        offset = float(record.get("scheduled_offset_s") or 0.0)
        if not self.trip_truncation(offset, record.get("actual_send_ts_ms"), record):
            if self.first_overflow_record is None and self._trunc_offset.value == offset:
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


class RunnerError(RuntimeError):
    """A run that failed part-way. ``records`` holds every record received before the
    failure, so a caller can still write them (marked failed) instead of losing them."""

    def __init__(self, message: str, records: list, workers: list) -> None:
        super().__init__(message)
        self.records = records
        self.workers = workers


#: ``make_sender(worker_index, in_flight_counter, on_record) -> sender``; runs in the worker.
SenderFactory = Callable[[int, Any, Callable[[dict], None]], Any]


def die_with_parent(parent_pid: int) -> bool:
    """Ask the kernel to SIGKILL this process when its parent dies; exit now if the
    parent is already gone. False where ``prctl`` is unavailable (not Linux): the
    worker's ``getppid`` watch is then the only guard."""
    armed = False
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            pr_set_pdeathsig = 1
            armed = libc.prctl(pr_set_pdeathsig, int(signal.SIGKILL), 0, 0, 0) == 0
        except Exception:  # noqa: BLE001 - the getppid watch still holds
            armed = False
    if os.getppid() != parent_pid:  # died between the fork and the prctl
        os._exit(1)
    return armed


def _worker_main(index: int, shard: list, make_sender: SenderFactory, conn, gate: Optional[StopGate],
                 in_flight: SharedCounter, parent_pid: int) -> None:
    die_with_parent(parent_pid)
    # Ctrl-C goes to the parent, which terminates the workers; a worker must not die
    # half-way through writing a record to the pipe.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    state = {"orphaned": False}
    counters = {"censored_truncation": 0, "censored_backlog": 0, "unsent_orphaned": 0}

    def on_record(record: dict) -> None:
        if gate is not None:
            gate.observe(record)
        if state["orphaned"]:
            return
        try:
            conn.send(("rec", record))
        except (BrokenPipeError, EOFError, OSError):
            # The parent is gone: whatever this worker still sends is load nobody records.
            state["orphaned"] = True

    try:
        sender = make_sender(index, in_flight, on_record)
        gated = gate.wrap(sender, counters) if gate is not None else sender

        async def call(request, scheduled_ts: float, actual_ts: float) -> None:
            if state["orphaned"] or os.getppid() != parent_pid:
                state["orphaned"] = True
                counters["unsent_orphaned"] += 1
                return
            await gated(request, scheduled_ts, actual_ts)

        async def watch_parent() -> None:
            while True:
                await asyncio.sleep(PARENT_POLL_S)
                if state["orphaned"] or os.getppid() != parent_pid:
                    os._exit(1)

        async def main():
            watcher = None
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
                watcher = asyncio.ensure_future(watch_parent())
                return await dispatch_open_loop(shard, call, start_at=float(message[1]))
            finally:
                if watcher is not None:
                    watcher.cancel()
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
    order, across workers); ``gate`` is checked in the workers before each send;
    ``validate(event)`` runs on every event in the parent before anything forks (a
    request that cannot be sent is refused up front, not discovered in a worker).
    Use it as a context manager, or call :meth:`close`, so workers that never ran are
    not left behind.
    """

    def __init__(self, events: Sequence, make_sender: SenderFactory, *, processes: int,
                 gate: Optional[StopGate] = None, observer: Optional[Callable[[dict], None]] = None,
                 start_delay_s: float = DEFAULT_START_DELAY_S, ctx=None,
                 validate: Optional[Callable[[Any], None]] = None) -> None:
        self.ctx = ctx or fork_context()
        ordered = sorted(events, key=lambda event: event.scheduled_offset_s)
        if validate is not None:
            for event in ordered:
                validate(event)
        self.events = ordered
        count = max(1, min(int(processes), len(ordered))) if ordered else 1
        self.processes = count
        self.gate = gate
        self.observer = observer
        self.start_delay_s = float(start_delay_s)
        self.in_flight = SharedCounter(self.ctx)
        self._procs: list = []
        self._conns: list = []
        self._ran = False
        parent = os.getpid()
        try:
            for index in range(count):
                shard = ordered[index::count]
                parent_conn, child_conn = self.ctx.Pipe(duplex=True)
                proc = self.ctx.Process(target=_worker_main, name=f"tre-send-{index}", daemon=True,
                                        args=(index, shard, make_sender, child_conn, gate, self.in_flight, parent))
                proc.start()
                child_conn.close()
                self._procs.append(proc)
                self._conns.append(parent_conn)
        except BaseException:
            self._terminate()
            raise

    def __enter__(self) -> "ProcessPoolRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Terminate any worker still alive (a runner that never ran, or failed)."""
        self._terminate()

    def _terminate(self) -> None:
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self) -> ShardedRun:
        if self._ran:
            raise RuntimeError("a ProcessPoolRunner runs once")
        self._ran = True
        records: list[dict] = []
        dispatch: list[tuple] = []
        workers: list[dict] = []
        previous = None
        on_main = threading.current_thread() is threading.main_thread()

        def on_sigterm(signum, frame):
            # Killed politely: take the workers down first, then die as SIGTERM would.
            self._terminate()
            raise SystemExit(128 + signum)

        try:
            if on_main:
                previous = signal.signal(signal.SIGTERM, on_sigterm)
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
        except BaseException as exc:
            self._terminate()
            if isinstance(exc, Exception):
                raise RunnerError(str(exc), records, workers) from exc
            raise
        finally:
            if on_main and previous is not None:
                signal.signal(signal.SIGTERM, previous)
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


def monotonic_to_wall_ms(mono_ts: float, now_ms: Callable[[], int]) -> int:
    """A ``time.monotonic()`` instant on the wall clock ``now_ms`` reads (ms)."""
    return int(now_ms() - (time.monotonic() - mono_ts) * 1000.0)


def run_open_loop(events: Sequence, make_sender: SenderFactory, *, processes: int,
                  gate: Optional[StopGate] = None, observer: Optional[Callable[[dict], None]] = None,
                  start_delay_s: float = DEFAULT_START_DELAY_S) -> ShardedRun:
    """Send ``events`` open-loop from ``processes`` workers (>= 1; 1 is still a worker
    process, so the parent stays free for the gate and the sidecars)."""
    with ProcessPoolRunner(events, make_sender, processes=processes, gate=gate, observer=observer,
                           start_delay_s=start_delay_s) as runner:
        return runner.run()
