"""Per-model replica floor of the service-manager (2026-09-29).

Every operation that takes a pod out of routing - a SafeScale hide
(``PUT /v2/models/{m}/routable``), any sleep (the sleep primitive's prepare, which
hides first) and the fleet repair quarantine - is checked against the model's
floor: the number of ROUTABLE replicas left must stay >= the registry
``min_replicas`` (both arms, TRE and APA, use it).

Routable (binding ids) = awake and not hidden in the SM store AND - when the SM has a
Kubernetes view - a Ready Pod carrying the routable label, and not in an unexpired
transient ``starting`` (or old ``waking``) GPU lease. A make-before-break move (defrag)
counts its already-routable destination explicitly (the store records it only once
the move is done). The routable view unreadable (a failed Pod LIST, Redis) refuses
the operation with 409 ``routable_unknown`` before anything is hidden - except on
the exempt / make-up paths below.

What happens when an operation would go below the floor depends on its path:

* a model-level shrink - ``PUT /v2/models/{m}/target`` on ANY path (2026-10-02;
  before only ``apa``) and ``POST /v2/transfers`` (the donor model) - is clamped so
  that the floor holds: HTTP 200 with ``taken`` and ``clamped_by_floor``, never 409;
* a named binding or pod - ``PUT /v2/bindings/{b}/power`` (``urgent``,
  ``scale_down``, ``safescale_commit``, ``default``), the SafeScale hide
  (``safescale_hide``) and ``defrag``: refused with :class:`FloorViolation` (HTTP
  409, ``error: floor_violation``; the controller treats it as permanent for this
  tick and re-plans on the next one).
* ``startup``: another replica of the model is woken first, best effort (its own
  writer phase, within max_awake_replicas); what cannot be made up is exempt and
  recorded - never RetryLater: a Pod held in its startup gate would be seen as fleet
  drift and trigger a fleet-wide repair (review 2026-09-29 P1-1).
* ``repair``: exempt (fleet repair quarantines the whole fleet), recorded.

Every refusal / clamp / exemption / make-up wake is counted (per path and model; the
counters live in Redis with the sleep stats when the SM has Redis, so they survive a
restart) and logged as a JSON event (WARNING at most once per outcome / path / model
per ``log_interval_s``; DEBUG otherwise).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Iterable

LOG = logging.getLogger(__name__)

#: The SafeScale hide (PUT /v2/models/{m}/routable) is not a sleep path; it is
#: recorded under this name.
HIDE_PATH = "safescale_hide"

#: Paths whose floor violation is refused (FloorViolation).
REJECT_PATHS = frozenset(
    {HIDE_PATH, "safescale_commit", "urgent", "scale_down", "default", "defrag"}
)
#: Historical: /target clamped only these paths; since 2026-10-02 it clamps every
#: path (model-level shrinks are clamped, binding-level sleeps refused).
CLAMP_PATHS = frozenset({"apa"})
#: Paths that wake another replica first (best effort) and are exempt otherwise.
MAKEUP_PATHS = frozenset({"startup"})
EXEMPT_PATHS = frozenset({"repair"})

#: GPU lease phases of a replica that is not serving yet.
TRANSIENT_LEASE_PHASES = frozenset({"waking", "starting"})

#: Outcomes recorded (counter name prefix ``floor_<outcome>``).
OUTCOMES = ("rejected", "clamped", "exempt", "makeup_wake")

RECENT_EVENTS = 50


@dataclass(frozen=True)
class FloorCheck:
    model: str
    floor: int
    #: binding ids routable now
    routable: tuple[str, ...]
    #: binding ids of the operation that are routable now (what it takes out of routing)
    removing: tuple[str, ...]

    @property
    def after(self) -> int:
        return len(set(self.routable) - set(self.removing))

    @property
    def ok(self) -> bool:
        """An operation that removes nothing routable never violates the floor (even
        a model already below it); one that removes something must leave >= floor."""
        return not self.removing or self.after >= self.floor

    @property
    def deficit(self) -> int:
        return max(0, self.floor - self.after) if self.removing else 0

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "floor": self.floor,
            "routable": sorted(self.routable),
            "removing": sorted(self.removing),
            "routable_after": self.after,
        }


class FloorViolation(RuntimeError):
    """An operation would leave a model with fewer routable replicas than its floor
    (HTTP 409, body ``error: floor_violation``). Not retriable as such: the caller
    re-plans (the controller on its next tick)."""

    def __init__(self, check: FloorCheck, *, path: str) -> None:
        super().__init__(
            f"replica floor: {path} would leave {check.model} with {check.after} routable "
            f"replica(s) < min_replicas {check.floor} (routable {sorted(check.routable)}, "
            f"taking {sorted(check.removing)})"
        )
        self.check = check
        self.path = path

    def body(self) -> dict:
        return {
            "detail": str(self),
            "error": "floor_violation",
            "path": self.path,
            "floor": self.check.as_dict(),
        }


def check_floor(
    model: str, floor: int, routable: Iterable[str], removing: Iterable[str]
) -> FloorCheck:
    routable_set = set(routable)
    return FloorCheck(
        model=model,
        floor=max(0, int(floor)),
        routable=tuple(sorted(routable_set)),
        removing=tuple(sorted(routable_set & set(removing))),
    )


@dataclass
class FloorRecorder:
    """Counters (per outcome, and per outcome / path / model) and recent events.

    ``incr(name, amount)``: a counter sink (the sleep journal's Redis-backed
    counters, so ``GET /v2/sleep`` shows them); None = in memory only.
    ``read_counts()``: the sink's persistent counters (the journal's ``stats()``);
    :meth:`counts` reads the ``floor_*`` ones from it, so ``floor.counts`` and
    ``stats`` of ``GET /v2/sleep`` are one set of numbers that survives a restart
    (None = the in-memory counts of this process).
    ``log_interval_s``: WARNING at most once per (outcome, path, model) in this
    many seconds, DEBUG in between (0 = every event); counting is unaffected."""

    incr: object = None
    read_counts: object = None
    log_interval_s: float = 0.0
    clock: object = time.monotonic
    _counts: dict[str, int] = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_EVENTS))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_logged: dict[tuple[str, str, str], float] = field(default_factory=dict)
    _suppressed: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def record(self, outcome: str, check: FloorCheck, *, path: str, **extra) -> dict:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown floor outcome {outcome!r}")
        names = (f"floor_{outcome}_total", f"floor_{outcome}:{path}:{check.model}")
        with self._lock:
            for name in names:
                self._counts[name] = self._counts.get(name, 0) + 1
        sink = self.incr
        if callable(sink):
            for name in names:
                try:
                    sink(name, 1)
                except Exception:  # noqa: BLE001 - counters never fail an operation
                    LOG.exception("recording floor counter %s failed", name)
        event = {
            "event": f"replica_floor_{outcome}",
            "path": path,
            "ts_ms": int(time.time() * 1000),
            **check.as_dict(),
            **extra,
        }
        key = (outcome, path, check.model)
        now = self.clock()
        with self._lock:
            self._recent.appendleft(event)
            last = self._last_logged.get(key)
            interval = float(self.log_interval_s or 0.0)
            if interval > 0 and last is not None and now - last < interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                warn = False
            else:
                self._last_logged[key] = now
                suppressed = self._suppressed.pop(key, 0)
                warn = True
        if warn:
            logged = {**event, "suppressed_since_last_log": suppressed} if suppressed else event
            LOG.warning(json.dumps(logged, sort_keys=True, default=str))
        else:
            LOG.debug(json.dumps(event, sort_keys=True, default=str))
        return event

    def counts(self) -> dict[str, int]:
        reader = self.read_counts
        if callable(reader):
            try:
                return {
                    name: int(value)
                    for name, value in reader().items()
                    if str(name).startswith("floor_")
                }
            except Exception:  # noqa: BLE001 - fall back to this process' counts
                LOG.exception("reading the persistent floor counters failed")
        with self._lock:
            return dict(self._counts)

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._recent)
