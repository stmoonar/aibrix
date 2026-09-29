"""Per-model replica floor of the service-manager (2026-09-29).

Every operation that takes a pod out of routing - a SafeScale hide
(``PUT /v2/models/{m}/routable``), any sleep (the sleep primitive's prepare, which
hides first) and the fleet repair quarantine - is checked against the model's
floor: the number of ROUTABLE replicas left must stay >= the registry
``min_replicas`` (both arms, TRE and APA, use it).

Routable = awake, not hidden, Ready, carrying the routable label (when the SM has a
Kubernetes view), without a sleep reservation and not in a transient ``waking`` /
``starting`` GPU lease. A replica that is being woken does not count.

What happens when an operation would go below the floor depends on its path:

* ``safescale_hide``, ``urgent``, ``scale_down``, ``safescale_commit``, ``default``,
  ``defrag``: refused with :class:`FloorViolation` (HTTP 409, ``error:
  floor_violation``; the controller treats it as permanent for this tick and
  re-plans on the next one).
* ``apa``: the target is clamped so that the floor holds (no error).
* ``startup``: another replica of the model is woken first; if none can be woken the
  admission is refused with RetryLater (409, the init gate polls again).
* ``repair``: exempt (fleet repair quarantines the whole fleet), recorded.

Every refusal / clamp / exemption / make-up wake is counted (per path and model)
and logged as a JSON event.
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
CLAMP_PATHS = frozenset({"apa"})
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
    #: serve_ids routable now
    routable: tuple[str, ...]
    #: serve_ids of the operation that are routable now (what it takes out of routing)
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
    counters, so ``GET /v2/sleep`` shows them); None = in memory only."""

    incr: object = None
    _counts: dict[str, int] = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_EVENTS))
    _lock: threading.Lock = field(default_factory=threading.Lock)

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
        with self._lock:
            self._recent.appendleft(event)
        LOG.warning(json.dumps(event, sort_keys=True, default=str))
        return event

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._recent)
