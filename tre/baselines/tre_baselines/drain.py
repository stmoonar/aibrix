"""Drain before sleep, for a policy that isolates instances on scale-in (PreServe §4.3.2:
"isolating ... allowing their ongoing requests to complete naturally").

Without it every baseline shrink sleeps on the shared abort path: the requests in flight
are cut and continued by the reissue sidecar (re-prefill). With it (policy attribute
``drain_on_shrink``) the shell turns the policy's desired count into these steps, one call
per model per tick, against the **routable** count (awake minus hidden):

1. ``desired < routable``: hide the ``routable - desired`` pods with the least work
   (``running + waiting``, then KV usage, then name; scraped pods only). The SM refuses a
   hide below the replica floor. A hidden pod gets no new requests and is not in
   ``ModelSnapshot.pods``, so the policy no longer counts it.
2. A hidden pod is drained once a scrape at least ``ROUTE_MARGIN_MS`` after the tick that
   first saw it hidden shows ``running + waiting == 0`` (a request routed just before the
   hide may reach the engine late); then it is slept by name (``down`` with ``pods``). Its
   sleep still goes through the abort path: a straggler is continued by the sidecar.
   Bound: a pod hidden for ``max_s`` (default the route timeout, 150 s: a request still
   running then is cut by the gateway anyway) is slept as it is (``timeout``, counted).
3. ``desired > routable``: draining pods come back first (``unhide``, most recently hidden
   first; their KV cache is still warm), a wake only for the rest (``up`` with
   ``unhide_first``: all draining pods are unhidden, then ``at_least``).

Only pods this shell hid are drained; at its first owner tick it adopts the awake hidden
pods of its models (a restarted shell resumes their drains). Another actor's hidden pod
is never unhidden or slept, but counts as not routable. :meth:`DrainTracker.release`
lists the pods to unhide when the shell stops.

The decision is made on the tick's snapshot only; nothing here calls the SM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tre_baselines.snapshot import ModelSnapshot

#: A scrape this long after the tick that first saw the hide shows the engine's own view
#: after any request routed before the hide reached it (as PreServe ``ENGINE_IDLE_MARGIN_MS``).
ROUTE_MARGIN_MS = 1000
#: Default bound of one drain (s): the gateway route timeout.
DEFAULT_DRAIN_MAX_S = 150.0


@dataclass(frozen=True)
class DrainStep:
    """The call of one model on one tick (``none`` = no call)."""

    direction: str
    target: int
    pods: tuple[str, ...] = ()
    unhide_first: tuple[str, ...] = ()
    info: dict = field(default_factory=dict)


@dataclass
class _Pod:
    #: Redis-clock ms of the first tick that saw it hidden (None = hide requested, not seen).
    hidden_ms: Optional[int]


class DrainTracker:
    def __init__(self, max_s: float = DEFAULT_DRAIN_MAX_S) -> None:
        self.max_ms = int(round(float(max_s) * 1000.0))
        self._ours: dict[str, dict[str, _Pod]] = {}
        self._adopted = False
        #: Per model: drains ended by a drained engine / by the bound / by an unhide.
        self.counts: dict[str, dict[str, int]] = {}

    def _count(self, model: str, key: str, n: int = 1) -> None:
        c = self.counts.setdefault(model, {"drained": 0, "timeout": 0, "unhidden": 0, "adopted": 0})
        c[key] += n

    def draining(self, model: str) -> dict[str, Optional[int]]:
        return {pod: p.hidden_ms for pod, p in self._ours.get(model, {}).items()}

    def reconcile(self, ms: ModelSnapshot, now_ms: int, *, inflight: bool, adopt: bool) -> None:
        """Track this tick's SM view: a pod of ours no longer awake + hidden is done (slept or
        unhidden); a requested hide not seen yet is kept while the model's call is in flight."""
        hidden = {p.pod for p in ms.hidden} | set(ms.hidden_unscraped)
        ours = self._ours.setdefault(ms.model, {})
        if adopt and not self._adopted:
            for pod in sorted(hidden - set(ours)):
                ours[pod] = _Pod(hidden_ms=int(now_ms))
                self._count(ms.model, "adopted")
        for pod in list(ours):
            entry = ours[pod]
            if pod in hidden:
                if entry.hidden_ms is None:
                    entry.hidden_ms = int(now_ms)
            elif entry.hidden_ms is not None or not inflight:
                del ours[pod]

    def adopted(self) -> None:
        """Adoption is done once (the first owner tick)."""
        self._adopted = True

    def step(self, ms: ModelSnapshot, desired: int, now_ms: int) -> DrainStep:
        """The call that moves ``ms`` toward ``desired`` routable replicas."""
        hidden = {p.pod for p in ms.hidden} | set(ms.hidden_unscraped)
        ours = self._ours.get(ms.model, {})
        draining = {pod: p for pod, p in ours.items() if pod in hidden and p.hidden_ms is not None}
        routable = ms.routable_awake
        scraped = {p.pod: p for p in ms.hidden}
        info: dict[str, Any] = {"routable": routable, "draining": {
            pod: {"age_s": round((int(now_ms) - p.hidden_ms) / 1000.0, 1),
                  "queued": None if pod not in scraped else scraped[pod].queued}
            for pod, p in sorted(draining.items())}}
        if desired > routable:
            back = sorted(draining, key=lambda pod: (-draining[pod].hidden_ms, pod))
            need = desired - routable
            if need <= len(back):
                pods = tuple(back[:need])
                info["unhide"] = list(pods)
                return DrainStep("unhide", routable + len(pods), pods=pods, info=info)
            # Every draining pod back, then a wake for the rest. ``at_least`` counts awake
            # pods, hidden ones included: the pods still hidden after the unhide (another
            # actor's) are added.
            target = desired + len(hidden) - len(back)
            info["unhide"] = back
            return DrainStep("up", target, unhide_first=tuple(back), info=info)
        if desired < routable:
            cands = sorted(ms.pods, key=lambda p: (p.queued is None, p.queued or 0.0,
                                                   p.kv_usage if p.kv_usage is not None else 1.0, p.pod))
            pods = tuple(p.pod for p in cands[:routable - desired])
            if not pods:
                info["hold"] = "no_scraped_pod"
                return DrainStep("none", routable, info=info)
            info["hide"] = list(pods)
            return DrainStep("hide", routable - len(pods), pods=pods, info=info)
        done: list[str] = []
        for pod, p in sorted(draining.items()):
            snap = scraped.get(pod)
            if snap is not None and snap.queued == 0 and snap.scraped_at_ms >= p.hidden_ms + ROUTE_MARGIN_MS:
                done.append(pod)
                info.setdefault("drained", []).append(pod)
            elif int(now_ms) - p.hidden_ms >= self.max_ms:
                done.append(pod)
                info.setdefault("timeout", []).append(pod)
        if done:
            return DrainStep("down", int(ms.awake) - len(done), pods=tuple(done), info=info)
        return DrainStep("none", routable, info=info)

    def sent(self, model: str, step: DrainStep, now_ms: int) -> None:
        """A step was handed to the SM: a hide makes its pods ours; counters."""
        if step.direction == "hide":
            ours = self._ours.setdefault(model, {})
            for pod in step.pods:
                ours[pod] = _Pod(hidden_ms=None)
        elif step.direction == "down":
            self._count(model, "drained", len(step.info.get("drained", ())))
            self._count(model, "timeout", len(step.info.get("timeout", ())))
        elif step.direction == "unhide" or step.unhide_first:
            self._count(model, "unhidden", len(step.info.get("unhide", ())))

    def release(self) -> dict[str, list[str]]:
        """Per model, the pods this shell hid (to unhide when it stops)."""
        return {m: sorted(pods) for m, pods in self._ours.items() if pods}
