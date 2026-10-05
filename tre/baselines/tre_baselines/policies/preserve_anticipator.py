"""PreServe instance load anticipator (§4.3.1): the per-pod load-look-ahead map.

``U[i]`` is the predicted KV usage (fraction of the pod's token capacity ``M``) ``i``
iterations from now. It is kept as a ring of ``length`` slots indexed by *absolute*
iteration ``a`` (slot ``a % length``); the head ``it`` is the current iteration, so the
live part is ``[it, it + length)`` and ``U[i]`` of the paper is slot ``(it + i) % length``.

* A request with ``P`` prompt tokens and predicted output ``D`` that finishes prefill at
  head ``s`` adds ``(P + i) / M`` at absolute iteration ``s + i`` for ``i`` in ``[0, D)``
  (the paper's ``U'_i = (U_i * M + (P + i)) / M``).
* Finishing early subtracts what is still ahead of the head (``i`` in ``[it - s, D)``).
* A request leaves the map only through :meth:`LookaheadMap.remove` (its ``done``).
  The head is an *estimate* (wall clock / mean TPOT): it may lower what is predicted
  ahead, but it never removes a request that has not finished. Reaching ``D`` without a
  ``done``:

  - ``out_len_is_upper_bound=True`` (default, ours): ``D_pred`` is the request's
    ``max_tokens``, a hard cap in vLLM, so the prediction ends at ``D`` and no virtual
    extension is needed. If the head gets there first (preemption, a pause, slower
    decoding than the mean), the request is *overdue*: it keeps its full size
    ``(P + D_pred) / M`` in every slot until its ``done`` (a floor, not a guess of when
    it ends).
  - ``out_len_is_upper_bound=False`` (the paper, section 4.3.1): extend ``D`` by
    ``ext_frac * D_pred`` (paper: 0.2), repeatedly, while the head is at or beyond the
    planned end. The paper needs this because its response-length predictor can
    under-estimate; ``max_tokens`` cannot.
* Advancing the head by ``k`` consumes (zeroes) ``k`` slots and reveals ``k`` new tail
  slots, which are filled from the requests still active. Contributions beyond the
  current tail are never written early, so a request longer than the ring (``D`` above
  ``length``, or after extensions) is still exact once its iterations come into view.

The map length and the read window ``l`` of the scaler are independent (plan §7 reviewer
fix): the scaler reads ``U[0:l]`` only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Optional

#: |value| below this after a subtraction is float noise and is snapped to 0.
_EPS = 1e-12


@dataclass
class ActiveRequest:
    P: int
    #: Predicted output length (max_tokens) - the base of every extension.
    D_pred: int
    #: Planned length including extensions.
    D: int
    #: Absolute iteration at which the request entered the map (prefill done).
    start: int
    M: float
    #: Absolute iteration up to which (exclusive) its contribution has been written.
    filled: int
    extensions: int = 0
    #: Redis-clock ms of the ``ft`` event.
    ts_ms: int = 0
    #: Upper-bound mode: the output length the value never grows past (``D_pred``);
    #: beyond it an overdue request holds ``(P + cap) / M``. None = paper mode.
    cap: Optional[int] = None

    def value(self, a: int) -> float:
        i = a - self.start
        if self.cap is not None and i > self.cap:
            i = self.cap
        return (self.P + i) / self.M

    @property
    def overdue(self) -> bool:
        return self.cap is not None and self.D > self.D_pred

    @property
    def end(self) -> int:
        return self.start + self.D


class LookaheadMap:
    def __init__(self, length: int, *, ext_frac: float = 0.2, out_len_is_upper_bound: bool = True) -> None:
        if length < 1:
            raise ValueError("look-ahead map length must be >= 1")
        self.length = int(length)
        self.ext_frac = float(ext_frac)
        self.upper_bound = bool(out_len_is_upper_bound)
        self.U = [0.0] * self.length
        self.it = 0
        self.requests: dict[Hashable, ActiveRequest] = {}

    # ---------------------------------------------------------------- mutation

    def _write(self, req: ActiveRequest, lo: int, hi: int, sign: float) -> None:
        L = self.length
        U = self.U
        for a in range(lo, hi):
            j = a % L
            v = U[j] + sign * req.value(a)
            U[j] = 0.0 if abs(v) < _EPS else v

    def _fill(self, req: ActiveRequest) -> None:
        lo = max(req.filled, self.it)
        hi = min(req.end, self.it + self.length)
        if hi > lo:
            self._write(req, lo, hi, +1.0)
            req.filled = hi
        else:
            req.filled = max(req.filled, lo)

    def add(self, key: Hashable, P: int, D: int, M: float, ts_ms: int = 0) -> ActiveRequest:
        """Prefill of ``key`` finished at the current head. Replaces an active ``key``."""
        if M <= 0:
            raise ValueError("KV capacity M must be > 0")
        if key in self.requests:
            self.remove(key)
        D = max(0, int(D))
        req = ActiveRequest(P=max(0, int(P)), D_pred=D, D=D, start=self.it, M=float(M),
                            filled=self.it, ts_ms=int(ts_ms), cap=D if self.upper_bound else None)
        self.requests[key] = req
        self._fill(req)
        return req

    def remove(self, key: Hashable) -> Optional[ActiveRequest]:
        """``key`` finished (or is dropped): subtract its part not yet walked past."""
        req = self.requests.pop(key, None)
        if req is None:
            return None
        lo = max(self.it, req.start)
        if req.filled > lo:
            self._write(req, lo, req.filled, -1.0)
        return req

    def _extend(self, req: ActiveRequest) -> None:
        step = max(1, int(math.ceil(self.ext_frac * req.D_pred)))
        while self.it >= req.end:
            req.D += step
            req.extensions += 1

    def advance(self, k: int) -> list[Hashable]:
        """Move the head ``k`` iterations forward; returns the keys that became overdue
        (upper-bound mode: the head passed ``D_pred`` before their ``done``)."""
        k = int(k)
        if k <= 0:
            return []
        if k >= self.length:
            self.U = [0.0] * self.length
        else:
            for a in range(self.it, self.it + k):
                self.U[a % self.length] = 0.0
        self.it += k
        overdue: list[Hashable] = []
        for key, req in self.requests.items():
            if self.upper_bound:
                if self.it >= req.start + req.D_pred:
                    if not req.overdue:
                        overdue.append(key)
                    # still not done: hold its full size over the whole visible ring
                    req.D = self.it + self.length - req.start
            elif self.it >= req.end:
                self._extend(req)
            self._fill(req)
        return overdue

    def total(self) -> float:
        """Sum of the whole ring (0 when no request is left)."""
        return sum(self.U)

    # ------------------------------------------------------------------- reads

    def at(self, i: int) -> float:
        """``U[i]``: predicted usage ``i`` iterations ahead (0 beyond the ring)."""
        if i < 0 or i >= self.length:
            return 0.0
        return self.U[(self.it + i) % self.length]

    def window(self, l: int) -> list[float]:
        n = max(0, min(int(l), self.length))
        return [self.U[(self.it + i) % self.length] for i in range(n)]
