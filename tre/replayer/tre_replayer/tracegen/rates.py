"""Time functions for rates and length means.

A *function spec* is a number (flat), a dict with ``kind``, or a list (the sum of its
items). Kinds (times in seconds from trace start; every edge is a linear ramp of
``ramp_s``, default 3 s, that *starts* at the edge):

* ``flat``      ``value``
* ``sinusoid``  ``mean + amp * sin(2 pi (t / period_s) + phase_deg)``
* ``square``    ``low`` outside, ``high`` inside the window ``[offset_s + k period_s,
                + high_s)`` for every k (the up-ramp lies inside the window, the down-ramp
                after it)
* ``pulses``    ``base``, plus pulse i lifting it to ``peaks[i]`` (or ``peak``) over
                ``[starts_s[i], starts_s[i] + width_s)``
* ``steps``     ``points`` = [[t, v], ...]: value v from t on (ramp from the previous value)
* ``piecewise`` ``points`` = [[t, v], ...]: linear interpolation, ends held
* ``ou``        ``level * (1 + x(t))``, x an Ornstein-Uhlenbeck process (mean 0, stationary
                sd ``rel_sd``, time constant ``tau_s``, sampled every ``dt_s`` = 1 s and
                linearly interpolated, clipped to ``+-clip_rel``); seeded from the trace seed.
"""
from __future__ import annotations

import bisect
import math
import random
from typing import Any, Callable

DEFAULT_RAMP_S = 3.0
Func = Callable[[float], float]


def _trapezoid(t: float, start: float, width: float, ramp: float) -> float:
    """0..1: up-ramp over [start, start+ramp], 1 until start+width, down-ramp after."""
    if t < start:
        return 0.0
    if ramp > 0 and t < start + ramp:
        up = (t - start) / ramp
    else:
        up = 1.0
    end = start + width
    if t < end:
        return up
    if ramp <= 0 or t >= end + ramp:
        return 0.0
    return min(up, 1.0 - (t - end) / ramp)


def _steps(points, ramp):
    ts = [float(p[0]) for p in points]
    vs = [float(p[1]) for p in points]

    def f(t: float) -> float:
        i = bisect.bisect_right(ts, t) - 1
        if i < 0:
            return vs[0]
        if i == 0 or ramp <= 0 or t >= ts[i] + ramp:
            return vs[i]
        return vs[i - 1] + (vs[i] - vs[i - 1]) * (t - ts[i]) / ramp
    return f


def _piecewise(points):
    ts = [float(p[0]) for p in points]
    vs = [float(p[1]) for p in points]

    def f(t: float) -> float:
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        i = bisect.bisect_right(ts, t) - 1
        return vs[i] + (vs[i + 1] - vs[i]) * (t - ts[i]) / (ts[i + 1] - ts[i])
    return f


def _ou(spec: dict, duration_s: float, rng: random.Random) -> Func:
    level = float(spec["level"])
    sd = float(spec["rel_sd"])
    tau = float(spec["tau_s"])
    dt = float(spec.get("dt_s", 1.0))
    clip = float(spec.get("clip_rel", 2 * sd))
    a = math.exp(-dt / tau)
    b = sd * math.sqrt(1 - a * a)
    x = rng.gauss(0.0, sd)
    n = int(math.ceil(duration_s / dt)) + 2
    xs = []
    for _ in range(n):
        xs.append(max(-clip, min(clip, x)))
        x = a * x + b * rng.gauss(0.0, 1.0)
    pts = [[i * dt, level * (1 + v)] for i, v in enumerate(xs)]
    return _piecewise(pts)


def build(spec: Any, *, duration_s: float, rng: random.Random | None = None) -> Func:
    """A callable ``f(t)`` for ``spec`` (see the module docstring)."""
    if isinstance(spec, (int, float)):
        v = float(spec)
        return lambda t: v
    if isinstance(spec, list):
        parts = [build(s, duration_s=duration_s, rng=rng) for s in spec]
        return lambda t: sum(p(t) for p in parts)
    kind = spec["kind"]
    ramp = float(spec.get("ramp_s", DEFAULT_RAMP_S))
    if kind == "flat":
        v = float(spec["value"])
        return lambda t: v
    if kind == "sinusoid":
        mean, amp, period = float(spec["mean"]), float(spec["amp"]), float(spec["period_s"])
        ph = math.radians(float(spec.get("phase_deg", 0.0)))
        return lambda t: mean + amp * math.sin(2 * math.pi * t / period + ph)
    if kind == "square":
        low, high = float(spec["low"]), float(spec["high"])
        period, width = float(spec["period_s"]), float(spec["high_s"])
        offset = float(spec.get("offset_s", 0.0))

        def sq(t: float) -> float:
            k = math.floor((t - offset) / period)
            # the current window and the previous one (its down-ramp may still be running)
            w = max(_trapezoid(t, offset + j * period, width, ramp) for j in (k - 1, k))
            return low + (high - low) * w
        return sq
    if kind == "pulses":
        base = float(spec["base"])
        starts = [float(s) for s in spec["starts_s"]]
        width = float(spec["width_s"])
        peaks = [float(p) for p in spec.get("peaks", [spec.get("peak")] * len(starts))]

        def pu(t: float) -> float:
            v = base
            for s, p in zip(starts, peaks):
                w = _trapezoid(t, s, width, ramp)
                if w:
                    v = max(v, base + (p - base) * w)
            return v
        return pu
    if kind == "steps":
        return _steps(spec["points"], ramp)
    if kind == "piecewise":
        return _piecewise(spec["points"])
    if kind == "ou":
        if rng is None:
            raise ValueError("an 'ou' function needs a seeded rng")
        return _ou(spec, duration_s, rng)
    raise ValueError(f"unknown function kind {kind!r}")


def sample(f: Func, duration_s: float, dt: float = 1.0) -> list[float]:
    """f at the centre of every ``dt`` step of [0, duration_s)."""
    n = int(round(duration_s / dt))
    return [f((i + 0.5) * dt) for i in range(n)]
