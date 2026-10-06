"""Per-request token lengths: truncated lognormal with a fitted shape and a target mean.

``sigma`` (the shape) comes from an Azure fit (``data/azure_fits.json``, e.g.
``conv2023:in``). For a target mean ``m`` the location ``mu`` is solved so that the mean of
the *truncated* distribution is ``m``; the upper truncation is the ``trunc_q`` quantile
(default p99.5) of the untruncated lognormal with that same ``mu`` (so the cut moves with
the mean), lowered to ``max_tokens`` when that is smaller (the route-timeout cap,
``capacity.route_out_max``; the mean is still the target, the tail is cut), the lower one
is ``min_tokens``. Draws are by inverse CDF in log space and
rounded to whole tokens. ``dist: fixed`` gives every request ``round(m)``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()
DEFAULT_TRUNC_Q = 0.995


@dataclass(frozen=True)
class TruncLogNormal:
    mu: float
    sigma: float
    lo: float
    hi: float

    @property
    def _bounds(self):
        a = (math.log(self.lo) - self.mu) / self.sigma
        b = (math.log(self.hi) - self.mu) / self.sigma
        return a, b

    @property
    def mean(self) -> float:
        a, b = self._bounds
        z = _N.cdf(b) - _N.cdf(a)
        return math.exp(self.mu + self.sigma ** 2 / 2) * (_N.cdf(b - self.sigma) - _N.cdf(a - self.sigma)) / z

    def draw(self, rng: random.Random) -> int:
        a, b = self._bounds
        ua, ub = _N.cdf(a), _N.cdf(b)
        u = ua + (ub - ua) * rng.random()
        u = min(max(u, 1e-15), 1 - 1e-15)
        x = math.exp(self.mu + self.sigma * _N.inv_cdf(u))
        return int(min(self.hi, max(self.lo, round(x))))


def solve(mean: float, sigma: float, *, min_tokens: float = 1.0, trunc_q: float = DEFAULT_TRUNC_Q,
          max_tokens: float | None = None) -> TruncLogNormal:
    """The truncated lognormal of shape ``sigma`` whose mean is ``mean``."""
    zq = _N.inv_cdf(trunc_q)

    def make(mu: float) -> TruncLogNormal:
        hi = math.exp(mu + zq * sigma)
        if max_tokens is not None:
            hi = min(hi, max_tokens)
        return TruncLogNormal(mu, sigma, float(min_tokens), max(hi, float(min_tokens) + 1))

    lo_mu, hi_mu = math.log(max(mean, 1.0)) - 4 * sigma - 2, math.log(max(mean, 1.0)) + 4 * sigma + 2
    for _ in range(100):
        mid = (lo_mu + hi_mu) / 2
        if make(mid).mean < mean:
            lo_mu = mid
        else:
            hi_mu = mid
    d = make((lo_mu + hi_mu) / 2)
    if abs(d.mean - mean) > max(0.5, 1e-3 * mean):
        raise ValueError(f"no truncated lognormal (sigma {sigma}, max {max_tokens}) has mean {mean}")
    return d


class LengthModel:
    """Lengths for one model and one direction (in or out) from a spec dict.

    spec: ``{"dist": "lognormal", "sigma_from": "conv2023:in" | "sigma": 0.98,
    "mean": <function spec>, "min": 1, "max": null, "trunc_q": 0.995}`` or
    ``{"dist": "fixed", "mean": <function spec>}``."""

    def __init__(self, spec: dict, fits: dict, mean_fn):
        self.spec = spec
        self.dist = spec.get("dist", "lognormal")
        self.mean_fn = mean_fn
        self.min = float(spec.get("min", 1))
        self.max = spec.get("max")
        self.trunc_q = float(spec.get("trunc_q", DEFAULT_TRUNC_Q))
        if self.dist == "lognormal":
            if "sigma" in spec:
                self.sigma = float(spec["sigma"])
            else:
                name, side = spec["sigma_from"].split(":")
                self.sigma = float(fits[name][side]["sigma"])
        elif self.dist != "fixed":
            raise ValueError(f"unknown length dist {self.dist!r}")
        self._cache: dict = {}

    def at(self, t: float):
        m = self.mean_fn(t)
        if self.dist == "fixed":
            return None, m
        key = round(m * 2) / 2  # 0.5-token resolution of the target mean
        d = self._cache.get(key)
        if d is None:
            d = solve(key, self.sigma, min_tokens=self.min, trunc_q=self.trunc_q, max_tokens=self.max)
            self._cache[key] = d
        return d, d.mean

    def mean(self, t: float) -> float:
        return self.at(t)[1]

    def draw(self, t: float, rng: random.Random) -> int:
        d, m = self.at(t)
        if d is None:
            return int(round(m))
        return d.draw(rng)
