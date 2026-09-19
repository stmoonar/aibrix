"""Utilisation-gated scale-down evidence (TRE_UTIL_SCALE_DOWN).

Z_m is ~ per-request token speed and barely moves with replica count or load while a
model is unsaturated, so it cannot say "one replica fewer would still be fine". This
path uses the per-replica load *after* removing one replica instead:

    q_after = Q_raw / (routable - 1),   Q_raw = avg_running + avg_waiting (unweighted)

Q_raw is deliberately the unweighted in-flight count (not the lambda_wait-weighted TRS
queue Q_ctl): the per-model defaults were derived from R3 single-replica sweeps binned on
exactly avg_running + avg_waiting (see deploy/registry.yaml scale_down_q_per_replica).

HIGH models are eligible on purpose (HIGH is the common state of a lightly loaded
model); the non-rising-load condition is the spike guard. TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE
governs only the Z_m-driven high_proactive_safescale path, NOT this one: ablations that
must not shrink receiver-less have to set TRE_UTIL_SCALE_DOWN explicitly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from tre_common.registry import ModelSpec

DEFAULT_Q_PER_REPLICA = 2.0
DEFAULT_WINDOWS = 6


@dataclass(frozen=True)
class UtilWindow:
    window_end_ms: int
    q_raw: float
    routable: int


class UtilScaleDown:
    """Shared across the rescue and fairness loops: one entry per distinct metrics window."""

    def __init__(
        self,
        *,
        windows: int = DEFAULT_WINDOWS,
        q_overrides: Mapping[str, float] | None = None,
        max_history: int = 64,
    ) -> None:
        self.windows = max(1, int(windows))
        self.q_overrides = dict(q_overrides or {})
        self._max_history = max(self.windows, int(max_history))
        self._history: dict[str, deque[UtilWindow]] = {}

    def threshold_for(self, spec: ModelSpec) -> float:
        # Precedence: env per-model > env global ("*") > registry model key > default.
        if spec.name in self.q_overrides:
            return self.q_overrides[spec.name]
        if "*" in self.q_overrides:
            return self.q_overrides["*"]
        if spec.scale_down_q_per_replica is not None:
            return spec.scale_down_q_per_replica
        return DEFAULT_Q_PER_REPLICA

    def observe(self, model: str, *, window_end_ms: int, q_raw: float, routable: int) -> None:
        history = self._history.setdefault(model, deque(maxlen=self._max_history))
        if history and int(window_end_ms) <= history[-1].window_end_ms:
            return  # same (or older) window already recorded by the other loop
        history.append(UtilWindow(int(window_end_ms), max(0.0, float(q_raw)), int(routable)))

    def history(self) -> dict[str, tuple[UtilWindow, ...]]:
        return {model: tuple(values) for model, values in self._history.items()}


def util_scale_down_ready(
    history: tuple[UtilWindow, ...],
    *,
    routable: int,
    threshold: float,
    windows: int,
) -> float | None:
    """q_after of the latest window when the last `windows` distinct windows all had the
    current replica count, q_after <= threshold, and load is not rising (latest Q_raw <=
    mean of those windows). None otherwise."""
    if routable < 2 or windows <= 0 or len(history) < windows:
        return None
    recent = history[-windows:]
    if any(item.routable != routable for item in recent):
        return None
    remaining = routable - 1
    if any(item.q_raw / remaining > threshold for item in recent):
        return None
    mean_q = sum(item.q_raw for item in recent) / len(recent)
    if recent[-1].q_raw > mean_q + 1e-9:
        return None
    return recent[-1].q_raw / remaining


def parse_q_per_replica(raw: str | None) -> dict[str, float]:
    """TRE_UTIL_SCALE_DOWN_Q_PER_REPLICA: "2.5" (all models) or "model=v,model=v" ("*=v" ok)."""
    text = (raw or "").strip()
    if not text:
        return {}
    overrides: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        model, sep, value = part.rpartition("=")
        key = model.strip() if sep else "*"
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"TRE_UTIL_SCALE_DOWN_Q_PER_REPLICA: invalid value {part!r}") from exc
        if not key or not parsed > 0.0 or parsed == float("inf"):
            raise ValueError(f"TRE_UTIL_SCALE_DOWN_Q_PER_REPLICA: invalid entry {part!r}")
        overrides[key] = parsed
    return overrides
