from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacitySample:
    model: str
    input_tokens: int
    output_tokens: int
    rps: float
    slo_met: bool


@dataclass(frozen=True)
class CapacityPoint:
    model: str
    input_tokens: int
    output_tokens: int
    rps: float
    low_confidence: bool
    reason: str


@dataclass(frozen=True)
class CapacitySurface:
    points: dict[tuple[str, int, int], float]

    def capacity_at(self, model: str, *, input_tokens: int, output_tokens: int) -> CapacityPoint:
        key = (model, input_tokens, output_tokens)
        if key in self.points:
            return CapacityPoint(model, input_tokens, output_tokens, self.points[key], False, "exact")

        candidates = [(candidate_key, rps) for candidate_key, rps in self.points.items() if candidate_key[0] == model]
        if not candidates:
            raise KeyError(f"no capacity data for model: {model}")
        nearest_key, nearest_rps = min(
            candidates,
            key=lambda item: abs(item[0][1] - input_tokens) + abs(item[0][2] - output_tokens),
        )
        return CapacityPoint(
            model=model,
            input_tokens=nearest_key[1],
            output_tokens=nearest_key[2],
            rps=nearest_rps,
            low_confidence=True,
            reason="nearest_extrapolated",
        )


def fit_capacity_surface(samples: list[CapacitySample]) -> CapacitySurface:
    points: dict[tuple[str, int, int], float] = {}
    for sample in samples:
        if not sample.slo_met:
            continue
        key = (sample.model, sample.input_tokens, sample.output_tokens)
        points[key] = max(points.get(key, 0.0), sample.rps)
    return CapacitySurface(points=points)
