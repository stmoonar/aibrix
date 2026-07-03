from __future__ import annotations

from collections.abc import Iterable

Bucket = tuple[float, float]


def histogram_percentile(
    cumulative_buckets: Iterable[Bucket],
    quantile: float,
    *,
    mode: str = "interpolated",
) -> float | None:
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be between 0 and 1")
    buckets = sorted((float(upper), float(count)) for upper, count in cumulative_buckets)
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0.0:
        return None
    target = quantile * total
    previous_upper = 0.0
    previous_count = 0.0
    for upper, count in buckets:
        if count >= target:
            if mode == "bucket_upper":
                return upper
            if mode != "interpolated":
                raise ValueError(f"unsupported percentile mode: {mode}")
            bucket_count = count - previous_count
            if bucket_count <= 0.0:
                return upper
            fraction = (target - previous_count) / bucket_count
            return previous_upper + (upper - previous_upper) * fraction
        previous_upper = upper
        previous_count = count
    return buckets[-1][0]
