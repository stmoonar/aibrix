from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tre_replayer.engine.schedule import RpsSegment


def load_trace_segments(path: str | Path) -> list[RpsSegment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("trace JSON must be a model-keyed object")

    segments: list[RpsSegment] = []
    for model, raw_segments in data.items():
        if not isinstance(raw_segments, list):
            raise ValueError(f"trace model {model!r} must contain a list of segments")
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise ValueError(f"trace model {model!r} contains a non-object segment")
            segments.append(_segment_from_mapping(str(model), raw))
    return segments


def _segment_from_mapping(model: str, raw: dict[str, Any]) -> RpsSegment:
    return RpsSegment(
        model=model,
        start_s=float(raw["start_time"]),
        end_s=float(raw["end_time"]),
        rps=float(raw["rps"]),
        input_tokens=_optional_int(raw.get("input_tokens")),
        max_output_tokens=_optional_int(raw.get("max_tokens")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
