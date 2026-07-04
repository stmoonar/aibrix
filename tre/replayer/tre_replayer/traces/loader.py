from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tre_replayer.engine.schedule import RpsSegment


@dataclass(frozen=True)
class TraceCase:
    name: str
    path: Path
    indexed: bool
    segments: list[RpsSegment]


@dataclass(frozen=True)
class TraceSet:
    root: Path
    version: str | None
    cases: list[TraceCase]


def discover_trace_set(root: str | Path) -> TraceSet:
    trace_root = Path(root)
    index_path = trace_root / "INDEX.json"
    version: str | None = None
    indexed_names: list[str] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        version = str(index.get("version")) if index.get("version") is not None else None
        indexed_names = [str(name) for name in index.get("workloads", [])]

    case_by_name: dict[str, TraceCase] = {}
    for name in indexed_names:
        trace_path = trace_root / name / "trace.json"
        if trace_path.exists():
            case_by_name[name] = TraceCase(
                name=name,
                path=trace_path,
                indexed=True,
                segments=load_trace_segments(trace_path),
            )

    for trace_path in sorted(trace_root.glob("*/trace.json")):
        name = trace_path.parent.name
        if name in case_by_name:
            continue
        case_by_name[name] = TraceCase(
            name=name,
            path=trace_path,
            indexed=False,
            segments=load_trace_segments(trace_path),
        )

    ordered_cases = [case_by_name[name] for name in indexed_names if name in case_by_name]
    ordered_cases.extend(case_by_name[name] for name in sorted(case_by_name) if name not in indexed_names)
    return TraceSet(root=trace_root, version=version, cases=ordered_cases)


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
