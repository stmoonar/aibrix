#!/usr/bin/env python3
"""Derive deterministic traceset-v2 workloads from public production traces.

The source files are never committed. Outputs use the model-keyed trace.json schema
consumed by tre_replayer. See docs/refactor/p11_evidence/real_traces_20260713/README.md
for provenance and the rationale for the locked parameters below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterator

AZURE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv"
)
AZURE_SHA256 = "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941"
BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
BURSTGPT_SHA256 = "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
DOWNLOAD_DATE = "2026-07-10"
UNIX_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()

TARGET_DURATION_S = 1120
BIN_WIDTH_S = 5
TARGET_PEAK_RPS = 29.426667
SEED = 20260713
MAX_INPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 2048
MODEL_WEIGHTS = (
    ("dsllama-8b", 0.40),
    ("dsqwen-7b", 0.35),
    ("dsqwen-14b", 0.25),
)
MODEL_ORDER = tuple(model for model, _ in MODEL_WEIGHTS)


@dataclass(frozen=True)
class RawRequest:
    timestamp: float
    assignment_key: str
    input_tokens: int
    output_tokens: int


@dataclass
class Bucket:
    count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def stable_model(assignment_key: str, *, seed: int = SEED) -> str:
    """Map one conversation/session key to a model with fixed weighted hashing."""
    digest = hashlib.sha256(f"{seed}:{assignment_key}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for model, weight in MODEL_WEIGHTS:
        cumulative += weight
        if unit < cumulative:
            return model
    return MODEL_WEIGHTS[-1][0]


def _azure_timestamp(value: str, day_cache: dict[str, int]) -> float:
    day_text = value[:10]
    day_start = day_cache.get(day_text)
    if day_start is None:
        day_start = (date.fromisoformat(day_text).toordinal() - UNIX_EPOCH_ORDINAL) * 86400
        day_cache[day_text] = day_start
    seconds_text = value[17:]
    timezone_at = min(
        (index for marker in ("+", "-") if (index := seconds_text.find(marker, 2)) >= 0),
        default=len(seconds_text),
    )
    return day_start + int(value[11:13]) * 3600 + int(value[14:16]) * 60 + float(seconds_text[:timezone_at])


def iter_azure(path: Path) -> Iterator[RawRequest]:
    """Read Azure's conversation subset; it has no conversation/session identifier."""
    day_cache: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        indexes = _column_indexes(next(reader), {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}, path)
        for row_number, row in enumerate(reader, start=2):
            timestamp_text = row[indexes["TIMESTAMP"]].strip()
            input_tokens = _positive_int(row[indexes["ContextTokens"]])
            output_tokens = _positive_int(row[indexes["GeneratedTokens"]])
            if input_tokens is None or output_tokens is None:
                continue
            timestamp = _azure_timestamp(timestamp_text, day_cache)
            # The published Azure conversation schema has no session ID. This stable
            # request identity is the explicit fallback; it is not claimed to recover sessions.
            key = f"azure-request:{timestamp_text}:{row_number}"
            yield RawRequest(timestamp, key, input_tokens, output_tokens)


def iter_burstgpt(path: Path) -> Iterator[RawRequest]:
    """Read successful BurstGPT v2 conversation requests with their real Session ID."""
    required = {"Timestamp", "Session ID", "Request tokens", "Response tokens", "Log Type"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        indexes = _column_indexes(next(reader), required, path)
        for row in reader:
            if not _burstgpt_selected(row, indexes):
                continue
            session_id = row[indexes["Session ID"]].strip()
            input_tokens = _positive_int(row[indexes["Request tokens"]])
            output_tokens = _positive_int(row[indexes["Response tokens"]])
            if not session_id or input_tokens is None or output_tokens is None:
                continue
            yield RawRequest(
                float(row[indexes["Timestamp"]]),
                f"burstgpt-session:{session_id}",
                input_tokens,
                output_tokens,
            )


def _burstgpt_selected(row: list[str], indexes: dict[str, int]) -> bool:
    return (
        row[indexes["Log Type"]].strip() == "Conversation log"
        and bool(row[indexes["Session ID"]].strip())
        and _positive_int(row[indexes["Request tokens"]]) is not None
        and _positive_int(row[indexes["Response tokens"]]) is not None
    )


def _reverse_lines(path: Path, block_size: int = 64 * 1024) -> Iterator[str]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        remainder = b""
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            parts = (handle.read(read_size) + remainder).split(b"\n")
            remainder = parts[0]
            for line in reversed(parts[1:]):
                if line.strip():
                    yield line.decode("utf-8")
        if remainder.strip():
            yield remainder.decode("utf-8")


def source_time_bounds(
    path: Path,
    timestamp_column: str,
    parser: Callable[[str], float],
    *,
    required_columns: set[str] | None = None,
    selected: Callable[[list[str], dict[str, int]], bool] | None = None,
) -> tuple[float, float]:
    """Read first/last selected rows without scanning a sorted production trace."""
    required = set(required_columns or ()) | {timestamp_column}
    accept = selected or (lambda _row, _indexes: True)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indexes = _column_indexes(header, required, path)
        first_row = next(row for row in reader if accept(row, indexes))
    for line in _reverse_lines(path):
        row = next(csv.reader([line]))
        if row != header and accept(row, indexes):
            last_row = row
            break
    else:
        raise ValueError(f"{path} contains no selected source rows")
    return parser(first_row[indexes[timestamp_column]]), parser(last_row[indexes[timestamp_column]])


def derive_trace(
    source_factory: Callable[[], Iterator[RawRequest]],
    output_path: Path,
    *,
    target_duration_s: int = TARGET_DURATION_S,
    bin_width_s: int = BIN_WIDTH_S,
    target_peak_rps: float = TARGET_PEAK_RPS,
    seed: int = SEED,
    source_bounds: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Compress, bin, model-map and peak-rescale one request stream."""
    if target_duration_s <= 0 or bin_width_s <= 0 or target_duration_s % bin_width_s:
        raise ValueError("target duration must be positive and divisible by bin width")
    if target_peak_rps <= 0:
        raise ValueError("target peak RPS must be positive")

    source_rows = 0
    if source_bounds is None:
        min_timestamp = math.inf
        max_timestamp = -math.inf
        for request in source_factory():
            source_rows += 1
            min_timestamp = min(min_timestamp, request.timestamp)
            max_timestamp = max(max_timestamp, request.timestamp)
    else:
        min_timestamp, max_timestamp = source_bounds
    if not math.isfinite(min_timestamp) or max_timestamp <= min_timestamp:
        raise ValueError("source must span at least two distinct timestamps")
    source_rows = 0

    bin_count = target_duration_s // bin_width_s
    buckets: dict[tuple[int, str], Bucket] = {}
    aggregate_counts = [0] * bin_count
    model_request_counts = {model: 0 for model in MODEL_ORDER}
    input_clamped = 0
    output_clamped = 0
    source_span = max_timestamp - min_timestamp
    previous_timestamp = -math.inf

    for request in source_factory():
        if request.timestamp < previous_timestamp:
            raise ValueError("source requests must be ordered by timestamp")
        previous_timestamp = request.timestamp
        source_rows += 1
        compressed = (request.timestamp - min_timestamp) / source_span * target_duration_s
        bin_index = min(bin_count - 1, max(0, int(compressed // bin_width_s)))
        model = stable_model(request.assignment_key, seed=seed)
        input_tokens = min(request.input_tokens, MAX_INPUT_TOKENS)
        output_tokens = min(request.output_tokens, MAX_OUTPUT_TOKENS)
        input_clamped += int(input_tokens != request.input_tokens)
        output_clamped += int(output_tokens != request.output_tokens)
        bucket = buckets.setdefault((bin_index, model), Bucket())
        bucket.count += 1
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        aggregate_counts[bin_index] += 1
        model_request_counts[model] += 1

    if source_rows == 0:
        raise ValueError("source contains no selected requests")
    raw_peak_rps = max(aggregate_counts) / bin_width_s
    rate_scale = target_peak_rps / raw_peak_rps
    trace: dict[str, list[dict[str, int | float]]] = {model: [] for model in MODEL_ORDER}
    for model in MODEL_ORDER:
        for bin_index in range(bin_count):
            bucket = buckets.get((bin_index, model))
            if bucket is None:
                continue
            trace[model].append({
                "start_time": bin_index * bin_width_s,
                "end_time": (bin_index + 1) * bin_width_s,
                "rps": round(bucket.count / bin_width_s * rate_scale, 6),
                "input_tokens": max(1, round(bucket.input_tokens / bucket.count)),
                "max_tokens": max(1, round(bucket.output_tokens / bucket.count)),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    peak_after_rounding = max(
        sum(
            next((segment["rps"] for segment in trace[model] if segment["start_time"] == start), 0.0)
            for model in MODEL_ORDER
        )
        for start in range(0, target_duration_s, bin_width_s)
    )
    return {
        "valid_source_rows": source_rows,
        "source_start": min_timestamp,
        "source_end": max_timestamp,
        "source_span_s": max_timestamp - min_timestamp,
        "target_duration_s": target_duration_s,
        "bin_width_s": bin_width_s,
        "raw_peak_rps_after_time_compression": raw_peak_rps,
        "rate_scale": rate_scale,
        "derived_peak_rps": peak_after_rounding,
        "model_request_counts": model_request_counts,
        "input_rows_clamped": input_clamped,
        "output_rows_clamped": output_clamped,
        "output_sha256": sha256_file(output_path),
    }


def derive_both(
    azure_source: Path,
    burstgpt_source: Path,
    trace_root: Path,
    evidence_dir: Path,
    *,
    target_duration_s: int = TARGET_DURATION_S,
    bin_width_s: int = BIN_WIDTH_S,
    target_peak_rps: float = TARGET_PEAK_RPS,
    seed: int = SEED,
) -> dict[str, object]:
    outputs = {
        "t8_azure_conv": trace_root / "t8_azure_conv" / "trace.json",
        "t9_burstgpt": trace_root / "t9_burstgpt" / "trace.json",
    }
    reports = {
        "t8_azure_conv": derive_trace(
            lambda: iter_azure(azure_source), outputs["t8_azure_conv"],
            target_duration_s=target_duration_s, bin_width_s=bin_width_s,
            target_peak_rps=target_peak_rps, seed=seed,
            source_bounds=source_time_bounds(
                azure_source, "TIMESTAMP", lambda value: _azure_timestamp(value, {}),
            ),
        ),
        "t9_burstgpt": derive_trace(
            lambda: iter_burstgpt(burstgpt_source), outputs["t9_burstgpt"],
            target_duration_s=target_duration_s, bin_width_s=bin_width_s,
            target_peak_rps=target_peak_rps, seed=seed,
            source_bounds=source_time_bounds(
                burstgpt_source,
                "Timestamp",
                float,
                required_columns={
                    "Session ID", "Request tokens", "Response tokens", "Log Type",
                },
                selected=_burstgpt_selected,
            ),
        ),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, output in outputs.items():
        shutil.copyfile(output, evidence_dir / f"{name}.trace.json")

    manifest: dict[str, object] = {
        "sources": {
            "azure": {"url": AZURE_URL, "sha256": AZURE_SHA256, "download_date": DOWNLOAD_DATE},
            "burstgpt": {"url": BURSTGPT_URL, "sha256": BURSTGPT_SHA256, "download_date": DOWNLOAD_DATE},
        },
        "parameters": {
            "target_duration_s": target_duration_s,
            "duration_basis": "maximum observed traceset-v2 t1-t7 duration (t2/t7)",
            "bin_width_s": bin_width_s,
            "target_peak_rps": target_peak_rps,
            "peak_basis": "observed aggregate peak of traceset-v2 t4_a4_spike_vs_burst",
            "seed": seed,
            "model_weights": dict(MODEL_WEIGHTS),
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "token_aggregation": "arithmetic mean per compressed time bin and assigned model",
        },
        "reports": reports,
    }
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _column_indexes(header: list[str], required: set[str], path: Path) -> dict[str, int]:
    indexes = {name: index for index, name in enumerate(header)}
    missing = required.difference(indexes)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return {name: indexes[name] for name in required}


def _verify_source(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"source hash mismatch for {path}: expected {expected_sha256}, got {actual}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azure-source", type=Path, required=True)
    parser.add_argument("--burstgpt-source", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--target-duration-s", type=int, default=TARGET_DURATION_S)
    parser.add_argument("--bin-width-s", type=int, default=BIN_WIDTH_S)
    parser.add_argument("--target-peak-rps", type=float, default=TARGET_PEAK_RPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-source-hash-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if not args.skip_source_hash_check:
        _verify_source(args.azure_source, AZURE_SHA256)
        _verify_source(args.burstgpt_source, BURSTGPT_SHA256)
    manifest = derive_both(
        args.azure_source, args.burstgpt_source, args.trace_root, args.evidence_dir,
        target_duration_s=args.target_duration_s, bin_width_s=args.bin_width_s,
        target_peak_rps=args.target_peak_rps, seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())