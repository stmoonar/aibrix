from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

SIGNAL_LOG_FIELDS = (
    "ts",
    "window_id",
    "model",
    "signal_source",
    "raw_signal",
    "theta",
    "z",
    "tss",
    "theta_m",
    "z_m",
    "queue_len",
    "decode_tps",
    "prefill_tps",
    "replicas_awake",
    "replicas_target",
    "tier",
    "eta_m",
    "action",
)
SIGNAL_LOG_KEY = "tre:v2:controller:signal_log"


def stream_rows(entries: Iterable[tuple[Any, dict[Any, Any]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _stream_id, raw_fields in entries:
        decoded = {_text(key): _text(value) for key, value in raw_fields.items()}
        rows.append({field: decoded.get(field, "") for field in SIGNAL_LOG_FIELDS})
    return rows


def write_signal_csv(entries: Iterable[tuple[Any, dict[Any, Any]]], output: Path) -> int:
    rows = stream_rows(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SIGNAL_LOG_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest the TRE signal stream to CSV")
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default=SIGNAL_LOG_KEY)
    parser.add_argument("--trim-before-ms", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import redis
    except ModuleNotFoundError as exc:
        raise SystemExit("redis package is required for live harvest") from exc
    client = redis.Redis.from_url(args.redis_url)
    entries = client.xrange(
        args.key,
        min=f"{args.start_ms}-0",
        max=f"{args.end_ms}-999999",
    )
    write_signal_csv(entries, args.output)
    if args.trim_before_ms is not None:
        client.xtrim(
            args.key,
            minid=f"{args.trim_before_ms}-0",
            approximate=False,
        )


if __name__ == "__main__":
    main()
