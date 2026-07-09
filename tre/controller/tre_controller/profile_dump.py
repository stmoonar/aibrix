from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Iterable

from tre_controller.profiling import PROFILE_STREAM_KEY

# Envelope fields first (stable leading columns), then any kind-specific fields.
_PREFERRED_ORDER = [
    "kind",
    "loop",
    "seq",
    "ts_ms",
    "n_models",
    "n_pods",
    "n_actions",
]


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def read_events(
    redis_client: Any,
    *,
    stream_key: str = PROFILE_STREAM_KEY,
    since_ms: int | None = None,
) -> list[dict]:
    entries = redis_client.xrange(stream_key, min="-", max="+")
    events: list[dict] = []
    for _entry_id, fields in entries:
        # flush_loop writes the whole event JSON into a single "data" field.
        decoded = {_decode(k): _decode(v) for k, v in fields.items()}
        raw = decoded.get("data")
        if raw is None:
            continue
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        # Filter on the event own ts_ms (not the stream entry id, which is the
        # ~1s-later flush time) so --since is precise against the recorded timestamp.
        if since_ms is not None and int(event.get("ts_ms", 0)) < int(since_ms):
            continue
        events.append(event)
    return events


def _ordered_fieldnames(events: Iterable[dict]) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for name in _PREFERRED_ORDER:
        seen.append(name)
        seen_set.add(name)
    for event in events:
        for key in event:
            if key not in seen_set:
                seen.append(key)
                seen_set.add(key)
    # Drop preferred columns that never actually appear, keeping output tidy.
    present: set[str] = set()
    for event in events:
        present.update(event.keys())
    return [name for name in seen if name in present]


def write_csv(events: list[dict], out_path: str) -> int:
    fieldnames = _ordered_fieldnames(events)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(event)
    return len(events)


def _build_redis_client(redis_url: str) -> Any:
    import redis  # type: ignore[import-not-found]

    return redis.Redis.from_url(redis_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump the TRE controller profile Redis stream to CSV.")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0"),
        help="Redis URL of the controller's primary redis (default: $TRE_REDIS_URL).",
    )
    parser.add_argument("--stream-key", default=PROFILE_STREAM_KEY)
    parser.add_argument("--since", type=int, default=None, help="Only events with ts_ms >= this (ms epoch).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    client = _build_redis_client(args.redis_url)
    events = read_events(client, stream_key=args.stream_key, since_ms=args.since)
    count = write_csv(events, args.out)
    print(f"wrote {count} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
