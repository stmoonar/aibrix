from __future__ import annotations

from tre_ui.sampler import decode_decision, diff_events, merge_hist


def test_decode_decision_parses_hash_and_json_fields() -> None:
    decoded = decode_decision({
        b"ts_ms": b"1710",
        b"loop": b"fairness",
        b"stale": b"false",
        b"model_states": b'{"m1":{"z_m":0.9,"state":"LOW"}}',
        b"actions": b'[{"kind":"scale","model":"m1","delta":1}]',
        b"events": b'["safescale_probe:m1"]',
    })
    assert decoded["ts_ms"] == 1710 and decoded["loop"] == "fairness"
    assert decoded["model_states"]["m1"]["state"] == "LOW"
    assert decoded["actions"][0]["delta"] == 1
    assert decode_decision({})["ts_ms"] is None


def test_diff_events_emits_only_on_new_decision() -> None:
    decision = decode_decision({
        b"ts_ms": b"2000",
        b"loop": b"rescue",
        b"actions": b'[{"kind":"scale","model":"m1","delta":2,"reason":"critical_sleeping_capacity"}]',
        b"events": b'["leak:node9","routine_ok"]',
    })
    events, key = diff_events(decision, None)
    kinds = {e["kind"] for e in events}
    assert kinds == {"scale", "event"}  # action carries its own kind; "routine_ok" has no marker -> filtered
    assert any("+2" in e["text"] for e in events if e["kind"] == "scale")
    assert any(e["text"] == "leak:node9" for e in events)
    # Same (ts_ms, loop) -> no duplicate emission.
    again, key2 = diff_events(decision, key)
    assert again == [] and key2 == key


def test_merge_hist_dedups_by_window_keeping_latest_ts() -> None:
    existing = [{"window_end_ms": 1000, "ts": 1000, "z_m": 1.0}]
    new = [
        b'{"window_end_ms":1000,"ts":1200,"z_m":1.4}',  # same window, newer ts -> replaces
        b'{"window_end_ms":2000,"ts":2000,"z_m":0.6}',  # new window -> appended
        b"not-json",                                      # ignored
    ]
    merged = merge_hist(existing, new)
    windows = [p["window_end_ms"] for p in merged]
    assert windows == [1000, 2000]
    assert merged[0]["z_m"] == 1.4  # newer ts won
