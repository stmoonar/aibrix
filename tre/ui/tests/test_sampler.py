from __future__ import annotations

from tre_ui.sampler import _RATES, Sampler, decode_decision, diff_events, merge_hist


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


# ---- sampler upstream sources (Task 1) ----


class FakeRedis:
    """Minimal Redis stand-in: every read the sampler makes returns empty."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict] = {}

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def zrangebyscore(self, key, minimum, maximum):
        return []

    def scan_iter(self, match):
        return iter(())

    def get(self, key):
        return None


class FakeSmClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_state(self):
        self.calls.append("/v2/state")
        return {"version": 1, "bindings": []}

    def request(self, method, path, payload=None):
        self.calls.append(path)
        if path == "/v2/fleet/state":
            return {"desired_version": 3, "observed_version": 3,
                    "desired": [], "observed": [], "mismatches": []}
        if path == "/v2/supervisor":
            return {"enabled": True, "running": True, "drift_observations": 0}
        if path.startswith("/v2/operations"):
            return {"operations": [{"id": "op-1", "status": "succeeded"}]}
        raise AssertionError(f"unexpected path {path}")


def test_sampler_publishes_fleet_supervisor_and_operations() -> None:
    sampler = Sampler(FakeRedis(), FakeSmClient(), model_names=["m1"])

    sampler.sample_once()
    snap = sampler.snapshot()

    assert snap["fleet"]["state"]["desired_version"] == 3
    assert snap["fleet"]["supervisor"]["running"] is True
    assert snap["operations"]["items"][0]["id"] == "op-1"


def test_sampler_never_polls_the_expensive_audit_endpoint() -> None:
    """Guard: /v2/audit lists k8s pods and HTTP-probes every vLLM pod.

    Polling it would load the model pods and perturb scaling experiments, so it
    must stay button-triggered only. See docs/design/20260725-console-redesign.md.
    """
    client = FakeSmClient()
    sampler = Sampler(FakeRedis(), client, model_names=["m1"])

    for _ in range(20):
        sampler.sample_once()

    assert not any("audit" in path for path in client.calls)
    assert not any("audit" in source for source in _RATES)


def test_sampler_publishes_gpu_leases() -> None:
    redis = FakeRedis()
    redis.hashes["tre:v2:sm:gpu_leases"] = {
        b"node-a/0": b'{"binding_id":"m1/node-a/0","phase":"awake","fencing_token":7}'
    }
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])

    sampler.sample_once()

    assert sampler.snapshot()["leases"]["gpus"]["node-a/0"]["fencing_token"] == 7


# ---- signal_log timeline (Task 2) ----


class FakeStreamRedis(FakeRedis):
    def __init__(self, entries) -> None:
        super().__init__()
        self.entries = list(entries)  # [(id, {field: value})]
        self.stream_calls: list[tuple] = []

    def xrevrange(self, key, max="+", min="-", count=None):
        self.stream_calls.append(("xrevrange", count))
        out = list(reversed(self.entries))
        return out[:count] if count else out

    def xrange(self, key, min="-", max="+", count=None):
        self.stream_calls.append(("xrange", count))
        start = str(min).lstrip("(")
        out = [e for e in self.entries if e[0] > start]
        return out[:count] if count else out


def _entry(entry_id, model, ts_ms, z, action="none"):
    return (entry_id, {
        b"ts": str(ts_ms / 1000).encode(), b"window_id": str(ts_ms).encode(),
        b"model": model.encode(), b"z_m": str(z).encode(), b"queue_len": b"3",
        b"decode_tps": b"120", b"prefill_tps": b"400", b"replicas_awake": b"1",
        b"replicas_target": b"2", b"tier": b"healthy", b"action": action.encode(),
    })


def test_timeline_backfills_then_reads_incrementally() -> None:
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, 0.5)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])
    sampler.sample_once()

    assert [p["z_m"] for p in sampler.timeline("m1")] == [0.5]
    assert redis.stream_calls[0][0] == "xrevrange"

    redis.entries.append(_entry("2-0", "m1", 2000, 0.9, action="scale_up"))
    sampler._next["timeline"] = 0.0
    sampler.sample_once()

    points = sampler.timeline("m1")
    assert [p["z_m"] for p in points] == [0.5, 0.9]
    assert points[1]["action"] == "scale_up"
    assert redis.stream_calls[-1][0] == "xrange"


def test_timeline_reads_are_always_count_bounded() -> None:
    """A full XRANGE would scan the 200k-entry stream and stall the sampler."""
    redis = FakeStreamRedis([_entry(f"{i}-0", "m1", i * 1000, 0.1) for i in range(1, 50)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])

    sampler.sample_once()

    assert redis.stream_calls
    assert all(call[1] is not None and call[1] > 0 for call in redis.stream_calls)


def test_timeline_decodes_nan_as_none() -> None:
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, float("nan"))])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])

    sampler.sample_once()

    assert sampler.timeline("m1")[0]["z_m"] is None
    assert sampler.timeline("m1")[0]["queue_len"] == 3.0


def test_timeline_filters_by_since_ms() -> None:
    redis = FakeStreamRedis([_entry("1-0", "m1", 1000, 0.5), _entry("2-0", "m1", 5000, 0.7)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])

    sampler.sample_once()

    assert [p["z_m"] for p in sampler.timeline("m1", since_ms=3000)] == [0.7]


def test_timeline_ignores_rows_for_unknown_models() -> None:
    redis = FakeStreamRedis([_entry("1-0", "other", 1000, 0.5)])
    sampler = Sampler(redis, FakeSmClient(), model_names=["m1"])

    sampler.sample_once()

    assert sampler.timeline("m1") == []
