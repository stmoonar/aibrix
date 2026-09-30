"""PreServe load-look-ahead map (§4.3.1): exact values, ring, extension, early done."""
from __future__ import annotations

import math

import pytest

from tre_baselines.policies.preserve_anticipator import LookaheadMap

M = 10_000.0


def u(P: int, i: int, m: float = M) -> float:
    return (P + i) / m


def test_long_request_inside_and_beyond_the_read_window() -> None:
    L = math.ceil(4096 * 1.2)
    amap = LookaheadMap(L)
    amap.add("r", P=1000, D=512, M=M)
    w = amap.window(100)
    assert len(w) == 100
    assert w == pytest.approx([u(1000, i) for i in range(100)])
    # beyond the scaler's read window the map still holds the request
    for i in (100, 300, 511):
        assert amap.at(i) == pytest.approx(u(1000, i))
    assert amap.at(512) == 0.0 and amap.at(L - 1) == 0.0
    amap.advance(100)
    assert amap.window(100) == pytest.approx([u(1000, 100 + i) for i in range(100)])
    assert amap.at(411) == pytest.approx(u(1000, 511)) and amap.at(412) == 0.0
    # a second request overlaps additively
    amap.add("s", P=200, D=50, M=M)
    assert amap.at(0) == pytest.approx(u(1000, 100) + u(200, 0))
    assert amap.at(49) == pytest.approx(u(1000, 149) + u(200, 49))
    assert amap.at(50) == pytest.approx(u(1000, 150))


def test_ring_wraps_around() -> None:
    amap = LookaheadMap(120, out_len_is_upper_bound=False)  # the paper's extension mode
    amap.advance(110)  # head at absolute 110 -> slot 110; the next 10 iterations wrap
    amap.add("r", P=0, D=50, M=100.0)
    assert amap.window(50) == pytest.approx([i / 100 for i in range(50)])
    assert amap.U[110] == pytest.approx(0.0) and amap.U[119] == pytest.approx(0.09)
    assert amap.U[0] == pytest.approx(0.10)  # absolute 120 lives in slot 0
    amap.advance(30)
    assert amap.window(20) == pytest.approx([(30 + i) / 100 for i in range(20)])
    assert amap.at(20) == 0.0
    # a request never finished is extended past the jump; once done, a big jump leaves
    # an empty ring
    amap.advance(500)
    assert amap.requests["r"].D > 640 - 110 and amap.at(0) == pytest.approx(640 / 100 - 110 / 100)
    amap.remove("r")
    amap.advance(500)
    assert amap.it == 1140 and all(x == 0.0 for x in amap.window(120))


def test_request_longer_than_the_ring_is_filled_as_it_comes_into_view() -> None:
    amap = LookaheadMap(120)
    amap.add("r", P=10, D=300, M=1000.0)
    assert amap.at(119) == pytest.approx(u(10, 119, 1000.0))
    amap.advance(150)
    assert amap.window(120) == pytest.approx([u(10, 150 + i, 1000.0) for i in range(120)])
    amap.advance(100)  # absolute 250: 50 iterations left
    assert amap.at(49) == pytest.approx(u(10, 299, 1000.0)) and amap.at(50) == 0.0


def test_virtual_extension_when_running_past_the_prediction() -> None:
    amap = LookaheadMap(120, ext_frac=0.2, out_len_is_upper_bound=False)
    amap.add("r", P=0, D=20, M=100.0)
    amap.advance(19)
    assert amap.at(0) == pytest.approx(0.19) and amap.at(1) == 0.0
    amap.advance(1)  # head at 20 = planned end, not done -> + ceil(0.2*20) = 4
    req = amap.requests["r"]
    assert req.D == 24 and req.extensions == 1
    assert amap.window(5) == pytest.approx([0.20, 0.21, 0.22, 0.23, 0.0])
    amap.advance(4)  # again: extension base is the prediction (20), not the extended length
    assert req.D == 28 and req.extensions == 2
    assert amap.at(0) == pytest.approx(0.24) and amap.at(4) == 0.0
    amap.advance(13)  # past several extension steps at once
    assert req.D == 40 and amap.at(0) == pytest.approx(0.37) and amap.at(3) == 0.0
    amap.remove("r")
    assert all(x == 0.0 for x in amap.window(120))


def test_early_done_subtracts_only_what_is_ahead() -> None:
    amap = LookaheadMap(200)
    amap.add("a", P=100, D=100, M=1000.0)
    amap.add("b", P=50, D=30, M=1000.0)
    amap.advance(10)
    removed = amap.remove("a")
    assert removed is not None and removed.D == 100
    # only b remains, exactly
    assert amap.window(25) == pytest.approx([u(50, 10 + i, 1000.0) if i < 20 else 0.0 for i in range(25)])
    assert amap.remove("a") is None
    amap.remove("b")
    assert all(x == 0.0 for x in amap.window(200))


def test_readd_replaces_and_bad_inputs() -> None:
    amap = LookaheadMap(50)
    amap.add("a", P=10, D=5, M=100.0)
    amap.add("a", P=20, D=5, M=100.0)
    assert amap.at(0) == pytest.approx(0.2)
    assert amap.at(-1) == 0.0 and amap.at(50) == 0.0
    amap.advance(0)
    amap.advance(-3)
    assert amap.it == 0
    with pytest.raises(ValueError):
        amap.add("x", P=1, D=1, M=0.0)
    with pytest.raises(ValueError):
        LookaheadMap(0)
    assert len(LookaheadMap(10).window(100)) == 10


def test_upper_bound_mode_never_extends_and_drops_phantoms() -> None:
    amap = LookaheadMap(200)  # default: max_tokens is a hard cap, margin 0.25
    amap.add("a", P=100, D=40, M=1000.0)
    amap.add("b", P=10, D=100, M=1000.0)
    assert amap.advance(40) == []  # a reached its cap: no extension, no load beyond D
    assert amap.requests["a"].D == 40 and amap.requests["a"].extensions == 0
    assert amap.at(0) == pytest.approx(u(10, 40, 1000.0))
    assert amap.advance(10) == []  # 50 = 40 * 1.25: not yet a phantom
    assert amap.advance(1) == ["a"]  # 51 > 50: its done event was lost
    assert "a" not in amap.requests and amap.phantom_dropped == 1
    assert amap.advance(100) == ["b"]  # 151 > 125
    assert amap.requests == {} and amap.total() == 0.0 and amap.phantom_dropped == 2
    with pytest.raises(ValueError):
        LookaheadMap(10, phantom_margin=-0.1)


def test_lost_done_events_do_not_accumulate_load() -> None:
    amap = LookaheadMap(120)
    peak = 0.0
    for step in range(200):  # one request per 5 iterations, none ever finishes
        amap.add(f"r{step}", P=50, D=30, M=1000.0)
        amap.advance(5)
        peak = max(peak, amap.total())
    # at most ceil(30 * 1.25 / 5) + 1 requests are ever alive: the load stays bounded
    assert len(amap.requests) <= 9 and peak < 9 * sum(u(50, i, 1000.0) for i in range(30))
    amap.advance(40)
    assert amap.requests == {} and amap.total() == 0.0 and amap.phantom_dropped == 200
