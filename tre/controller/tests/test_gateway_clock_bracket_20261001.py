"""O1 review P3 (2026-10-01): a written_ms measurement whose poll bracket is wider than
2 x poll_ms (a stalled poll) is discarded, never turned into an offset."""

from __future__ import annotations

import json

from tre_controller.gateway_clock import measure_written_offset_ms


class _Docs:
    """The newest instant doc's written_ms per read, from a script (None = no doc)."""

    def __init__(self, values):
        self.values = list(values)
        self.last = None

    def zrange(self, key, start, end, withscores=False):
        if self.values:
            self.last = self.values.pop(0)
        if self.last is None:
            return []
        return [(json.dumps({"timestamp": self.last, "written_ms": self.last}).encode(), float(self.last))]


def _clock(steps_ms):
    """Clock advancing by the scripted amounts on every sleep."""
    state = {"ms": 1_000_000, "steps": list(steps_ms)}

    def sleep(_seconds):
        state["ms"] += state["steps"].pop(0) if state["steps"] else 250

    return state, sleep


def test_a_normal_bracket_gives_the_midpoint_offset():
    docs = _Docs([10_000, 10_000, 1_000_400])        # initial, one poll unchanged, then new
    state, sleep = _clock([250, 250])
    offset = measure_written_offset_ms(docs, "k", clock_ms=lambda: state["ms"], sleep_s=sleep, poll_ms=250,
                                       max_wait_ms=12_000)
    # bracket [1_000_250, 1_000_500] -> midpoint 1_000_375
    assert offset == 1_000_400 - 1_000_375


def test_a_stalled_poll_is_discarded_and_the_next_write_measured():
    # initial; a poll that stalled 2 s and sees a new doc (discarded); then normal polls;
    # the next doc is measured on a 250 ms bracket
    docs = _Docs([10_000, 1_001_000, 1_001_000, 1_003_300])
    state, sleep = _clock([2_000, 250, 250])
    offset = measure_written_offset_ms(docs, "k", clock_ms=lambda: state["ms"], sleep_s=sleep, poll_ms=250,
                                       max_wait_ms=12_000)
    # discarded bracket [1_000_000, 1_002_000]; measured bracket [1_002_250, 1_002_500]
    assert offset == 1_003_300 - 1_002_375


def test_only_stalled_brackets_give_no_measurement():
    docs = _Docs([10_000] + [1_000_000 + 10_000 * i for i in range(1, 10)])
    state, sleep = _clock([1_000] * 20)               # every poll overshoots 4 x poll_ms
    assert measure_written_offset_ms(docs, "k", clock_ms=lambda: state["ms"], sleep_s=sleep, poll_ms=250,
                                     max_wait_ms=5_000) is None
