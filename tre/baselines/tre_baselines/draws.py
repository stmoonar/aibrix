"""Per-request random draws as pure functions of a key (v1 ``tre_tokenscale.buckets._unit``).

A draw (TokenScale's simulated length-prediction error, PreServe's Tab. 2 length error) is a
hash of ``(seed, purpose, model, request key)``: it does not depend on arrival order or
timing, a reissued request (same client id) gets the draw of its original, and every arm
replaying the same trace with the same seed sees the same draws. The request key is the
client's request id (:attr:`tre_baselines.snapshot.RequestEvent.draw_key`).
"""
from __future__ import annotations

import hashlib
from typing import Any


def unit(*parts: Any) -> float:
    """Deterministic uniform number in [0, 1) from ``parts`` (blake2b, 64 bits; the same
    encoding as v1, so equal parts give v1's draw)."""
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)
