"""Redis keys of the baseline arms (``tre:v2:bl:*``). One place, shared with the gateway
event writer (Go) and the campaign, which repeat these literals."""
from __future__ import annotations

#: STREAM per model, written by the gateway (XADD * MAXLEN ~): request events
#: ``kind=arr|ft|done`` with ``pod, req_id, in_tokens, in_src, max_tokens, out_tokens,
#: status, reissue``. Entry ID = Redis server clock.
REQ_STREAM_PREFIX = "tre:v2:bl:req:"
#: STRING (JSON ``{"t0_ms", "trace_path"}``) written by the campaign at replay start.
REPLAY_T0_KEY = "tre:v2:bl:replay_t0"
#: STRING (JSON, TTL 1 h): the shell's latest decision line per model.
DECISION_PREFIX = "tre:v2:bl:decision:"
DECISION_TTL_S = 3600
#: STRING, SET NX PX: the one shell allowed to actuate.
OWNER_KEY = "tre:v2:bl:owner"


def req_stream_key(model: str) -> str:
    return f"{REQ_STREAM_PREFIX}{model}"


def decision_key(model: str) -> str:
    return f"{DECISION_PREFIX}{model}"
