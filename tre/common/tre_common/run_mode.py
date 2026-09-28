"""Run mode of the TRE controller and of the service-manager supervisor.

``observe`` = compute and record only, no scaling side effects; ``active`` =
actuate. The console writes the controller mode and the SM actuation switch
together (one MULTI) so the two never disagree. Readers that cannot read the
keys keep their last known mode and fail closed (observe) before the first
successful read. See tre/docs/design/20260928-observe-mode-semantics.md.
"""
from __future__ import annotations

from typing import Any

from tre_common.rediskeys import CONTROLLER_MODE_KEY, SM_ACTUATION_KEY

ACTIVE = "active"
OBSERVE = "observe"
RUN_MODES = (ACTIVE, OBSERVE)


def parse_mode(raw: Any) -> str | None:
    """``active`` / ``observe`` from a raw Redis value; None when the key is
    absent or holds something else (callers treat that as unknown)."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    value = str(raw).strip().lower()
    return value if value in RUN_MODES else None


def write_run_mode(redis_client: Any, mode: str) -> None:
    """Set the controller mode AND the SM actuation switch in one transaction."""
    if mode not in RUN_MODES:
        raise ValueError(f"mode must be one of {RUN_MODES}, got {mode!r}")
    pipe = redis_client.pipeline(transaction=True)
    pipe.set(CONTROLLER_MODE_KEY, mode)
    pipe.set(SM_ACTUATION_KEY, mode)
    pipe.execute()


def read_run_modes(redis_client: Any) -> dict[str, str | None]:
    """Raw view for displays: {"controller": mode | None, "sm_actuation": mode |
    None} (None = absent / not a mode). Raises on a Redis error."""
    return {
        "controller": parse_mode(redis_client.get(CONTROLLER_MODE_KEY)),
        "sm_actuation": parse_mode(redis_client.get(SM_ACTUATION_KEY)),
    }


def effective_mode(raw_mode: str | None) -> str:
    """What a reader acts on when the key read succeeded: an absent / unknown
    value is observe (fail-closed)."""
    return ACTIVE if raw_mode == ACTIVE else OBSERVE
