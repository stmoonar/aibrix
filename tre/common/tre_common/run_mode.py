"""Run mode of the TRE controller and actuation switch of the service-manager.

Two INDEPENDENT switches (user decision 2026-09-28), each ``active`` | ``observe``:

* ``tre:v2:controller:mode`` -- observe = the TRE controller computes and
  records its decisions only, no scaling side effects.
* ``tre:v2:sm:actuation`` -- whether the SM supervisor may perform the
  self-heal actions that restore declared state (B7 recreate, drift fleet
  repair, resume a stale repair, reap rejected Deployments, sleep residents for
  an unrequested startup admission). It is NOT derived from the controller
  mode.

Experiments run the SM in ``active`` in both arms (symmetric self-heal): TRE arm
= (controller active, SM active), APA arm = (controller observe, SM active).
Calibration and maintenance = (observe, observe). A missing key is observe for
its reader (fail-closed); deployments must set both explicitly. Readers that
cannot read a key keep their last known mode and fail closed (observe) before
the first successful read. See tre/docs/design/20260928-observe-mode-semantics.md.
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


def _check(name: str, value: str | None) -> None:
    if value is not None and value not in RUN_MODES:
        raise ValueError(f"{name} must be one of {RUN_MODES}, got {value!r}")


def write_run_mode(
    redis_client: Any,
    *,
    controller_mode: str | None = None,
    sm_actuation: str | None = None,
) -> dict[str, str]:
    """Set the controller mode and / or the SM actuation switch. None = leave
    that key unchanged; both given = one MULTI transaction. Returns the keys
    written ({"controller": ..., "sm_actuation": ...}, only those set)."""
    _check("controller_mode", controller_mode)
    _check("sm_actuation", sm_actuation)
    writes: list[tuple[str, str, str]] = []
    if controller_mode is not None:
        writes.append(("controller", CONTROLLER_MODE_KEY, controller_mode))
    if sm_actuation is not None:
        writes.append(("sm_actuation", SM_ACTUATION_KEY, sm_actuation))
    if not writes:
        raise ValueError("write_run_mode needs controller_mode and/or sm_actuation")
    if len(writes) == 1:
        _name, key, value = writes[0]
        redis_client.set(key, value)
    else:
        pipe = redis_client.pipeline(transaction=True)
        for _name, key, value in writes:
            pipe.set(key, value)
        pipe.execute()
    return {name: value for name, _key, value in writes}


def read_run_modes(redis_client: Any) -> dict[str, str | None]:
    """Raw view for displays: {"controller": mode | None, "sm_actuation": mode |
    None} (None = absent / not a mode). Raises on a Redis error."""
    return {
        "controller": parse_mode(redis_client.get(CONTROLLER_MODE_KEY)),
        "sm_actuation": parse_mode(redis_client.get(SM_ACTUATION_KEY)),
    }


def missing_key_warnings(raw: dict[str, str | None]) -> list[str]:
    """Operator warnings for keys that are absent (or hold no valid mode): each
    reader then acts as observe. Deployments must set both keys explicitly."""
    warnings: list[str] = []
    if raw.get("controller") is None:
        warnings.append(f"{CONTROLLER_MODE_KEY} missing -> controller treats as observe")
    if raw.get("sm_actuation") is None:
        warnings.append(f"{SM_ACTUATION_KEY} missing -> SM treats as observe")
    return warnings


def effective_mode(raw_mode: str | None) -> str:
    """What a reader acts on when the key read succeeded: an absent / unknown
    value is observe (fail-closed)."""
    return ACTIVE if raw_mode == ACTIVE else OBSERVE


def run_mode_view(redis_client: Any) -> dict[str, Any]:
    """Display payload (console): ``controller`` (alias ``mode``) and
    ``sm_actuation`` = what each reader acts on, ``raw`` = the stored values
    (None = absent), ``warnings`` = missing keys. Never raises: an unreadable
    Redis shows observe with a warning (readers keep their last known mode)."""
    try:
        raw = read_run_modes(redis_client)
    except Exception as exc:  # noqa: BLE001 - display only
        return {
            "mode": OBSERVE, "controller": OBSERVE, "sm_actuation": OBSERVE, "raw": None,
            "warnings": [f"run mode unreadable ({exc}) -> readers keep their last known mode "
                         "(observe if never read)"],
        }
    controller = effective_mode(raw["controller"])
    return {
        "mode": controller,
        "controller": controller,
        "sm_actuation": effective_mode(raw["sm_actuation"]),
        "raw": raw,
        "warnings": missing_key_warnings(raw),
    }
