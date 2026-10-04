from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


#: :meth:`VllmOps.sidecar_waking` when the pod has no reissue sidecar (404).
NO_SIDECAR = "no_sidecar"


class HttpTransport(Protocol):
    def post(self, url: str, *, timeout: float, headers: dict[str, str] | None = None): ...
    def get(self, url: str, *, timeout: float): ...


@dataclass(frozen=True)
class VllmOpResult:
    success: bool
    action: str
    url: str
    attempts: int
    status_code: int | None = None
    message: str = ""
    idempotent: bool = False


class VllmOps:
    def __init__(
        self,
        *,
        http: HttpTransport | None = None,
        timeout_s: float = 5.0,
        max_attempts: int = 3,
        default_port: int = 8000,
        wake_timeout_s: float | None = None,
    ) -> None:
        """``timeout_s``: every probe (and, without ``wake_timeout_s``, each of
        up to ``max_attempts`` /wake_up attempts). ``wake_timeout_s``
        (registry ``service_manager.wake.call_timeout_s``): ONE /wake_up attempt
        with this timeout - the service-manager holds its writer lock through
        the call (2026-10-02), so its length is bounded once."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._http = http or _RequestsTransport()
        self._timeout_s = timeout_s
        self._wake_timeout_s = wake_timeout_s
        self._max_attempts = max_attempts
        self._default_port = default_port

    def sleep(
        self,
        pod_ip: str,
        *,
        port: int | None = None,
        mode: str | None = None,
        timeout_s: float | None = None,
        hidden: bool = False,
    ) -> VllmOpResult:
        """POST /sleep.

        ``mode`` (vLLM >= 0.18: ``wait`` | ``abort`` | ``keep``; see
        ``sleep_primitive.SLEEP_MODE_MIN_VERSION``) is sent as a query parameter
        only when given. ``timeout_s`` replaces the default
        HTTP timeout and makes the call single-shot (a timed-out ``mode=wait``
        must not be blindly retried). ``hidden`` adds ``X-TRE-Hidden: 1``: the
        caller hid the pod first (plan D2: a sidecar refuses /sleep without it).
        Only the service-manager sleep primitive may call this.
        """
        if mode is not None and mode not in {"wait", "abort", "keep"}:
            raise ValueError(f"unknown vLLM sleep mode: {mode}")
        headers = {"X-TRE-Hidden": "1"} if hidden else None
        return self._post(
            pod_ip,
            "sleep",
            port=port,
            query=f"mode={mode}" if mode else None,
            headers=headers,
            timeout_s=timeout_s,
            max_attempts=1 if (timeout_s is not None or mode == "wait") else None,
        )

    def metrics(self, pod_ip: str, *, port: int | None = None) -> str | None:
        """Prometheus text from GET /metrics, or None on any failure / non-2xx."""
        url = f"http://{pod_ip}:{port or self._default_port}/metrics"
        try:
            response = self._http.get(url, timeout=self._timeout_s)
            status = int(response.status_code)
        except Exception:
            return None
        if not (200 <= status < 300):
            return None
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else None

    def version(self, pod_ip: str, *, port: int | None = None) -> str | None:
        """vLLM version string from ``GET /version`` (None on any failure)."""
        url = f"http://{pod_ip}:{port or self._default_port}/version"
        try:
            response = self._http.get(url, timeout=self._timeout_s)
            status = int(response.status_code)
        except Exception:
            return None
        if not (200 <= status < 300):
            return None
        payload = None
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                payload = json_method()
            except Exception:
                payload = None
        if isinstance(payload, dict) and payload.get("version") is not None:
            return str(payload["version"])
        return None

    def wake_up(self, pod_ip: str, *, port: int | None = None) -> VllmOpResult:
        """POST /wake_up. Never retried after a call without an answer (2026-10-04):
        the first one may still be running on the engine, and a retry only makes
        the outcome harder to read. Such a result has status_code None: the
        caller treats the wake as uncertain, not failed."""
        if self._wake_timeout_s is not None:
            return self._post(pod_ip, "wake_up", port=port, timeout_s=self._wake_timeout_s, max_attempts=1)
        return self._post(pod_ip, "wake_up", port=port, retry_unanswered=False)

    def sidecar_waking(self, pod_ip: str, *, port: int | None = None) -> int | str | None:
        """``waking`` of the reissue sidecar's ``GET /tre-reissue/state`` (the
        sidecar serves the pod's port and counts the /wake_up calls it is
        forwarding to the engine).

        * an int: the in-flight count;
        * :data:`NO_SIDECAR`: HTTP 404 - no sidecar in front of the engine
          (reissue disabled: the path reaches vLLM); the caller may fall back to
          weaker evidence;
        * None: no answer (timeout, connection error), another status or an
          undecodable body - a wake may be in flight; the engine on the same
          port is not worth probing either."""
        url = f"http://{pod_ip}:{port or self._default_port}/tre-reissue/state"
        try:
            response = self._http.get(url, timeout=self._timeout_s)
            status = int(response.status_code)
        except Exception:
            return None
        if status == 404:
            return NO_SIDECAR
        if not (200 <= status < 300):
            return None
        json_method = getattr(response, "json", None)
        try:
            payload = json_method() if callable(json_method) else None
        except Exception:
            return None
        value = payload.get("waking") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def is_paused(self, pod_ip: str, *, port: int | None = None) -> bool | None:
        """``GET /is_paused`` (vLLM dev-mode router, next to /sleep): whether the
        engine's scheduler is paused - e.g. a /sleep paused it and then failed.
        None when unreachable, not served (older engines) or undecodable."""
        url = f"http://{pod_ip}:{port or self._default_port}/is_paused"
        try:
            response = self._http.get(url, timeout=self._timeout_s)
            status = int(response.status_code)
        except Exception:
            return None
        if not (200 <= status < 300):
            return None
        json_method = getattr(response, "json", None)
        try:
            payload = json_method() if callable(json_method) else None
        except Exception:
            return None
        value = payload.get("is_paused") if isinstance(payload, dict) else None
        return value if isinstance(value, bool) else None

    def resume(self, pod_ip: str, *, port: int | None = None) -> VllmOpResult:
        """``POST /resume``: un-pause the scheduler (single attempt, probe timeout)."""
        return self._post(pod_ip, "resume", port=port, timeout_s=self._timeout_s, max_attempts=1)

    def is_sleeping(self, pod_ip: str, *, port: int | None = None) -> bool | None:
        """Physical /is_sleeping probe (OBSERVED ground truth).

        Returns True/False for the physical sleep state, or None when the
        pod is unreachable or returns an undecodable/non-2xx response.
        """
        url = f"http://{pod_ip}:{port or self._default_port}/is_sleeping"
        try:
            response = self._http.get(url, timeout=self._timeout_s)
        except Exception:  # pragma: no cover - exact transport exceptions vary.
            return None
        try:
            status = int(response.status_code)
        except Exception:
            return None
        if not (200 <= status < 300):
            return None
        return _parse_is_sleeping(response)

    def wait_until_ready(
        self,
        pod_ip: str,
        *,
        port: int | None = None,
        timeout_s: float = 180.0,
        interval_s: float = 2.0,
    ) -> VllmOpResult:
        import time

        url = f"http://{pod_ip}:{port or self._default_port}/is_sleeping"
        deadline = time.monotonic() + timeout_s
        attempts = 0
        last_status: int | None = None
        last_message = ""
        while time.monotonic() < deadline:
            attempts += 1
            try:
                response = self._http.get(url, timeout=self._timeout_s)
            except Exception as exc:  # pragma: no cover - exact transport exceptions vary.
                last_message = str(exc)
            else:
                last_status = int(response.status_code)
                last_message = getattr(response, "text", "") or ""
                if 200 <= last_status < 300:
                    return VllmOpResult(
                        success=True,
                        action="wait_until_ready",
                        url=url,
                        attempts=attempts,
                        status_code=last_status,
                        message=last_message,
                    )
            time.sleep(interval_s)

        return VllmOpResult(
            success=False,
            action="wait_until_ready",
            url=url,
            attempts=attempts,
            status_code=last_status,
            message=last_message or "timed out waiting for vLLM HTTP readiness",
        )

    def _post(
        self,
        pod_ip: str,
        action: str,
        *,
        port: int | None,
        query: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        max_attempts: int | None = None,
        retry_unanswered: bool = True,
    ) -> VllmOpResult:
        url = f"http://{pod_ip}:{port or self._default_port}/{action}"
        if query:
            url = f"{url}?{query}"
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        attempts = self._max_attempts if max_attempts is None else max_attempts
        last_status: int | None = None
        last_message = ""
        for attempt in range(1, attempts + 1):
            try:
                if headers:
                    response = self._http.post(url, timeout=timeout, headers=headers)
                else:
                    response = self._http.post(url, timeout=timeout)
            except Exception as exc:  # pragma: no cover - exact transport exceptions vary.
                # The last attempt decides: no answer = status_code None.
                last_status = None
                last_message = str(exc)
                if not retry_unanswered:
                    attempts = attempt
                    break
                continue

            last_status = int(response.status_code)
            last_message = getattr(response, "text", "") or ""
            if 200 <= last_status < 300:
                return VllmOpResult(
                    success=True,
                    action=action,
                    url=url,
                    attempts=attempt,
                    status_code=last_status,
                    message=last_message,
                )
            if last_status == 409:
                return VllmOpResult(
                    success=True,
                    action=action,
                    url=url,
                    attempts=attempt,
                    status_code=last_status,
                    message=last_message,
                    idempotent=True,
                )

        return VllmOpResult(
            success=False,
            action=action,
            url=url,
            attempts=attempts,
            status_code=last_status,
            message=last_message,
        )


def _parse_is_sleeping(response) -> bool | None:
    payload = None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            payload = json_method()
        except Exception:
            payload = None
    if payload is None:
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return None
        import json as _json

        try:
            payload = _json.loads(text)
        except Exception:
            return None
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, dict) and "is_sleeping" in payload:
        return bool(payload["is_sleeping"])
    return None


class _RequestsTransport:
    def get(self, url: str, *, timeout: float):
        import requests

        return requests.get(url, timeout=timeout)

    def post(self, url: str, *, timeout: float, headers: dict[str, str] | None = None):
        import requests

        return requests.post(url, timeout=timeout, headers=headers)
