from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HttpTransport(Protocol):
    def post(self, url: str, *, timeout: float): ...
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
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._http = http or _RequestsTransport()
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._default_port = default_port

    def sleep(self, pod_ip: str, *, port: int | None = None) -> VllmOpResult:
        return self._post(pod_ip, "sleep", port=port)

    def wake_up(self, pod_ip: str, *, port: int | None = None) -> VllmOpResult:
        return self._post(pod_ip, "wake_up", port=port)

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

    def _post(self, pod_ip: str, action: str, *, port: int | None) -> VllmOpResult:
        url = f"http://{pod_ip}:{port or self._default_port}/{action}"
        last_status: int | None = None
        last_message = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.post(url, timeout=self._timeout_s)
            except Exception as exc:  # pragma: no cover - exact transport exceptions vary.
                last_message = str(exc)
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
            attempts=self._max_attempts,
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

    def post(self, url: str, *, timeout: float):
        import requests

        return requests.post(url, timeout=timeout)
