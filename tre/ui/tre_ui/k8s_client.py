"""Minimal in-cluster Kubernetes client (stdlib only -- no `kubernetes` dependency).

The console needs exactly four operations to support restart-to-apply param editing:
read/replace the tre-v2-registry ConfigMap, and read/annotate the controller Deployment.
Auth is the pod's ServiceAccount: the projected token rotates, so it is re-read from disk
on EVERY request (never cached), and TLS is pinned to the cluster CA. All calls are
namespace-scoped and the RBAC Role (tre-v2-ui-params) is resourceName-bound, so this client
is incapable of touching anything but those two objects in the tre-v2 namespace.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
_TOKEN_PATH = f"{_SA_DIR}/token"
_CA_PATH = f"{_SA_DIR}/ca.crt"
_NS_PATH = f"{_SA_DIR}/namespace"


class K8sError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"k8s {status}: {message}")
        self.status = status
        self.message = message


class InClusterK8sClient:
    def __init__(
        self,
        *,
        namespace: str | None = None,
        host: str | None = None,
        port: str | None = None,
        ca_path: str = _CA_PATH,
        token_path: str = _TOKEN_PATH,
    ) -> None:
        host = host or os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = port or os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self._base = f"https://{host}:{port}"
        self._token_path = token_path
        self._namespace = namespace or _read_namespace()
        # Pin TLS to the cluster CA when present (in-cluster); fall back to default verify.
        self._ctx = ssl.create_default_context(cafile=ca_path) if os.path.exists(ca_path) else ssl.create_default_context()

    @property
    def namespace(self) -> str:
        return self._namespace

    def get_configmap(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/namespaces/{self._namespace}/configmaps/{name}")

    def replace_configmap(self, name: str, data: dict[str, str], resource_version: str) -> dict[str, Any]:
        body = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name, "namespace": self._namespace, "resourceVersion": resource_version},
            "data": data,
        }
        return self._request("PUT", f"/api/v1/namespaces/{self._namespace}/configmaps/{name}", body)

    def get_deployment(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/apis/apps/v1/namespaces/{self._namespace}/deployments/{name}")

    def patch_deployment(self, name: str, patch: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/apis/apps/v1/namespaces/{self._namespace}/deployments/{name}?fieldManager=tre-v2-ui",
            patch,
            content_type="application/strategic-merge-patch+json",
        )

    def _request(self, method: str, path: str, body: dict | None = None, *, content_type: str = "application/json") -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {self._read_token()}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self._base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=10, context=self._ctx) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # 409 conflict, 403 forbidden, 422, ...
            detail = exc.read().decode("utf-8", "replace")
            raise K8sError(exc.code, _reason(detail)) from exc
        return json.loads(raw) if raw else {}

    def _read_token(self) -> str:
        with open(self._token_path, encoding="utf-8") as handle:  # re-read: projected tokens rotate
            return handle.read().strip()


def _read_namespace() -> str:
    try:
        with open(_NS_PATH, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ.get("TRE_NAMESPACE", "tre-v2")


def _reason(detail: str) -> str:
    try:
        return json.loads(detail).get("message", detail)[:300]
    except (json.JSONDecodeError, AttributeError):
        return detail[:300]
