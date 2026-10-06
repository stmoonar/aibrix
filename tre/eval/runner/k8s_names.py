"""Kubernetes pod name -> component name (shared by sampler.py and components.py)."""
from __future__ import annotations

_K8S_SAFE = set("bcdfghjklmnpqrstvwxz2456789")  # alphabet of generated pod-name suffixes (no vowels)


def component_of(pod: str) -> str:
    """Pod name -> component: ``tre-v2-service-manager-<rs hash>-<id>`` -> ``service-manager``
    (Deployment), ``tre-v2-gpu-truth-<id>`` -> ``gpu-truth`` (DaemonSet), ``x-0`` -> ``x``."""
    parts = pod.split("-")

    def generated(p: str, lo: int, hi: int) -> bool:
        return lo <= len(p) <= hi and set(p) <= _K8S_SAFE

    if len(parts) >= 3 and generated(parts[-1], 5, 5) and generated(parts[-2], 6, 10):
        base = "-".join(parts[:-2])
    elif len(parts) >= 2 and (generated(parts[-1], 5, 5) or parts[-1].isdigit()):
        base = "-".join(parts[:-1])
    else:
        base = pod
    for prefix in ("tre-v2-", "tre-"):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base
