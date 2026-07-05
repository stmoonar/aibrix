#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_GATEWAY_URL = "http://10.99.21.145/v1/completions"
DEFAULT_SERVICE_MANAGER_URL = "http://10.111.21.116:8000"
DEFAULT_MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
PHASE_SECONDS = 60
DEFAULT_DURATION_SECONDS = 900
DEFAULT_WORKERS = 4
REQUEST_TIMEOUT = 45


def parse_models(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the N4b three-model zm precheck load.")
    parser.add_argument("--gateway-url", default=os.environ.get("N4B_GATEWAY_URL", DEFAULT_GATEWAY_URL))
    parser.add_argument("--service-manager-url", default=os.environ.get("N4B_SERVICE_MANAGER_URL", DEFAULT_SERVICE_MANAGER_URL))
    parser.add_argument("--models", type=parse_models, default=parse_models(os.environ.get("N4B_MODELS", ",".join(DEFAULT_MODELS))))
    parser.add_argument("--duration-seconds", type=int, default=int(os.environ.get("N4B_DURATION_SECONDS", str(DEFAULT_DURATION_SECONDS))))
    parser.add_argument("--phase-seconds", type=int, default=int(os.environ.get("N4B_PHASE_SECONDS", str(PHASE_SECONDS))))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("N4B_WORKERS", str(DEFAULT_WORKERS))))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("N4B_MAX_TOKENS", "96")))
    parser.add_argument("--sample-seconds", type=int, default=int(os.environ.get("N4B_SAMPLE_SECONDS", "30")))
    parser.add_argument("--request-timeout", type=int, default=int(os.environ.get("N4B_REQUEST_TIMEOUT", str(REQUEST_TIMEOUT))))
    return parser.parse_args(argv)


def http_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None, timeout: int = 10):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else None


def kubectl_json(args: list[str]):
    out = subprocess.check_output(["kubectl", *args], text=True)
    return json.loads(out)


def pod_restarts(namespace: str, selector: str | None = None) -> dict[str, int]:
    args = ["-n", namespace, "get", "pods", "-o", "json"]
    if selector:
        args[3:3] = ["-l", selector]
    data = kubectl_json(args)
    result: dict[str, int] = {}
    for item in data.get("items", []):
        total = 0
        for status in item.get("status", {}).get("containerStatuses", []) or []:
            total += int(status.get("restartCount", 0))
        result[item["metadata"]["name"]] = total
    return result


def rss_kb(namespace: str, selector: str) -> int | None:
    pods = kubectl_json(["-n", namespace, "get", "pods", "-l", selector, "-o", "json"]).get("items", [])
    if not pods:
        return None
    pod = pods[0]["metadata"]["name"]
    try:
        out = subprocess.check_output(
            ["kubectl", "-n", namespace, "exec", pod, "--", "sh", "-c", "awk '/VmRSS/ {print $2}' /proc/1/status"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return int(out.strip())
    except Exception:
        return None


def redis_dbsize() -> int | None:
    pods = kubectl_json(["-n", "tre-v2", "get", "pods", "-l", "app.kubernetes.io/name=tre-v2-redis", "-o", "json"]).get("items", [])
    if not pods:
        return None
    pod = pods[0]["metadata"]["name"]
    try:
        out = subprocess.check_output(
            ["kubectl", "-n", "tre-v2", "exec", pod, "--", "redis-cli", "DBSIZE"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return int(out.strip())
    except Exception:
        return None


class Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ok = defaultdict(int)
        self.errors = defaultdict(int)
        self.lat_ms = defaultdict(list)
        self.error_samples = defaultdict(list)

    def add_ok(self, model: str, lat_ms: float) -> None:
        with self.lock:
            self.ok[model] += 1
            self.lat_ms[model].append(lat_ms)

    def add_error(self, model: str, error: str) -> None:
        with self.lock:
            self.errors[model] += 1
            if len(self.error_samples[model]) < 10:
                self.error_samples[model].append(error[:300])


def worker(
    stop: threading.Event,
    active_model: list[str],
    counters: Counters,
    *,
    gateway_url: str,
    max_tokens: int,
    request_timeout: int,
) -> None:
    prompt = "Return exactly one short sentence about capacity planning."
    while not stop.is_set():
        model = active_model[0]
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        started = time.perf_counter()
        try:
            http_json("POST", gateway_url, payload, headers={"model": model}, timeout=request_timeout)
            counters.add_ok(model, (time.perf_counter() - started) * 1000.0)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception) as exc:
            counters.add_error(model, repr(exc))
            time.sleep(0.25)


def summarize_latencies(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "avg": None, "p95": None, "max": None}
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(values),
        "min": round(ordered[0], 2),
        "avg": round(statistics.fmean(ordered), 2),
        "p95": round(ordered[idx], 2),
        "max": round(ordered[-1], 2),
    }


def run_precheck(args: argparse.Namespace) -> dict:
    if not args.models:
        raise ValueError("at least one model is required")
    start = time.time()
    active_model = [args.models[0]]
    counters = Counters()
    stop = threading.Event()
    samples = []
    phases = []
    initial_restarts = {
        "tre-v2": pod_restarts("tre-v2"),
        "default": pod_restarts("default"),
    }
    initial_state = http_json("GET", f"{args.service_manager_url}/v2/state")
    threads = [
        threading.Thread(
            target=worker,
            args=(stop, active_model, counters),
            kwargs={
                "gateway_url": args.gateway_url,
                "max_tokens": args.max_tokens,
                "request_timeout": args.request_timeout,
            },
            daemon=True,
        )
        for _ in range(args.workers)
    ]
    for thread in threads:
        thread.start()

    try:
        next_sample = start
        while time.time() - start < args.duration_seconds:
            elapsed = time.time() - start
            phase_idx = int(elapsed // args.phase_seconds)
            model = args.models[phase_idx % len(args.models)]
            if active_model[0] != model:
                active_model[0] = model
                phases.append({"elapsed_s": round(elapsed, 1), "model": model})
            if time.time() >= next_sample:
                state = http_json("GET", f"{args.service_manager_url}/v2/state")
                samples.append(
                    {
                        "elapsed_s": round(elapsed, 1),
                        "active": active_model[0],
                        "state": state,
                        "controller_rss_kb": rss_kb("tre-v2", "app.kubernetes.io/name=tre-v2-controller"),
                        "service_manager_rss_kb": rss_kb("tre-v2", "app.kubernetes.io/name=tre-v2-service-manager"),
                        "redis_dbsize": redis_dbsize(),
                    }
                )
                next_sample += args.sample_seconds
            time.sleep(1)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)

    final_state = http_json("GET", f"{args.service_manager_url}/v2/state")
    final_restarts = {
        "tre-v2": pod_restarts("tre-v2"),
        "default": pod_restarts("default"),
    }
    result = {
        "started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "duration_s": round(time.time() - start, 1),
        "gateway_url": args.gateway_url,
        "service_manager_url": args.service_manager_url,
        "models": list(args.models),
        "workers": args.workers,
        "phase_seconds": args.phase_seconds,
        "max_tokens": args.max_tokens,
        "sample_seconds": args.sample_seconds,
        "initial_state": initial_state,
        "final_state": final_state,
        "phases": phases,
        "samples": samples,
        "ok": dict(counters.ok),
        "errors": dict(counters.errors),
        "error_samples": dict(counters.error_samples),
        "latency_ms": {model: summarize_latencies(counters.lat_ms[model]) for model in args.models},
        "initial_restarts": initial_restarts,
        "final_restarts": final_restarts,
    }


def main(argv: list[str] | None = None) -> int:
    result = run_precheck(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if sum(result["errors"].values()) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
