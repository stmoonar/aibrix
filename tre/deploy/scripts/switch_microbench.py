#!/usr/bin/env python3
"""Measure exact-binding vLLM sleep/wake latency through the service-manager API."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


COLUMNS = [
    "model", "serve_id", "node", "gpu_ids", "cycle_idx", "direction",
    "cold", "t_api_start", "t_api_return", "t_log_marker", "t_ready",
    "dur_api_s", "dur_engine_s", "dur_e2e_s", "mem_before_mb",
    "mem_after_mb", "inflight_load", "inflight_total", "inflight_errors",
    "inflight_p99_ms",
]
MARKERS = {
    "sleep": re.compile(r"It took ([0-9.]+) seconds to fall asleep\."),
    "wake": re.compile(r"It took ([0-9.]+) seconds to wake up"),
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
GPU_RE = re.compile(
    r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?),\s*(\d+),\s*(\d+)"
)
ORPHAN_KEY = "tre:v2:controller:alerts:hidden_orphans"


@dataclass(frozen=True)
class Target:
    model: str
    serve_id: str
    node: str
    gpu_ids: tuple[int, ...]
    pod_ip: str


@dataclass
class TimedRow:
    row: dict[str, Any]
    start_epoch: float
    ready_epoch: float


@dataclass(frozen=True)
class GpuSample:
    epoch: float
    gpu_id: int
    used_mb: int


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_epoch(value: str) -> float:
    value = re.sub(r"(\.\d{6})\d+(?=Z|[+-])", r"\1", value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def parse_engine_marker(
    text: str,
    direction: str,
    *,
    node_clock_offset_s: float,
    not_before_epoch: float,
) -> tuple[float, float, str] | None:
    pattern = MARKERS[direction]
    candidates: list[tuple[float, float, str]] = []
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line)
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            raw_epoch = parse_iso_epoch(parts[0])
        except ValueError:
            continue
        match = pattern.search(parts[1])
        if match is None:
            continue
        normalized_epoch = raw_epoch - node_clock_offset_s
        if normalized_epoch >= not_before_epoch:
            candidates.append((normalized_epoch, float(match.group(1)), raw_line))
    return candidates[-1] if candidates else None


def parse_gpu_samples(
    text: str,
    *,
    utc_offset: timezone,
    node_clock_offset_s: float,
) -> list[GpuSample]:
    samples: list[GpuSample] = []
    for line in text.splitlines():
        match = GPU_RE.match(line.strip())
        if match is None:
            continue
        local_dt = datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S.%f")
        raw_epoch = local_dt.replace(tzinfo=utc_offset).timestamp()
        samples.append(
            GpuSample(
                epoch=raw_epoch - node_clock_offset_s,
                gpu_id=int(match.group(2)),
                used_mb=int(match.group(3)),
            )
        )
    return samples


def transition_memory(
    samples: list[GpuSample],
    gpu_ids: tuple[int, ...],
    *,
    start_epoch: float,
    ready_epoch: float,
) -> tuple[int | None, int | None]:
    grouped: dict[float, dict[int, int]] = {}
    for sample in samples:
        grouped.setdefault(sample.epoch, {})[sample.gpu_id] = sample.used_mb
    complete = [
        (epoch, values)
        for epoch, values in grouped.items()
        if all(gpu_id in values for gpu_id in gpu_ids)
    ]
    before = [item for item in complete if item[0] <= start_epoch]
    after = [item for item in complete if item[0] >= ready_epoch]
    before_values = max(before, default=None, key=lambda item: item[0])
    after_values = min(after, default=None, key=lambda item: item[0])
    before_mb = (
        sum(before_values[1][gpu_id] for gpu_id in gpu_ids)
        if before_values else None
    )
    after_mb = (
        sum(after_values[1][gpu_id] for gpu_id in gpu_ids)
        if after_values else None
    )
    return before_mb, after_mb


def summarize_inflight(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    errors = sum(1 for row in rows if not row["ok"])
    p99 = percentile_nearest_rank(
        [float(row["latency_ms"]) for row in rows if row["ok"]], 0.99
    )
    return {
        "inflight_total": len(rows),
        "inflight_errors": errors,
        "inflight_p99_ms": round(p99, 3) if p99 is not None else None,
    }


def run_json(command: list[str]) -> dict[str, Any]:
    return json.loads(subprocess.check_output(command, text=True))


def http_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
    response.raise_for_status()
    return response.json()


def measure_clock_offset(host: str, count: int = 5) -> dict[str, Any]:
    samples = []
    for _ in range(count):
        start = time.time()
        remote = float(subprocess.check_output(
            ["ssh", host, "date +%s.%N"], text=True
        ).strip())
        end = time.time()
        samples.append({
            "start": start,
            "end": end,
            "rtt_s": end - start,
            "offset_s": remote - ((start + end) / 2.0),
        })
    offset = statistics.median(item["offset_s"] for item in samples)
    return {"host": host, "offset_s": offset, "samples": samples}


def resolve_targets(
    sm_state: dict[str, Any],
    requested: dict[str, str],
    *,
    namespace: str,
    expected_node: str,
) -> list[Target]:
    bindings = {item["serve_id"]: item for item in sm_state["bindings"]}
    targets = []
    for model, serve_id in requested.items():
        binding = bindings.get(serve_id)
        if binding is None or binding["model"] != model:
            raise ValueError(f"binding {serve_id} does not belong to {model}")
        if binding["node"] != expected_node:
            raise ValueError(f"binding {serve_id} is on {binding['node']}, not {expected_node}")
        if not binding["awake"] or binding["hidden"]:
            raise ValueError(f"binding {serve_id} must start awake and routable")
        pod = run_json(["kubectl", "-n", namespace, "get", "pod", serve_id, "-o", "json"])
        pod_ip = pod.get("status", {}).get("podIP")
        if not pod_ip:
            raise ValueError(f"pod {serve_id} has no pod IP")
        targets.append(Target(
            model=model,
            serve_id=serve_id,
            node=binding["node"],
            gpu_ids=tuple(int(value) for value in binding["gpu_ids"]),
            pod_ip=pod_ip,
        ))
    return targets


def binding_layout(state: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("serve_id", "model", "node", "gpu_ids", "awake", "hidden")
    return [
        {key: binding[key] for key in keys}
        for binding in sorted(state["bindings"], key=lambda item: item["serve_id"])
    ]


class GpuSampler:
    def __init__(self, host: str, output: Path) -> None:
        self.host = host
        self.output = output
        self._stream = None
        self._process = None

    def start(self) -> None:
        self._stream = self.output.open("w", encoding="utf-8", newline="")
        command = (
            "exec stdbuf -oL nvidia-smi "
            "--query-gpu=timestamp,index,memory.used "
            "--format=csv,noheader,nounits -l 1"
        )
        self._process = subprocess.Popen(
            ["ssh", self.host, command],
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1.2)
        if self._process.poll() is not None:
            raise RuntimeError("nvidia-smi sampler exited during startup")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._stream is not None:
            self._stream.close()


class InflightLoad:
    def __init__(self, gateway: str, model: str, cycle_idx: int, rps: float) -> None:
        self.gateway = gateway
        self.model = model
        self.cycle_idx = cycle_idx
        self.interval_s = 1.0 / rps
        self.rows: list[dict[str, Any]] = []
        self._rows_lock = threading.Lock()
        self._stop = threading.Event()
        self._scheduler: threading.Thread | None = None
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        self._scheduler = threading.Thread(target=self._schedule, daemon=True)
        self._scheduler.start()

    def _schedule(self) -> None:
        deadline = time.perf_counter()
        request_idx = 0
        while not self._stop.is_set():
            worker = threading.Thread(
                target=self._request, args=(request_idx,), daemon=True
            )
            self._workers.append(worker)
            worker.start()
            request_idx += 1
            deadline += self.interval_s
            self._stop.wait(max(0.0, deadline - time.perf_counter()))

    def _request(self, request_idx: int) -> None:
        started = time.time()
        ok = False
        status = None
        error = ""
        try:
            response = requests.post(
                self.gateway,
                headers={"Content-Type": "application/json", "model": self.model},
                json={
                    "model": self.model,
                    "prompt": "microbenchmark probe",
                    "max_tokens": 1,
                    "temperature": 0,
                },
                timeout=15,
            )
            status = response.status_code
            ok = response.status_code == 200
            if not ok:
                error = response.text[:200]
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ended = time.time()
        row = {
            "model": self.model,
            "cycle_idx": self.cycle_idx,
            "request_idx": request_idx,
            "t_start": iso_utc(started),
            "t_end": iso_utc(ended),
            "latency_ms": round((ended - started) * 1000.0, 3),
            "ok": ok,
            "status": status,
            "error": error,
        }
        with self._rows_lock:
            self.rows.append(row)

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._scheduler is not None:
            self._scheduler.join(timeout=5)
        for worker in self._workers:
            worker.join(timeout=20)
        return sorted(self.rows, key=lambda row: row["request_idx"])


def wait_ready(target: Target, awake: bool, timeout_s: float) -> float:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if not awake:
                state = http_json("GET", f"http://{target.pod_ip}:8000/is_sleeping", timeout=2)
                if state.get("is_sleeping") is True:
                    return time.time()
            else:
                response = requests.post(
                    f"http://{target.pod_ip}:8000/v1/completions",
                    json={
                        "model": target.model,
                        "prompt": "ready",
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                    timeout=3,
                )
                if response.status_code == 200:
                    return time.time()
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"readiness timeout for {target.serve_id} awake={awake}")


def find_marker(
    target: Target,
    direction: str,
    *,
    namespace: str,
    node_clock_offset_s: float,
    start_epoch: float,
    excerpt_path: Path,
) -> tuple[float, float]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        text = subprocess.check_output(
            ["kubectl", "-n", namespace, "logs", target.serve_id,
             "--timestamps", "--since=10m"],
            text=True,
            errors="replace",
        )
        marker = parse_engine_marker(
            text,
            direction,
            node_clock_offset_s=node_clock_offset_s,
            not_before_epoch=start_epoch - 2.0,
        )
        if marker is not None:
            marker_epoch, engine_s, raw_line = marker
            with excerpt_path.open("a", encoding="utf-8") as stream:
                stream.write(raw_line.rstrip() + "\n")
            return marker_epoch, engine_s
        time.sleep(0.2)
    raise TimeoutError(f"missing {direction} log marker for {target.serve_id}")


def run_transition(
    target: Target,
    *,
    awake: bool,
    cycle_idx: int,
    sm_url: str,
    namespace: str,
    node_clock_offset_s: float,
    excerpt_path: Path,
    timeout_s: float,
) -> TimedRow:
    direction = "wake" if awake else "sleep"
    start = time.time()
    response = http_json(
        "PUT",
        f"{sm_url}/v2/bindings/{target.serve_id}/power",
        json={"awake": awake},
        timeout=timeout_s,
    )
    returned = time.time()
    expected = [{"action": direction, "serve_id": target.serve_id}]
    if response.get("actions") != expected:
        raise RuntimeError(f"unexpected SM response: {response}")
    ready = wait_ready(target, awake, timeout_s)
    marker, engine_s = find_marker(
        target,
        direction,
        namespace=namespace,
        node_clock_offset_s=node_clock_offset_s,
        start_epoch=start,
        excerpt_path=excerpt_path,
    )
    row = {
        "model": target.model,
        "serve_id": target.serve_id,
        "node": target.node,
        "gpu_ids": ";".join(str(value) for value in target.gpu_ids),
        "cycle_idx": cycle_idx,
        "direction": direction,
        "cold": cycle_idx == 1,
        "t_api_start": iso_utc(start),
        "t_api_return": iso_utc(returned),
        "t_log_marker": iso_utc(marker),
        "t_ready": iso_utc(ready),
        "dur_api_s": round(returned - start, 6),
        "dur_engine_s": round(engine_s, 6),
        "dur_e2e_s": round(ready - start, 6),
        "mem_before_mb": None,
        "mem_after_mb": None,
        "inflight_load": False,
        "inflight_total": 0,
        "inflight_errors": 0,
        "inflight_p99_ms": None,
    }
    return TimedRow(row=row, start_epoch=start, ready_epoch=ready)


def parse_mapping(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use KEY=VALUE: {value}")
        key, mapped = value.split("=", 1)
        result[key] = mapped
    return result


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def redis_hlen(redis_url: str) -> int:
    import redis

    return int(redis.Redis.from_url(redis_url).hlen(ORPHAN_KEY))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", action="append", required=True,
                        help="MODEL=SERVE_ID; repeat once per model")
    parser.add_argument("--sm-url", default="http://10.97.239.243:8000")
    parser.add_argument("--gateway", default="http://192.168.223.76:31094/v1/completions")
    parser.add_argument("--redis-url", default="redis://10.109.196.83:6379/0")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--controller-namespace", default="tre-v2")
    parser.add_argument("--controller-deployment", default="tre-v2-controller")
    parser.add_argument("--expected-node", default="nscc-ds-4a100-node9")
    parser.add_argument("--node-ssh", default="192.168.223.75")
    parser.add_argument("--node-utc-offset", default="+08:00")
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--load-start-cycle", type=int, default=21)
    parser.add_argument("--load-rps", type=float, default=2.0)
    parser.add_argument("--load-pre-s", type=float, default=30.0)
    parser.add_argument("--load-post-s", type=float, default=30.0)
    parser.add_argument("--transition-timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = parse_mapping(args.target, "target")
    state_before = http_json("GET", f"{args.sm_url}/v2/state")
    targets = resolve_targets(
        state_before, requested, namespace=args.namespace,
        expected_node=args.expected_node,
    )
    if len(targets) != 3:
        raise ValueError("exactly three model targets are required")
    for target in targets:
        counts = state_before["models"].get(target.model, {})
        if counts.get("awake") != 1:
            raise ValueError(f"{target.model} must have exactly one awake binding")
    (output / "targets.json").write_text(
        json.dumps(
            [
                {
                    **target.__dict__,
                    "gpu_ids": list(target.gpu_ids),
                    "vllm": http_json("GET", f"http://{target.pod_ip}:8000/version"),
                }
                for target in targets
            ],
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output / "sm_state_before.json").write_text(
        json.dumps(state_before, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    clock = measure_clock_offset(args.node_ssh)
    (output / "clock_sync.json").write_text(
        json.dumps(clock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sign = 1 if args.node_utc_offset[0] == "+" else -1
    hours, minutes = (int(value) for value in args.node_utc_offset[1:].split(":"))
    node_tz = timezone(sign * timedelta(hours=hours, minutes=minutes))

    orphan_before = redis_hlen(args.redis_url)
    if orphan_before != 0:
        raise RuntimeError(f"orphan alert precheck failed: {orphan_before}")

    session_started = time.time()
    gpu_log = output / "nvidia_smi_node9.csv"
    sampler = GpuSampler(args.node_ssh, gpu_log)
    timed_rows: list[TimedRow] = []
    inflight_rows: list[dict[str, Any]] = []
    sampler.start()
    try:
        for target in targets:
            excerpt = output / f"pod_log_{target.serve_id}.txt"
            for cycle_idx in range(1, args.cycles + 1):
                load = None
                if cycle_idx >= args.load_start_cycle:
                    load = InflightLoad(args.gateway, target.model, cycle_idx, args.load_rps)
                    load.start()
                    time.sleep(args.load_pre_s)
                pair = []
                try:
                    pair.append(run_transition(
                        target, awake=False, cycle_idx=cycle_idx,
                        sm_url=args.sm_url, namespace=args.namespace,
                        node_clock_offset_s=clock["offset_s"],
                        excerpt_path=excerpt, timeout_s=args.transition_timeout_s,
                    ))
                    pair.append(run_transition(
                        target, awake=True, cycle_idx=cycle_idx,
                        sm_url=args.sm_url, namespace=args.namespace,
                        node_clock_offset_s=clock["offset_s"],
                        excerpt_path=excerpt, timeout_s=args.transition_timeout_s,
                    ))
                    if load is not None:
                        time.sleep(args.load_post_s)
                finally:
                    if load is not None:
                        current = load.stop()
                        inflight_rows.extend(current)
                        metrics = summarize_inflight(current)
                        for item in pair:
                            item.row.update(metrics)
                            item.row["inflight_load"] = True
                timed_rows.extend(pair)
    finally:
        sampler.stop()
        current_state = http_json("GET", f"{args.sm_url}/v2/state")
        by_id = {item["serve_id"]: item for item in current_state["bindings"]}
        for target in targets:
            binding = by_id.get(target.serve_id)
            if binding is not None and not binding["awake"]:
                http_json(
                    "PUT", f"{args.sm_url}/v2/bindings/{target.serve_id}/power",
                    json={"awake": True}, timeout=args.transition_timeout_s,
                )

    samples = parse_gpu_samples(
        gpu_log.read_text(encoding="utf-8", errors="replace"),
        utc_offset=node_tz,
        node_clock_offset_s=clock["offset_s"],
    )
    target_by_id = {target.serve_id: target for target in targets}
    for item in timed_rows:
        target = target_by_id[item.row["serve_id"]]
        before_mb, after_mb = transition_memory(
            samples, target.gpu_ids,
            start_epoch=item.start_epoch, ready_epoch=item.ready_epoch,
        )
        item.row["mem_before_mb"] = before_mb
        item.row["mem_after_mb"] = after_mb

    write_csv(output / "cycles.csv", COLUMNS, [item.row for item in timed_rows])
    inflight_columns = [
        "model", "cycle_idx", "request_idx", "t_start", "t_end",
        "latency_ms", "ok", "status", "error",
    ]
    write_csv(output / "inflight_requests.csv", inflight_columns, inflight_rows)
    state_after = http_json("GET", f"{args.sm_url}/v2/state")
    (output / "sm_state_after.json").write_text(
        json.dumps(state_after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    controller_log = subprocess.check_output(
        [
            "kubectl", "-n", args.controller_namespace, "logs",
            f"deployment/{args.controller_deployment}",
            f"--since-time={iso_utc(session_started - 1.0)}",
        ],
        text=True,
        errors="replace",
    )
    (output / "controller_session.log").write_text(controller_log, encoding="utf-8")
    orphan_log_events = controller_log.count("TRE_ORPHAN_HIDDEN")
    orphan_after = redis_hlen(args.redis_url)
    layout_restored = binding_layout(state_after) == binding_layout(state_before)
    summary = {
        "models": len(targets),
        "cycles_per_model": args.cycles,
        "complete_cycles": len(timed_rows) // 2,
        "transition_rows": len(timed_rows),
        "orphan_alerts_before": orphan_before,
        "orphan_alerts_after": orphan_after,
        "orphan_log_events": orphan_log_events,
        "baseline_layout_restored": layout_restored,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected_cycles = len(targets) * args.cycles
    if len(timed_rows) != expected_cycles * 2:
        raise RuntimeError(f"incomplete transitions: {len(timed_rows)}")
    if orphan_after != 0 or orphan_log_events != 0:
        raise RuntimeError(
            f"orphan alerts fired: redis={orphan_after} log_events={orphan_log_events}"
        )
    if not layout_restored:
        raise RuntimeError("service-manager binding layout did not return to baseline")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())