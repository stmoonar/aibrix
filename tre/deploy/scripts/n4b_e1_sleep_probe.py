#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def parse_gpu_memory(text: str) -> dict[str, list[dict[str, int | str]]]:
    rows: dict[str, list[dict[str, int | str]]] = {}
    for raw_line in text.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid, used_raw = parts
        try:
            used_mib = int(used_raw.removesuffix(" MiB").strip())
        except ValueError:
            continue
        rows.setdefault(uuid, []).append({"pid": pid, "used_mib": used_mib})
    return rows


def summarize_used_mib(rows: dict[str, list[dict[str, int | str]]], gpu_uuid: str) -> dict[str, int]:
    selected = rows.get(gpu_uuid, [])
    return {
        "processes": len(selected),
        "used_mib": sum(int(row["used_mib"]) for row in selected),
    }


def summarize_many_gpus(rows: dict[str, list[dict[str, int | str]]], gpu_uuids: list[str]) -> dict[str, Any]:
    per_gpu = {gpu_uuid: summarize_used_mib(rows, gpu_uuid) for gpu_uuid in gpu_uuids}
    return {
        "total_used_mib": sum(item["used_mib"] for item in per_gpu.values()),
        "gpus": per_gpu,
    }


def percentile95(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def run_request_batch(*, total: int, concurrency: int, sender) -> None:
    if total <= 0:
        return
    workers = max(1, min(int(concurrency), int(total)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(sender, idx) for idx in range(total)]
        for future in as_completed(futures):
            future.result()


def run(cmd: list[str], *, timeout_s: int = 120) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout_s)


def ssh(host: str, command: str, *, timeout_s: int = 120) -> str:
    return run(["ssh", host, command], timeout_s=timeout_s)


def kubectl(kubectl_host: str, args: str, *, timeout_s: int = 120) -> str:
    return ssh(kubectl_host, f"kubectl {args}", timeout_s=timeout_s)


def http_json(kubectl_host: str, method: str, url: str, payload: dict[str, Any] | None = None, *, timeout_s: int = 60) -> Any:
    body = "None" if payload is None else repr(json.dumps(payload).encode("utf-8"))
    script = f"""
import json, urllib.request
data = {body}
req = urllib.request.Request({url!r}, data=data, method={method!r})
req.add_header('content-type', 'application/json')
with urllib.request.urlopen(req, timeout={timeout_s}) as resp:
    raw = resp.read()
    print(raw.decode('utf-8') if raw else 'null')
"""
    out = ssh(kubectl_host, "python3 - <<'PY'\n" + script + "\nPY", timeout_s=timeout_s + 10)
    return json.loads(out)


def parse_gpu_uuids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def sample_gpu(node_host: str, gpu_uuids: list[str]) -> dict[str, Any]:
    text = ssh(
        node_host,
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits",
        timeout_s=30,
    )
    rows = parse_gpu_memory(text)
    summary = summarize_many_gpus(rows, gpu_uuids)
    summary["raw"] = {gpu_uuid: rows.get(gpu_uuid, []) for gpu_uuid in gpu_uuids}
    return summary


def wait_http_ready(kubectl_host: str, pod_ip: str, *, timeout_s: int = 240) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            http_json(kubectl_host, "GET", f"http://{pod_ip}:8000/is_sleeping", timeout_s=5)
            return
        except Exception as exc:  # noqa: BLE001 - operational probe records final error.
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"vLLM HTTP readiness timed out for {pod_ip}: {last_error}")


def post_vllm(kubectl_host: str, pod_ip: str, path: str) -> Any:
    return http_json(kubectl_host, "POST", f"http://{pod_ip}:8000{path}", timeout_s=120)


def send_completion(kubectl_host: str, pod_ip: str, model: str, *, max_tokens: int = 32) -> Any:
    return http_json(
        kubectl_host,
        "POST",
        f"http://{pod_ip}:8000/v1/completions",
        {"model": model, "prompt": "Return one short sentence about GPU memory.", "max_tokens": max_tokens, "temperature": 0},
        timeout_s=120,
    )


def get_pod(kubectl_host: str, app: str) -> dict[str, str]:
    data = json.loads(kubectl(kubectl_host, f"-n default get pod -l app={app} -o json", timeout_s=60))
    items = data.get("items", [])
    if len(items) != 1:
        raise RuntimeError(f"expected one pod for app={app}, got {len(items)}")
    item = items[0]
    return {"name": item["metadata"]["name"], "ip": item["status"].get("podIP", "")}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    gpu_uuids = parse_gpu_uuids(args.gpu_uuid)
    kubectl(args.kubectl_host, f"apply -f {args.manifest}", timeout_s=60)
    kubectl(args.kubectl_host, f"-n default rollout status deploy/{args.deployment} --timeout=600s", timeout_s=660)
    pod = get_pod(args.kubectl_host, args.deployment)
    wait_http_ready(args.kubectl_host, pod["ip"], timeout_s=args.ready_timeout_s)

    samples: list[dict[str, Any]] = [{"event": "ready", "gpu": sample_gpu(args.node_host, gpu_uuids)}]

    if args.initial_sleep:
        post_vllm(args.kubectl_host, pod["ip"], f"/sleep?level={args.sleep_level}")
        time.sleep(args.sleep_wait_s)
        samples.append({"event": "initial_sleep", "gpu": sample_gpu(args.node_host, gpu_uuids)})

    sleep_used: list[float] = []
    for round_idx in range(args.rounds):
        post_vllm(args.kubectl_host, pod["ip"], "/wake_up")
        time.sleep(args.wake_wait_s)
        samples.append({"event": f"round_{round_idx}_wake", "gpu": sample_gpu(args.node_host, gpu_uuids)})
        run_request_batch(
            total=args.requests,
            concurrency=args.concurrency,
            sender=lambda _idx: send_completion(args.kubectl_host, pod["ip"], args.model, max_tokens=args.max_tokens),
        )
        post_vllm(args.kubectl_host, pod["ip"], f"/sleep?level={args.sleep_level}")
        time.sleep(args.sleep_wait_s)
        sleep_sample = sample_gpu(args.node_host, gpu_uuids)
        samples.append({"event": f"round_{round_idx}_sleep", "gpu": sleep_sample})
        sleep_used.append(float(sleep_sample["total_used_mib"]))

    return {
        "deployment": args.deployment,
        "model": args.model,
        "pod": pod,
        "gpu_uuid": args.gpu_uuid,
        "gpu_uuids": gpu_uuids,
        "rounds": args.rounds,
        "requests_per_round": args.requests,
        "concurrency": args.concurrency,
        "sleep_level": args.sleep_level,
        "duration_s": round(time.time() - started, 3),
        "sleep_used_mib_p95": percentile95(sleep_used),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N4b E1 vLLM sleep leak probe.")
    parser.add_argument("--kubectl-host", default="A100_76")
    parser.add_argument("--node-host", default="A100_75")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--sleep-level", type=int, default=1)
    parser.add_argument("--initial-sleep", action="store_true")
    parser.add_argument("--ready-timeout-s", type=int, default=300)
    parser.add_argument("--wake-wait-s", type=float, default=2.0)
    parser.add_argument("--sleep-wait-s", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run_probe(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
