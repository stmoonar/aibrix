from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

import requests


GATEWAY_URL = "http://10.99.21.145/v1/completions"
SM_DEFRAG_URL = "http://10.111.21.116:8000/v2/defrag"
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
PROMPTS = {
    "dsqwen-7b": "qwen health check",
    "dsllama-8b": "llama health check",
    "dsqwen-14b": "qwen fourteen health check",
}


def request_model(model: str, *, timeout: float = 10.0) -> tuple[bool, str]:
    payload = {
        "model": model,
        "prompt": PROMPTS[model],
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        response = requests.post(GATEWAY_URL, json=payload, headers={"model": model}, timeout=timeout)
    except Exception as exc:
        return False, f"exception:{type(exc).__name__}:{exc}"
    if 200 <= response.status_code < 300:
        return True, f"http:{response.status_code}"
    return False, f"http:{response.status_code}:{response.text[:240]}"


def main() -> None:
    evidence = Path("/data/nfs_shared_data/xxy/aibrix/docs/refactor/p11_evidence/f3_live_defrag_route_guard_20260705")
    stop = threading.Event()
    results = {
        model: {
            "ok": 0,
            "errors": 0,
            "error_counts": Counter(),
            "samples": [],
        }
        for model in MODELS
    }
    lock = threading.Lock()

    smoke = {}
    for model in MODELS:
        model_results = [request_model(model) for _ in range(5)]
        smoke[model] = {
            "ok": sum(1 for ok, _ in model_results if ok),
            "errors": [message for ok, message in model_results if not ok],
        }
    (evidence / "pre_defrag_gateway_smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))
    if any(item["errors"] for item in smoke.values()):
        raise SystemExit("pre-defrag smoke failed")

    def worker(model: str) -> None:
        while not stop.is_set():
            ok, message = request_model(model)
            with lock:
                bucket = results[model]
                if ok:
                    bucket["ok"] += 1
                else:
                    bucket["errors"] += 1
                    bucket["error_counts"][message] += 1
                    if len(bucket["samples"]) < 20:
                        bucket["samples"].append({"t": time.time(), "message": message})

    threads = [threading.Thread(target=worker, args=(model,), daemon=True) for model in MODELS for _ in range(2)]
    started_at = time.time()
    for thread in threads:
        thread.start()
    time.sleep(2.0)
    try:
        defrag_started_at = time.time()
        response = requests.post(SM_DEFRAG_URL, json={"tp_size": 2}, timeout=600)
        defrag_finished_at = time.time()
        defrag = {
            "status_code": response.status_code,
            "body": _json_or_text(response),
            "started_at": defrag_started_at,
            "finished_at": defrag_finished_at,
            "duration_s": defrag_finished_at - defrag_started_at,
        }
        time.sleep(8.0)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    finished_at = time.time()

    serializable_results = {}
    for model, bucket in results.items():
        serializable_results[model] = {
            "ok": bucket["ok"],
            "errors": bucket["errors"],
            "error_counts": dict(bucket["error_counts"]),
            "samples": bucket["samples"],
        }
    output = {
        "gateway_url": GATEWAY_URL,
        "models": MODELS,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": finished_at - started_at,
        "defrag": defrag,
        "results": serializable_results,
    }
    (evidence / "defrag_with_header_probes.json").write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


def _json_or_text(response):
    try:
        return response.json()
    except Exception:
        return response.text


if __name__ == "__main__":
    main()
