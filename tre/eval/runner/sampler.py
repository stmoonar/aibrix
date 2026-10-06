#!/usr/bin/env python3
"""Smoke-E1 sampler: SM layout (1 s), APA CR status + per-pod vLLM gauges (5 s).
Runs until <out>/STOP exists. Read-only against the cluster."""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

OUT = sys.argv[1]
SM = sys.argv[2]
STOP = os.path.join(OUT, "STOP")
GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:num_requests_paused")


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def layout_loop():
    with open(os.path.join(OUT, "layout.jsonl"), "a") as f:
        while not os.path.exists(STOP):
            t = time.time()
            try:
                s = get_json(SM + "/v2/state")
                row = {"ts": t, "version": s.get("version"), "models": {}}
                for b in s["bindings"]:
                    m = row["models"].setdefault(b["model"], {"awake": [], "hidden": []})
                    if b["awake"]:
                        m["awake"].append(b["binding_id"])
                    if b["hidden"]:
                        m["hidden"].append(b["binding_id"])
                f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"ts": t, "error": str(e)[:200]}) + "\n")
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


def pods():
    raw = subprocess.check_output(["kubectl", "-n", "default", "get", "pods", "-l",
                                   "tre.aibrix.io/managed=true", "-o", "json"], text=True)
    out = {}
    for p in json.loads(raw)["items"]:
        out[p["metadata"]["name"]] = {
            "ip": p["status"].get("podIP"),
            "model": p["metadata"]["labels"].get("model.aibrix.ai/name"),
            "routable": p["metadata"]["labels"].get("tre.aibrix.io/routable"),
        }
    return out


def scrape(ip):
    vals = {}
    with urllib.request.urlopen(f"http://{ip}:8000/metrics", timeout=3) as r:
        for line in r.read().decode().splitlines():
            for g in GAUGES:
                if line.startswith(g + "{"):
                    vals[g.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return vals


def slow_loop():
    fk = open(os.path.join(OUT, "pod_gauges.jsonl"), "a")
    fa = open(os.path.join(OUT, "apa_status.jsonl"), "a")
    while not os.path.exists(STOP):
        t = time.time()
        try:
            ps = pods()
            rows = {}
            for name, p in ps.items():
                if p["routable"] != "true" or not p["ip"]:
                    continue
                try:
                    rows[name] = {"model": p["model"], **scrape(p["ip"])}
                except Exception as e:  # noqa: BLE001
                    rows[name] = {"model": p["model"], "error": str(e)[:100]}
            fk.write(json.dumps({"ts": t, "pods": rows}) + "\n")
            fk.flush()
        except Exception as e:  # noqa: BLE001
            fk.write(json.dumps({"ts": t, "error": str(e)[:200]}) + "\n")
        try:
            raw = subprocess.run(["kubectl", "-n", "default", "get",
                                  "podautoscalers.autoscaling.aibrix.ai", "-o", "json"],
                                 text=True, capture_output=True, timeout=20).stdout
            items = json.loads(raw)["items"] if raw else []
            fa.write(json.dumps({"ts": t, "pa": {
                i["metadata"]["name"]: {k: i.get("status", {}).get(k) for k in
                                        ("desiredScale", "actualScale", "lastScaleTime")}
                for i in items}}) + "\n")
            fa.flush()
        except Exception as e:  # noqa: BLE001
            fa.write(json.dumps({"ts": t, "error": str(e)[:200]}) + "\n")
        time.sleep(max(0.0, 5.0 - (time.time() - t)))


threads = [threading.Thread(target=layout_loop), threading.Thread(target=slow_loop)]
for th in threads:
    th.start()
for th in threads:
    th.join()
