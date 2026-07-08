#!/usr/bin/env python3
"""S1.2 authoritative W-freeze measurement (ADR-0012 / plan15 §1.2).

Drives step loads against single-replica models and measures the lag from load
onset to the first controller decision entry whose z_m becomes non-null (i.e. the
shared TSS signal first *reflects* the added load). Controller stays in observe
mode; nothing on the fleet is mutated. Reads the per-model decision history zset
tre:v2:decision:hist:<model> from tre-v2-redis (same node10 host clock as the
load driver, so onset and decision ts share one clock).

Metric (apples-to-apples with the pre-change 60-120s number, which was the
first-reflection lag under the old 60s tumbling window):
    lag_first = t(first decision entry after onset with z_m != null) - t(onset)
Secondary context recorded per trial: state-elevation lag, warm-transition lag,
settle-to-plateau lag, load client stats.
"""
from __future__ import annotations
import json, subprocess, sys, threading, time, urllib.request

GATEWAY = "http://192.168.223.76:31592/v1/completions"
REDIS_POD = "tre-v2-redis-6c47ddfb8d-zc46g"
NS = "tre-v2"
MODELS = ["dsqwen-7b", "dsllama-8b", "dsqwen-14b"]
ROUNDS = 4
WORKERS = 20
MAX_TOKENS = 128
INPUT_TOKENS = 64
LOAD_DUR = 60.0
COOLDOWN_MIN = 45.0          # min seconds of idle-drain after load stops
IDLE_GUARD_TIMEOUT = 120.0   # max wait for model to return to idle before a trial

def now_ms() -> int:
    return int(time.time() * 1000)

def redis_zrange_by_score(key: str, lo, hi) -> list[str]:
    out = subprocess.run(
        ["kubectl", "-n", NS, "exec", REDIS_POD, "--",
         "redis-cli", "ZRANGEBYSCORE", key, str(lo), str(hi)],
        capture_output=True, text=True, timeout=60)
    return [l for l in out.stdout.splitlines() if l.strip()]

def last_hist_entry(model: str):
    lines = redis_zrange_by_score(f"tre:v2:decision:hist:{model}", "(%d" % (now_ms() - 30000), "+inf")
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None

def wait_idle(model: str) -> dict:
    """Block until the model's latest decision entry is idle (z_m null, trs ~0)."""
    t0 = time.time()
    while True:
        e = last_hist_entry(model)
        idle = e is not None and e.get("z_m") is None and (e.get("trs") or 0) == 0
        if idle:
            return {"idle": True, "waited_s": round(time.time() - t0, 1), "last_ts": e.get("ts")}
        if time.time() - t0 > IDLE_GUARD_TIMEOUT:
            return {"idle": False, "waited_s": round(time.time() - t0, 1),
                    "last": e}
        time.sleep(5)

def run_load(model: str) -> dict:
    prompt = " ".join(["token"] * INPUT_TOKENS)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": 0, "ignore_eos": True}).encode()
    stop = threading.Event()
    lock = threading.Lock()
    ok = [0]; err = [0]; lats: list[float] = []; errs: dict[str, int] = {}
    def worker() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            req = urllib.request.Request(GATEWAY, data=body,
                headers={"Content-Type": "application/json", "model": model})
            try:
                urllib.request.urlopen(req, timeout=max(30.0, MAX_TOKENS / 4.0)).read()
                dt = (time.perf_counter() - t0) * 1000.0
                with lock:
                    ok[0] += 1; lats.append(dt)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    err[0] += 1
                    k = type(exc).__name__; errs[k] = errs.get(k, 0) + 1
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    onset = now_ms()
    for t in threads:
        t.start()
    time.sleep(LOAD_DUR)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    load_end = now_ms()
    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(len(lats) * q))] if lats else 0.0
    return {"onset_ms": onset, "load_end_ms": load_end, "ok": ok[0], "err": err[0],
            "rps": round(ok[0] / LOAD_DUR, 2), "lat_p50_ms": round(p(0.5), 1),
            "lat_p95_ms": round(p(0.95), 1), "errors": errs}

def main() -> int:
    trials = []
    tidx = 0
    for r in range(ROUNDS):
        for model in MODELS:
            tidx += 1
            print(f"[trial {tidx}] round {r+1} model {model}: waiting for idle...", flush=True)
            ig = wait_idle(model)
            print(f"           idle-guard: {ig}", flush=True)
            print(f"           firing {WORKERS}-worker step load for {LOAD_DUR:.0f}s...", flush=True)
            ld = run_load(model)
            print(f"           load: ok={ld['ok']} err={ld['err']} rps={ld['rps']} "
                  f"p50={ld['lat_p50_ms']}ms p95={ld['lat_p95_ms']}ms", flush=True)
            trials.append({"trial": tidx, "round": r + 1, "model": model,
                           "idle_guard": ig, **ld})
            print(f"           cooldown {COOLDOWN_MIN:.0f}s...", flush=True)
            time.sleep(COOLDOWN_MIN)
    # dump raw decision hist per model spanning the whole experiment
    lo = min(t["onset_ms"] for t in trials) - 40000
    hi = now_ms() + 5000
    raw = {}
    for model in MODELS:
        lines = redis_zrange_by_score(f"tre:v2:decision:hist:{model}", lo, hi)
        raw[model] = [json.loads(l) for l in lines]
    with open("/tmp/s1wf_trials.json", "w") as f:
        json.dump(trials, f, indent=2)
    with open("/tmp/s1wf_raw_hist.json", "w") as f:
        json.dump(raw, f)
    print("SAVED /tmp/s1wf_trials.json /tmp/s1wf_raw_hist.json", flush=True)
    print(f"trials={len(trials)} raw_entries=" +
          ",".join(f"{m}:{len(raw[m])}" for m in MODELS), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
