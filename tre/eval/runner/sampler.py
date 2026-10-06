#!/usr/bin/env python3
"""E1 sampler: read-only against the cluster, runs until <out>/STOP exists.

Usage: sampler.py <out_dir> <SM url> [<redis url>]      (settings from the environment, see
runner.env.example; the runner exports them)

Files (one JSON object per line, ``ts`` = this host's epoch seconds = the reference clock):

* ``layout.jsonl`` (SAMPLE_LAYOUT_S, 1 s): SM ``/v2/state`` per model ``awake`` / ``hidden``
  binding ids (as the pilot) plus ``routable`` (SM's own routable flag per binding) and
  ``models_sm`` (the SM per-model counts);
* ``gpu_map.jsonl`` (on change): per ``<node>/<gpu>`` the awake / hidden binding ids, plus the
  SM ``gpus`` / ``nodes`` blocks verbatim;
* ``pod_metrics_1s.jsonl`` (SAMPLE_PODS_S, 1 s): every model pod the SM shows awake or hidden,
  or labelled routable - hidden pods still decode - with ``routable_label`` (the
  ``tre.aibrix.io/routable`` label the gateway routes on), the SM flags, the vLLM gauges and
  counters (prompt / generation tokens, preemptions, successes); pod list refreshed every
  POD_LIST_REFRESH_S;
* ``pod_gauges.jsonl`` (SAMPLE_GAUGES_S, 5 s): routable pods only, gauges only (the pilot's
  file, unchanged, for comparison with earlier runs);
* ``apa_status.jsonl`` (SAMPLE_APA_S, 1 s): APA PodAutoscaler status;
* ``gpu_truth.jsonl`` (SAMPLE_GPU_TRUTH_S, 1 s): Redis ``tre:gpu_truth:<node>`` (needs the
  redis url and redis-py);
* ``resource_usage.jsonl`` (SAMPLE_RESOURCES_S, 5 s): per container of the TRE / AIBrix /
  Envoy namespaces, kubelet ``/stats/summary`` (no metrics-server): ``cpu_cores`` from the
  cumulative ``usageCoreNanoSeconds`` difference (kubelet's own ``usageNanoCores`` on the first
  sample), ``rss_mib``, ``working_set_mib``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

from k8s_names import component_of

GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:num_requests_paused")
COUNTERS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total",
            "vllm:request_success_total")
ROUTABLE_LABEL = "tre.aibrix.io/routable"
MODEL_LABEL = "model.aibrix.ai/name"


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def get_json(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def kubectl_json(*args: str, timeout: float = 20):
    out = subprocess.run(["kubectl", *args, "-o", "json"], text=True, capture_output=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return json.loads(out.stdout)


def parse_metrics(text: str) -> dict:
    """Gauges and counters (summed over label sets) of one vLLM /metrics page."""
    vals: dict = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in GAUGES or name in COUNTERS:
            try:
                v = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            key = name.split(":", 1)[1]
            vals[key] = vals.get(key, 0.0) + v if name in COUNTERS else v
    return vals


def gpu_map(state: dict) -> dict:
    """``<node>/<gpu>`` -> {"awake": [binding ids], "hidden": [binding ids]} from SM bindings."""
    out: dict = {}
    for b in state.get("bindings") or ():
        if not (b.get("awake") or b.get("hidden")):
            continue
        node = b.get("node") or str(b.get("binding_id", "?/?/?")).rsplit("/", 2)[-2]
        gpus = b.get("gpu_ids") or str(b.get("binding_id", "")).rsplit("/", 1)[-1].split(",")
        for g in gpus:
            slot = out.setdefault(f"{node}/{g}", {"awake": [], "hidden": []})
            if b.get("awake"):
                slot["awake"].append(b["binding_id"])
            if b.get("hidden"):
                slot["hidden"].append(b["binding_id"])
    for slot in out.values():
        slot["awake"].sort()
        slot["hidden"].sort()
    return out


def layout_row(ts: float, state: dict) -> dict:
    row = {"ts": ts, "version": state.get("version"), "models": {}}
    for b in state.get("bindings") or ():
        m = row["models"].setdefault(b["model"], {"awake": [], "hidden": [], "routable": []})
        if b.get("awake"):
            m["awake"].append(b["binding_id"])
        if b.get("hidden"):
            m["hidden"].append(b["binding_id"])
        if b.get("routable") is True:
            m["routable"].append(b["binding_id"])
    if isinstance(state.get("models"), (dict, list)):
        row["models_sm"] = state["models"]
    return row


def select_pods(pods: dict, state: dict | None, *, routable_only: bool = False) -> dict:
    """Pods to scrape: routable_only -> labelled routable (the pilot's pod_gauges set); else
    every pod the SM shows awake or hidden (hidden pods still decode) or labelled routable.
    Each value gets ``binding_id`` / ``sm_awake`` / ``sm_hidden`` (None: no SM binding found)."""
    by_serve = {}
    for b in (state or {}).get("bindings") or ():
        if b.get("serve_id"):
            by_serve[b["serve_id"]] = b
    out = {}
    for name, p in pods.items():
        if not p.get("ip"):
            continue
        b = next((bb for sid, bb in by_serve.items() if name == sid or name.startswith(sid + "-")), None)
        info = dict(p, binding_id=b.get("binding_id") if b else None,
                    sm_awake=bool(b.get("awake")) if b else None, sm_hidden=bool(b.get("hidden")) if b else None)
        labelled = p.get("routable_label") == "true"
        if routable_only:
            if labelled:
                out[name] = info
        elif labelled or (b is not None and (b.get("awake") or b.get("hidden"))):
            out[name] = info
    return out


class Sampler:
    def __init__(self, out: str, sm: str, redis_url: str | None):
        self.out, self.sm, self.redis_url = out, sm, redis_url
        self.stop_path = os.path.join(out, "STOP")
        self.model_ns = env("MODEL_NS", "default")
        self.selector = env("MODEL_SELECTOR", "tre.aibrix.io/managed=true")
        self.apa_ns = env("APA_NS", "default")
        self.port = env("ENGINE_PORT", "8000")
        self.res_ns = [n for n in {env("TRE_NS", "tre-v2"), env("AIBRIX_NS", "aibrix-system"),
                                   env("ENVOY_NS", "envoy-gateway-system"), env("BL_NS", "tre-v2"),
                                   env("GW_NS", "tre-v2")} if n]
        self.lock = threading.Lock()
        self.pods_lock = threading.Lock()
        self.state: dict | None = None
        self.pods: dict = {}
        self.pods_ts = 0.0

    def stopped(self) -> bool:
        return os.path.exists(self.stop_path)

    def every(self, period: float, fn) -> None:
        while not self.stopped():
            t = time.time()
            fn(t)
            time.sleep(max(0.0, period - (time.time() - t)))

    def append(self, fname: str, row: dict) -> None:
        with open(os.path.join(self.out, fname), "a") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")

    # ---- SM layout + GPU map
    def layout_loop(self) -> None:
        last_map = [None]

        def tick(t):
            try:
                s = get_json(self.sm + "/v2/state")
            except Exception as e:  # noqa: BLE001
                self.append("layout.jsonl", {"ts": t, "error": str(e)[:200]})
                return
            with self.lock:
                self.state = s
            self.append("layout.jsonl", layout_row(t, s))
            m = gpu_map(s)
            if m != last_map[0]:
                self.append("gpu_map.jsonl", {"ts": t, "version": s.get("version"), "map": m,
                                              "gpus": s.get("gpus"), "nodes": s.get("nodes")})
                last_map[0] = m
        self.every(envf("SAMPLE_LAYOUT_S", 1.0), tick)

    # ---- model pods
    def pod_list(self) -> dict:
        with self.pods_lock:
            return self._pod_list()

    def _pod_list(self) -> dict:
        now = time.time()
        if now - self.pods_ts >= envf("POD_LIST_REFRESH_S", 10.0) or not self.pods:
            items = kubectl_json("-n", self.model_ns, "get", "pods", "-l", self.selector)["items"]
            self.pods = {p["metadata"]["name"]: {
                "ip": p["status"].get("podIP"), "node": p["spec"].get("nodeName"),
                "model": p["metadata"].get("labels", {}).get(MODEL_LABEL),
                "routable_label": p["metadata"].get("labels", {}).get(ROUTABLE_LABEL)} for p in items}
            self.pods_ts = now
        return self.pods

    def scrape(self, ip: str) -> dict:
        with urllib.request.urlopen(f"http://{ip}:{self.port}/metrics", timeout=3) as r:
            return parse_metrics(r.read().decode())

    def pods_loop(self) -> None:
        def tick(t):
            try:
                with self.lock:
                    state = self.state
                chosen = select_pods(self.pod_list(), state)
                rows = {}
                for name, p in chosen.items():
                    base = {"model": p["model"], "node": p["node"], "binding_id": p["binding_id"],
                            "routable_label": p["routable_label"], "routable": p["routable_label"] == "true",
                            "sm_awake": p["sm_awake"], "sm_hidden": p["sm_hidden"]}
                    try:
                        rows[name] = {**base, **self.scrape(p["ip"])}
                    except Exception as e:  # noqa: BLE001
                        rows[name] = {**base, "error": str(e)[:100]}
                self.append("pod_metrics_1s.jsonl", {"ts": t, "pods": rows})
            except Exception as e:  # noqa: BLE001
                self.append("pod_metrics_1s.jsonl", {"ts": t, "error": str(e)[:200]})
        self.every(envf("SAMPLE_PODS_S", 1.0), tick)

    def gauges_loop(self) -> None:   # the pilot's 5 s routable-only file, unchanged format
        def tick(t):
            try:
                rows = {}
                for name, p in select_pods(self.pod_list(), None, routable_only=True).items():
                    try:
                        vals = self.scrape(p["ip"])
                        rows[name] = {"model": p["model"], **{k: v for k, v in vals.items()
                                                              if "vllm:" + k in GAUGES}}
                    except Exception as e:  # noqa: BLE001
                        rows[name] = {"model": p["model"], "error": str(e)[:100]}
                self.append("pod_gauges.jsonl", {"ts": t, "pods": rows})
            except Exception as e:  # noqa: BLE001
                self.append("pod_gauges.jsonl", {"ts": t, "error": str(e)[:200]})
        self.every(envf("SAMPLE_GAUGES_S", 5.0), tick)

    # ---- APA CR status
    def apa_loop(self) -> None:
        def tick(t):
            try:
                items = kubectl_json("-n", self.apa_ns, "get", "podautoscalers.autoscaling.aibrix.ai")["items"]
                self.append("apa_status.jsonl", {"ts": t, "pa": {
                    i["metadata"]["name"]: {k: i.get("status", {}).get(k) for k in
                                            ("desiredScale", "actualScale", "lastScaleTime")} for i in items}})
            except Exception as e:  # noqa: BLE001
                self.append("apa_status.jsonl", {"ts": t, "error": str(e)[:200]})
        self.every(envf("SAMPLE_APA_S", 1.0), tick)

    # ---- gpu-truth (Redis)
    def gpu_truth_loop(self) -> None:
        if not self.redis_url:
            return
        try:
            import redis  # noqa: PLC0415
            r = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=3.0)
        except Exception as e:  # noqa: BLE001
            self.append("gpu_truth.jsonl", {"ts": time.time(), "error": f"redis unavailable: {e}"[:200]})
            return

        def tick(t):
            try:
                nodes = {}
                for k in sorted(r.scan_iter(match="tre:gpu_truth:*", count=100)):
                    raw = r.get(k)
                    try:
                        nodes[k.split(":", 2)[2]] = json.loads(raw) if raw else None
                    except ValueError:
                        nodes[k.split(":", 2)[2]] = {"raw": str(raw)[:200]}
                self.append("gpu_truth.jsonl", {"ts": t, "nodes": nodes})
            except Exception as e:  # noqa: BLE001
                self.append("gpu_truth.jsonl", {"ts": t, "error": str(e)[:200]})
        self.every(envf("SAMPLE_GPU_TRUTH_S", 1.0), tick)

    # ---- container resources (kubelet summary API through the API server proxy)
    def resources_loop(self) -> None:
        prev: dict = {}

        def tick(t):
            try:
                nodes = [n["metadata"]["name"] for n in kubectl_json("get", "nodes")["items"]]
            except Exception as e:  # noqa: BLE001
                self.append("resource_usage.jsonl", {"ts": t, "error": str(e)[:200]})
                return
            for node in nodes:
                try:
                    raw = subprocess.run(["kubectl", "get", "--raw", f"/api/v1/nodes/{node}/proxy/stats/summary"],
                                         text=True, capture_output=True, timeout=20).stdout
                    summary = json.loads(raw)
                except Exception as e:  # noqa: BLE001
                    self.append("resource_usage.jsonl", {"ts": t, "node": node, "error": str(e)[:200]})
                    continue
                for row in resource_rows(t, node, summary, self.res_ns, prev):
                    self.append("resource_usage.jsonl", row)
        self.every(envf("SAMPLE_RESOURCES_S", 5.0), tick)

    def run(self) -> None:
        loops = [self.layout_loop, self.pods_loop, self.gauges_loop, self.apa_loop, self.gpu_truth_loop,
                 self.resources_loop]
        threads = [threading.Thread(target=f, name=f.__name__, daemon=True) for f in loops]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


def resource_rows(ts: float, node: str, summary: dict, namespaces, prev: dict) -> list[dict]:
    rows = []
    for pod in summary.get("pods") or ():
        ref = pod.get("podRef") or {}
        if ref.get("namespace") not in namespaces:
            continue
        for c in pod.get("containers") or ():
            cpu, mem = c.get("cpu") or {}, c.get("memory") or {}
            key = (ref.get("namespace"), ref.get("name"), c.get("name"))
            total, stamp = cpu.get("usageCoreNanoSeconds"), cpu.get("time")
            cores = None
            if total is not None:
                last = prev.get(key)
                now_s = _rfc3339(stamp) if stamp else ts
                if last and now_s > last[1] and total >= last[0]:
                    cores = (total - last[0]) / 1e9 / (now_s - last[1])
                prev[key] = (total, now_s)
            if cores is None and cpu.get("usageNanoCores") is not None:
                cores = cpu["usageNanoCores"] / 1e9
            rows.append({"ts": ts, "stats_ts": stamp, "node": node, "namespace": ref.get("namespace"),
                         "pod": ref.get("name"), "container": c.get("name"), "component": component_of(ref.get("name", "")),
                         "cpu_cores": None if cores is None else round(cores, 4),
                         "rss_mib": None if mem.get("rssBytes") is None else round(mem["rssBytes"] / 2**20, 2),
                         "working_set_mib": None if mem.get("workingSetBytes") is None
                         else round(mem["workingSetBytes"] / 2**20, 2)})
    return rows


def _rfc3339(stamp: str) -> float:
    import datetime as _dt

    s = stamp.rstrip("Z")
    frac = 0.0
    if "." in s:
        s, f = s.split(".", 1)
        digits = "".join(ch for ch in f if ch.isdigit())
        frac = float("0." + digits) if digits else 0.0
    return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp() + frac


if __name__ == "__main__":
    Sampler(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None).run()
