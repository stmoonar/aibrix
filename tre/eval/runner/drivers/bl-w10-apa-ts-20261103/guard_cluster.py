#!/usr/bin/env python3
"""READ-ONLY cluster guard for the baseline campaigns with the fleet at rest (APA / TokenScale) (kubectl get only; no exec, no Redis, no writes).

Checks what the runner's own pre-checks cannot repair: both nodes Ready and not under DiskPressure, the tre-v2
components up with the images of the first sample (SM, gateway, baseline-scaler; the controller is only printed),
baseline-scaler at 0 replicas, no APA CR, 20/20 model pods 2/2 Running, no Failed pod, node file-system headroom
above the runner's own guard (kubelet hard-eviction threshold x 1.1) plus 1 GB. Run mode and the awake layout are NOT
checked here: the runner's first pre-check resets them (reset_canonical.sh) when they are not canonical.

Exit 0 = all hard checks ok, 1 = a hard check failed (lines starting FAIL). Environment: EXPECT_SM, EXPECT_GW, EXPECT_BL,
EXPECT_CTL override the expected image tags.
"""
import json
import os
import subprocess
import sys

NS = os.environ.get("TRE_NS", "tre-v2")
MODEL_NS = os.environ.get("MODEL_NS", "default")
SELECTOR = os.environ.get("MODEL_SELECTOR", "tre.aibrix.io/managed=true")
EXPECT = {
    "tre-v2-service-manager": os.environ.get("EXPECT_SM", "tre-v2-service-manager:20261029-d9fa1571"),
    "tre-gateway-plugins": os.environ.get("EXPECT_GW", "aibrix/gateway-plugins:20261029-8b60253f-nozmq2"),
    "tre-v2-baseline-scaler": os.environ.get("EXPECT_BL", "tre-v2-baseline-scaler:20261029-8b60253f"),
}
EXPECT_CTL = os.environ.get("EXPECT_CTL", "tre-v2-controller:20261102-420c69ef")
READY_ONE = ("tre-v2-controller", "tre-v2-service-manager", "tre-gateway-plugins", "tre-v2-redis")
EVICT_FRAC, MARGIN, EXTRA_GB = 0.10, 1.1, 1.0

fails = []


def k(*args, timeout=90):
    r = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:5])}: {r.stderr.strip()[:200]}")
    return r.stdout


def ok(msg):
    print(f"ok   {msg}")


def fail(msg):
    fails.append(msg)
    print(f"FAIL {msg}")


def info(msg):
    print(f"info {msg}")


def main() -> int:
    # nodes
    nodes = json.loads(k("get", "nodes", "-o", "json"))["items"]
    for n in nodes:
        name = n["metadata"]["name"]
        conds = {c["type"]: c["status"] for c in n["status"]["conditions"]}
        if conds.get("Ready") != "True" or conds.get("DiskPressure") == "True":
            fail(f"node {name}: Ready={conds.get('Ready')} DiskPressure={conds.get('DiskPressure')}")
        else:
            ok(f"node {name}: Ready, no DiskPressure")
    # tre-v2 deployments
    deps = {d["metadata"]["name"]: d for d in json.loads(k("-n", NS, "get", "deploy", "-o", "json"))["items"]}
    for name in READY_ONE:
        d = deps.get(name)
        if not d:
            fail(f"deploy {name}: missing")
            continue
        spec, ready = d["spec"].get("replicas"), d["status"].get("readyReplicas") or 0
        (ok if spec == 1 and ready == 1 else fail)(f"deploy {name}: spec {spec} ready {ready}")
    bl = deps.get("tre-v2-baseline-scaler")
    if bl:
        (ok if bl["spec"].get("replicas") == 0 else fail)(f"baseline-scaler replicas {bl['spec'].get('replicas')} (want 0)")
    for name, want in EXPECT.items():
        d = deps.get(name)
        got = d["spec"]["template"]["spec"]["containers"][0]["image"] if d else None
        (ok if got == want else fail)(f"image {name}: {got} (want {want})")
    d = deps.get("tre-v2-controller")
    got = d["spec"]["template"]["spec"]["containers"][0]["image"] if d else None
    info(f"controller image {got} ({'as expected' if got == EXPECT_CTL else 'differs from ' + EXPECT_CTL + ': observe mode, logging only'})")
    if d:
        env = {e["name"]: e.get("value") for e in d["spec"]["template"]["spec"]["containers"][0].get("env", [])}
        info(f"controller env: TRE_SAFESCALE_SCOPE={env.get('TRE_SAFESCALE_SCOPE')} TRE_SIGNAL_SOURCE={env.get('TRE_SIGNAL_SOURCE')} "
             f"TRE_METRICS_WINDOW_MS={env.get('TRE_METRICS_WINDOW_MS')} (the runner resets the switches to default)")
    # APA CRs
    apa = k("-n", MODEL_NS, "get", "podautoscalers.autoscaling.aibrix.ai", "-o", "name").split()
    (ok if not apa else fail)(f"APA CRs in {MODEL_NS}: {len(apa)}")
    # model pods
    pods = json.loads(k("-n", MODEL_NS, "get", "pods", "-l", SELECTOR, "-o", "json"))["items"]
    ready = 0
    for p in pods:
        cs = p["status"].get("containerStatuses") or []
        if p["status"].get("phase") == "Running" and cs and all(c.get("ready") for c in cs):
            ready += 1
    (ok if len(pods) == 20 and ready == 20 else fail)(f"model pods Running + all containers ready: {ready}/{len(pods)} (want 20/20)")
    failed = json.loads(k("-n", MODEL_NS, "get", "pods", "--field-selector=status.phase=Failed", "-o", "json"))["items"]
    (ok if not failed else fail)(f"Failed / Evicted pods in {MODEL_NS}: {len(failed)}"
                                  + ("" if not failed else f" (kubectl -n {MODEL_NS} delete pods --field-selector=status.phase=Failed)"))
    # node file system headroom (the runner's node_fs_free check: kubelet nodefs.available threshold x margin)
    for n in nodes:
        name = n["metadata"]["name"]
        try:
            fs = json.loads(k("get", "--raw", f"/api/v1/nodes/{name}/proxy/stats/summary", timeout=60))["node"]["fs"]
        except Exception as exc:  # noqa: BLE001
            info(f"node {name}: file-system stats unavailable ({exc}); the runner checks again")
            continue
        avail, cap = fs["availableBytes"] / 1e9, fs["capacityBytes"] / 1e9
        need = MARGIN * EVICT_FRAC * cap + EXTRA_GB
        (ok if avail >= need else fail)(f"node {name} nodefs: {avail:.1f} GB free of {cap:.0f} GB, need >= {need:.1f} GB "
                                         f"(kubelet eviction at {EVICT_FRAC * cap:.1f} GB, runner guard {MARGIN * EVICT_FRAC * cap:.1f} GB)")
    print(f"guard_cluster: {len(fails)} hard failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL guard_cluster crashed: {type(exc).__name__}: {exc}")
        sys.exit(1)
