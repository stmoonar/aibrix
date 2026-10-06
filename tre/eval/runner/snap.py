#!/usr/bin/env python3
"""snap.py <out.json> <SM url>: sidecar tre_reissue_* counters of every model pod + SM /v2/sleep."""
import json
import subprocess
import sys
import time
import urllib.request

out, SM = sys.argv[1], sys.argv[2]
raw = subprocess.check_output(["kubectl", "-n", "default", "get", "pods", "-l",
                               "tre.aibrix.io/managed=true", "-o", "json"], text=True)
res = {"ts": time.time(), "pods": {}, "sleep": None}
for p in json.loads(raw)["items"]:
    name, ip = p["metadata"]["name"], p["status"].get("podIP")
    ctr = {}
    try:
        with urllib.request.urlopen(f"http://{ip}:8000/tre-reissue/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or "_bucket" in line or "_seconds" in line:
                    continue
                if line.startswith("tre_reissue"):
                    k, v = line.rsplit(" ", 1)
                    ctr[k] = float(v)
    except Exception as e:  # noqa: BLE001
        ctr = {"error": str(e)[:100]}
    res["pods"][name] = ctr
with urllib.request.urlopen(SM + "/v2/sleep", timeout=15) as r:
    res["sleep"] = json.load(r)
json.dump(res, open(out, "w"), indent=1)
print("snap", out, len(res["pods"]), "pods")
