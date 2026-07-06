import json
import sys

path = "/root/tre-experiments/valid_20260706/decisions.log"
model = sys.argv[1] if len(sys.argv) > 1 else "dsqwen-7b"
latest = (None, None, None, None)  # ts, z_m, trs, submitted
try:
    lines = open(path).read().splitlines()
except FileNotFoundError:
    print("none none none none")
    raise SystemExit(0)
for line in lines:
    if "trs_calc_result" not in line:
        continue
    try:
        msg = json.loads(json.loads(line)["message"])
        ms = json.loads(msg.get("model_states", "{}"))
        entry = ms.get(model, {})
        latest = (msg.get("ts_ms"), entry.get("z_m"), entry.get("trs_z_m"), msg.get("submitted"))
    except Exception:
        pass
print("%s %s %s %s" % latest)
