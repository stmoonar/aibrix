"""Load one arm directory into an :class:`Arm` (read-only, every input optional).

Only ``client/performance_metrics.json`` is required. Every other file is optional: a
missing or unparsable file adds a line to ``Arm.warnings`` and leaves its field empty, so
the report degrades (fewer panels / columns) instead of failing.

Time base: every timestamp is converted to seconds on the *reference clock* (the host
that runs the client and the sampler, whose ``load_start_epoch`` is t = 0). Sources
written on another node are shifted by that node's clock offset (seconds the node is
ahead of the reference): ``Arm.clock_offsets`` from ``clock_offsets.json`` in the arm
directory or ``--clock-offset NODE=SECONDS`` on the command line.
"""

from __future__ import annotations

import csv
import calendar
import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

try:  # PyYAML is in requirements-runtime.txt; the report still runs without a registry
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def parse_binding(binding_id: str) -> tuple[str, str, list[str]]:
    """``model/node/gpu[,gpu]`` -> (model, node, [gpu, ...])."""
    parts = binding_id.rsplit("/", 2)
    if len(parts) != 3:
        return binding_id, "?", [binding_id]
    return parts[0], parts[1], [g for g in parts[2].split(",") if g != ""]


_SM_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) +\w+ +([\w.]+): (.*)$")


def _utc_epoch(stamp: str, millis: str) -> float:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")) + int(millis) / 1000.0


@dataclass
class Arm:
    dir: str
    name: str
    label: str
    trace_name: str
    warnings: list[str] = field(default_factory=list)
    present: dict[str, bool] = field(default_factory=dict)
    t_load: float | None = None
    t_end: float | None = None
    t_restart: float | None = None
    requests: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    traces_sha256: str | None = None
    registry: dict = field(default_factory=dict)
    registry_sha256: str | None = None
    models: list[str] = field(default_factory=list)
    tp: dict[str, int] = field(default_factory=dict)
    slo: dict[str, dict] = field(default_factory=dict)
    trs: dict[str, dict] = field(default_factory=dict)
    layout: list[tuple[float, dict]] = field(default_factory=list)
    gauges: list[tuple[float, dict]] = field(default_factory=list)
    signal: list[dict] = field(default_factory=list)
    ctrl_ticks: list[dict] = field(default_factory=list)
    ctrl_events: list[dict] = field(default_factory=list)
    sm_events: list[dict] = field(default_factory=list)
    sidecar: list[dict] = field(default_factory=list)
    bl_decisions: list[dict] = field(default_factory=list)
    apa_status: list[tuple[float, dict]] = field(default_factory=list)
    profile_rows: list[dict] = field(default_factory=list)
    resource_rows: list[dict] = field(default_factory=list)
    trace_segments: dict | None = None
    clock_offsets: dict[str, float] = field(default_factory=dict)
    components: dict[str, str] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)  # small provenance files (text / json)
    # ---- newer collectors (empty when the file is absent; old arm directories never set them)
    pod_metrics: list[tuple[float, dict]] = field(default_factory=list)   # pod_metrics_1s.jsonl (1 Hz, incl. hidden)
    layout_routable: list[tuple[float, dict]] = field(default_factory=list)  # layout.jsonl models[m].routable (SM flag)
    gpu_map: list[tuple[float, dict]] = field(default_factory=list)       # gpu_map.jsonl {node/gpu: {awake, hidden}}
    gpu_truth: list[tuple[float, dict]] = field(default_factory=list)     # gpu_truth.jsonl {node: {gpus: [...]}}
    pod_nodes: dict[str, str] = field(default_factory=dict)               # components.json pods: pod -> node
    component_meta: dict = field(default_factory=dict)                    # components.json image ids, node moves
    clock_meta: dict = field(default_factory=dict)                        # clock_offsets.json drift / rtt
    sm_access: list[dict] = field(default_factory=list)                   # sm.ts.log uvicorn access lines
    gw_requests: dict[str, dict] = field(default_factory=dict)            # gateway per-request events by req_id

    # ---- helpers
    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def rel(self, ts: float | None) -> float | None:
        return None if ts is None or self.t_load is None else ts - self.t_load

    def node_of(self, name: str) -> str | None:
        """Node whose name occurs in a pod / serve id (longest match wins)."""
        best = None
        for node in self.nodes():
            if node in name and (best is None or len(node) > len(best)):
                best = node
        return best

    def nodes(self) -> list[str]:
        seen = set()
        for _, models in self.layout:
            for awake, hidden in models.values():
                for b in list(awake) + list(hidden):
                    seen.add(parse_binding(b)[1])
        seen.update(self.clock_offsets)
        return sorted(seen)

    def offset_for(self, name: str | None) -> float:
        if not name:
            return 0.0
        node = name if name in self.clock_offsets else (self.pod_nodes.get(name) or self.node_of(name))
        return float(self.clock_offsets.get(node, 0.0)) if node else 0.0


def _read_text(path: str) -> str | None:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _read_json(path: str) -> Any:
    txt = _read_text(path)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except ValueError:
        return None


def _jsonl(path: str) -> list[dict]:
    out = []
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _num(x: Any) -> float | None:
    if x is None or x == "" or x == "None" or x == "null":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_arm(path: str, clock_offsets: dict[str, float] | None = None,
             trace_segments: str | None = None) -> Arm:
    d = os.path.abspath(path.rstrip("/"))
    name = os.path.basename(d)
    label_txt = _read_text(os.path.join(d, "arm_label"))
    arm = Arm(dir=d, name=name, label=(label_txt or name).strip() or name,
              trace_name=os.path.basename(os.path.dirname(d)))
    j = lambda *p: os.path.join(d, *p)  # noqa: E731

    def have(key: str, p: str) -> bool:
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        arm.present[key] = ok
        return ok

    def have_new(key: str, p: str) -> bool:  # newer collector: recorded in `present` only when there
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        if ok:
            arm.present[key] = True
        return ok

    # ---- clock offsets (manifest first, CLI overrides)
    co = _read_json(j("clock_offsets.json"))
    if isinstance(co, dict):
        arm.clock_offsets.update({k: float(v) for k, v in (co.get("offsets_s") or co).items()
                                  if isinstance(v, (int, float))})
    if clock_offsets:
        arm.clock_offsets.update(clock_offsets)
    if isinstance(co, dict) and (isinstance(co.get("start"), dict) or isinstance(co.get("end"), dict)):
        _clock_meta(arm, co)
    comp = _read_json(j("components.json"))
    if isinstance(comp, dict) and isinstance(comp.get("component_nodes"), dict):
        _components_v2(arm, comp)
    elif isinstance(comp, dict):
        arm.components = {k: str(v) for k, v in comp.items()}

    # ---- epochs
    for attr, fname in (("t_load", "load_start_epoch"), ("t_end", "load_end_epoch"),
                        ("t_restart", "controller_restart_epoch")):
        v = _num((_read_text(j(fname)) or "").strip())
        setattr(arm, attr, v)
        if v is None and attr != "t_restart":
            arm.warn(f"{fname} missing: time base falls back to the first client send")

    # ---- client (required)
    pm = j("client", "performance_metrics.json")
    if not have("client", pm):
        raise FileNotFoundError(f"{pm}: no client results (the only required input)")
    arm.requests = _jsonl(pm)
    starts = [r["start_time"] for r in arm.requests if r.get("start_time")]
    if arm.t_load is None and starts:
        arm.t_load = min(starts)
    if arm.t_end is None and starts:
        arm.t_end = max(starts)
    tp = j("client", "traces.json")
    if have("traces", tp):
        try:
            raw = json.load(open(tp))
            arm.traces = [{k: r.get(k) for k in ("request_id", "timestamp", "model_name", "max_output_tokens",
                                                   "prompt_length", "phase_type")} for r in raw]
            arm.traces_sha256 = sha256_file(tp)
        except (OSError, ValueError) as e:
            arm.warn(f"client/traces.json unreadable: {e}")
    else:
        arm.warn("client/traces.json missing: offered load and max_tokens checks from client sends only")

    # ---- registry
    reg_path = j("live-registry.yaml")
    if have("registry", reg_path) and yaml is not None:
        try:
            arm.registry = yaml.safe_load(open(reg_path)) or {}
            arm.registry_sha256 = sha256_file(reg_path)
        except Exception as e:  # noqa: BLE001
            arm.warn(f"live-registry.yaml unreadable: {e}")
    else:
        arm.warn("live-registry.yaml missing: SLO thresholds unavailable (V_req not computed)")
    for m in arm.registry.get("models") or []:
        arm.models.append(m["name"])
        arm.tp[m["name"]] = int(m.get("tp_size") or 1)
        if m.get("slo"):
            arm.slo[m["name"]] = m["slo"]
        if m.get("trs"):
            arm.trs[m["name"]] = m["trs"]
    for r in arm.requests:
        if r.get("model_name") and r["model_name"] not in arm.models:
            arm.models.append(r["model_name"])

    # ---- layout (SM /v2/state, 1 s)
    if have("layout", j("layout.jsonl")):
        for r in _jsonl(j("layout.jsonl")):
            if "models" not in r:
                continue
            arm.layout.append((float(r["ts"]), {m: (list(v.get("awake") or []), list(v.get("hidden") or []))
                                                for m, v in r["models"].items()}))
            if any(isinstance(v, dict) and "routable" in v for v in r["models"].values()):
                arm.layout_routable.append((float(r["ts"]), {m: list(v.get("routable") or [])
                                                             for m, v in r["models"].items()}))
        arm.layout.sort(key=lambda x: x[0])
        arm.layout_routable.sort(key=lambda x: x[0])
    else:
        arm.warn("layout.jsonl missing: no replica / GPU timelines, GPU-seconds or onset latencies")

    # ---- per-pod vLLM gauges
    if have("pod_gauges", j("pod_gauges.jsonl")):
        for r in _jsonl(j("pod_gauges.jsonl")):
            if "pods" in r:
                arm.gauges.append((float(r["ts"]), r["pods"]))
        arm.gauges.sort(key=lambda x: x[0])
    elif not have_new("pod_metrics_1s", j("pod_metrics_1s.jsonl")):
        arm.warn("pod_gauges.jsonl missing: no KV / running / waiting panels from the sampler")
    # 1 Hz, every awake pod incl. hidden (replaces pod_gauges for the KV / running / waiting columns)
    if have_new("pod_metrics_1s", j("pod_metrics_1s.jsonl")):
        for r in _jsonl(j("pod_metrics_1s.jsonl")):
            if isinstance(r.get("pods"), dict) and _num(r.get("ts")) is not None:
                arm.pod_metrics.append((float(r["ts"]), r["pods"]))
        arm.pod_metrics.sort(key=lambda x: x[0])

    # ---- SM per-GPU map (on change) and physical GPU memory (gpu-truth, 1 Hz)
    if have_new("gpu_map", j("gpu_map.jsonl")):
        for r in _jsonl(j("gpu_map.jsonl")):
            if isinstance(r.get("map"), dict) and _num(r.get("ts")) is not None:
                arm.gpu_map.append((float(r["ts"]), r["map"]))
        arm.gpu_map.sort(key=lambda x: x[0])
    if have_new("gpu_truth", j("gpu_truth.jsonl")):
        for r in _jsonl(j("gpu_truth.jsonl")):
            if isinstance(r.get("nodes"), dict) and _num(r.get("ts")) is not None:
                arm.gpu_truth.append((float(r["ts"]), r["nodes"]))
        arm.gpu_truth.sort(key=lambda x: x[0])
    elif arm.layout and arm.pod_metrics:  # a new-collector arm without gpu-truth
        arm.warn("gpu_truth.jsonl missing: no physical-vs-ledger GPU check")

    # ---- gateway signal log (TSS / Z per model per window)
    if have("signal_log", j("signal_log.jsonl")):
        for r in _jsonl(j("signal_log.jsonl")):
            ts = _num(r.get("ts"))
            if ts is None:
                continue
            arm.signal.append({"ts": ts, "model": r.get("model"),
                               **{k: _num(r.get(k)) for k in ("z_m", "z", "tss", "theta_m", "queue_len",
                                                             "decode_tps", "prefill_tps", "replicas_awake",
                                                             "replicas_target", "raw_signal", "eta_m")},
                               "tier": r.get("tier"), "action": r.get("action"),
                               "signal_source": r.get("signal_source")})
    else:
        arm.warn("signal_log.jsonl empty or missing: no TSS / Z panel for this arm")

    # ---- controller log (trs_calc_result ticks + safescale events)
    if have("controller_log", j("controller.log")):
        _parse_controller(arm, j("controller.log"))

    # ---- SM log (wake / sleep events)
    if have("sm_log", j("sm.log")):
        _parse_sm(arm, j("sm.log"))
    elif have_new("sm_ts_log", j("sm.ts.log")):
        _parse_sm(arm, j("sm.ts.log"), kubelet_ts=True)
    else:
        arm.warn("sm.log missing: no wake / sleep durations")
    # kubectl logs --timestamps copy: uvicorn access lines (PUT /target etc.) with exact times
    if have_new("sm_ts_log", j("sm.ts.log")):
        _parse_sm_access(arm, j("sm.ts.log"))

    # ---- sidecar logs
    sc = sorted(glob.glob(j("sidecar", "*.log")))
    arm.present["sidecar"] = bool(sc)
    for f in sc:
        pod = os.path.basename(f)[:-4]
        for r in _jsonl(f):
            if not isinstance(r, dict) or "event" not in r:
                continue
            r = dict(r)
            r.setdefault("pod", pod)
            ts = _num(r.get("ts"))
            off = arm.offset_for(r.get("pod"))
            r["ts_raw"] = ts
            r["ts"] = None if ts is None else ts - off
            if "abort_ts" in r:  # epoch s on the pod's node clock, like ts
                ab = _num(r.get("abort_ts"))
                r["abort_ts_raw"] = ab
                r["abort_ts"] = None if ab is None else ab - off
            arm.sidecar.append(r)

    # ---- gateway per-request events (baseline arms: baseline/, other arms: gateway_events/)
    gw = sorted(glob.glob(j("baseline", "bl_req_events.*.jsonl")) + glob.glob(j("gateway_events", "bl_req_events.*.jsonl")))
    if gw:
        arm.present["gateway_events"] = True
        _load_gateway_events(arm, gw)

    # ---- baseline decisions (baseline arms)
    dec = sorted(glob.glob(j("baseline", "*", "decisions-*.jsonl")))
    arm.present["bl_decisions"] = bool(dec)
    for f in dec:
        for r in _jsonl(f):
            if "ts_ms" in r:
                arm.bl_decisions.append(r)
    arm.bl_decisions.sort(key=lambda r: (r.get("ts_ms", 0), r.get("model", "")))

    # ---- APA CR status (5 s in older runs, 1 Hz since the 1 Hz sampler; nothing assumes a period)
    if have("apa_status", j("apa_status.jsonl")):
        for r in _jsonl(j("apa_status.jsonl")):
            if isinstance(r.get("pa"), dict):
                arm.apa_status.append((float(r["ts"]), r["pa"]))

    # ---- profiler / resource usage
    for key, fname, attr in (("controller_profile", "controller_profile.csv", "profile_rows"),
                             ("resource_usage", "resource_usage.csv", "resource_rows")):
        if have(key, j(fname)):
            try:
                setattr(arm, attr, list(csv.DictReader(open(j(fname)))))
            except Exception as e:  # noqa: BLE001
                arm.warn(f"{fname} unreadable: {e}")
    if have("resource_usage_jsonl", j("resource_usage.jsonl")):
        arm.resource_rows.extend(_jsonl(j("resource_usage.jsonl")))

    # ---- trace segments (phase table): CLI path, else next to the arm
    seg_path = trace_segments or next((p for p in (j("trace_segments.json"), j("trace.json")) if os.path.exists(p)), None)
    if seg_path:
        seg = _read_json(seg_path)
        if isinstance(seg, dict):
            arm.trace_segments = seg
            arm.files["trace_segments_path"] = seg_path
            arm.files["trace_segments_sha256"] = sha256_file(seg_path)

    # ---- small provenance files
    for fname in ("tre_sha", "loadgen_sha", "runner_sha", "images.txt", "image_ids_node10.txt", "model_images.txt",
                  "policy_cm_sha", "start_iso", "sidecar_nofile.tsv", "EMFILE_PODS", "profile_xlen.txt"):
        txt = _read_text(j(fname))
        if txt is not None:
            arm.files[fname] = txt
    for p in sorted(glob.glob(j("image_ids_*.txt"))):
        arm.files.setdefault(os.path.basename(p), _read_text(p))
    for fname in ("score.json", "summary.json", "safescale_summary.json", "arm_meta.json",
                  "policy_check.json", "profile_summary_load.json", "trace_manifest.json",
                  "trace_source_manifest.json"):
        v = _read_json(j(fname))
        if v is not None:
            arm.files[fname] = v
    v = _read_json(j("baseline", "run_validity.json"))
    if v is not None:
        arm.files["run_validity.json"] = v
    for p in (j("policy-chiron.yaml"), j("policy-tokenscale.yaml"), j("policy-preserve.yaml")):
        if os.path.exists(p):
            arm.files["policy_sha256"] = sha256_file(p)
            arm.files["policy_file"] = os.path.basename(p)
    arm.files["trace_seed"] = _trace_seed(arm)
    if not arm.clock_offsets:
        arm.warn("no clock offsets (clock_offsets.json / --clock-offset): sidecar and other-node timestamps are not corrected")
    return arm


def _parse_controller(arm: Arm, path: str) -> None:
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
                m = json.loads(j.get("message", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(m, dict):
                continue
            ev = m.get("event")
            if ev == "trs_calc_result":
                try:
                    tick = {
                        "ts": int(m["ts_ms"]) / 1000.0, "loop": m.get("loop"),
                        "submitted": int(m.get("submitted") or 0),
                        "actions": json.loads(m.get("actions") or "[]"),
                        "events": json.loads(m.get("events") or "[]"),
                        "model_states": json.loads(m.get("model_states") or "{}"),
                    }
                except (ValueError, KeyError, TypeError):
                    continue
                now = _num(m.get("now_ms"))  # decision wall clock (ts_ms is the window boundary)
                if now is not None:
                    tick["now"] = now / 1000.0
                arm.ctrl_ticks.append(tick)
            elif ev and "ts_ms" in m:
                arm.ctrl_events.append({"ts": int(m["ts_ms"]) / 1000.0, "event": ev,
                                        "events": m.get("events"), "submitted": m.get("submitted")})


def _parse_sm(arm: Arm, path: str, kubelet_ts: bool = False) -> None:
    off = arm.offset_for(arm.components.get("service-manager"))
    with open(path, errors="replace") as fh:
        for line in fh:
            if kubelet_ts:  # `kubectl logs --timestamps`: drop the kubelet stamp, keep the original line
                line = line.split(" ", 1)[1] if " " in line else line
            mt = _SM_LINE.match(line.rstrip("\n"))
            if not mt:
                continue
            body = mt.group(4)
            i = body.find("{")
            if i < 0:
                continue
            try:
                rec = json.loads(body[i:])
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            kind = rec.get("event") or (body[:i].strip() or None)
            if kind not in ("wake_start", "wake_done", "wake_failed", "sleep"):
                continue
            bid = rec.get("binding_id")
            model = rec.get("model") or (parse_binding(bid)[0] if bid else None)
            arm.sm_events.append({"ts": _utc_epoch(mt.group(1), mt.group(2)) - off, "kind": kind,
                                  "logger": mt.group(3), "binding_id": bid, "model": model,
                                  "serve_id": rec.get("serve_id"), "rec": rec})
    arm.sm_events.sort(key=lambda e: e["ts"])


# ---------------------------------------------------------------- newer collectors
_KUBELET_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d) (.*)$")
_ACCESS = re.compile(r'INFO:\s+\S+ - "([A-Z]+) (\S+) HTTP/[\d.]+" (\d{3})')
_MODEL_PATH = re.compile(r"^(/v\d+/models/)([^/?]+)(.*)$")


def parse_rfc3339(stamp: str, frac: str | None, tz: str) -> float:
    """``2026-10-07T01:02:03.123456789Z`` (kubelet RFC3339Nano) -> epoch s."""
    t = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S")) + (float("0." + frac) if frac else 0.0)
    if tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        t -= sign * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
    return t


def _parse_sm_access(arm: Arm, path: str) -> None:
    """uvicorn access lines of ``sm.ts.log`` -> ``arm.sm_access`` (ts on the reference clock)."""
    off = arm.offset_for(arm.components.get("service-manager"))
    with open(path, errors="replace") as fh:
        for line in fh:
            mt = _KUBELET_TS.match(line.rstrip("\n"))
            if not mt:
                continue
            ma = _ACCESS.search(mt.group(4))
            if not ma:
                continue
            p = ma.group(2).split("?", 1)[0]
            mm = _MODEL_PATH.match(p)
            arm.sm_access.append({"ts": parse_rfc3339(mt.group(1), mt.group(2), mt.group(3)) - off,
                                  "method": ma.group(1), "path": p, "status": int(ma.group(3)),
                                  "model": mm.group(2) if mm else None,
                                  "template": f"{mm.group(1)}{{model}}{mm.group(3)}" if mm else p})
    arm.sm_access.sort(key=lambda e: e["ts"])


def _load_gateway_events(arm: Arm, paths: list[str]) -> None:
    """Gateway per-request stream exports (``{id: "<redis_ms>-<seq>", kind: arr|ft|done, req_id, pod, ...}``)
    -> ``arm.gw_requests[req_id] = {arr, ft, done, pod, model, n_arr}`` (Redis server clock, corrected by
    the offset of the node running component ``redis``)."""
    off = arm.offset_for(arm.components.get("redis"))
    for f in paths:
        model = os.path.basename(f)[len("bl_req_events."):-len(".jsonl")]
        for r in _jsonl(f):
            rid, kind = r.get("req_id"), r.get("kind")
            ms = _num(str(r.get("id", "")).split("-", 1)[0])
            if not rid or ms is None or kind not in ("arr", "ft", "done"):
                continue
            ts = ms / 1000.0 - off
            g = arm.gw_requests.setdefault(rid, {"arr": None, "ft": None, "done": None, "pod": None,
                                                 "model": model, "n_arr": 0})
            if kind == "arr":
                g["n_arr"] += 1
            if kind == "done":  # last completion wins (a reissued request may pass the gateway again)
                g["done"] = ts if g["done"] is None else max(g["done"], ts)
            elif g[kind] is None or ts < g[kind]:
                g[kind] = ts
            if r.get("pod") and (g["pod"] is None or kind == "done"):
                g["pod"] = r["pod"]


def _components_v2(arm: Arm, comp: dict) -> None:
    """components.json {component_nodes, image_ids_by_node, start: {pods}, end: {pods}}."""
    arm.components = {k: str(v) for k, v in comp["component_nodes"].items() if v}
    if isinstance(comp.get("image_ids_by_node"), dict):
        arm.component_meta["image_ids_by_node"] = comp["image_ids_by_node"]
    where: dict[str, dict[str, set]] = {}
    for phase in ("start", "end"):
        for p in ((comp.get(phase) or {}).get("pods") or []):
            if not isinstance(p, dict) or not p.get("name") or not p.get("node"):
                continue
            arm.pod_nodes.setdefault(p["name"], p["node"])
            if p.get("component"):
                where.setdefault(p["component"], {}).setdefault(phase, set()).add(p["node"])
    moved = {c: {"start": sorted(v["start"]), "end": sorted(v["end"])} for c, v in sorted(where.items())
             if "start" in v and "end" in v and v["start"] != v["end"]}
    if moved:
        arm.component_meta["moved"] = moved
        for c, v in moved.items():
            arm.warn(f"component {c} changed node between arm start and end: {v['start']} -> {v['end']}")


def _clock_meta(arm: Arm, co: dict) -> None:
    """clock_offsets.json start / end measurements: drift (end - start) and RTT per node."""
    st = (co.get("start") or {}).get("nodes") or {}
    en = (co.get("end") or {}).get("nodes") or {}
    drift, rtt = {}, {}
    for node in sorted(set(st) | set(en)):
        a, b = _num((st.get(node) or {}).get("offset_s")), _num((en.get(node) or {}).get("offset_s"))
        if a is not None and b is not None:
            drift[node] = round(b - a, 4)
            if abs(b - a) > 1.0:
                arm.warn(f"clock offset of {node} drifted {b - a:+.3f} s between arm start and end")
        rs = [x for x in (_num((st.get(node) or {}).get("rtt_s")), _num((en.get(node) or {}).get("rtt_s"))) if x is not None]
        if rs:
            rtt[node] = max(rs)
            if max(rs) > 0.5:
                arm.warn(f"clock offset of {node} measured with RTT {max(rs):.3f} s (> 0.5 s): offset uncertain")
    arm.clock_meta = {"ref_host": co.get("ref_host"), "method": co.get("method"),
                      "drift_s": drift or None, "rtt_s_max": rtt or None}


def _trace_seed(arm: Arm):
    """trace_manifest.json seed, else trace_source_manifest.json seed, else the ``_s<seed>`` dir suffix."""
    for k in ("trace_manifest.json", "trace_source_manifest.json"):
        v = arm.files.get(k)
        if isinstance(v, dict) and v.get("seed") is not None:
            return v["seed"]
    mt = re.search(r"_s(\d+)$", arm.trace_name)
    return int(mt.group(1)) if mt else None


def trace_base(trace_name: str) -> str:
    """Trace name without the per-seed ``_s<seed>`` suffix (multi-seed grouping)."""
    return re.sub(r"_s\d+$", "", trace_name)
