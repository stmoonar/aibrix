#!/usr/bin/env python3
"""campaign.py: one-click E1 campaign over traces x arms (tre/eval/runner).

Entry point: ``run_campaign.sh <campaign.yaml> [--dry-run [--probe]] [--retry-failed]`` (the
wrapper exports the campaign's ``env:`` block, then sources runner.env, then runs
``campaign.py run``). Subcommands: ``env`` (shell exports of the env block), ``run``
(``--dry-run`` = plan + input validation, no cluster access), ``status``.

Order: the traces run in the file's order; the arms of one trace run back to back before the
next trace. The arm order inside a trace is rotated per trace (Williams Latin square by default:
every arm in every position, first-order carry-over balanced over a full square) and recorded.

Per arm: pre-checks (marker, run mode, canonical awake layout, decision source NONE, gpu-truth
fresh, no ``breakpoint_window_suspended``, disk) -> ``run_arm_pilot.sh`` (all collectors of the
eval spec) -> validity verdict -> ``campaign_manifest.json`` + done marker -> post-arm reset to
the canonical state -> report for the arm + cross-arm report of the trace + campaign index
(plot failures are logged, never stop the campaign). A failed / invalid arm gets one automatic
retry after a full reset, then a failed marker and the campaign continues.

Resume: rerun the same command; arms with a valid done marker are skipped, a directory without
a marker (interrupted) is moved aside and counts as a failed attempt. ``<results_root>/STOP``
(or SIGINT / SIGTERM to this process) = stop cleanly after the current arm. Monitoring:
``<results_root>/status.json`` and the one-line-per-event ``<results_root>/campaign.log``.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import html
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

HERE = Path(__file__).resolve().parent
TRE_REPO_DEFAULT = HERE.parents[1]          # .../tre
BASE_ARMS = ("tre", "apa", "chiron", "tokenscale", "preserve")
BASELINE_ARMS = ("chiron", "tokenscale", "preserve")
DEFAULT_LABELS = {"tre": "TRE", "apa": "APA-kvcache", "chiron": "Chiron-global",
                  "tokenscale": "TokenScale-colocated", "preserve": "PreServe-oracle"}
DONE_MARKER = "CAMPAIGN_DONE.json"
FAILED_SUFFIX = ".CAMPAIGN_FAILED.json"
ORDER_METHODS = ("williams", "cyclic", "fixed")
PRESERVE_KEYS = {"trace_path", "trace_seed", "trace_schedule", "trace_match_parts", "window_s", "lead_s",
                 "tier1", "noise_sigma", "noise_seed", "mu", "max_output_len", "map_factor", "lookahead_iters",
                 "kv_high", "overload_frac", "t_f", "ext_frac", "out_len_is_upper_bound", "kv_capacity_tokens",
                 "kv_agree_tol", "hold_mode", "req_ttl_s"}
TIER1_MODES = ("oracle_noisy", "oracle", "last_window")
# collectors of docs/eval-metrics-spec §7 that every arm directory should hold (missing -> suspect)
COLLECTOR_FILES = ("layout.jsonl", "pod_metrics_1s.jsonl", "pod_gauges.jsonl", "gpu_map.jsonl", "gpu_truth.jsonl",
                   "resource_usage.jsonl", "apa_status.jsonl", "signal_log.jsonl", "clock_offsets.json",
                   "components.json", "trace_manifest.json", "controller.log", "sm.ts.log", "live-registry.yaml")

DEFAULTS: dict = {
    "trace_file": "traces_tre.effective.json",
    "seed": 1,
    "order": "williams",
    "max_attempts": 2,
    "gap_s": 0,
    "eta": {"overhead_s": 300, "report_s": 120},
    "env": {},
    "arm_defs": {},
    "preserve": {"trace_host_root": None, "trace_pod_root": "/etc/tre-baselines-traces", "match_parts": 3},
    "report": {"enabled": True, "per_arm": True, "capacity": None, "decision_points": None, "ref": "tre",
               "reps": 1000, "timeout_s": 900, "nice": 10, "extra_args": []},
    "precheck": {"gpu_truth_max_age_s": 30, "breakpoint_lookback_s": 60, "wait_s": 300, "poll_s": 30,
                 "disk_min_free_gb": {"nfs": 50, "nodes": {}}},
    "validity": {"fail_frac_suspect": 0.01, "fail_frac_invalid": 0.20, "sent_frac_min": 0.98,
                 "emfile": "invalid", "max_tokens_hit_min": 0.99, "clock_drift_max_s": 1.0},
    "marker": {"extend": True, "margin_s": 1800},
}


class CampaignError(Exception):
    pass


# ------------------------------------------------------------------ small helpers
def now_iso(t: Optional[float] = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t).astimezone().isoformat(timespec="seconds")


def hms(s: float) -> str:
    s = int(round(max(0.0, s)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def write_json_atomic(path: str | Path, doc: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


def write_text_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


_VAR = re.compile(r"\$\{(\w+)\}")


def expand(value: Any, env: dict) -> Any:
    """``${VAR}`` from ``env`` in every string of ``value`` (dict / list recursively)."""
    if isinstance(value, str):
        def sub(m):
            if m.group(1) not in env:
                raise CampaignError(f"${{{m.group(1)}}} is not set (campaign env block / environment)")
            return str(env[m.group(1)])
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    return value


# ------------------------------------------------------------------ campaign file
def _yaml():
    import yaml  # noqa: PLC0415
    return yaml


def load_campaign(path: str | Path, environ: Optional[dict] = None) -> dict:
    """Parse + normalise a campaign file. ``env:`` values are expanded in order (each may use
    the earlier ones, ``${TRE_REPO}`` and the caller's environment) and win over both the
    caller's environment and runner.env (recorded per arm)."""
    raw = _yaml().safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise CampaignError(f"{path}: top level must be a mapping")
    for k in ("name", "results_root", "traces"):
        if not raw.get(k):
            raise CampaignError(f"{path}: '{k}' is required")
    base_env = dict(os.environ if environ is None else environ)
    base_env.setdefault("TRE_REPO", str(TRE_REPO_DEFAULT))
    env_block: dict = {}
    for k, v in (raw.get("env") or {}).items():
        env_block[k] = str(expand(v if v is not None else "", {**base_env, **env_block}))
    scope = {**base_env, **env_block}
    c = deep_merge(DEFAULTS, {k: v for k, v in raw.items() if k != "env"})
    c = expand(c, scope)
    c["env"] = env_block
    c["round"] = str(c.get("round") or c["name"])
    c["campaign_file"] = str(Path(path).resolve())
    c["campaign_sha256"] = sha256_file(path)
    c.setdefault("arms", list(BASE_ARMS))
    if c["order"] not in ORDER_METHODS:
        raise CampaignError(f"order must be one of {ORDER_METHODS}")
    if int(c["max_attempts"]) < 1:
        raise CampaignError("max_attempts must be >= 1")
    c["_scope"] = scope
    return c


@dataclass
class ArmDef:
    id: str
    base: str
    label: str
    policy: dict = field(default_factory=dict)
    oracle_shift: Optional[dict] = None


def arm_def(c: dict, arm_id: str) -> ArmDef:
    d = (c.get("arm_defs") or {}).get(arm_id)
    if d is None:
        if arm_id not in BASE_ARMS:
            raise CampaignError(f"arm {arm_id!r}: not one of {BASE_ARMS} and not in arm_defs")
        return ArmDef(arm_id, arm_id, DEFAULT_LABELS[arm_id])
    base = d.get("base")
    if base not in BASE_ARMS:
        raise CampaignError(f"arm_defs.{arm_id}.base must be one of {BASE_ARMS}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", arm_id):
        raise CampaignError(f"arm id {arm_id!r}: letters, digits, _ . - only (it is a directory name)")
    pol = dict(d.get("policy") or {})
    shift = d.get("oracle_shift")
    if (pol or shift) and base != "preserve":
        raise CampaignError(f"arm_defs.{arm_id}: policy / oracle_shift are only supported for base preserve")
    bad = sorted(set(pol) - PRESERVE_KEYS) if pol else []
    if bad:
        raise CampaignError(f"arm_defs.{arm_id}.policy: unknown PreServe keys {bad}")
    if "tier1" in pol and pol["tier1"] not in TIER1_MODES:
        raise CampaignError(f"arm_defs.{arm_id}.policy.tier1 must be one of {TIER1_MODES}")
    if shift is not None:
        if float(shift.get("period_s", 0)) <= 0 or not (0 < float(shift.get("frac", 0.25)) <= 0.5):
            raise CampaignError(f"arm_defs.{arm_id}.oracle_shift needs period_s > 0 and 0 < frac <= 0.5")
    return ArmDef(arm_id, base, str(d.get("label") or DEFAULT_LABELS[base]), pol, shift)


# ------------------------------------------------------------------ arm order
def williams_rows(n: int) -> list[list[int]]:
    """Williams design: n rows (n even) or 2n rows (n odd, plus the mirrored rows). Every
    treatment is in every position equally often and every ordered adjacent pair occurs
    equally often (first-order carry-over balanced)."""
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    first, lo, hi = [0], 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo); lo += 1
        else:
            first.append(hi); hi -= 1
        take_lo = not take_lo
    rows = [[(x + i) % n for x in first] for i in range(n)]
    if n % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


def arm_order(arms: list[str], index: int, method: str) -> list[str]:
    n = len(arms)
    if method == "fixed" or n <= 1:
        return list(arms)
    if method == "cyclic":
        k = index % n
        return list(arms[k:] + arms[:k])
    rows = williams_rows(n)
    return [arms[j] for j in rows[index % len(rows)]]


# ------------------------------------------------------------------ plan
@dataclass
class TraceEntry:
    index: int
    trace: str
    seed: int
    key: str                 # trace dir name in the results ("<key>_s<seed>" is the out name)
    out_name: str
    src_dir: str             # generated trace (design.json, manifest.json, effective file)
    client_dir: str          # directory the client replays (src_dir, or the slice under results)
    trace_file: str
    config: str              # loadgen config of the trace
    client_config: str       # config given to the client (slice: duration rewritten)
    arms: list
    slice: Optional[dict]
    duration_s: float
    requests: Optional[int]
    note: str = ""
    preserve_policy: dict = field(default_factory=dict)   # per-trace PreServe params (e.g. window_s)


@dataclass
class PlanItem:
    n: int
    entry: int
    trace: str
    out_name: str
    arm: str
    base_arm: str
    label: str
    position: int
    k: int
    order: list
    arm_dir: str
    duration_s: float
    est_s: float

    @property
    def key(self) -> str:
        return f"{self.out_name}/{self.arm}"


def _manifest_info(src_dir: str) -> dict:
    m = read_json(Path(src_dir) / "manifest.json") or {}
    design = m.get("design") or {}
    return {"trace": m.get("trace"), "seed": m.get("seed"), "duration_s": m.get("duration_s"),
            "requests": design.get("requests"), "generator_sha": (m.get("generator") or {}).get("git_sha"),
            "out_cap": bool(m.get("out_cap")), "present": bool(m)}


def resolve_traces(c: dict) -> list[TraceEntry]:
    out = []
    root = Path(c["results_root"])
    for i, t in enumerate(c["traces"]):
        if isinstance(t, str):
            t = {"trace": t}
        name = t.get("trace")
        if not name:
            raise CampaignError(f"traces[{i}]: 'trace' is required")
        seed = int(t.get("seed", c["seed"]))
        src_dir = str(t.get("dir") or f"{c.get('trace_root', '')}/{name}/seed{seed}")
        sl = t.get("slice")
        if sl is not None:
            start, dur = float(sl["start_s"]), float(sl["duration_s"])
            if dur <= 0 or start < 0:
                raise CampaignError(f"traces[{i}].slice: start_s >= 0, duration_s > 0")
            sl = {"start_s": start, "duration_s": dur}
            key = t.get("key") or f"{name}_slice{int(start)}-{int(start + dur)}"
        else:
            key = t.get("key") or name
        out_name = f"{key}_s{seed}"
        config = str(t.get("config") or f"{c.get('loadgen_config_root', '')}/{name}/config.yaml")
        info = _manifest_info(src_dir)
        if sl:
            client_dir = str(root / "_traces" / key / f"seed{seed}")
            client_config = str(root / "_traces" / key / "config.yaml")
            duration = sl["duration_s"]
            requests = None
        else:
            client_dir, client_config = src_dir, config
            duration = float(t.get("duration_s") or info["duration_s"] or 0)
            requests = info["requests"]
        arms = list(t.get("arms") or c["arms"])
        if len(set(arms)) != len(arms):
            raise CampaignError(f"traces[{i}] ({name}): duplicate arms {arms}")
        pp = dict(t.get("preserve_policy") or {})
        bad = sorted(set(pp) - PRESERVE_KEYS)
        if bad:
            raise CampaignError(f"traces[{i}].preserve_policy: unknown PreServe keys {bad}")
        out.append(TraceEntry(i, name, seed, key, out_name, src_dir, client_dir, c["trace_file"], config,
                              client_config, arms, sl, duration, requests, str(t.get("note") or ""), pp))
    seen: dict = {}
    for e in out:
        for a in e.arms:
            k = f"{e.out_name}/{a}"
            if k in seen:
                raise CampaignError(f"{k} is planned twice (traces[{seen[k]}] and traces[{e.index}])")
            seen[k] = e.index
    return out


def est_arm_s(c: dict, duration_s: float, overhead_s: Optional[float] = None) -> float:
    eta = c["eta"]
    over = float(eta["overhead_s"] if overhead_s is None else overhead_s)
    rep = float(eta["report_s"]) if c["report"].get("enabled") else 0.0
    return float(duration_s) + over + rep + float(c.get("gap_s") or 0)


def build_plan(c: dict, entries: Optional[list[TraceEntry]] = None) -> list[PlanItem]:
    entries = resolve_traces(c) if entries is None else entries
    items, n = [], 0
    for e in entries:
        order = arm_order(e.arms, e.index, c["order"])
        for pos, a in enumerate(order, 1):
            d = arm_def(c, a)
            n += 1
            items.append(PlanItem(n, e.index, e.trace, e.out_name, a, d.base, d.label, pos, len(order), order,
                                  str(Path(c["results_root"]) / e.out_name / a), e.duration_s,
                                  est_arm_s(c, e.duration_s)))
    return items


# ------------------------------------------------------------------ markers / resume
def failed_marker_path(item: PlanItem) -> Path:
    return Path(item.arm_dir).parent / f"{item.arm}{FAILED_SUFFIX}"


def done_marker(item: PlanItem, campaign: str) -> Optional[dict]:
    """The done marker when it is valid for this plan item: same campaign + trace + arm,
    a non-invalid verdict and the client results still present."""
    m = read_json(Path(item.arm_dir) / DONE_MARKER)
    if not isinstance(m, dict):
        return None
    if m.get("campaign") != campaign or m.get("out_name") != item.out_name or m.get("arm") != item.arm:
        return None
    if (m.get("validity") or {}).get("verdict") not in ("valid", "suspect"):
        return None
    if not (Path(item.arm_dir) / "client" / "performance_metrics.json").is_file():
        return None
    return m


def item_status(item: PlanItem, campaign: str) -> str:
    if done_marker(item, campaign):
        return "done"
    if failed_marker_path(item).is_file():
        return "failed"
    if Path(item.arm_dir).exists():
        return "interrupted"
    return "pending"


def move_aside(path: Path, tag: str) -> Optional[Path]:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = path.with_name(f"{path.name}.{tag}-{stamp}")
    i = 1
    while dst.exists():
        dst = path.with_name(f"{path.name}.{tag}-{stamp}-{i}"); i += 1
    os.rename(path, dst)
    return dst


# ------------------------------------------------------------------ exclusive-window marker
def parse_marker(text: str) -> dict:
    lines = text.splitlines()
    first = lines[0].split() if lines else []
    out = {"type": first[0] if first else None, "start": first[1] if len(first) > 1 else None,
           "end": first[2] if len(first) > 2 else None, "tokens": first[3:], "end_epoch": None}
    if out["end"]:
        with contextlib.suppress(ValueError):
            out["end_epoch"] = dt.datetime.fromisoformat(out["end"]).timestamp()
    return out


def marker_path(c: dict) -> str:
    return c["env"].get("MARKER") or os.environ.get("MARKER", "")


def marker_is_ours(info: dict, round_: str) -> bool:
    return info.get("type") == "validation" and f"round={round_}" in (info.get("tokens") or [])


def extend_marker_text(text: str, new_end_epoch: float) -> tuple[str, Optional[str], Optional[str]]:
    """Line 1 field 3 (end time) -> max(old, new_end) rounded up to the minute, in the old
    end's UTC offset. Only the end time changes (rest of line 1 and every other line kept)."""
    lines = text.split("\n")
    parts = lines[0].split(" ") if lines else []
    fields = [p for p in parts if p]
    if len(fields) < 3:
        return text, None, None
    try:
        old = dt.datetime.fromisoformat(fields[2])
    except ValueError:
        return text, None, None
    if old.tzinfo is None:
        old = old.astimezone()
    if new_end_epoch <= old.timestamp():
        return text, fields[2], None
    new_epoch = math.ceil(new_end_epoch / 60.0) * 60
    new = dt.datetime.fromtimestamp(new_epoch, tz=old.tzinfo).isoformat(timespec="seconds")
    seen = 0
    for i, p in enumerate(parts):
        if p:
            seen += 1
            if seen == 3:
                parts[i] = new
                break
    lines[0] = " ".join(parts)
    return "\n".join(lines), fields[2], new


def extend_marker(path: str | Path, round_: str, eta_end_epoch: float, margin_s: float) -> Optional[str]:
    """Extend the marker's end when the ETA (+ margin) passes it. Returns a log line or None."""
    p = Path(path)
    try:
        text = p.read_text()
    except OSError:
        return None
    info = parse_marker(text)
    if not marker_is_ours(info, round_):
        return None
    new_text, old, new = extend_marker_text(text, eta_end_epoch + margin_s)
    if not new:
        return None
    write_text_atomic(p, new_text)
    return f"marker end {old} -> {new}"


# ------------------------------------------------------------------ trace preparation (files only)
def slice_trace(records: list, start_s: float, dur_s: float) -> list:
    """Requests with ``start_s <= timestamp < start_s + dur_s``, timestamps rebased to 0,
    order and every other field unchanged (same request ids)."""
    out = []
    for r in records:
        t = float(r["timestamp"])
        if start_s <= t < start_s + dur_s:
            out.append(dict(r, timestamp=round(t - start_s, 6)))
    out.sort(key=lambda r: r["timestamp"])
    return out


def strip_trace(records: list) -> list:
    """The PreServe oracle copy (no prompts): the fields the oracle reads (as baseline-traces/strip_trace.py)."""
    keep = ("request_id", "timestamp", "model_name", "prompt_length", "max_output_tokens")
    return [{k: r[k] for k in keep if k in r} for r in records]


def shift_delta(campaign: str, key: str, seed: int, shift: dict) -> float:
    """Deterministic U(-frac*T, frac*T) phase shift of the oracle (recorded)."""
    h = hashlib.sha256(f"{campaign}|{key}|{seed}|{shift.get('seed', 0)}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    half = float(shift.get("frac", 0.25)) * float(shift["period_s"])
    return round(rng.uniform(-half, half), 3)


def shift_trace(records: list, delta_s: float, duration_s: float) -> list:
    """Every arrival moved by ``delta_s``, wrapped into [0, duration): the oracle's view of the
    load is the true load phase-shifted by -delta (request count and lengths unchanged)."""
    out = [dict(r, timestamp=round((float(r["timestamp"]) + delta_s) % duration_s, 6)) for r in records]
    out.sort(key=lambda r: r["timestamp"])
    return out


def write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n]\n")
    os.replace(tmp, path)


def prepare_slice(e: TraceEntry, log: Callable[[str], None]) -> None:
    """The smoke slice: <results>/_traces/<key>/seed<k>/{traces_tre.effective.json, manifest.json}
    + <results>/_traces/<key>/config.yaml (duration rewritten). Rebuilt only when the source changed."""
    src = Path(e.src_dir) / e.trace_file
    dst = Path(e.client_dir) / e.trace_file
    man_dst = Path(e.client_dir) / "manifest.json"
    src_sha = sha256_file(src)
    old = read_json(man_dst) or {}
    if dst.is_file() and (old.get("slice") or {}).get("source_sha256") == src_sha \
            and (old.get("slice") or {}).get("start_s") == e.slice["start_s"] \
            and (old.get("slice") or {}).get("duration_s") == e.slice["duration_s"]:
        e.requests = (old.get("slice") or {}).get("requests")
        return
    recs = json.loads(src.read_text())
    sl = slice_trace(recs, e.slice["start_s"], e.slice["duration_s"])
    if not sl:
        raise CampaignError(f"slice {e.slice} of {src} is empty")
    write_records(dst, sl)
    man = read_json(Path(e.src_dir) / "manifest.json") or {}
    man = dict(man, duration_s=e.slice["duration_s"],
               slice={"source_file": str(src), "source_sha256": src_sha, "source_manifest_sha256":
                      sha256_file(Path(e.src_dir) / "manifest.json"), "start_s": e.slice["start_s"],
                      "duration_s": e.slice["duration_s"], "requests": len(sl), "made_by": "tre/eval/runner/campaign.py"})
    write_json_atomic(man_dst, man)
    cfg = Path(e.config).read_text()
    cfg2, n = re.subn(r"(?m)^(\s*duration_seconds:\s*)\d+(\.\d+)?", lambda m: f"{m.group(1)}{int(e.slice['duration_s'])}", cfg)
    if n != 1:
        raise CampaignError(f"{e.config}: expected one duration_seconds line, found {n}")
    write_text_atomic(e.client_config, f"# slice {e.slice} of {e.config} (campaign.py)\n" + cfg2)
    e.requests = len(sl)
    log(f"slice {e.out_name}: {len(sl)} requests [{e.slice['start_s']}, +{e.slice['duration_s']}) of {src}")


def render_preserve_policy(bl_cm_file: str, trace_pod_path: str, match_parts: int, overrides: dict) -> tuple[str, dict]:
    """The frozen policy-configmaps file with the PreServe ConfigMap's params changed: trace_path
    (this trace's oracle copy), trace_match_parts, then the arm's overrides. Every other ConfigMap
    (and every other PreServe key) is unchanged. Returns (yaml text, preserve params)."""
    yaml = _yaml()
    docs = list(yaml.safe_load_all(Path(bl_cm_file).read_text()))
    params = None
    for d in docs:
        if d and d.get("kind") == "ConfigMap" and (d.get("metadata") or {}).get("name") == "tre-v2-baseline-preserve":
            params = yaml.safe_load((d.get("data") or {}).get("preserve.yaml") or "{}") or {}
            params["trace_path"] = trace_pod_path
            params["trace_match_parts"] = int(match_parts)
            params.update(overrides or {})
            d["data"]["preserve.yaml"] = yaml.safe_dump(params, sort_keys=False)
    if params is None:
        raise CampaignError(f"{bl_cm_file} has no ConfigMap tre-v2-baseline-preserve")
    head = f"# rendered by tre/eval/runner/campaign.py from {bl_cm_file} (sha256 {sha256_file(bl_cm_file)})\n"
    return head + yaml.safe_dump_all([d for d in docs if d], sort_keys=False), params


def _tail(path: str, parts: int) -> tuple:
    bits = [b for b in str(path).replace("\\", "/").split("/") if b]
    return tuple(bits[-parts:]) if parts > 0 else ()


def prepare_preserve(c: dict, e: TraceEntry, d: ArmDef, log: Callable[[str], None]) -> dict:
    """Oracle copy of the replayed trace (stripped; phase-shifted for an oracle_shift arm) under
    the shell's trace volume + the rendered policy-configmaps file. Returns the env + record."""
    pc = c["preserve"]
    host_root, pod_root = pc.get("trace_host_root"), pc.get("trace_pod_root")
    if not host_root or not pod_root:
        raise CampaignError("preserve.trace_host_root / trace_pod_root are required for PreServe arms")
    bl_cm = os.environ.get("BL_CM_FILE") or c["env"].get("BL_CM_FILE")
    if not bl_cm:
        raise CampaignError("BL_CM_FILE (frozen policy-configmaps.yaml) is required for PreServe arms")
    parts = int(pc.get("match_parts", 3))
    client_trace = Path(e.client_dir) / e.trace_file
    tail = _tail(str(client_trace), parts)
    rec: dict = {"client_trace": str(client_trace), "match_parts": parts}
    if d.oracle_shift:
        delta = shift_delta(c["name"], e.key, e.seed, d.oracle_shift)
        oracle_id = f"shift{delta:+.3f}"
        rec["oracle_shift"] = dict(d.oracle_shift, delta_s=delta, duration_s=e.duration_s)
    else:
        delta, oracle_id = None, "plain"
    rel = Path(c["name"]) / oracle_id / Path(*tail)
    host_path, pod_path = Path(host_root) / rel, str(Path(pod_root) / rel)
    meta_path = host_path.with_name(host_path.name + ".meta.json")
    src_sha = sha256_file(client_trace)
    meta = read_json(meta_path) or {}
    if not (host_path.is_file() and meta.get("source_sha256") == src_sha and meta.get("delta_s") == delta):
        recs = strip_trace(json.loads(client_trace.read_text()))
        if delta is not None:
            recs = shift_trace(recs, delta, e.duration_s)
        write_records(host_path, recs)
        write_json_atomic(meta_path, {"source": str(client_trace), "source_sha256": src_sha, "delta_s": delta,
                                      "requests": len(recs), "sha256": sha256_file(host_path)})
        log(f"preserve oracle {e.out_name}/{d.id}: {host_path} ({len(recs)} requests, shift {delta})")
    overrides = {**e.preserve_policy, **d.policy}
    text, params = render_preserve_policy(bl_cm, pod_path, parts, overrides)
    pol_path = Path(c["results_root"]) / "_policy" / e.out_name / f"{d.id}.policy-configmaps.yaml"
    write_text_atomic(pol_path, text)
    rec.update(oracle_host_path=str(host_path), oracle_pod_path=pod_path, oracle_sha256=sha256_file(host_path),
               policy_file=str(pol_path), policy_sha256=sha256_file(pol_path), policy_params_changed=
               {"trace_path": pod_path, "trace_match_parts": parts, **overrides})
    return {"env": {"BL_CM_FILE": str(pol_path), "BL_CM_APPLY": "1"}, "record": rec}


# ------------------------------------------------------------------ validity
def _count_lines(path: Path, pattern: re.Pattern) -> int:
    try:
        with open(path, errors="replace") as f:
            return sum(1 for line in f if pattern.search(line))
    except OSError:
        return 0


_HTTP_409 = re.compile(r'HTTP/1\.[01]" 409\b')
_HTTP_5XX = re.compile(r'HTTP/1\.[01]" 5\d\d\b')


def assess_arm(arm_dir: str | Path, rc: Optional[int], expected_requests: Optional[int], thr: dict,
               baseline: bool = False) -> dict:
    """Verdict of one finished arm: ``valid`` | ``suspect`` (recorded, kept) | ``invalid``
    (retried once). Overload outcomes of the system under test (SLO misses, 150 s cuts) are
    results, not invalidity; invalid means the run itself broke (rc, client gone, sidecar
    EMFILE, baseline event loss, gross client failure)."""
    d = Path(arm_dir)
    inv, sus, m = [], [], {}
    m["rc"] = rc
    if rc != 0:
        inv.append(f"run_arm_pilot rc={rc}")
    if not (d / "client" / "performance_metrics.json").is_file():
        inv.append("no client/performance_metrics.json")
    score = read_json(d / "score.json")
    if isinstance(score, dict):
        a = (score.get("models") or {}).get("ALL") or {}
        n, trimmed, fail = int(a.get("n") or 0), int(a.get("trimmed") or 0), int(a.get("fail") or 0)
        m.update(n=n, trimmed=trimmed, fail=fail, V_req_pct=a.get("V_req_pct"),
                 max_tokens_hit_frac=a.get("max_tokens_hit_frac"), ttft_p95_ms=a.get("ttft_p95_ms"))
        if expected_requests:
            sent = (n + trimmed) / float(expected_requests)
            m["sent_frac"] = round(sent, 4)
            if sent < float(thr["sent_frac_min"]):
                inv.append(f"client recorded {n + trimmed}/{expected_requests} requests ({sent:.1%})")
        if n:
            ff = fail / float(n)
            m["fail_frac"] = round(ff, 5)
            if ff > float(thr["fail_frac_invalid"]):
                inv.append(f"client failures {fail}/{n} ({ff:.1%}) > {thr['fail_frac_invalid']:.0%}")
            elif ff > float(thr["fail_frac_suspect"]):
                sus.append(f"client failures {fail}/{n} ({ff:.2%})")
        mth = a.get("max_tokens_hit_frac")
        if mth is not None and float(mth) < float(thr["max_tokens_hit_min"]):
            sus.append(f"max_tokens_hit_frac {mth} < {thr['max_tokens_hit_min']} (ignore_eos not effective?)")
        if score.get("valid") is False:
            inv.append(f"score.json valid=false: {(score.get('run_validity') or {}).get('invalid_because')}")
    elif rc == 0:
        sus.append("no score.json (score_pilot failed)")
    emf = d / "EMFILE_PODS"
    if emf.is_file() and emf.read_text().strip():
        pods = [Path(x).stem for x in emf.read_text().split()]
        m["emfile_pods"] = pods
        (inv if thr.get("emfile", "invalid") == "invalid" else sus).append(f"EMFILE in sidecar logs: {pods}")
    sm_log = d / "sm.ts.log" if (d / "sm.ts.log").is_file() else d / "sm.log"
    m["sm_409"] = _count_lines(sm_log, _HTTP_409)
    m["sm_5xx"] = _count_lines(sm_log, _HTTP_5XX)
    if m["sm_5xx"]:
        sus.append(f"SM answered {m['sm_5xx']} requests with 5xx")
    if baseline:
        rv = read_json(d / "baseline" / "run_validity.json")
        if not isinstance(rv, dict):
            if rc == 0:
                inv.append("baseline arm without baseline/run_validity.json")
        else:
            if rv.get("events_valid") is False:
                inv.append(f"baseline events invalid: {rv.get('invalid_because')}")
            smf = float(((rv.get("sm") or {}).get("tre_bl_sm_failures_total")) or 0)
            m["bl_sm_failures"] = smf
            if smf:
                sus.append(f"baseline shell: {int(smf)} SM call failures")
    co = read_json(d / "clock_offsets.json")
    if isinstance(co, dict):
        for node in (co.get("offsets_s") or {}):
            vals = [((co.get(ph) or {}).get("nodes") or {}).get(node, {}).get("offset_s") for ph in ("start", "end")]
            if None not in vals and abs(vals[0] - vals[1]) > float(thr["clock_drift_max_s"]):
                sus.append(f"clock offset of {node} moved {vals[0]} -> {vals[1]} s during the arm")
    missing = [f for f in COLLECTOR_FILES if not (d / f).is_file() or (d / f).stat().st_size == 0]
    if rc == 0 and missing:
        m["missing_collectors"] = missing
        sus.append(f"missing / empty collector files: {missing}")
    verdict = "invalid" if inv else ("suspect" if sus else "valid")
    return {"verdict": verdict, "invalid": inv, "suspect": sus, "metrics": m}


# ------------------------------------------------------------------ per-arm manifest
def _kv_file(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep:
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def arm_manifest(c: dict, item: PlanItem, e: TraceEntry, attempt: int, t0: float, t1: float, rc: Optional[int],
                 verdict: dict, extra: dict) -> dict:
    d = Path(item.arm_dir)
    comp = read_json(d / "components.json") or {}
    tm = read_json(d / "trace_manifest.json") or {}
    src_man = read_json(d / "trace_source_manifest.json") or {}
    clock = read_json(d / "clock_offsets.json") or {}
    try:
        images = [ln.split(" ", 1) for ln in (d / "images.txt").read_text().splitlines() if ln.strip()]
    except OSError:
        images = []
    return {
        "campaign": c["name"], "campaign_file": c["campaign_file"], "campaign_sha256": c["campaign_sha256"],
        "round": c["round"], "item": item.n, "out_name": item.out_name, "trace": item.trace, "seed": e.seed,
        "arm": item.arm, "base_arm": item.base_arm, "label": item.label,
        "order_position": item.position, "order_k": item.k, "order": item.order, "order_method": c["order"],
        "attempt": attempt, "started_at": now_iso(t0), "ended_at": now_iso(t1), "wall_s": round(t1 - t0, 1), "rc": rc,
        "runner": _kv_file(d / "runner_sha"), "campaign_runner_sha": git_sha(HERE),
        "tre_sha": (d / "tre_sha").read_text().strip() if (d / "tre_sha").is_file() else None,
        "loadgen": _kv_file(d / "loadgen_sha"),
        "images": {k: v for k, v in (x for x in images if len(x) == 2)},
        "image_ids_by_node": comp.get("image_ids_by_node"), "component_nodes": comp.get("component_nodes"),
        "registry_sha256": sha256_file(d / "live-registry.yaml"),
        "trace_file": tm.get("trace_file"), "trace_sha256": tm.get("trace_sha256"),
        "config_file": tm.get("config_file"), "config_sha256": tm.get("config_sha256"),
        "trace_source_manifest_sha256": tm.get("source_manifest_sha256"),
        "trace_generator_sha": (src_man.get("generator") or {}).get("git_sha"),
        "trace_design_sha256": (src_man.get("design") or {}).get("sha256"),
        "trace_slice": src_man.get("slice"), "expected_requests": e.requests,
        "clock_offsets_s": clock.get("offsets_s"),
        "env": {k: c["env"][k] for k in sorted(c["env"])},
        "validity": verdict, **extra,
    }


def git_sha(path: Path) -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(path), *a], text=True, capture_output=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"sha": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len((g("status", "--porcelain", "--", ".") or "").splitlines())}


# ------------------------------------------------------------------ cluster (pre-checks)
class Cluster:
    """Read-only probes of the pre-checks + the scripts that change state (reset, arm)."""

    def __init__(self, env: Optional[dict] = None):
        self.e = dict(os.environ if env is None else env)
        self._sm = None

    def sh(self, args: list, timeout: float = 60) -> subprocess.CompletedProcess:
        return subprocess.run(args, text=True, capture_output=True, timeout=timeout)

    def kubectl(self, *args, timeout: float = 60) -> str:
        r = self.sh(["kubectl", *args], timeout)
        if r.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args[:4])}: {r.stderr.strip()[:200]}")
        return r.stdout

    def redis_get(self, key: str) -> str:
        r = self.sh(["kubectl", "-n", self.e.get("TRE_NS", "tre-v2"), "exec", f"deploy/{self.e.get('REDIS_DEPLOY', 'tre-v2-redis')}",
                     "--", "redis-cli", "--raw", "GET", key], 30)
        if r.returncode != 0:
            raise RuntimeError(f"redis GET {key}: {r.stderr.strip()[:200]}")
        return r.stdout.strip()

    def sm_url(self) -> str:
        if self._sm:
            return self._sm
        if self.e.get("SM_URL"):
            self._sm = self.e["SM_URL"].rstrip("/")
        else:
            ip = self.kubectl("-n", self.e.get("TRE_NS", "tre-v2"), "get", "svc", self.e.get("SM_SVC", "tre-v2-service-manager"),
                              "-o", "jsonpath={.spec.clusterIP}").strip()
            self._sm = f"http://{ip}:{self.e.get('SM_PORT', '8000')}"
        return self._sm

    def sm_state(self) -> dict:
        import urllib.request  # noqa: PLC0415
        with urllib.request.urlopen(f"{self.sm_url()}/v2/state", timeout=10) as r:
            return json.load(r)

    def clock_offset(self, node: str) -> Optional[float]:
        targets = dict(p.split("=", 1) for p in self.e.get("CLOCK_NODES", "").split() if "=" in p)
        t = targets.get(node)
        if t is None:
            return None
        sys.path.insert(0, str(HERE))
        import clock_probe  # noqa: PLC0415
        return clock_probe.probe(t, shlex.split(self.e.get("CLOCK_SSH_OPTS", "-o BatchMode=yes -o ConnectTimeout=5")))["offset_s"]

    # each check -> {"name", "ok", "detail", "kind": reset | wait | hard}
    def checks(self, c: dict, need_baseline: bool) -> list[dict]:
        out = []

        def add(name, kind, fn):
            try:
                ok, detail = fn()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {str(exc)[:200]}"
            out.append({"name": name, "ok": bool(ok), "detail": detail, "kind": kind})

        pc = c["precheck"]
        tre = self.e.get("TRE_DIR", "")

        def marker():
            p = Path(marker_path(c))
            if not p.is_file():
                return False, f"no exclusive-window marker {p}"
            info = parse_marker(p.read_text())
            return marker_is_ours(info, c["round"]), f"{info['type']} end={info['end']} tokens={info['tokens']}"
        add("marker", "hard", marker)

        def no_other_runner():
            r = self.sh(["pgrep", "-af", r"^bash .*run_arm_pilot\.sh"], 10)
            pids = [ln for ln in r.stdout.splitlines() if ln.strip()]
            return not pids, f"{len(pids)} run_arm_pilot.sh process(es) running" + (f": {pids[0][:160]}" if pids else "")
        add("no_other_runner", "hard", no_other_runner)

        def run_mode():
            r = self.sh(["bash", f"{tre}/deploy/scripts/set_run_mode.sh", "status"], 60)
            kv = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln)
            got = (kv.get("controller_mode"), kv.get("sm_actuation"))
            return got == ("observe", "active"), f"controller={got[0]} sm={got[1]} (want observe active)"
        add("run_mode", "reset", run_mode)

        def layout():
            st = self.sm_state()
            awake = sorted(b["binding_id"] for b in st["bindings"] if b.get("awake"))
            hidden = sorted(b["binding_id"] for b in st["bindings"] if b.get("hidden"))
            want = sorted(self.e.get("BASELINE", "").split())
            return awake == want and not hidden, f"awake={awake} hidden={hidden} want={want}"
        add("layout", "reset", layout)

        def decision_source():
            crs = [x for x in self.kubectl("-n", self.e.get("APA_NS", "default"), "get",
                                           "podautoscalers.autoscaling.aibrix.ai", "-o", "name").split() if x]
            probs = [f"{len(crs)} APA CR(s)"] if crs else []
            try:
                rep = self.kubectl("-n", self.e.get("BL_NS", "tre-v2"), "get", "deploy", self.e.get("BL_DEPLOY", "tre-v2-baseline-scaler"),
                                   "-o", "jsonpath={.spec.replicas}").strip()
                if rep not in ("0", ""):
                    probs.append(f"baseline scaler replicas={rep}")
            except RuntimeError as exc:
                if need_baseline:
                    probs.append(f"baseline scaler: {exc}")
            owner = self.redis_get(self.e.get("BL_OWNER_KEY", "tre:v2:bl:owner"))
            if owner:
                probs.append(f"baseline owner lock held: {owner[:80]}")
            return not probs, "; ".join(probs) or "NONE (0 APA CRs, scaler 0, no owner)"
        add("decision_source", "reset", decision_source)

        def gpu_truth():
            nodes = [p.split("=", 1)[0] for p in self.e.get("CLOCK_NODES", "").split() if "=" in p]
            if not nodes:
                return False, "CLOCK_NODES empty: no node list"
            bad, det = [], []
            for node in nodes:
                raw = self.redis_get(f"tre:gpu_truth:{node}")
                if not raw:
                    bad.append(node); det.append(f"{node}: missing"); continue
                ts = float(json.loads(raw).get("timestamp") or 0)
                off = self.clock_offset(node) or 0.0
                age = time.time() - (ts - off)
                det.append(f"{node}: age {age:.0f}s (offset {off:+.1f})")
                if age > float(pc["gpu_truth_max_age_s"]):
                    bad.append(node)
            return not bad, "; ".join(det)
        add("gpu_truth", "wait", gpu_truth)

        def breakpoint_ok():
            logs = self.kubectl("-n", self.e.get("TRE_NS", "tre-v2"), "logs", f"deploy/{self.e.get('CONTROLLER_DEPLOY', 'tre-v2-controller')}",
                                f"--since={int(pc['breakpoint_lookback_s'])}s", timeout=60)
            n = logs.count("breakpoint_window_suspended")
            return n == 0, f"{n} breakpoint_window_suspended line(s) in the last {pc['breakpoint_lookback_s']} s"
        add("breakpoint_window", "wait", breakpoint_ok)

        def disk():
            want = pc.get("disk_min_free_gb") or {}
            det, bad = [], []
            nfs_min = want.get("nfs")
            if nfs_min is not None:
                p = Path(c["results_root"])
                while not p.exists():
                    p = p.parent
                free = shutil.disk_usage(p).free / 1e9
                det.append(f"results fs {free:.0f} GB free")
                if free < float(nfs_min):
                    bad.append("results fs")
            targets = dict(p.split("=", 1) for p in self.e.get("CLOCK_NODES", "").split() if "=" in p)
            for node, gb in (want.get("nodes") or {}).items():
                t = targets.get(node)
                cmd = ["df", "-Pk", "/"] if t in (None, "local") else \
                    ["ssh", *shlex.split(self.e.get("CLOCK_SSH_OPTS", "-o BatchMode=yes -o ConnectTimeout=5")), t, "df -Pk /"]
                r = self.sh(cmd, 30)
                free = int(r.stdout.splitlines()[-1].split()[3]) * 1024 / 1e9
                det.append(f"{node} / {free:.0f} GB free")
                if free < float(gb):
                    bad.append(node)
            return not bad, "; ".join(det)
        add("disk", "hard", disk)
        return out


def precheck_until_ready(checks: Callable[[], list], reset: Callable[[bool], int], c: dict,
                         log: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
                         clock: Callable[[], float] = time.time) -> tuple[bool, list]:
    """Pre-checks with recovery: a ``reset`` failure triggers one full reset, a ``wait`` failure
    (gpu-truth stale, O1 window suspended) is polled up to precheck.wait_s, a ``hard`` failure
    (marker, disk) blocks at once. Returns (ready, last results)."""
    pc = c["precheck"]
    deadline = clock() + float(pc["wait_s"])
    did_reset = False
    while True:
        res = checks()
        bad = [r for r in res if not r["ok"]]
        if not bad:
            return True, res
        log("precheck: " + "; ".join(f"{r['name']}: {r['detail']}" for r in bad))
        if any(r["kind"] == "hard" for r in bad):
            return False, res
        if any(r["kind"] == "reset" for r in bad):
            if did_reset:
                return False, res
            did_reset = True
            log("precheck: full reset")
            reset(True)
            continue
        if clock() >= deadline:
            return False, res
        sleep(float(pc["poll_s"]))


# ------------------------------------------------------------------ input validation (offline)
def validate_inputs(c: dict, entries: list[TraceEntry], plan: list[PlanItem]) -> tuple[list, list]:
    problems, warnings = [], []
    env = {**os.environ, **c["env"]}
    for e in entries:
        src = Path(e.src_dir) / e.trace_file
        if not src.is_file():
            problems.append(f"{e.out_name}: missing trace {src}")
        info = _manifest_info(e.src_dir)
        if not info["present"]:
            problems.append(f"{e.out_name}: missing {e.src_dir}/manifest.json")
        else:
            if info["trace"] != e.trace:
                problems.append(f"{e.out_name}: manifest trace={info['trace']!r} != {e.trace!r}")
            if info["seed"] != e.seed:
                problems.append(f"{e.out_name}: manifest seed={info['seed']} != {e.seed}")
            if not info["out_cap"]:
                warnings.append(f"{e.out_name}: manifest has no out_cap (generated before the route-timeout cap?)")
        if not Path(e.config).is_file():
            problems.append(f"{e.out_name}: missing loadgen config {e.config}")
        if e.slice:
            if info["duration_s"] and e.slice["start_s"] + e.slice["duration_s"] > float(info["duration_s"]):
                problems.append(f"{e.out_name}: slice {e.slice} beyond the trace ({info['duration_s']} s)")
        elif not e.duration_s:
            problems.append(f"{e.out_name}: duration unknown (manifest duration_s)")
    for a in sorted({i.arm for i in plan}):
        try:
            arm_def(c, a)
        except CampaignError as exc:
            problems.append(str(exc))
    for f in ("run_arm_pilot.sh", "reset_canonical.sh", "lib_env.sh", "sampler.py", "clock_probe.py", "components.py"):
        if not (HERE / f).is_file():
            problems.append(f"runner file missing: {HERE / f}")
    tre = env.get("TRE_DIR")
    if not tre:
        problems.append("TRE_DIR not set (campaign env / runner.env)")
    else:
        for f in ("deploy/scripts/set_run_mode.sh", "deploy/scripts/toggle_tre_apa.sh", "deploy/scripts/release/awake_ctl.py"):
            if not (Path(tre) / f).is_file():
                problems.append(f"TRE_DIR {tre}: missing {f}")
        if any(i.base_arm in BASELINE_ARMS for i in plan) and not (Path(tre) / "baselines/tre_baselines/tools/arm.py").is_file():
            problems.append(f"TRE_DIR {tre}: no baselines arm tool")
    lg = env.get("LOADGEN_TRE_DIR")
    cli = Path(lg or "") / "loadgen_v1/tre_loadgen_v1/cli.py"
    if not lg or not cli.is_file():
        problems.append(f"LOADGEN_TRE_DIR {lg!r}: no loadgen_v1 cli")
    else:
        txt = cli.read_text()
        for flag in ("--ignore-eos", "--send-in-tokens"):
            if flag not in txt:
                problems.append(f"{cli}: no {flag}")
    if any(i.base_arm in BASELINE_ARMS for i in plan):
        bl = env.get("BL_CM_FILE")
        if not bl or not Path(bl).is_file():
            problems.append(f"BL_CM_FILE {bl!r} missing (frozen baseline policy ConfigMaps)")
        else:
            names = set(re.findall(r"(?m)^\s*name:\s*(tre-v2-baseline-\w+)", Path(bl).read_text()))
            for base in sorted({i.base_arm for i in plan if i.base_arm in BASELINE_ARMS}):
                if f"tre-v2-baseline-{base}" not in names:
                    problems.append(f"BL_CM_FILE {bl}: no ConfigMap tre-v2-baseline-{base}")
    if any(i.base_arm == "preserve" for i in plan) and env.get("BL_CM_FILE") and Path(env["BL_CM_FILE"]).is_file():
        for a in sorted({i.arm for i in plan if i.base_arm == "preserve"}):
            try:
                d = arm_def(c, a)
                _, params = render_preserve_policy(env["BL_CM_FILE"], "/x/t/seed1/f.json",
                                                   int(c["preserve"].get("match_parts", 3)), d.policy)
                warnings.append(f"preserve arm {a}: tier1={params.get('tier1')} noise_sigma={params.get('noise_sigma')} "
                                f"window_s={params.get('window_s')} match_parts={params.get('trace_match_parts')}"
                                f"{' oracle_shift=' + str(d.oracle_shift) if d.oracle_shift else ''}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"preserve arm {a}: policy render failed: {exc}")
    if any(i.base_arm == "preserve" for i in plan):
        hr = c["preserve"].get("trace_host_root")
        if not hr or not Path(hr).is_dir():
            problems.append(f"preserve.trace_host_root {hr!r} is not a directory (the shell's trace volume)")
    rep = c["report"]
    if rep.get("enabled"):
        if not (HERE.parent / "report.py").is_file():
            problems.append(f"no report tool {HERE.parent / 'report.py'}")
        for k in ("capacity", "decision_points"):
            if rep.get(k) and not Path(rep[k]).is_file():
                problems.append(f"report.{k} {rep[k]} missing")
    if not Path(c["results_root"]).parent.is_dir():
        problems.append(f"results_root parent {Path(c['results_root']).parent} does not exist")
    for label, path in (("runner", HERE), ("TRE_DIR", tre), ("LOADGEN_TRE_DIR", lg)):
        if path and Path(path).exists():
            w = provenance_warning(label, Path(path))
            if w:
                warnings.append(w)
    return problems, warnings


def provenance_warning(label: str, path: Path) -> Optional[str]:
    g = git_sha(path)
    if not g["sha"]:
        return f"{label} {path}: not a git checkout (sha not recorded)"
    r = subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor", g["sha"], "origin/main"],
                       capture_output=True)
    notes = []
    if r.returncode != 0:
        notes.append("not in origin/main")
    if g["dirty_files"]:
        notes.append(f"{g['dirty_files']} dirty files")
    if notes:
        return (f"{label} {path} @ {g['sha'][:10]} ({g['branch']}): {', '.join(notes)} - AGENTS.md rule 6: "
                f"data-producing runs need the sha in main and pushed (the smoke may run from a branch)")
    return None


# ------------------------------------------------------------------ status / log / index
class Recorder:
    def __init__(self, c: dict, quiet: bool = False):
        self.c = c
        self.root = Path(c["results_root"])
        self.quiet = quiet

    def line(self, msg: str) -> None:
        ln = f"{now_iso()} {msg}"
        if not self.quiet:
            print(ln, flush=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "campaign.log", "a") as f:
            f.write(ln + "\n")

    def status(self, doc: dict) -> None:
        write_json_atomic(self.root / "status.json", dict(doc, updated_at=now_iso()))


def eta_seconds(c: dict, plan: list[PlanItem], statuses: dict, observed_over: list[float],
                current: Optional[PlanItem] = None, current_elapsed: float = 0.0) -> float:
    """Remaining wall seconds: pending arms (and the current one) at duration + overhead; the
    overhead is the median observed (wall - trace duration) of this campaign once known."""
    over = sorted(observed_over)[len(observed_over) // 2] if observed_over else None
    rem = 0.0
    for it in plan:
        if current is not None and it.key == current.key:
            rem += max(0.0, est_arm_s(c, it.duration_s, over) - current_elapsed)
        elif statuses.get(it.key) in ("pending", "interrupted", None):
            rem += est_arm_s(c, it.duration_s, over)
    return rem


def write_index(c: dict, plan: list[PlanItem], status: dict) -> Path:
    root = Path(c["results_root"])
    rows_by_trace: dict = {}
    for it in plan:
        rows_by_trace.setdefault(it.out_name, []).append(it)
    parts = [f"<h1>{html.escape(c['name'])}</h1>",
             f"<p class=meta>state <b>{html.escape(str(status.get('state')))}</b> · updated {html.escape(now_iso())} · "
             f"ETA end {html.escape(str(status.get('eta_end') or '-'))} · campaign file "
             f"<code>{html.escape(c['campaign_file'])}</code></p>"]
    cur = status.get("current") or {}
    if cur:
        parts.append(f"<p class=meta>running: <b>{html.escape(str(cur.get('out_name')))}/{html.escape(str(cur.get('arm')))}</b> "
                     f"(position {cur.get('position')}/{cur.get('k')}, attempt {cur.get('attempt')}, expected end {html.escape(str(cur.get('expected_end')))})</p>")
    for out_name, items in rows_by_trace.items():
        rep = root / out_name / "report" / "index.html"
        link = f' · <a href="{html.escape(out_name)}/report/index.html">cross-arm report</a>' if rep.is_file() else ""
        parts.append(f"<h2>{html.escape(out_name)}{link}</h2><table><tr><th>#</th><th>pos</th><th>arm</th><th>label</th>"
                     "<th>status</th><th>verdict</th><th>V_req %</th><th>fail</th><th>P95 TTFT ms</th><th>wall</th><th>notes</th></tr>")
        for it in items:
            m = read_json(Path(it.arm_dir) / DONE_MARKER) or read_json(failed_marker_path(it)) or {}
            met = (m.get("validity") or {}).get("metrics") or {}
            st = item_status(it, c["name"])
            if cur.get("out_name") == it.out_name and cur.get("arm") == it.arm:
                st = "running"
            arm_rep = Path(it.arm_dir) / "report" / "index.html"
            arm_cell = (f'<a href="{html.escape(out_name)}/{html.escape(it.arm)}/report/index.html">{html.escape(it.arm)}</a>'
                        if arm_rep.is_file() else html.escape(it.arm))
            notes = "; ".join((m.get("validity") or {}).get("invalid", []) + (m.get("validity") or {}).get("suspect", []))
            parts.append(f"<tr class={st}><td>{it.n}</td><td>{it.position}/{it.k}</td><td>{arm_cell}</td><td>{html.escape(it.label)}</td>"
                         f"<td>{st}</td><td>{html.escape(str((m.get('validity') or {}).get('verdict', '')))}</td>"
                         f"<td>{met.get('V_req_pct', '')}</td><td>{met.get('fail', '')}</td><td>{met.get('ttft_p95_ms', '')}</td>"
                         f"<td>{hms(m['wall_s']) if m.get('wall_s') else ''}</td><td>{html.escape(notes[:300])}</td></tr>")
        parts.append("</table>")
    css = (":root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--ok:#1a7f37;--bad:#c62828;--run:#7b3fe4}"
           "@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--ok:#4cc26b;--bad:#ff6b6b;--run:#b48cff}}"
           "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1200px}"
           "table{border-collapse:collapse;width:100%;margin-bottom:12px;display:block;overflow-x:auto}"
           "td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}"
           ".meta{color:var(--mut)}a{color:var(--run)}tr.done td:nth-child(5){color:var(--ok)}"
           "tr.failed td:nth-child(5){color:var(--bad)}tr.running td:nth-child(5){color:var(--run);font-weight:600}")
    doc = (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Campaign {html.escape(c['name'])}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
    p = root / "index.html"
    write_text_atomic(p, doc)
    return p


# ------------------------------------------------------------------ reports
def report_cmds(c: dict, item: PlanItem, trace_arm_dirs: list[str]) -> list[tuple[str, list]]:
    rep = c["report"]
    py = [sys.executable, str(HERE.parent / "report.py")]
    common: list = []
    if rep.get("capacity"):
        common += ["--capacity", rep["capacity"]]
    if rep.get("decision_points"):
        common += ["--decision-points", rep["decision_points"]]
    common += ["--reps", str(rep.get("reps", 1000))] + [str(x) for x in rep.get("extra_args") or []]
    nice = ["nice", "-n", str(rep["nice"])] if rep.get("nice") else []
    cmds = []
    if rep.get("per_arm"):
        cmds.append(("arm", nice + py + [item.arm_dir, "--out", str(Path(item.arm_dir) / "report")] + common))
    if trace_arm_dirs:
        names = [Path(d).name for d in trace_arm_dirs]
        ref = rep.get("ref") if rep.get("ref") in names else names[0]
        cmds.append(("trace", nice + py + trace_arm_dirs + ["--out", str(Path(item.arm_dir).parent / "report"), "--ref", ref] + common))
    return cmds


# ------------------------------------------------------------------ the campaign loop
class Deps:
    """What the loop does to the world (replaced by fakes in the tests)."""

    def __init__(self, c: dict, rec: Recorder):
        self.c, self.rec = c, rec
        self.cluster = Cluster()
        self.child: Optional[subprocess.Popen] = None

    def precheck(self, need_baseline: bool) -> tuple[bool, list]:
        return precheck_until_ready(lambda: self.cluster.checks(self.c, need_baseline), self.reset, self.c, self.rec.line)

    def reset(self, force: bool) -> int:
        args = ["bash", str(HERE / "reset_canonical.sh")] + (["--force"] if force else [])
        log = Path(self.c["results_root"]) / "logs" / "reset.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"=== {now_iso()} {' '.join(args)}\n"); f.flush()
            rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT)
        self.rec.line(f"reset{' --force' if force else ''} rc={rc} (logs/reset.log)")
        return rc

    def run_arm(self, item: PlanItem, env: dict, attempt: int) -> int:
        out = Path(self.c["results_root"]) / "logs" / f"{item.out_name}.{item.arm}.a{attempt}.out"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            self.child = subprocess.Popen(["bash", str(HERE / "run_arm_pilot.sh"), item.base_arm, item.out_name],
                                          env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT,
                                          start_new_session=True)   # Ctrl-C / SIGTERM of the campaign never hits the arm
            rc = self.child.wait()
        self.child = None
        return rc

    def report(self, item: PlanItem, trace_arm_dirs: list[str]) -> None:
        for kind, cmd in report_cmds(self.c, item, trace_arm_dirs):
            log = Path(self.c["results_root"]) / "logs" / f"report.{item.out_name}.{item.arm if kind == 'arm' else 'cross'}.log"
            try:
                with open(log, "w") as f:
                    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=float(self.c["report"]["timeout_s"]))
                self.rec.line(f"report {kind} {item.out_name}{'/' + item.arm if kind == 'arm' else ''} rc={rc}"
                              f"{'' if rc == 0 else ' (see ' + str(log) + ')'}")
            except Exception as exc:  # noqa: BLE001  plotting must never stop the campaign
                self.rec.line(f"report {kind} {item.out_name} FAILED {type(exc).__name__}: {str(exc)[:200]} (continuing)")

    def sleep(self, s: float) -> None:
        time.sleep(s)


class Campaign:
    def __init__(self, c: dict, deps: Optional[Deps] = None, rec: Optional[Recorder] = None,
                 retry_failed: bool = False, clock: Callable[[], float] = time.time):
        self.c = c
        self.root = Path(c["results_root"])
        self.rec = rec or Recorder(c)
        self.deps = deps or Deps(c, self.rec)
        self.entries = resolve_traces(c)
        self.by_index = {e.index: e for e in self.entries}
        self.plan = build_plan(c, self.entries)
        self.retry_failed = retry_failed
        self.clock = clock
        self.stop_signal = False
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path, {}) or {}
        self.state.setdefault("attempts", {})
        self.state.setdefault("observed_overhead_s", [])
        self.started = clock()
        self.completed: list = []
        self.failed: list = []
        self.last_validity: dict = {}

    # -- stop
    def stop_requested(self) -> bool:
        return self.stop_signal or (self.root / "STOP").exists()

    def _save_state(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def statuses(self) -> dict:
        return {it.key: item_status(it, self.c["name"]) for it in self.plan}

    def publish(self, state: str, current: Optional[dict] = None, cur_item: Optional[PlanItem] = None,
                cur_t0: Optional[float] = None) -> None:
        sts = self.statuses()
        elapsed = (self.clock() - cur_t0) if cur_t0 else 0.0
        rem = eta_seconds(self.c, self.plan, sts, self.state["observed_overhead_s"], cur_item, elapsed)
        eta_end = self.clock() + rem
        doc = {"campaign": self.c["name"], "campaign_file": self.c["campaign_file"], "state": state,
               "pid": os.getpid(), "host": socket.gethostname(), "started_at": now_iso(self.started),
               "current": current, "eta_end": now_iso(eta_end) if state == "running" else None,
               "remaining_s": round(rem) if state == "running" else 0,
               "counts": {k: sum(1 for v in sts.values() if v == k) for k in ("done", "failed", "pending", "interrupted")},
               "completed": self.completed, "failed": self.failed, "last_validity": self.last_validity,
               "plan": [{"n": it.n, "key": it.key, "position": f"{it.position}/{it.k}", "status": sts[it.key]} for it in self.plan]}
        self.rec.status(doc)
        if state == "running" and self.c["marker"].get("extend"):
            msg = extend_marker(marker_path(self.c), self.c["round"], eta_end, float(self.c["marker"]["margin_s"]))
            if msg:
                self.rec.line(msg)
        try:
            write_index(self.c, self.plan, doc)
        except Exception as exc:  # noqa: BLE001
            self.rec.line(f"index FAILED {type(exc).__name__}: {exc}")

    # -- one arm
    def arm_env(self, item: PlanItem, e: TraceEntry, prep: dict) -> dict:
        env = dict(self.c["env"])
        env.update({"PILOT_ROOT": str(self.root), "PILOT_ROUND": self.c["round"], "RUN_TAG": "",
                    "ARM_TRACE_DIR": e.client_dir, "ARM_LOADGEN_CONFIG": e.client_config, "TRACE_FILE": e.trace_file,
                    "ARM_OUT_DIR": item.arm_dir, "ARM_LABEL_OVERRIDE": item.label, "ARM_VARIANT": item.arm,
                    "TRACE_SEED": str(e.seed)})
        env.update(prep.get("env") or {})
        return env

    def prepare(self, item: PlanItem, e: TraceEntry) -> dict:
        if e.slice:
            prepare_slice(e, self.rec.line)
        if item.base_arm == "preserve":
            return prepare_preserve(self.c, e, arm_def(self.c, item.arm), self.rec.line)
        return {}

    def trace_done_dirs(self, out_name: str) -> list[str]:
        return [it.arm_dir for it in self.plan if it.out_name == out_name and done_marker(it, self.c["name"])]

    def run_item(self, item: PlanItem) -> str:
        """Returns done | failed | stopped | blocked."""
        e = self.by_index[item.entry]
        key = item.key
        max_att = int(self.c["max_attempts"])
        while True:
            att = int(self.state["attempts"].get(key, 0))
            if att >= max_att:
                self._mark_failed(item, att, "attempts exhausted")
                return "failed"
            if self.stop_requested():
                return "stopped"
            if att > 0:
                self.rec.line(f"[{item.n}/{len(self.plan)}] RETRY {key}: full reset before attempt {att + 1}")
                self.deps.reset(True)
            ok, res = self.deps.precheck(item.base_arm in BASELINE_ARMS)
            if not ok:
                self.rec.line(f"[{item.n}/{len(self.plan)}] BLOCKED {key}: pre-checks failed: "
                              + "; ".join(f"{r['name']}: {r['detail']}" for r in res if not r["ok"]))
                return "blocked"
            try:
                prep = self.prepare(item, e)
            except Exception as exc:  # noqa: BLE001  an input problem of this arm: no retry helps
                self.state["attempts"][key] = max_att
                self._save_state()
                self._mark_failed(item, att + 1, f"prepare failed: {type(exc).__name__}: {exc}")
                return "failed"
            aside = move_aside(Path(item.arm_dir), "interrupted")
            if aside:
                self.rec.line(f"{key}: earlier directory without a done marker moved to {aside.name}")
            att += 1
            self.state["attempts"][key] = att
            self._save_state()
            t0 = self.clock()
            cur = {"n": item.n, "out_name": item.out_name, "trace": item.trace, "arm": item.arm, "label": item.label,
                   "position": item.position, "k": item.k, "order": item.order, "attempt": att, "started_at": now_iso(t0),
                   "expected_end": now_iso(t0 + est_arm_s(self.c, item.duration_s)), "arm_dir": item.arm_dir}
            self.publish("running", cur, item, t0)
            self.rec.line(f"[{item.n}/{len(self.plan)}] START {key} pos={item.position}/{item.k} order={','.join(item.order)} "
                          f"attempt={att} expected_end={cur['expected_end']}")
            rc = self.deps.run_arm(item, self.arm_env(item, e, prep), att)
            t1 = self.clock()
            verdict = assess_arm(item.arm_dir, rc, e.requests, self.c["validity"], item.base_arm in BASELINE_ARMS)
            self.last_validity = {"key": key, "attempt": att, **verdict}
            extra = {"preserve": prep.get("record")} if prep.get("record") else {}
            man = arm_manifest(self.c, item, e, att, t0, t1, rc, verdict, extra)
            if Path(item.arm_dir).is_dir():
                write_json_atomic(Path(item.arm_dir) / "campaign_manifest.json", man)
            met = verdict["metrics"]
            self.rec.line(f"[{item.n}/{len(self.plan)}] {'DONE' if verdict['verdict'] != 'invalid' else 'INVALID'} {key} "
                          f"verdict={verdict['verdict']} rc={rc} V_req={met.get('V_req_pct')}% fail={met.get('fail')} "
                          f"wall={hms(t1 - t0)} reasons={'; '.join(verdict['invalid'] + verdict['suspect']) or '-'}")
            if verdict["verdict"] != "invalid":
                write_json_atomic(Path(item.arm_dir) / DONE_MARKER, man)
                self.state["observed_overhead_s"].append(round(t1 - t0 - item.duration_s, 1))
                self._save_state()
                self.completed.append({"key": key, "verdict": verdict["verdict"], "wall_s": round(t1 - t0)})
                self.deps.reset(False)              # post-arm: back to the canonical state
                return "done"
            if Path(item.arm_dir).exists():
                moved = move_aside(Path(item.arm_dir), f"failed-a{att}")
                self.rec.line(f"{key}: attempt {att} kept in {moved.name if moved else '?'}")
            if att >= max_att:
                self.deps.reset(True)               # leave the canonical state for the next arm
                self._mark_failed(item, att, "; ".join(verdict["invalid"]))
                return "failed"
            # else loop: full reset + retry (the STOP file is honoured before it)

    def _mark_failed(self, item: PlanItem, attempts: int, why: str) -> None:
        write_json_atomic(failed_marker_path(item), {"campaign": self.c["name"], "out_name": item.out_name, "arm": item.arm,
                                                     "attempts": attempts, "reason": why, "at": now_iso(),
                                                     "validity": self.last_validity if self.last_validity.get("key") == item.key else None})
        self.failed.append({"key": item.key, "reason": why[:300]})
        self.rec.line(f"[{item.n}/{len(self.plan)}] FAILED {item.key} after {attempts} attempt(s): {why[:300]} (campaign continues)")

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.retry_failed:
            for it in self.plan:
                fm = failed_marker_path(it)
                if fm.is_file():
                    move_aside(fm, "retried")
                    self.state["attempts"].pop(it.key, None)
            self._save_state()
        if (self.root / "STOP").exists():
            self.rec.line(f"campaign {self.c['name']}: {self.root / 'STOP'} is present; remove it to start")
            return 0
        self.rec.line(f"campaign {self.c['name']} start: {len(self.plan)} arms, file {self.c['campaign_file']} "
                      f"(sha256 {str(self.c['campaign_sha256'])[:12]}), pid {os.getpid()}")
        final = "done"
        for item in self.plan:
            st = item_status(item, self.c["name"])
            if st == "done":
                continue
            if st == "failed":
                self.rec.line(f"[{item.n}/{len(self.plan)}] skip {item.key}: failed marker (--retry-failed to run it again)")
                continue
            if self.stop_requested():
                final = "stopped"
                break
            self.publish("running")
            out = self.run_item(item)
            if out == "done" and self.c["report"].get("enabled"):
                self.deps.report(item, self.trace_done_dirs(item.out_name))
                self.publish("running")             # the index links the new reports
            if out in ("stopped", "blocked"):
                final = out
                break
            if float(self.c.get("gap_s") or 0) > 0 and not self.stop_requested():
                self.deps.sleep(float(self.c["gap_s"]))
        if final == "done" and self.stop_requested():
            final = "stopped" if any(item_status(i, self.c["name"]) in ("pending", "interrupted") for i in self.plan) else "done"
        self.publish(final)
        sts = self.statuses()
        self.rec.line(f"campaign {self.c['name']} {final.upper()}: done {sum(v == 'done' for v in sts.values())}, "
                      f"failed {sum(v == 'failed' for v in sts.values())}, pending {sum(v in ('pending', 'interrupted') for v in sts.values())}")
        return {"done": 0, "stopped": 0, "blocked": 3}.get(final, 1)


# ------------------------------------------------------------------ dry-run
def print_plan(c: dict, entries: list[TraceEntry], plan: list[PlanItem], start: Optional[float] = None,
               out=sys.stdout) -> float:
    start = time.time() if start is None else start
    w = out.write
    w(f"campaign {c['name']}  round={c['round']}  results={c['results_root']}\n")
    w(f"order={c['order']}  max_attempts={c['max_attempts']}  eta: arm = trace + {c['eta']['overhead_s']} s overhead"
      f" + {c['eta']['report_s'] if c['report'].get('enabled') else 0} s report + {c.get('gap_s') or 0} s gap\n\n")
    w(f"{'#':>3}  {'trace (out dir)':<34} {'pos':>4}  {'arm':<17} {'label':<22} {'trace':>7} {'est':>8}  {'start':<16} {'end':<16} status\n")
    t = start
    sts = {}
    for it in plan:
        st = item_status(it, c["name"])
        sts[it.key] = st
        est = 0.0 if st in ("done", "failed") else it.est_s
        a, b = t, t + est
        t = b
        w(f"{it.n:>3}  {it.out_name:<34} {it.position}/{it.k:<2}  {it.arm:<17} {it.label[:22]:<22} {hms(it.duration_s):>7} {hms(est):>8}  "
          f"{time.strftime('%m-%d %H:%M', time.localtime(a)):<16} {time.strftime('%m-%d %H:%M', time.localtime(b)):<16} {st}\n")
    total = t - start
    w(f"\ntotal: {len(plan)} arms over {len(entries)} trace entries, est {hms(total)} ({total / 3600:.1f} h), "
      f"end ~ {time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}\n")
    return total


def dry_run(c: dict, probe: bool = False, out=sys.stdout) -> int:
    entries = resolve_traces(c)
    plan = build_plan(c, entries)
    total = print_plan(c, entries, plan, out=out)
    problems, warnings = validate_inputs(c, entries, plan)
    w = out.write
    w("\ninputs:\n")
    for e in entries:
        src = Path(e.src_dir) / e.trace_file
        size = f"{src.stat().st_size / 1e6:.0f} MB" if src.is_file() else "MISSING"
        info = _manifest_info(e.src_dir)
        w(f"  {e.out_name:<34} {src} ({size}); requests {info['requests']}; generator {str(info['generator_sha'])[:10]}; "
          f"config {e.config}{'; slice ' + str(e.slice) + ' -> ' + e.client_dir if e.slice else ''}{'; ' + e.note if e.note else ''}\n")
    marker = marker_path(c)
    w("\nexclusive-window marker:\n")
    if marker and Path(marker).is_file():
        info = parse_marker(Path(marker).read_text())
        ours = marker_is_ours(info, c["round"])
        w(f"  {marker}: {info['type']} end={info['end']} round ok={ours}\n")
        if not ours:
            warnings.append(f"marker {marker} is not this campaign's (want 'validation ... round={c['round']}'): the run would refuse")
        elif info["end_epoch"] and time.time() + total + float(c["marker"]["margin_s"]) > info["end_epoch"]:
            w(f"  would extend its end to ~ {now_iso(time.time() + total + float(c['marker']['margin_s']))} (ETA + margin)\n")
    else:
        w(f"  {marker or '(MARKER unset)'}: absent. Create it before the run (one line, then notify the parallel sessions):\n"
          f"    validation {now_iso()} {now_iso(time.time() + total + float(c['marker']['margin_s']))} round={c['round']} owner=<session>\n")
    if c["env"]:
        w("\ncampaign env (wins over the caller and runner.env):\n")
        for k in sorted(c["env"]):
            w(f"  {k}={c['env'][k]}\n")
    if probe:
        w("\nread-only cluster probe (pre-checks, nothing is changed):\n")
        for r in Cluster().checks(c, any(i.base_arm in BASELINE_ARMS for i in plan)):
            w(f"  {'ok ' if r['ok'] else 'BAD'} {r['name']:<18} [{r['kind']}] {r['detail']}\n")
    if (Path(c["results_root"]) / "STOP").exists():
        warnings.append(f"{Path(c['results_root']) / 'STOP'} is present: the run would stop at once")
    for x in warnings:
        w(f"{'INFO' if x.startswith('preserve arm') else 'WARN'} {x}\n")
    for x in problems:
        w(f"PROBLEM {x}\n")
    n_warn = sum(1 for x in warnings if not x.startswith("preserve arm"))
    w(f"\ndry-run: {len(problems)} problem(s), {n_warn} warning(s); nothing was changed\n")
    return 1 if problems else 0


# ------------------------------------------------------------------ CLI
def _lock(root: Path):
    import fcntl  # noqa: PLC0415
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "campaign.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise CampaignError(f"another campaign process holds {root / 'campaign.lock'}")
    f.seek(0); f.truncate(); f.write(f"{os.getpid()} {socket.gethostname()} {now_iso()}\n"); f.flush()
    return f


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_env = sub.add_parser("env", help="shell exports of the campaign env block (used by run_campaign.sh)")
    p_env.add_argument("campaign")
    p_run = sub.add_parser("run")
    p_run.add_argument("campaign")
    p_run.add_argument("--dry-run", action="store_true", help="plan + ETAs + input validation; no cluster access")
    p_run.add_argument("--probe", action="store_true", help="with --dry-run: also run the read-only pre-checks")
    p_run.add_argument("--retry-failed", action="store_true", help="run failed arms again (fresh attempts)")
    p_st = sub.add_parser("status")
    p_st.add_argument("campaign")
    a = ap.parse_args(argv)
    try:
        c = load_campaign(a.campaign)
        if a.cmd == "env":
            for k, v in c["env"].items():
                print(f"export {k}={shlex.quote(v)}")
            return 0
        if a.cmd == "status":
            st = read_json(Path(c["results_root"]) / "status.json")
            if not st:
                print("no status.json yet"); return 1
            print(json.dumps({k: st.get(k) for k in ("state", "updated_at", "current", "eta_end", "remaining_s", "counts",
                                                     "last_validity")}, indent=1))
            return 0
        if a.dry_run:
            return dry_run(c, probe=a.probe)
        lock = _lock(Path(c["results_root"]))   # noqa: F841  held for the life of the process
        camp = Campaign(c, retry_failed=a.retry_failed)

        def on_signal(sig, _frm):
            camp.stop_signal = True
            camp.rec.line(f"signal {sig}: stopping after the current arm (the arm itself keeps running)")
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        return camp.run()
    except CampaignError as exc:
        print(f"campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
