"""Profile PreServe's per-replica serving capability mu_p, mu_d, mu_t (Alg.1 lines 8-11).

PreServe keeps, over its history, the largest prefill / decode / total token rate of one
instance in any time window in which *all* requests completed without an SLO violation.
We read that from a TRE calibration capture (layout: ``tre/calibration/README.md``):

* ``<run>/<model>/cells.jsonl`` (the ladder ledger) names the driven attempts; we keep the
  ``hold`` cells (``--primitive``) that are not void, not truncated and, unless
  ``--include-contaminated``, not ``possibly_contaminated``;
* ``<run>/<model>/raw/<stem>/*.jsonl`` is each attempt's client per-request log, read with
  the re-windower's own loader (``scripts.rewindow_from_raw.load_cell_capture``, which
  attaches each request's outcome);
* tumbling ``--window-s`` windows (default 30 s, the calibration window) from the cell
  start + its warm-up to its end; a request belongs to the window its completion falls in
  (``[start, end)``, like the re-windower's token totals), an *unserved* request to the
  window it was sent in (``openloop.mark_unserved_request_windows``, like the labels);
* the SLO is the model's label definition (``tre_common.slo_labels`` via the same reader
  the baseline shell uses, ``tre_baselines.config``): ``--criterion request`` (default, the
  paper's "all requests complete without an SLO violation") requires every request sent
  in the window to be served and every served request completed in it to have
  ``ttft <= ttft_slo(L)`` and ``tpot <= tpot_slo``; ``--criterion label`` instead requires
  the window's calibration label (p95 over the window, min-n guard) to be ``healthy``;
* rates are per replica: tokens / window seconds / replicas (``--replicas``, default 1,
  the calibration runs a model at one replica; a ledger record's ``replicas`` wins).

Prints YAML in the policy's ``mu`` format on stdout, a summary on stderr. Exit 1 when a
model has no violation-free window (the policy refuses to start without its mu).

**alg1** (``preserve_mu alg1 --config preserve_mu_alg1.yaml --data-root R --nl-slo NL.yaml
--envelope ENV.yaml --out DIR``; user decisions 2026-11-04; the whole chain from raw data:
``tre/baselines/docs/preserve-mu-calibration.md``): PreServe Alg. 1 (arXiv 2504.03702v2) literally, on
``calibration_dataset`` dirs (``windows.csv`` / ``requests.csv`` / ``cells.csv``; the TSS / theta training set):

* windows T_i = the dataset's own tumbling windows (``window_start_ms`` / ``window_end_ms``, all
  ``window_s`` long), kept by the config's training-set filters (split, cell status, held-out
  shapes, warm-up, contamination, window inside the cell);
* r in T_i = the cell's requests whose ``send_ts_ms`` lies in [start, end);
* T_i is clean when it has a request and none violates PreServe's NL SLO
  (``slo_judge.judge_window_nl``: failed / timed out, or e2e / output tokens > ``nl_slo_ms`` of
  the model, read from the ``--nl-slo`` file of ``preserve_nl_slo`` and refused unless equal to the
  config's ``nl_slo_expect_ms`` pins);
* with ``--envelope`` (required when the config names ``envelope_default``; the file of
  ``preserve_eval_envelope``), only windows whose arrived requests' shape lies inside it are judged;
* mu_p / mu_d / mu_t = max over clean windows of sum p(r) / sum d(r) / sum (p + d) over the window
  seconds (no P-share filter; each cell ran on one pod, checked, so the rate is per instance).

Writes ``result.json`` (provenance: tool sha, config + sha256, every file read + sha256; per model
the source window of each term, clean / dirty counts, dirty windows above, top-5 clean values),
``mu.yaml`` (policy format), ``windows.csv.gz`` (every kept window) and ``summary.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

LEDGER = "cells.jsonl"
DEFAULT_WINDOW_S = 30.0
CRITERIA = ("request", "label")


@dataclass
class ModelProfile:
    model: str
    windows: int = 0
    clean: int = 0
    cells: int = 0
    skipped_cells: dict = field(default_factory=dict)
    p: float = 0.0
    d: float = 0.0
    t: float = 0.0

    def as_mu(self) -> dict:
        return {"p": round(self.p, 3), "d": round(self.d, 3), "t": round(self.t, 3)}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def model_dirs(paths: Iterable[str]) -> list[Path]:
    """Each path is a model dir (holds ``cells.jsonl``) or a run dir of model dirs."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if (p / LEDGER).is_file():
            out.append(p)
            continue
        found = sorted(d for d in p.iterdir() if d.is_dir() and (d / LEDGER).is_file()) if p.is_dir() else []
        if not found:
            raise SystemExit(f"preserve_mu: no {LEDGER} under {p}")
        out.extend(found)
    return out


def _served(record: Mapping) -> bool:
    from scripts.rewindow_from_raw import is_served

    return is_served(record)


def _window_unserved(records: Sequence[dict], ws: int, we: int) -> int:
    from scripts import openloop
    from tre_common import slo_labels

    row = openloop.mark_unserved_request_windows(
        [{"window_start_ms": ws, "window_end_ms": we}], records)[0]
    return slo_labels.unserved_requests(row)


def request_clean(done: Sequence[dict], unserved: int, label: Any, min_requests: int) -> Optional[bool]:
    """Paper criterion. None = not judgeable (no completions / unknown prompt length)."""
    if unserved > 0:
        return False
    served = [r for r in done if _served(r)]
    if len(served) < max(1, min_requests):
        return None
    for r in served:
        ttft = r.get("ttft_ms")
        if ttft is not None:
            L = r.get("input_tokens")
            if label.slowdown and L is None:
                return None
            if float(ttft) > label.ttft_slo_ms(L):
                return False
        tpot = r.get("tpot_ms")
        if tpot is not None and float(tpot) > float(label.tpot_p95_ms):
            return False
    return True


def label_clean(records: Sequence[dict], ws: int, we: int, label: Any,
                min_latency_samples: int) -> Optional[bool]:
    """Calibration-label criterion: the window's primary label is ``healthy``."""
    from scripts import openloop, rewindow_from_raw as rw
    from tre_common import slo_labels

    done = [r for r in records if rw.in_window(r.get("done_ts_ms"), ws, we) and rw.is_served(r)]
    mode = label.percentile_mode
    row = {
        "window_start_ms": ws,
        "window_end_ms": we,
        slo_labels.SLO_COLUMNS["ttft_p95"]: rw._guarded_p95(
            [r["ttft_ms"] for r in done if r.get("ttft_ms") is not None], mode, min_latency_samples),
        slo_labels.SLO_COLUMNS["tpot_p95"]: rw._guarded_p95(
            [r["tpot_ms"] for r in done if r.get("tpot_ms") is not None], mode, min_latency_samples),
        **rw.window_request_evidence(records, ws, we),
    }
    row = openloop.mark_unserved_request_windows([row], records)[0]
    verdict = label.window_label(row)
    if verdict == slo_labels.LABEL_UNLABELED:
        return None
    return verdict == slo_labels.LABEL_HEALTHY


def profile_model(model_dir: Path, label: Any, *, window_s: float = DEFAULT_WINDOW_S,
                  criterion: str = "request", primitive: str = "hold", replicas: int = 1,
                  include_contaminated: bool = False, min_requests: int = 1,
                  min_latency_samples: int = 10, model: Optional[str] = None) -> ModelProfile:
    from scripts import r3_grid, rewindow_from_raw as rw

    ledger = _read_jsonl(model_dir / LEDGER)
    name = model or next((str(r["model"]) for r in ledger if r.get("model")), model_dir.name)
    prof = ModelProfile(model=name)
    W = int(round(window_s * 1000))

    def skip(why: str) -> None:
        prof.skipped_cells[why] = prof.skipped_cells.get(why, 0) + 1

    for rec in ledger:
        if rec.get("model") not in (None, name):
            continue
        if str(rec.get("primitive") or rec.get("profile") or "") != primitive:
            skip("primitive")
            continue
        if rec.get("void_reasons"):
            skip("void")
            continue
        if rec.get("truncated"):
            skip("truncated")
            continue
        if rec.get("possibly_contaminated") and not include_contaminated:
            skip("contaminated")
            continue
        stem = str(rec.get("stem") or "")
        raw_dir = model_dir / "raw" / stem
        if not stem or not raw_dir.is_dir():
            skip("no_raw")
            continue
        files, _ = rw.discover_cell_files(raw_dir)
        if not files:
            skip("no_raw")
            continue
        n_rep = int(rec.get("replicas") or replicas)
        prof.cells += 1
        for path in files:
            records, _inst, guard, _unmatched = rw.load_cell_capture(path)
            if not records:
                continue
            start = rec.get("start_ms") or guard.get("start_ms")
            end = rec.get("end_ms") or guard.get("end_ms")
            if start is None:
                start = min(int(r["send_ts_ms"]) for r in records if r.get("send_ts_ms") is not None)
            if end is None:
                end = max(int(r["done_ts_ms"]) for r in records if r.get("done_ts_ms") is not None) + 1
            start = int(start) + int(round(float(rec.get("warmup_s") or 0.0) * 1000))
            windows = rw.enumerate_windows(start, int(end), W, W)
            rows, _dropped = r3_grid.censor_after(
                [{"window_start_ms": ws, "window_end_ms": we} for ws, we in windows],
                guard.get("truncated_at_ts_ms"))
            for row in rows:
                ws, we = int(row["window_start_ms"]), int(row["window_end_ms"])
                prof.windows += 1
                done = [r for r in records if rw.in_window(r.get("done_ts_ms"), ws, we)]
                if criterion == "request":
                    ok = request_clean(done, _window_unserved(records, ws, we), label, min_requests)
                else:
                    ok = label_clean(records, ws, we, label, min_latency_samples)
                if not ok:
                    continue
                prof.clean += 1
                P = sum(int(r["input_tokens"]) for r in done if r.get("input_tokens") is not None)
                D = sum(int(r["output_tokens"]) for r in done if r.get("output_tokens") is not None)
                denom = (we - ws) / 1000.0 * max(1, n_rep)
                prof.p = max(prof.p, P / denom)
                prof.d = max(prof.d, D / denom)
                prof.t = max(prof.t, (P + D) / denom)
    return prof


def label_for(model: str, registry: Any) -> Any:
    """The model's label definition, through the shell's own reader (registry SLO, the
    shared ``slo_labels`` definition). mu is exported for actuating runs, so a registry
    without the live idle-TTFT fit (c/b) is refused, as the shell does."""
    from tre_baselines.config import _slo_definition

    return _slo_definition(model, registry.model(model), registry, strict=True)


# ------------------------------------------------------------------ alg1: Alg. 1 on a calibration dataset

ALG1_TERMS = ("p", "d", "t")
ALG1_FILTER_KEYS = ("splits", "cell_status", "held_out_shapes", "exclude_in_warmup",
                    "exclude_possibly_contaminated", "window_inside_cell")
ALG1_WINDOW_COLS = ("src", "cell_id", "attempt", "shape", "primitive", "rho", "split", "window_start_ms",
                    "window_end_ms", "status", "n", "violations", "v_failed", "v_nl", "nl_p95_ms", "nl_max_ms",
                    "v_ttft", "v_tpot", "ttft_p95_ms", "tpot_p95_ms",
                    "p", "d", "t")
ENVELOPE_KEYS = ("share", "mean_in", "mean_out")   # P share = sum(in) / (sum(in) + sum(out)); mean lengths per request
DATASET_FILES = ("windows.csv", "requests.csv", "cells.csv")


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def load_alg1_config(path: str) -> dict:
    """The alg1 input file (``preserve_mu_alg1.yaml``): window_s, route_timeout_s, filters, sources per model
    (paths relative to ``--data-root``), optional ``nl_slo_expect_ms`` (pins of the ``--nl-slo`` file) and
    ``envelope_default`` (``{name, config}``: the envelope a run must name). Refuses an incomplete file and a
    hand-typed ``nl_slo_ms`` / inline ``envelope`` (both come from tool outputs, ``--nl-slo`` / ``--envelope``)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("window_s", "filters", "sources"):
        if key not in cfg:
            raise SystemExit(f"preserve_mu alg1: {path}: missing {key!r}")
    for key, flag in (("nl_slo_ms", "--nl-slo (preserve_nl_slo output)"), ("envelope", "--envelope")):
        if key in cfg:
            raise SystemExit(f"preserve_mu alg1: {path}: {key!r} is not a config key; give it with {flag}")
    unknown = sorted(set(cfg["filters"]) - set(ALG1_FILTER_KEYS))
    missing = sorted(set(ALG1_FILTER_KEYS) - set(cfg["filters"]))
    if unknown or missing:
        raise SystemExit(f"preserve_mu alg1: {path}: filters unknown {unknown} / missing {missing}")
    for model, srcs in cfg["sources"].items():
        if not srcs or any(not s.get("name") or not s.get("path") for s in srcs):
            raise SystemExit(f"preserve_mu alg1: {path}: sources of {model} need name and path")
    if not float(cfg["window_s"]) > 0:
        raise SystemExit(f"preserve_mu alg1: {path}: window_s must be > 0")
    if "envelope_default" in cfg and not (cfg["envelope_default"] or {}).get("name"):
        raise SystemExit(f"preserve_mu alg1: {path}: envelope_default needs a name")
    return cfg


def apply_nl_slo(cfg: dict, nl_path: str, models: Sequence[str], rule: str = "nl") -> None:
    """Put the ``--nl-slo`` file's ``nl_slo_ms`` (``preserve_nl_slo`` output) into ``cfg`` for ``models``.
    Refuses a model without a positive value and, when the config pins ``nl_slo_expect_ms``, any value
    that differs from its pin."""
    with open(nl_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    got = doc.get("nl_slo_ms") or {}
    pins = cfg.get("nl_slo_expect_ms")
    use: dict = {}
    for m in models:
        v = got.get(m)
        if v is None or not float(v) > 0:
            raise SystemExit(f"preserve_mu alg1: {nl_path}: no positive nl_slo_ms for {m}")
        if pins is not None and (pins.get(m) is None or abs(float(v) - float(pins[m])) > 1e-9):
            raise SystemExit(f"preserve_mu alg1: {nl_path}: nl_slo_ms {m} = {v} but the config pins "
                             f"nl_slo_expect_ms {pins.get(m)}; refusing (change the pin only on purpose)")
        use[m] = float(v)
    cfg["nl_slo_ms"] = use
    cfg["rule"] = rule
    if rule == "ttft_tpot":   # the TTFT / TPOT limits behind the NL value, same file (no hand-typed numbers)
        tt: dict = {}
        for m in models:
            q = (doc.get("models") or {}).get(m) or {}
            try:
                tt[m] = {"c_ms": float(q["ttft_idle_c_ms"]), "b_ms_per_token": float(q["ttft_idle_b_ms_per_token"]),
                         "k_ttft": float(q["k_ttft"]), "k_tpot": float(q["k_tpot"]),
                         "ttft_floor_ms": float(q["ttft_floor_ms"]), "tpot_idle_ms": float(q["tpot_idle_ms"])}
            except (KeyError, TypeError, ValueError):
                raise SystemExit(f"preserve_mu alg1: {nl_path}: models.{m} lacks the TTFT / TPOT limits "
                                 "(ttft_idle_c_ms, ttft_idle_b_ms_per_token, k_ttft, k_tpot, ttft_floor_ms, tpot_idle_ms)")
        cfg["ttft_tpot"] = tt
    cfg["nl_slo_file"] = {"path": str(Path(nl_path).resolve()), "sha256": _sha256(Path(nl_path))}


def apply_envelope(cfg: dict, config_path: str, env_path: str, no_envelope: bool, allow_variant: bool) -> None:
    """``--envelope`` / ``--no-envelope`` against the config's ``envelope_default``: with a default, a run
    names an envelope file (or says ``--no-envelope``); its ``name`` must be the default's unless
    ``--envelope-variant``; its ``source`` (window, trace sha256) must equal the envelope config's
    (``envelope_default.config``, relative to the alg1 config)."""
    if env_path and no_envelope:
        raise SystemExit("preserve_mu alg1: --envelope and --no-envelope exclude each other")
    dflt = cfg.get("envelope_default") or {}
    if dflt and not env_path and not no_envelope:
        raise SystemExit(f"preserve_mu alg1: the config's default envelope is {dflt['name']!r}: give its file "
                         "with --envelope (preserve_eval_envelope output) or --no-envelope")
    if not env_path:
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        env = yaml.safe_load(fh) or {}
    check_envelope(env, env_path)
    if dflt and env.get("name") != dflt["name"] and not allow_variant:
        raise SystemExit(f"preserve_mu alg1: {env_path}: envelope {env.get('name')!r} is not the default "
                         f"{dflt['name']!r}; --envelope-variant for a sensitivity row")
    if dflt.get("config"):
        ec_path = Path(config_path).resolve().parent / dflt["config"]
        with ec_path.open("r", encoding="utf-8") as fh:
            ec = yaml.safe_load(fh) or {}
        want = {t["trace"]: t["sha256"] for t in ec.get("traces") or []}
        src = env.get("source") or {}
        if src.get("trace_sha256") != want or float(src.get("window_s", -1)) != float(ec.get("window_s", -2)):
            raise SystemExit(f"preserve_mu alg1: {env_path}: source (window, trace sha256) differs from {ec_path}")
    env["file_sha256"] = _sha256(Path(env_path))
    cfg["envelope"] = env


def check_envelope(env: Mapping, where: str = "envelope") -> None:
    """Optional shape envelope: ``{name, bounds: {model: {share|mean_in|mean_out: [lo, hi]}}}`` (inclusive)."""
    for model, b in (env.get("bounds") or {}).items():
        bad = sorted(set(b) - set(ENVELOPE_KEYS))
        if bad or not b or any(len(v) != 2 or not float(v[0]) <= float(v[1]) for v in b.values()):
            raise SystemExit(f"preserve_mu alg1: {where}: envelope of {model}: keys {bad} / need [lo, hi] each")
    if not env.get("bounds"):
        raise SystemExit(f"preserve_mu alg1: {where}: envelope has no bounds")


def inside_envelope(res: Mapping, bounds: Optional[Mapping], secs: float) -> bool:
    """Window shape (arrived requests: P share, mean input, mean output tokens) inside ``bounds`` (None = no
    filter). A window without requests has no shape and is outside any envelope."""
    if not bounds:
        return True
    n = res["n"]
    if not n:
        return False
    P, D = res["p"] * secs, res["d"] * secs
    shape = {"share": P / (P + D) if P + D else 0.0, "mean_in": P / n, "mean_out": D / n}
    return all(float(lo) <= shape[k] <= float(hi) for k, (lo, hi) in bounds.items())


def _cell_key(cell_id: Any, attempt: Any) -> tuple:
    return (str(cell_id), str(int(float(attempt))))


def _window_drop_reason(row: Mapping, filters: Mapping, cell_end: Optional[float]) -> Optional[str]:
    """The first filter (training-set rule) a dataset window fails, None = kept."""
    if row.get("split") not in set(filters["splits"]):
        return "split"
    if row.get("cell_status") not in set(filters["cell_status"]):
        return "cell_status"
    if row.get("shape") in set(filters["held_out_shapes"]):
        return "held_out_shape"
    if filters["exclude_in_warmup"] and _truthy(row.get("in_warmup")):
        return "in_warmup"
    if filters["exclude_possibly_contaminated"] and _truthy(row.get("possibly_contaminated")):
        return "possibly_contaminated"
    if filters["window_inside_cell"] and (cell_end is None or float(row["window_end_ms"]) > cell_end + 1):
        return "outside_cell"
    return None


def read_dataset_source(ddir: Path, model: str, name: str, filters: Mapping, window_ms: int):
    """One ``calibration_dataset`` dir: (kept windows, requests by cell sorted by send, stats).
    Every window must be ``window_ms`` long (the dataset's own grid); a cell whose requests went to
    more than one pod is refused (the rates are per instance)."""
    import csv

    from tre_baselines.tools import slo_judge as sj

    cell_end: dict = {}
    with (ddir / "cells.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            end = sj._num(row.get("end_ms"))
            cell_end[_cell_key(row["cell_id"], row["attempt"])] = end
    stats = {"src": name, "path": str(ddir), "windows": 0, "kept": 0, "dropped": {}, "requests": 0,
             "pods_per_cell_max": 0}
    kept: list[dict] = []
    with (ddir / "windows.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") != model:
                continue
            stats["windows"] += 1
            lo, hi = int(float(row["window_start_ms"])), int(float(row["window_end_ms"]))
            if hi - lo != window_ms:
                raise SystemExit(f"preserve_mu alg1: {ddir}: window {lo}..{hi} is not {window_ms} ms "
                                 "(dataset built at another window?)")
            key = _cell_key(row["cell_id"], row["attempt"])
            why = _window_drop_reason(row, filters, cell_end.get(key))
            if why:
                stats["dropped"][why] = stats["dropped"].get(why, 0) + 1
                continue
            kept.append({"src": name, "key": key, "cell_id": key[0], "attempt": key[1],
                         "shape": row.get("shape"), "primitive": row.get("primitive"), "rho": row.get("rho"),
                         "split": row.get("split"), "window_start_ms": lo, "window_end_ms": hi})
    stats["kept"] = len(kept)
    want = {w["key"] for w in kept}
    reqs: dict = {}
    pods: dict = {}
    with (ddir / "requests.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") != model:
                continue
            key = _cell_key(row["cell_id"], row["attempt"])
            if key not in want:
                continue
            reqs.setdefault(key, []).append(sj.from_dataset_row(row))
            if row.get("target_pod"):
                pods.setdefault(key, set()).add(row["target_pod"])
            stats["requests"] += 1
    stats["pods_per_cell_max"] = max((len(p) for p in pods.values()), default=0)
    if stats["pods_per_cell_max"] > 1:
        raise SystemExit(f"preserve_mu alg1: {ddir}: a cell sent to {stats['pods_per_cell_max']} pods; "
                         "the window rates would not be per instance")
    for key in reqs:
        reqs[key].sort(key=lambda r: r.send)
    return kept, reqs, stats


def alg1_window(reqs: Sequence[Any], sends: Sequence[float], lo_ms: int, hi_ms: int, nl_slo: float,
                route_timeout_s: float, tt_params: Optional[Mapping] = None) -> dict:
    """Alg. 1 for one window: the requests that ARRIVE (send time) in [lo, hi); clean = at least one
    request and none violates the NL rule (failed = violation); rates = the arrived requests'
    prompt / output / total tokens over the window length (``p(r)`` / ``d(r)`` / sum of both).
    ``tt_params`` (rule ``ttft_tpot``): clean = no failure, TTFT and TPOT within the idle-multiple limits."""
    from bisect import bisect_left

    from tre_baselines.tools import slo_judge as sj

    lo, hi = lo_ms / 1000.0, hi_ms / 1000.0
    win = reqs[bisect_left(sends, lo):bisect_left(sends, hi)]
    secs = (hi_ms - lo_ms) / 1000.0
    P = sum(int(r.prompt_tokens or 0) for r in win)
    D = sum(int(r.completion_tokens or 0) for r in win)
    out = {"n": len(win), "p": P / secs, "d": D / secs, "t": (P + D) / secs}
    try:
        if tt_params is not None:
            v = sj.judge_window_ttft_tpot(win, lo, hi, tt_params, route_timeout_s=route_timeout_s)
        else:
            v = sj.judge_window_nl(win, lo, hi, nl_slo, route_timeout_s=route_timeout_s)
    except sj.UnjudgeableWindow as exc:
        return {**out, "status": "unjudgeable", "why": str(exc)}
    status = "empty" if not win else ("clean" if v["meets_slo"] else "dirty")
    if tt_params is not None:
        return {**out, "status": status, **{k: v[k] for k in ("violations", "v_failed", "v_ttft", "v_tpot",
                                                              "ttft_p95_ms", "tpot_p95_ms")}}
    return {**out, "status": status, **{k: v[k] for k in ("violations", "v_failed", "v_nl", "nl_p95_ms",
                                                           "nl_max_ms")}}


def _window_id(w: Mapping) -> dict:
    return {k: w[k] for k in ("src", "cell_id", "attempt", "shape", "rho", "window_start_ms")}


def alg1_summary(rows: Sequence[Mapping], top_k: int = 5) -> dict:
    """mu_x = max over clean windows (Alg. 1); per term the window it came from, the top-k clean values
    (reporting only) and the dirty windows above it (any shape / the same source + shape)."""
    clean = [r for r in rows if r["status"] == "clean"]
    dirty = [r for r in rows if r["status"] == "dirty"]
    out: dict = {"windows_kept": len(rows), "clean": len(clean), "dirty": len(dirty),
                 "empty": sum(1 for r in rows if r["status"] == "empty"),
                 "outside_envelope": sum(1 for r in rows if r["status"] == "outside"),
                 "unjudgeable": sum(1 for r in rows if r["status"] == "unjudgeable"), "terms": {}}
    for term in ALG1_TERMS:
        if not clean:
            out["terms"][term] = None
            continue
        best = max(clean, key=lambda r: r[term])  # first maximum in window order
        mu = best[term]
        above = [r for r in dirty if r[term] > mu]
        same = [r for r in above if r["src"] == best["src"] and r["shape"] == best["shape"]]
        top = sorted(clean, key=lambda r: -r[term])[:top_k]
        out["terms"][term] = {
            "mu": round(mu, 1), "window": _window_id(best), "n_requests": best["n"],
            "top": [{"value": round(r[term], 1), **_window_id(r)} for r in top],
            "dirty_above": len(above), "dirty_above_same_shape": len(same),
            "max_dirty": round(max((r[term] for r in dirty), default=0.0), 1) if dirty else None,
            "bracketed": bool(above)}
    return out


def profile_alg1(cfg: Mapping, data_root: Path, models: Sequence[str]) -> tuple:
    """(per-model summaries, all window rows, per-source stats)."""
    window_ms = int(round(float(cfg["window_s"]) * 1000))
    route_timeout_s = float(cfg.get("route_timeout_s", 150.0))
    summaries: dict = {}
    all_rows: list[dict] = []
    sources: list[dict] = []
    bounds = (cfg.get("envelope") or {}).get("bounds") or {}
    for model in models:
        nl_slo = float(cfg["nl_slo_ms"][model])
        rows: list[dict] = []
        for src in cfg["sources"][model]:
            ddir = data_root / src["path"]
            kept, reqs, stats = read_dataset_source(ddir, model, src["name"], cfg["filters"], window_ms)
            stats["model"] = model
            sends = {k: [r.send for r in v] for k, v in reqs.items()}
            for w in kept:
                res = alg1_window(reqs.get(w["key"], []), sends.get(w["key"], []), w["window_start_ms"],
                                  w["window_end_ms"], nl_slo, route_timeout_s,
                                  {model: cfg["ttft_tpot"][model]} if cfg.get("rule") == "ttft_tpot" else None)
                if res["status"] != "empty" and not inside_envelope(
                        res, bounds.get(model), (w["window_end_ms"] - w["window_start_ms"]) / 1000.0):
                    res = {**res, "status": "outside"}   # shape outside the envelope: not judged, not in mu
                rows.append({**{k: v for k, v in w.items() if k != "key"}, **res})
            sources.append(stats)
            print(f"# {model} {src['name']}: {stats['kept']}/{stats['windows']} windows kept, "
                  f"{stats['requests']} requests, dropped {stats['dropped']}", file=sys.stderr)
        summ = alg1_summary(rows)
        summ["nl_slo_ms"] = nl_slo
        if cfg.get("rule") == "ttft_tpot":
            summ["ttft_tpot_limits"] = cfg["ttft_tpot"][model]
        summ["windows_in_files"] = sum(s["windows"] for s in sources if s["model"] == model)
        summaries[model] = summ
        all_rows.extend({"model": model, **r} for r in rows)
    return summaries, all_rows, sources


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> Optional[str]:
    import subprocess

    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return f"{sha}{'-dirty' if dirty else ''}" if sha else None


def _summary_md(summaries: Mapping, prov: Mapping, env: Optional[Mapping] = None) -> str:
    lines = ["# PreServe mu, Alg. 1 on the TSS / theta training set", "",
             f"tool `{prov['tool_sha']}`, config `{prov['config']}` (sha256 `{prov['config_sha256'][:12]}`), "
             f"window {prov['window_s']:g} s, arrival attribution, {'TTFT + TPOT rule' if prov.get('rule') == 'ttft_tpot' else 'NL rule'}, "
             f"{'envelope ' + str(env.get('name')) if env else 'no shape filter'}.", "",
             f"| model | {'TTFT / TPOT limits' if prov.get('rule') == 'ttft_tpot' else 'NL SLO ms'} | clean / judged (kept) "
             "| mu_p | mu_d | mu_t | bracketed p/d/t |",
             "|---|---|---|---|---|---|---|"]
    for m, s in summaries.items():
        T = s["terms"]
        mus = ["-" if T[k] is None else f"{T[k]['mu']:.0f}" for k in ALG1_TERMS]
        br = "/".join("-" if T[k] is None else ("Y" if T[k]["bracketed"] else "n") for k in ALG1_TERMS)
        q = s.get("ttft_tpot_limits")
        lim = (f"TTFT max({q['ttft_floor_ms']:g}, {q['k_ttft']:g}*({q['c_ms']:g}+{q['b_ms_per_token']:g}L)); "
               f"TPOT {q['k_tpot'] * q['tpot_idle_ms']:.2f} ms" if q else f"{s['nl_slo_ms']:g}")
        lines.append(f"| {m} | {lim} | {s['clean']} / {s['clean'] + s['dirty']} ({s['windows_kept']}) "
                     f"| {mus[0]} | {mus[1]} | {mus[2]} | {br} |")
    lines += ["", "## Source window and top-5 clean values per term", ""]
    for m, s in summaries.items():
        for k in ALG1_TERMS:
            t = s["terms"][k]
            if t is None:
                continue
            w = t["window"]
            top = ", ".join(f"{x['value']:.0f} ({x['src']}/{x['shape']}/rho {x['rho']})" for x in t["top"])
            lines.append(f"- {m} mu_{k} = {t['mu']:.1f}: {w['src']} cell {w['cell_id']} a{w['attempt']} shape "
                         f"{w['shape']} rho {w['rho']} window {w['window_start_ms']} ({t['n_requests']} req); "
                         f"dirty above {t['dirty_above']} (same shape {t['dirty_above_same_shape']}, "
                         f"max dirty {t['max_dirty']}); top-5: {top}")
    return "\n".join(lines) + "\n"


def main_alg1(argv: Sequence[str]) -> int:
    """``preserve_mu alg1 --config C --data-root R --out DIR``: Alg. 1 literally (module docstring)."""
    import csv
    import gzip
    import platform
    import time

    ap = argparse.ArgumentParser(prog="preserve_mu alg1", description=main_alg1.__doc__)
    ap.add_argument("--config", required=True, help="alg1 input file (preserve_mu_alg1.yaml)")
    ap.add_argument("--data-root", required=True, help="root the config's source paths are relative to")
    ap.add_argument("--out", required=True, help="output dir (must not hold a result.json)")
    ap.add_argument("--models", default="", help="comma list (default: every model of the config)")
    ap.add_argument("--nl-slo", required=True, help="NL SLO file (preserve_nl_slo output, nl_slo.yaml); must "
                    "equal the config's nl_slo_expect_ms pins")
    ap.add_argument("--rule", choices=("nl", "ttft_tpot"), default="nl",
                    help="clean-window rule: nl (default, PreServe's e2e / output tokens <= NL SLO) or ttft_tpot "
                    "(sensitivity: TTFT <= max(floor, k_ttft (c + b L)) and TPOT <= k_tpot TPOT_idle, limits read "
                    "from the --nl-slo file)")
    ap.add_argument("--envelope", default="", help="envelope file (preserve_eval_envelope output: name, bounds "
                    "per model); keep only windows whose shape is inside it. Required when the config names an "
                    "envelope_default (or --no-envelope)")
    ap.add_argument("--envelope-variant", action="store_true", help="accept an envelope other than the default")
    ap.add_argument("--no-envelope", action="store_true", help="no shape filter (sensitivity row)")
    a = ap.parse_args(list(argv))
    cfg = load_alg1_config(a.config)
    only = [m.strip() for m in a.models.split(",") if m.strip()]
    models = only or list(cfg["sources"])
    bad = [m for m in models if m not in cfg["sources"]]
    if bad:
        raise SystemExit(f"preserve_mu alg1: no sources for {bad}")
    apply_nl_slo(cfg, a.nl_slo, models, a.rule)
    apply_envelope(cfg, a.config, a.envelope, a.no_envelope, a.envelope_variant)
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"preserve_mu alg1: {out}/result.json exists; use a new --out")
    root = Path(a.data_root)
    files = []
    for m in models:
        for s in cfg["sources"][m]:
            for f in DATASET_FILES:
                p = root / s["path"] / f
                if not p.is_file():
                    raise SystemExit(f"preserve_mu alg1: missing {p}")
                files.append({"model": m, "src": s["name"], "path": str(p), "bytes": p.stat().st_size,
                              "sha256": _sha256(p)})
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summaries, rows, sources = profile_alg1(cfg, root, models)
    prov = {"started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool_sha": _git_sha(),
            "argv": ["alg1", *argv], "rule": a.rule, "host": platform.node(), "python": platform.python_version(),
            "config": str(Path(a.config).resolve()), "config_sha256": _sha256(Path(a.config)),
            "data_root": str(root.resolve()), "window_s": float(cfg["window_s"]),
            "route_timeout_s": float(cfg.get("route_timeout_s", 150.0)), "files": files}
    rule = {"window": f"tumbling {float(cfg['window_s']):g} s, the dataset's own grid (window_start_ms / end_ms)",
            "requests": "requests whose send_ts_ms falls in [start, end) of the window, same cell",
            "rule": a.rule,
            "clean": ("at least one request and none violates: failed / timed out, TTFT > max(floor, k_ttft * (c + b * L)), "
                      "or TPOT = (e2e - ttft) / (out - 1) > k_tpot * TPOT_idle (out = 1: TTFT only); limits in "
                      "config.ttft_tpot (from the nl_slo file)") if a.rule == "ttft_tpot" else
                     "at least one request and none violates: failed / timed out, or NL = e2e / output tokens > NL SLO",
            "rates": "sum of the arrived requests' prompt (p) / output (d) / both (t) tokens / window seconds; "
                     "a failed request counts expected_prompt_tokens and 0 output tokens (dirty windows only)",
            "mu": "max over clean windows per term (Alg. 1 lines 8-11); one pod per cell; "
                  + (f"windows outside envelope {cfg['envelope'].get('name')!r} (shape of the arrived requests: P share, "
                     "mean input, mean output tokens) are dropped before judging" if cfg.get("envelope")
                     else "no shape filter"),
            "empty windows": "skipped (no requests: Alg. 1's max is unchanged by them)"}
    out.mkdir(parents=True, exist_ok=True)
    mu = {m: {k: s["terms"][k]["mu"] for k in ALG1_TERMS} for m, s in summaries.items() if s["clean"]}
    result = {"tool": "tre_baselines.tools.preserve_mu alg1", "provenance": prov, "rule": rule, "config": cfg,
              "sources": sources, "models": summaries, "mu": mu}
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    with (out / "mu.yaml").open("w", encoding="utf-8") as fh:
        fh.write(f"# preserve_mu alg1, tool {prov['tool_sha']}, config sha256 {prov['config_sha256']}; "
                 "provenance in result.json\n")
        fh.write(yaml.safe_dump({"mu": mu}, sort_keys=True, default_flow_style=None))
    with gzip.open(out / "windows.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["model", *ALG1_WINDOW_COLS, "why"], extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (out / "summary.md").write_text(_summary_md(summaries, prov, cfg.get("envelope")), encoding="utf-8")
    sys.stdout.write(yaml.safe_dump({"mu": mu}, sort_keys=True, default_flow_style=None))
    return 0 if len(mu) == len(models) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "alg1":
        return main_alg1(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="calibration model dir(s) (with cells.jsonl) or run dir(s)")
    ap.add_argument("--registry", default=os.environ.get("TRE_REGISTRY_PATH") or "/etc/tre/registry.yaml")
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--criterion", choices=CRITERIA, default="request")
    ap.add_argument("--primitive", default="hold")
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--include-contaminated", action="store_true")
    ap.add_argument("--min-requests", type=int, default=1,
                    help="request criterion: completed requests a window needs to be judged")
    ap.add_argument("--min-latency-samples", type=int, default=10,
                    help="label criterion: the re-windower's p95 guard")
    ap.add_argument("--models", default="", help="comma list of models to keep (default: all)")
    args = ap.parse_args(argv)

    from tre_common.registry import load_registry

    registry = load_registry(args.registry)
    only = {m.strip() for m in args.models.split(",") if m.strip()}
    mu: dict[str, dict] = {}
    status = 0
    for mdir in model_dirs(args.paths):
        ledger = _read_jsonl(mdir / LEDGER)
        model = next((str(r["model"]) for r in ledger if r.get("model")), mdir.name)
        if only and model not in only:
            continue
        prof = profile_model(
            mdir, label_for(model, registry), window_s=args.window_s, criterion=args.criterion,
            primitive=args.primitive, replicas=args.replicas,
            include_contaminated=args.include_contaminated, min_requests=args.min_requests,
            min_latency_samples=args.min_latency_samples, model=model,
        )
        print(f"# {model}: {prof.clean}/{prof.windows} violation-free windows in {prof.cells} "
              f"{args.primitive} cells (criterion={args.criterion}, window={args.window_s:g}s); "
              f"skipped cells {prof.skipped_cells}", file=sys.stderr)
        if prof.clean == 0:
            print(f"# {model}: no violation-free window, no mu", file=sys.stderr)
            status = 1
            continue
        mu[model] = prof.as_mu()
    sys.stdout.write(yaml.safe_dump({"mu": mu}, sort_keys=True, default_flow_style=None))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
