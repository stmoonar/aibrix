#!/usr/bin/env python3
"""Per-cell evidence the calibration campaign keeps next to its measurements.

Why this exists
---------------
Until 2026-09-30 a calibration run kept the client's per-request log, the 1 Hz queue
sidecar, the online window CSV, the prompts and the schedules - and nothing of what the
*system* saw. Three things were lost for good:

* the vLLM counters and histograms of each pod (the server-side view of the same
  requests; the sidecar parsed three gauges out of every ``/metrics`` body and dropped
  the rest);
* the gateway's Redis docs ``tre:v2:hist:<pod>`` / ``tre:v2:inst:<pod>`` - the exact
  input of the controller's ``MetricsStore`` - which the gateway expires after 30 min;
* the controller's own per-tick TSS / EMA / Z / band.

With those, a later question about a window ("what did the controller compute here, and
from which samples?") is answered from disk instead of being re-derived.

Layout (``CAPTURE_LAYOUT_VERSION`` 1)
-------------------------------------
Everything new lives under ``<model dir>/cells/<stem>/`` (``<model dir>`` = the campaign
``--out-dir``, ``<stem>`` = the attempt name ``<model>_<shape>_<primitive>_c<code>_a<n>``
that ``raw/<stem>/`` and ``<stem>.csv`` already use)::

    <run>/<model>/
      manifest.json                       run manifest: code, registry, images, config
      <stem>.csv                          online 30 s windows            (unchanged)
      raw/<stem>/<cell_id>.jsonl          per-request client log         (unchanged)
      raw/<stem>/<cell_id>.instant.jsonl  1 Hz queue sidecar            (unchanged)
      raw/<stem>/<cell_id>.{guard.json,rps.csv,failures.jsonl}          (unchanged)
      cells/<stem>/
        cell_meta.json                    identity, time span, pods, what was captured,
                                          relative paths of the unchanged files above
        vllm_metrics_1hz/<ns>_<pod>.jsonl vLLM counters / gauges / cumulative histogram
                                          buckets per pod, 1 Hz, delta-encoded
        gateway_redis_dump/hist/<ns>_<pod>.jsonl   gateway hist docs covering the cell
        gateway_redis_dump/inst/<ns>_<pod>.jsonl   gateway inst docs covering the cell
        controller_ticks.jsonl            the controller's decision history over the cell

The unchanged files stay where every reader (``rewindow_from_raw``,
``calibration_dataset``, the fit) already looks; ``cell_meta.json`` points at them and
:func:`resolve_cell_artifacts` resolves one attempt's files for a run of either layout.
``cells/`` is outside ``raw/`` so the recursive ``*.jsonl`` discovery of the re-window
never sees it (and :func:`scripts.rewindow_from_raw.discover_cell_files` also skips any
``cells`` directory, in case a raw root is pointed higher).

Nothing here may fail or void a cell: every capture step records its error in
``cell_meta.json`` and the cell goes on.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from tre_common.rediskeys import (
    SCRAPE_INTERVAL_MS,
    decision_hist_key,
    hist_key,
    inst_key,
    pods_key,
)
from tre_common.vllm_metrics import vllm_candidates

CAPTURE_LAYOUT_VERSION = 1
CELLS_DIRNAME = "cells"
CELL_META = "cell_meta.json"
RUN_MANIFEST = "manifest.json"
VLLM_METRICS_DIRNAME = "vllm_metrics_1hz"
GATEWAY_DUMP_DIRNAME = "gateway_redis_dump"
GATEWAY_KINDS = ("hist", "inst")
CONTROLLER_TICKS = "controller_ticks.jsonl"

VLLM_METRICS_SCHEMA = "tre.vllm_metrics_1hz/v1"
GATEWAY_DUMP_SCHEMA = "tre.gateway_redis_dump/v1"
CONTROLLER_TICKS_SCHEMA = "tre.controller_ticks/v1"

#: Histograms kept per pod (every name vLLM has used for each, newest first). Only the
#: cumulative ``_bucket`` counts plus ``_sum`` / ``_count`` are stored.
DEFAULT_VLLM_HISTOGRAMS: tuple[str, ...] = tuple(
    name
    for quantity in (
        "time_to_first_token_seconds",
        "inter_token_latency_seconds",
        "e2e_request_latency_seconds",
        "request_prompt_tokens",
        "request_generation_tokens",
    )
    for name in vllm_candidates(quantity)
)
#: Per-pod gauges (the queue sidecar only keeps their sum over pods).
DEFAULT_VLLM_GAUGES: tuple[str, ...] = (
    *vllm_candidates("num_requests_running"),
    *vllm_candidates("num_requests_waiting"),
    *vllm_candidates("kv_cache_usage_perc"),
    "vllm:num_requests_swapped",
    "vllm:engine_sleep_state",
)
#: Counters: every ``vllm:*_total`` family (prompt / generation tokens, preemptions,
#: request_success by finish reason, prefix-cache queries / hits, ...).
COUNTER_SUFFIX = "_total"
HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")
#: Label dropped from series keys: constant per pod and recorded once in the header.
DROPPED_LABELS = frozenset({"model_name"})
#: A full snapshot row every this many rows, so a truncated file loses at most this much.
DEFAULT_KEYFRAME_EVERY = 300

#: How long to wait, after a cell, for the gateway's next Redis write (its ticker has an
#: arbitrary phase against the 10 s round, so up to one interval plus the write itself).
DEFAULT_GATEWAY_FLUSH_WAIT_S = SCRAPE_INTERVAL_MS / 1000.0 + 2.0
DEFAULT_GATEWAY_POLL_S = 0.25


# ----------------------------------------------------------------------------- layout


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_name(key: str) -> str:
    """A file name for a pod key (``<namespace>/<pod>``) or an endpoint: every character
    outside ``[A-Za-z0-9.-]`` becomes ``_``. Kubernetes names cannot contain ``_``, so
    ``<ns>_<pod>`` stays unambiguous; the original key is kept inside the file."""
    return re.sub(r"[^A-Za-z0-9.-]", "_", str(key))


@dataclass(frozen=True)
class CellLayout:
    """Where one attempt's artifacts live. ``model_dir`` is the campaign ``--out-dir``
    (the directory holding ``<stem>.csv``); ``raw_root`` its ``raw/`` unless the run used
    another ``--raw-dir``."""

    model_dir: Path
    stem: str
    cell_id: str
    raw_root: Optional[Path] = None

    @property
    def raw_dir(self) -> Path:
        return Path(self.raw_root if self.raw_root is not None else Path(self.model_dir) / "raw") / self.stem

    @property
    def cell_dir(self) -> Path:
        return Path(self.model_dir) / CELLS_DIRNAME / self.stem

    @property
    def meta_path(self) -> Path:
        return self.cell_dir / CELL_META

    @property
    def vllm_metrics_dir(self) -> Path:
        return self.cell_dir / VLLM_METRICS_DIRNAME

    def vllm_metrics_path(self, pod_key: str) -> Path:
        return self.vllm_metrics_dir / f"{safe_name(pod_key)}.jsonl"

    def gateway_dump_path(self, kind: str, pod_key: str) -> Path:
        if kind not in GATEWAY_KINDS:
            raise ValueError(f"gateway doc kind must be one of {GATEWAY_KINDS}, got {kind!r}")
        return self.cell_dir / GATEWAY_DUMP_DIRNAME / kind / f"{safe_name(pod_key)}.jsonl"

    @property
    def controller_ticks_path(self) -> Path:
        return self.cell_dir / CONTROLLER_TICKS

    def legacy_paths(self) -> dict[str, Path]:
        """The files every existing reader uses, at their unchanged locations."""
        raw = self.raw_dir
        return {
            "requests_jsonl": raw / f"{self.cell_id}.jsonl",
            "queue_1hz_jsonl": raw / f"{self.cell_id}.instant.jsonl",
            "guard_json": raw / f"{self.cell_id}.guard.json",
            "rps_csv": raw / f"{self.cell_id}.rps.csv",
            "failures_jsonl": raw / f"{self.cell_id}.failures.jsonl",
            "windows_csv": Path(self.model_dir) / f"{self.stem}.csv",
        }


def run_manifest_path(model_dir: Path) -> Path:
    return Path(model_dir) / RUN_MANIFEST


def _rel(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return os.path.relpath(Path(path), Path(base))
    except ValueError:  # another drive (Windows); keep it absolute
        return str(path)


def resolve_cell_artifacts(model_dir: Path, stem: str, *, raw_root: Optional[Path] = None) -> dict[str, Any]:
    """Every artifact of one attempt, for a run of either layout.

    A capture-layout run has ``cells/<stem>/cell_meta.json``: its recorded relative paths
    win. A run made before it (every run up to 2026-09-30) has only ``raw/<stem>/``; the
    per-request file is found the way ``calibration_dataset`` finds it (the one
    ``*.jsonl`` that is not a sidecar, or its ``.void`` quarantine) and the rest follow the
    naming convention. The new artifacts are None there. Paths that do not exist are
    returned as None, so a caller never opens a file that was not written.
    """
    model_dir = Path(model_dir)
    meta_path = model_dir / CELLS_DIRNAME / stem / CELL_META
    out: dict[str, Any] = {"layout_version": None, "stem": stem, "cell_meta": None}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_dir = meta_path.parent
        out.update({"layout_version": meta.get("layout_version"), "cell_id": meta.get("cell_id"),
                    "cell_meta": meta_path})
        for key, rel in (meta.get("legacy") or {}).items():
            p = None if rel is None else (cell_dir / rel).resolve()
            out[key] = p if p is not None and p.exists() else None
        vm = meta.get("vllm_metrics") or {}
        out["vllm_metrics"] = {
            pod: cell_dir / rel for pod, rel in (vm.get("files") or {}).items() if (cell_dir / rel).exists()
        }
        gd = meta.get("gateway_redis_dump") or {}
        out["gateway_redis_dump"] = {
            kind: {pod: cell_dir / rel for pod, rel in ((gd.get("files") or {}).get(kind) or {}).items()
                   if (cell_dir / rel).exists()}
            for kind in GATEWAY_KINDS
        }
        ct = (meta.get("controller_ticks") or {}).get("file")
        out["controller_ticks"] = (cell_dir / ct) if ct and (cell_dir / ct).exists() else None
        return out
    raw_dir = Path(raw_root if raw_root is not None else model_dir / "raw") / stem
    cell_id = None
    if raw_dir.is_dir():
        from scripts.rewindow_from_raw import SIDECAR_JSONL_SUFFIXES

        for path in sorted(raw_dir.iterdir()):
            name = path.name
            if name.endswith(SIDECAR_JSONL_SUFFIXES):
                continue
            if name.endswith(".jsonl") or name.endswith(".jsonl.void"):
                cell_id = name.split(".", 1)[0]
                break
    out["cell_id"] = cell_id
    if cell_id is not None:
        layout = CellLayout(model_dir, stem, cell_id, raw_root=raw_root)
        for key, p in layout.legacy_paths().items():
            if key == "requests_jsonl" and not p.exists():
                void = Path(str(p) + ".void")
                p = void if void.exists() else p
            out[key] = p if p.exists() else None
    else:
        csv_path = model_dir / f"{stem}.csv"
        out["windows_csv"] = csv_path if csv_path.exists() else None
    out["vllm_metrics"] = {}
    out["gateway_redis_dump"] = {kind: {} for kind in GATEWAY_KINDS}
    out["controller_ticks"] = None
    return out


# ------------------------------------------------------------ vLLM /metrics (1 Hz)

_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_value(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _json_number(value: float) -> Any:
    """JSON has no NaN / Inf: those are stored as strings (``"NaN"``, ``"+Inf"``)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 2**53:
        return int(value)
    return value


def _series_key(name: str, labels: Sequence[tuple[str, str]]) -> str:
    kept = [(k, v) for k, v in labels if k not in DROPPED_LABELS]
    if not kept:
        return name
    return name + "{" + ",".join(f'{k}="{v}"' for k, v in sorted(kept)) + "}"


def parse_vllm_metrics(
    text: str,
    *,
    histograms: Iterable[str] = DEFAULT_VLLM_HISTOGRAMS,
    gauges: Iterable[str] = DEFAULT_VLLM_GAUGES,
) -> dict[str, Any]:
    """One ``/metrics`` body -> ``{"c": counters, "g": gauges, "h": histograms, "model_name"}``.

    Series keys are the family name plus its labels in Prometheus form, minus
    ``model_name`` (constant per pod; returned separately) and, for a bucket, ``le``:
    ``vllm:prompt_tokens_total{engine="0"}``. A histogram is
    ``{"b": {le: cumulative count}, "s": sum, "n": count}``. ``_created`` series and
    everything outside ``vllm:`` are skipped. Values are JSON numbers (NaN / Inf as
    strings).
    """
    hist_names = frozenset(histograms)
    gauge_names = frozenset(gauges)
    counters: dict[str, Any] = {}
    gauge_out: dict[str, Any] = {}
    hists: dict[str, dict] = {}
    model_name: Optional[str] = None
    for raw in text.splitlines():
        if not raw.startswith("vllm:"):
            continue
        brace = raw.find("{")
        if brace >= 0:
            close = raw.rfind("}")
            name = raw[:brace]
            labels = _LABEL_RE.findall(raw[brace + 1:close])
            rest = raw[close + 1:].split()
        else:
            parts = raw.split()
            if len(parts) < 2:
                continue
            name, labels, rest = parts[0], [], parts[1:]
        if not rest:
            continue
        value = _parse_value(rest[0])
        if value is None:
            continue
        if model_name is None:
            for k, v in labels:
                if k == "model_name":
                    model_name = v
                    break
        base = None
        suffix = None
        for suf in HISTOGRAM_SUFFIXES:
            if name.endswith(suf) and name[: -len(suf)] in hist_names:
                base, suffix = name[: -len(suf)], suf
                break
        if base is not None:
            le = None
            other = []
            for k, v in labels:
                if k == "le":
                    le = v
                else:
                    other.append((k, v))
            entry = hists.setdefault(_series_key(base, other), {"b": {}, "s": None, "n": None})
            if suffix == "_bucket":
                if le is not None:
                    entry["b"][le] = _json_number(value)
            elif suffix == "_sum":
                entry["s"] = _json_number(value)
            else:
                entry["n"] = _json_number(value)
            continue
        if name in gauge_names:
            gauge_out[_series_key(name, labels)] = _json_number(value)
        elif name.endswith(COUNTER_SUFFIX):
            counters[_series_key(name, labels)] = _json_number(value)
    return {"c": counters, "g": gauge_out, "h": hists, "model_name": model_name}


def _flatten(state: Mapping[str, Any]) -> dict[tuple, Any]:
    flat: dict[tuple, Any] = {}
    for key, value in (state.get("c") or {}).items():
        flat[("c", key)] = value
    for key, value in (state.get("g") or {}).items():
        flat[("g", key)] = value
    for key, h in (state.get("h") or {}).items():
        for le, v in (h.get("b") or {}).items():
            flat[("h", key, "b", le)] = v
        flat[("h", key, "s")] = h.get("s")
        flat[("h", key, "n")] = h.get("n")
    return flat


def _unflatten_into(row: dict, flat_items: Iterable[tuple[tuple, Any]]) -> None:
    for key, value in flat_items:
        if key[0] in ("c", "g"):
            row.setdefault(key[0], {})[key[1]] = value
        elif key[2] == "b":
            row.setdefault("h", {}).setdefault(key[1], {}).setdefault("b", {})[key[3]] = value
        else:
            row.setdefault("h", {}).setdefault(key[1], {})[key[2]] = value


class DeltaEncoder:
    """Rows that carry only what changed since the previous row of the same file.

    A *keyframe* row (``"full": true``) carries the whole state; it is written for the
    first sample, whenever the set of series changes (a restarted pod, a new label value)
    and every ``keyframe_every`` rows. Any other row carries the series whose value
    differs from the previous row - for a histogram, only the buckets that moved, and
    ``s`` / ``n`` only when they moved. :func:`decode_vllm_metrics` inverts it.
    """

    def __init__(self, keyframe_every: int = DEFAULT_KEYFRAME_EVERY) -> None:
        self.keyframe_every = max(1, int(keyframe_every))
        self._prev: Optional[dict[tuple, Any]] = None
        self._since_key = 0

    def encode(self, ts_ms: int, state: Mapping[str, Any]) -> dict:
        flat = _flatten(state)
        row: dict[str, Any] = {"ts_ms": int(ts_ms)}
        keyframe = (
            self._prev is None
            or set(flat) != set(self._prev)
            or self._since_key >= self.keyframe_every
        )
        if keyframe:
            row["full"] = True
            _unflatten_into(row, sorted(flat.items(), key=lambda kv: kv[0]))
            self._since_key = 1
        else:
            changed = [(k, v) for k, v in flat.items() if self._prev.get(k) != v]
            _unflatten_into(row, sorted(changed, key=lambda kv: kv[0]))
            self._since_key += 1
        self._prev = flat
        return row


def _flatten_row(row: Mapping[str, Any]) -> dict[tuple, Any]:
    """Flatten one written row. Unlike :func:`_flatten` (a parsed state, where a missing
    ``s`` / ``n`` is a real None), a delta row's histogram entry holds only what moved."""
    out: dict[tuple, Any] = {}
    for key, value in (row.get("c") or {}).items():
        out[("c", key)] = value
    for key, value in (row.get("g") or {}).items():
        out[("g", key)] = value
    for key, entry in (row.get("h") or {}).items():
        for le, v in (entry.get("b") or {}).items():
            out[("h", key, "b", le)] = v
        for part in ("s", "n"):
            if part in entry:
                out[("h", key, part)] = entry[part]
    return out


def decode_vllm_metrics(lines: Iterable[str]) -> Iterator[dict]:
    """The full state after every sample row of a ``vllm_metrics_1hz`` file:
    ``{"ts_ms", "c", "g", "h"}`` - what :func:`parse_vllm_metrics` returned for that
    second. Header and error rows are skipped."""
    state: dict[tuple, Any] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "header" or "error" in row:
            continue
        if row.get("full"):
            state = {}
        state.update(_flatten_row(row))
        full: dict[str, Any] = {"ts_ms": row["ts_ms"], "c": {}, "g": {}, "h": {}}
        _unflatten_into(full, sorted(state.items(), key=lambda kv: kv[0]))
        yield full


class VllmMetricsRecorder:
    """Writes one ``vllm_metrics_1hz/<pod>.jsonl`` per scraped pod from the bodies the
    queue sidecar already fetches (:func:`scripts.openloop.make_pod_metrics_sampler`
    hands every body to :meth:`record`), so there is no second scrape.

    Each file: a header line (``"kind": "header"``: schema, pod, endpoint, model,
    encoding), then one row per sample (:class:`DeltaEncoder`), or ``{"ts_ms", "error"}``
    when the scrape failed. Rows are appended as they come, so a killed driver keeps
    everything up to its last second.
    """

    def __init__(
        self,
        out_dir: Path,
        pods: Mapping[str, str],
        *,
        model: Optional[str] = None,
        histograms: Sequence[str] = DEFAULT_VLLM_HISTOGRAMS,
        gauges: Sequence[str] = DEFAULT_VLLM_GAUGES,
        keyframe_every: int = DEFAULT_KEYFRAME_EVERY,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.pods = dict(pods)  # endpoint URL -> pod key (<ns>/<pod>)
        self.model = model
        self.histograms = tuple(histograms)
        self.gauges = tuple(gauges)
        self.keyframe_every = int(keyframe_every)
        self._encoders: dict[str, DeltaEncoder] = {}
        self._files: dict[str, Any] = {}
        self.samples: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.bytes_written: dict[str, int] = {}
        self.parse_seconds = 0.0
        self.late_after_close = 0
        self._closed = False
        # The sidecar thread records, the driver closes: a sample that arrives after
        # close() (a scrape outliving the sidecar's join) is counted and dropped, never
        # written to a re-opened file.
        self._lock = threading.Lock()

    def pod_key(self, url: str) -> str:
        return self.pods.get(url) or url

    def path_for(self, url: str) -> Path:
        return self.out_dir / f"{safe_name(self.pod_key(url))}.jsonl"

    def _write(self, url: str, row: dict, model_name: Optional[str] = None) -> None:
        pod = self.pod_key(url)
        fh = self._files.get(url)
        if fh is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fh = self.path_for(url).open("a", encoding="utf-8")
            self._files[url] = fh
            header = {
                "kind": "header", "schema": VLLM_METRICS_SCHEMA, "pod": pod, "endpoint": url,
                "model": self.model, "model_name_label": model_name,
                "histograms": list(self.histograms), "gauges": list(self.gauges),
                "counters": f"every vllm:*{COUNTER_SUFFIX} family",
                "keyframe_every": self.keyframe_every,
                "encoding": ("delta: a row with full=true carries every series; any other row "
                             "only the series (for a histogram: the buckets, s, n) whose value "
                             "changed since the previous row of this file; buckets are "
                             "cumulative counts keyed by le; label model_name dropped"),
            }
            line = json.dumps(header, separators=(",", ":")) + "\n"
            fh.write(line)
            self.bytes_written[pod] = self.bytes_written.get(pod, 0) + len(line)
        line = json.dumps(row, separators=(",", ":")) + "\n"
        fh.write(line)
        fh.flush()
        self.bytes_written[pod] = self.bytes_written.get(pod, 0) + len(line)

    def record(self, url: str, ts_ms: int, body: str) -> None:
        with self._lock:
            if self._closed:
                self.late_after_close += 1
                return
            t0 = time.perf_counter()
            state = parse_vllm_metrics(body, histograms=self.histograms, gauges=self.gauges)
            enc = self._encoders.setdefault(url, DeltaEncoder(self.keyframe_every))
            row = enc.encode(ts_ms, state)
            self.parse_seconds += time.perf_counter() - t0
            self._write(url, row, state.get("model_name"))
            pod = self.pod_key(url)
            self.samples[pod] = self.samples.get(pod, 0) + 1

    def record_error(self, url: str, ts_ms: int, error: str) -> None:
        with self._lock:
            if self._closed:
                self.late_after_close += 1
                return
            self._write(url, {"ts_ms": int(ts_ms), "error": str(error)[:500]})
            pod = self.pod_key(url)
            self.errors[pod] = self.errors.get(pod, 0) + 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for fh in self._files.values():
                try:
                    fh.close()
                except OSError:
                    pass
            self._files.clear()

    def summary(self, cell_dir: Path) -> dict:
        files = {}
        for url, pod in sorted(self.pods.items(), key=lambda kv: kv[1]):
            p = self.path_for(url)
            if p.exists():
                files[pod] = _rel(p, cell_dir)
        return {
            "dir": VLLM_METRICS_DIRNAME,
            "schema": VLLM_METRICS_SCHEMA,
            "files": files,
            "histograms": list(self.histograms),
            "gauges": list(self.gauges),
            "keyframe_every": self.keyframe_every,
            "samples": dict(sorted(self.samples.items())),
            "errors": dict(sorted(self.errors.items())),
            "bytes": dict(sorted(self.bytes_written.items())),
            "parse_seconds_total": round(self.parse_seconds, 4),
            "late_after_close": self.late_after_close,
        }


# ------------------------------------------------------------------ pod discovery


def discover_pod_targets(
    model: str,
    namespace: str,
    port: int,
    *,
    selector_extra: str = "tre.aibrix.io/routable=true",
    run: Callable[..., Any] = subprocess.run,
) -> list[dict]:
    """The model's routable pods: ``{"key": "<ns>/<pod>", "name", "namespace", "ip",
    "node", "url", "containers": [{"name", "image", "image_id"}]}`` via ``kubectl``.

    Same selector as :func:`scripts.r3_grid.discover_pod_metrics_endpoints` (routable
    only: a sleeping resident is not serving this load)."""
    selector = f"model.aibrix.ai/name={model}"
    if selector_extra:
        selector += "," + selector_extra
    out = run(
        ["kubectl", "-n", namespace, "get", "pods", "-l", selector, "-o", "json"],
        capture_output=True, text=True, check=True,
    ).stdout
    doc = json.loads(out or "{}")
    targets: list[dict] = []
    for item in doc.get("items") or []:
        meta = item.get("metadata") or {}
        status = item.get("status") or {}
        ip = status.get("podIP")
        if not ip:
            continue
        ids = {c.get("name"): c.get("imageID") for c in status.get("containerStatuses") or []}
        containers = [
            {"name": c.get("name"), "image": c.get("image"), "image_id": ids.get(c.get("name"))}
            for c in (item.get("spec") or {}).get("containers") or []
        ]
        ns = meta.get("namespace") or namespace
        targets.append({
            "key": f"{ns}/{meta.get('name')}",
            "name": meta.get("name"),
            "namespace": ns,
            "ip": ip,
            "node": (item.get("spec") or {}).get("nodeName"),
            "url": f"http://{ip}:{port}/metrics",
            "containers": containers,
        })
    targets.sort(key=lambda t: t["key"])
    return targets


def endpoint_targets(urls: Sequence[str]) -> list[dict]:
    """Targets for explicit ``--pod-endpoint`` URLs: no pod name, the URL is the key."""
    return [{"key": u, "name": None, "namespace": None, "ip": None, "node": None, "url": u,
             "containers": []} for u in urls]


# -------------------------------------------------------------------- redis dumps

#: A stamp source (gateway rounds, controller window ends) whose clock is further off
#: redis TIME than this is treated as skewed and its dump range is shifted (see
#: :func:`source_clock`); below it, offsets are recorded but ignored.
CLOCK_SKEW_TOLERANCE_MS = 2_000
#: The controller's newest window end normally trails redis TIME by one gateway round
#: plus its own tick; a lag beyond this is flagged (stale controller or slow clock).
CONTROLLER_MAX_LAG_MS = 120_000
#: ... and in sync it trails by about one round plus its tick (measured 11.6 s).
CONTROLLER_EXPECTED_LAG_MS = SCRAPE_INTERVAL_MS + CLOCK_SKEW_TOLERANCE_MS
#: Pods whose newest doc is older than this before the dump range are not dumped or
#: polled (wider than the largest known node skew, ~160 s).
ACTIVE_POD_MARGIN_MS = 300_000
#: The redis the gateway and the controller write to, as seen from inside the cluster
#: (r3_grid's default too); a driver on a host passes ``--redis-url``.
DEFAULT_REDIS_URL = "redis://tre-v2-redis:6379/0"


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)


def redis_time_ms(redis_client: Any) -> Optional[int]:
    """Redis ``TIME`` in epoch ms (None if the server cannot say)."""
    try:
        sec, usec = redis_client.time()
        return int(sec) * 1000 + int(usec) // 1000
    except Exception:  # noqa: BLE001
        return None


def clock_probe(redis_client: Any, now_ms: Callable[[], int]) -> dict:
    """Redis TIME bracketed by the local clock: offset = redis - local midpoint."""
    before = int(now_ms())
    server = redis_time_ms(redis_client)
    after = int(now_ms())
    return {
        "local_before_ms": before,
        "redis_time_ms": server,
        "local_after_ms": after,
        "redis_minus_local_ms": None if server is None else server - (before + after) / 2.0,
        "round_trip_ms": after - before,
    }


def model_pod_keys(redis_client: Any, model: str) -> list[str]:
    """The gateway's pod keys (``<ns>/<pod>``) for ``model``: ``SMEMBERS tre:v2:pods:<model>``
    (never pruned: it also lists pods long gone - see :func:`active_pods`)."""
    return sorted(_text(p) for p in (redis_client.smembers(pods_key(model)) or ()))


def _latest_score(redis_client: Any, key: str) -> Optional[int]:
    got = redis_client.zrevrangebyscore(key, "+inf", "-inf", start=0, num=1, withscores=True)
    for _member, score in got or ():
        return int(float(score))
    return None


def _latest_inst_score(redis_client: Any, pods: Sequence[str]) -> Optional[int]:
    best: Optional[int] = None
    for pod in pods:
        s = _latest_score(redis_client, inst_key(pod))
        if s is not None and (best is None or s > best):
            best = s
    return best


def active_pods(redis_client: Any, pods: Sequence[str], since_ms: int) -> list[str]:
    """The pods of ``pods`` whose newest inst or hist doc is at or after ``since_ms`` -
    the ones that can hold a doc of the cell (the pod set also lists dead pods)."""
    out = []
    for pod in pods:
        latest = [s for s in (_latest_score(redis_client, inst_key(pod)),
                              _latest_score(redis_client, hist_key(pod))) if s is not None]
        if latest and max(latest) >= since_ms:
            out.append(pod)
    return out


def source_clock(
    latest_stamp_ms: Optional[int],
    redis_now_ms: Optional[int],
    *,
    redis_minus_local_ms: Optional[float],
    stamp_period_ms: int = SCRAPE_INTERVAL_MS,
    max_lag_ms: int = SCRAPE_INTERVAL_MS + CLOCK_SKEW_TOLERANCE_MS,
    expected_lag_ms: int = SCRAPE_INTERVAL_MS // 2,
    write_phase_ms: Optional[int] = None,
) -> dict:
    """How far a stamp source's clock is from the *driver's*, and the shift its dump range
    and tail need.

    Everything is judged against the driver's clock (redis TIME minus redis's own offset
    from the driver, ``redis_minus_local_ms`` from :func:`clock_probe`), because the
    cell's ``start_ms`` / ``end_ms`` are in it. A source stamps with its own clock floored
    to ``stamp_period_ms``; in sync its newest stamp trails the driver's now by
    0 .. ``max_lag_ms`` (typically ``expected_lag_ms``), and a live gateway's write phase
    (``write_phase_ms``, redis TIME at first sight minus the stamp) lies in
    ``[0, period]`` once redis's offset is removed. A stamp more than half a period (plus
    the tolerance) closer to the driver's now than ``expected_lag_ms``, or
    a phase outside one period, means the source's clock is off: its offset is estimated
    as ``expected lag - lag`` (from the phase: ``period / 2 - phase``), to within about half
    a period. A lag beyond ``max_lag_ms`` without a measured phase is ambiguous (stale
    source or slow clock): flagged, not shifted. The shift is applied only when it exceeds
    :data:`CLOCK_SKEW_TOLERANCE_MS`."""
    rml = float(redis_minus_local_ms or 0.0)
    out: dict[str, Any] = {"latest_stamp_ms": latest_stamp_ms, "redis_time_ms": redis_now_ms,
                           "redis_minus_local_ms": redis_minus_local_ms, "write_phase_ms": write_phase_ms,
                           "expected_lag_ms": expected_lag_ms}
    lag = None if latest_stamp_ms is None or redis_now_ms is None else int(redis_now_ms) - int(latest_stamp_ms)
    lag_local = None if lag is None else int(round(lag - rml))
    phase_local = None if write_phase_ms is None else int(round(write_phase_ms - rml))
    out.update({"lag_ms": lag, "lag_vs_driver_ms": lag_local, "write_phase_vs_driver_ms": phase_local})
    source_minus_local = 0
    suspect = None
    if phase_local is not None and not (-CLOCK_SKEW_TOLERANCE_MS <= phase_local
                                        <= stamp_period_ms + CLOCK_SKEW_TOLERANCE_MS):
        source_minus_local = stamp_period_ms // 2 - phase_local
        suspect = "write phase outside one period"
    elif phase_local is None and lag_local is not None and lag_local < (
            int(expected_lag_ms) - stamp_period_ms // 2 - CLOCK_SKEW_TOLERANCE_MS):
        source_minus_local = int(expected_lag_ms) - lag_local
        suspect = "newest stamp ahead of the driver's clock"
    elif phase_local is None and lag_local is not None and lag_local > max_lag_ms:
        suspect = "newest stamp far behind the driver's clock (stale source or slow clock; not shifted)"
    shift = int(source_minus_local) if abs(source_minus_local) > CLOCK_SKEW_TOLERANCE_MS else 0
    out.update({"source_minus_local_est_ms": source_minus_local, "shift_ms": shift, "suspect": suspect})
    return out


def wait_for_gateway_write(
    redis_client: Any,
    pods: Sequence[str],
    *,
    timeout_s: float = DEFAULT_GATEWAY_FLUSH_WAIT_S,
    poll_s: float = DEFAULT_GATEWAY_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Wait (bounded) until the gateway writes its next round, and time that write.

    The gateway stamps each round with ``now - now % 10 s`` (its own clock) but its
    ticker runs at an arbitrary phase, so ``redis TIME at first sight - round stamp`` is
    the *write phase* (resolution: ``poll_s``; it includes the gateway-vs-redis clock
    offset, see :func:`source_clock`). ``timeout_s <= 0`` skips the wait."""
    out: dict[str, Any] = {"timeout_s": timeout_s, "poll_s": poll_s, "waited_s": 0.0,
                           "new_round_seen": False}
    if not pods or timeout_s <= 0:
        out["reason"] = "no pods" if not pods else "wait disabled"
        return out
    start = monotonic()
    first = _latest_inst_score(redis_client, pods)
    out["latest_round_before_ms"] = first
    while True:
        latest = _latest_inst_score(redis_client, pods)
        if latest is not None and (first is None or latest > first):
            seen = redis_time_ms(redis_client)
            out.update({
                "new_round_seen": True,
                "new_round_ms": latest,
                "seen_at_redis_ms": seen,
                "write_phase_ms": None if seen is None else seen - latest,
            })
            break
        if monotonic() - start >= timeout_s:
            break
        sleep(poll_s)
    out["waited_s"] = round(monotonic() - start, 3)
    return out


#: Where a run keeps its last measured gateway write phase (under ``cells/``), so the
#: flush wait runs once per gateway instance set and hour rather than once per cell.
GATEWAY_PHASE_CACHE = ".gateway_write_phase.json"
DEFAULT_PHASE_CACHE_S = 3600.0


def gateway_instances(redis_client: Any) -> list[str]:
    """The live gateway plugin instances (``tre:v2:gw:instances``): a restart changes the
    set and, with it, the ticker phase."""
    from tre_common.rediskeys import GW_INSTANCES_KEY

    try:
        return sorted(_text(m) for m in (redis_client.zrange(GW_INSTANCES_KEY, 0, -1) or ()))
    except Exception:  # noqa: BLE001
        return []


def cached_phase(cells_dir: Path, instances: Sequence[str], *, now_ms: int,
                 max_age_s: float = DEFAULT_PHASE_CACHE_S) -> Optional[dict]:
    """The run's cached write-phase measurement, if it is younger than ``max_age_s`` and
    was made with the same gateway instances (and there are any)."""
    if not instances or max_age_s <= 0:
        return None
    try:
        doc = json.loads((Path(cells_dir) / GATEWAY_PHASE_CACHE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("instances") != list(instances) or doc.get("write_phase_ms") is None:
        return None
    if now_ms - int(doc.get("measured_at_ms") or 0) > max_age_s * 1000.0:
        return None
    return doc


def store_phase(cells_dir: Path, instances: Sequence[str], wait: Mapping[str, Any], *, now_ms: int) -> None:
    if not wait.get("new_round_seen") or not instances:
        return
    doc = {"instances": list(instances), "measured_at_ms": int(now_ms),
           "write_phase_ms": wait.get("write_phase_ms"), "round_ms": wait.get("new_round_ms")}
    path = Path(cells_dir) / GATEWAY_PHASE_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_member(member: Any) -> Any:
    text = _text(member)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _write_jsonl_atomic(path: Path, header: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write header + rows to ``path`` via a temporary sibling; returns the bytes of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for row in rows:
            line = json.dumps(row, separators=(",", ":"), default=str) + "\n"
            fh.write(line)
            n += len(line)
    tmp.replace(path)
    return n


class NotASuperset(RuntimeError):
    """A re-dump would not contain every row of the file it replaces (retention trimmed
    the range): the existing file is kept."""


def _file_scores(path: Path) -> Counter:
    """The multiset of ``score`` values of a dump file (header excluded)."""
    out: Counter = Counter()
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("kind") != "header" and "score" in row:
                    out[int(row["score"])] += 1
    except OSError:
        pass
    return out


def _require_superset(old_path: Path, new_scores: Iterable[int], what: str) -> None:
    old = _file_scores(old_path)
    missing = old - Counter(int(s) for s in new_scores)
    if missing:
        raise NotASuperset(f"{what}: {sum(missing.values())} row(s) of the existing dump are no longer "
                           f"in redis (e.g. score {min(missing)})")


def dump_gateway_docs(
    redis_client: Any,
    layout: CellLayout,
    pods: Sequence[str],
    *,
    lo_ms: int,
    hi_ms: int,
    previous: Optional[Mapping[str, Any]] = None,
) -> dict:
    """``ZRANGEBYSCORE [lo_ms, hi_ms]`` of every pod's hist and inst key, one JSONL per
    pod and kind: ``{"score": round stamp ms, "doc": parsed JSON}``. Pods with no doc in
    the range get no file.

    ``previous`` (the summary of an earlier dump of the same cell): every row of the
    existing files must still be in the new dump, else :class:`NotASuperset` is raised
    before anything is written - a backfill only ever widens a dump."""
    fetched: dict[str, dict[str, list]] = {kind: {} for kind in GATEWAY_KINDS}
    for kind in GATEWAY_KINDS:
        key_of = hist_key if kind == "hist" else inst_key
        for pod in pods:
            fetched[kind][pod] = list(redis_client.zrangebyscore(key_of(pod), lo_ms, hi_ms, withscores=True) or [])
    if previous:
        for kind, per_pod in (previous.get("files") or {}).items():
            for pod, rel in per_pod.items():
                _require_superset(layout.cell_dir / rel,
                                  (int(float(s)) for _m, s in fetched.get(kind, {}).get(pod, [])),
                                  f"{kind} {pod}")
    files: dict[str, dict[str, str]] = {kind: {} for kind in GATEWAY_KINDS}
    counts: dict[str, dict[str, int]] = {kind: {} for kind in GATEWAY_KINDS}
    stamps: list[int] = []
    total_bytes = 0
    for kind in GATEWAY_KINDS:
        key_of = hist_key if kind == "hist" else inst_key
        for pod in pods:
            members = fetched[kind][pod]
            if not members:
                continue
            counts[kind][pod] = len(members)
            rows = []
            for member, score in members:
                s = int(float(score))
                stamps.append(s)
                rows.append({"score": s, "doc": _parse_member(member)})
            path = layout.gateway_dump_path(kind, pod)
            header = {"kind": "header", "schema": GATEWAY_DUMP_SCHEMA, "redis_key": key_of(pod),
                      "pod": pod, "doc_kind": kind, "range_ms": [lo_ms, hi_ms]}
            total_bytes += _write_jsonl_atomic(path, header, rows)
            files[kind][pod] = _rel(path, layout.cell_dir)
    phases = sorted({s % SCRAPE_INTERVAL_MS for s in stamps})
    return {
        "dir": GATEWAY_DUMP_DIRNAME,
        "schema": GATEWAY_DUMP_SCHEMA,
        "range_ms": [lo_ms, hi_ms],
        "pods": list(pods),
        "files": files,
        "docs": counts,
        "bytes": total_bytes,
        "round_stamp_phases_ms": phases,
        "round_interval_ms": SCRAPE_INTERVAL_MS,
        "last_round_ms": max(stamps) if stamps else None,
    }


TSS_RAW_FROM_CONTROLLER = "controller trs_raw"
TSS_RAW_DERIVED = "derived y_m / q_ctl (serving window: replica factor 1)"


def tss_raw_of(member: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """(pre-EMA TSS, where it came from) of one decision-history member.

    The controller writes ``trs_raw`` since 2026-09-30 (read-only field, see
    ``tre_controller.loops.decision_snapshot``; 0.0 also marks an idle / undefined
    window, as ``trs`` does). A member written before has only ``trs`` (after the EMA),
    ``y_m`` (numerator) and ``q_ctl`` (floored queue); its raw TSS is
    ``y_m / q_ctl * factor`` where the factor is the assigned/routable correction of the
    *serving window* - 1 whenever the controller has its fleet view (the normal path:
    ``serving_window`` sets both counts to the awake-and-not-hidden pods) - and is NOT
    the member's ``assigned_replicas`` (that is the bound count). The derived value
    assumes factor 1 and says so; None when the controller had no token data."""
    if "trs_raw" in member:
        raw = member.get("trs_raw")
        return (None if raw is None else float(raw)), TSS_RAW_FROM_CONTROLLER
    y, q = member.get("y_m"), member.get("q_ctl")
    if y is None or q is None:
        return None, None
    if float(q) > 0.0:
        return float(y) / float(q), TSS_RAW_DERIVED
    return (math.inf if float(y) > 0 else 0.0), TSS_RAW_DERIVED


def dump_controller_ticks(
    redis_client: Any,
    layout: CellLayout,
    model: str,
    *,
    lo_ms: int,
    hi_ms: int,
    previous: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The controller's per-model decision history (``tre:v2:decision:hist:<model>``,
    score = ``window_end_ms``, kept ~24 h) over ``[lo_ms, hi_ms]``, one line per member:
    the member as written (``trs`` = EMA'd TSS, ``trs_z_m`` = its Z, ``z_m`` = the active
    signal's Z, ``state`` = band, ``trs_raw`` = pre-EMA TSS (controllers from 2026-09-30),
    ``y_m``, ``q_ctl``, replicas, ``window_end_ms``, ``ts``) plus ``tss_raw`` /
    ``tss_raw_source`` (:func:`tss_raw_of`). The rescue and the fairness loop both write
    per tick, so one ``window_end_ms`` can hold two members; readers dedup. ``previous``:
    as in :func:`dump_gateway_docs`."""
    key = decision_hist_key(model)
    members = list(redis_client.zrangebyscore(key, lo_ms, hi_ms, withscores=True) or [])
    if previous and previous.get("file"):
        _require_superset(layout.cell_dir / previous["file"], (int(float(s)) for _m, s in members), key)
    rows = []
    for member, score in members:
        doc = _parse_member(member)
        row: dict[str, Any] = {"score": int(float(score))}
        if isinstance(doc, dict):
            row.update(doc)
            row["tss_raw"], row["tss_raw_source"] = tss_raw_of(doc)
        else:
            row["raw"] = doc
        rows.append(row)
    header = {"kind": "header", "schema": CONTROLLER_TICKS_SCHEMA, "redis_key": key, "model": model,
              "range_ms": [lo_ms, hi_ms],
              "fields": {"trs": "TSS after the EMA", "trs_z_m": "trs / theta_m",
                         "z_m": "Z of the active signal source", "state": "band",
                         "tss_raw": "pre-EMA TSS (tss_raw_source says whether the controller "
                                    "wrote it or it was derived from y_m / q_ctl)"}}
    _write_jsonl_atomic(layout.controller_ticks_path, header, rows)
    ends = [r["score"] for r in rows]
    return {
        "file": _rel(layout.controller_ticks_path, layout.cell_dir),
        "schema": CONTROLLER_TICKS_SCHEMA,
        "redis_key": key,
        "range_ms": [lo_ms, hi_ms],
        "members": len(rows),
        "first_window_end_ms": min(ends) if ends else None,
        "last_window_end_ms": max(ends) if ends else None,
    }


# --------------------------------------------------------------------- manifests


def file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_state(worktree: Path, run: Callable[..., Any] = subprocess.run) -> dict:
    def git(*argv: str) -> Optional[str]:
        try:
            proc = run(["git", "-C", str(worktree), *argv], capture_output=True, text=True,
                       timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def control_plane_images(namespace: str, run: Callable[..., Any] = subprocess.run) -> dict:
    """``{deployment: [{"container", "image"}]}`` of ``namespace`` (controller, SM,
    gateway plugins, redis, ...); ``{"error": ...}`` when kubectl cannot say."""
    try:
        out = run(["kubectl", "-n", namespace, "get", "deploy", "-o", "json"],
                  capture_output=True, text=True, check=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    images: dict[str, list] = {}
    for item in json.loads(out or "{}").get("items") or []:
        name = (item.get("metadata") or {}).get("name")
        spec = ((item.get("spec") or {}).get("template") or {}).get("spec") or {}
        images[name] = [{"container": c.get("name"), "image": c.get("image")}
                        for c in spec.get("containers") or []]
    return dict(sorted(images.items()))


def live_configmap(namespace: str, name: str, run: Callable[..., Any] = subprocess.run) -> dict:
    """sha256 of every data key of a ConfigMap - for ``tre-v2-registry`` this is the
    registry the controller and the SM actually read (its ``registry.yaml`` hash compares
    directly with the repo file's); ``{"error": ...}`` when kubectl cannot say."""
    try:
        out = run(["kubectl", "-n", namespace, "get", "configmap", name, "-o", "json"],
                  capture_output=True, text=True, check=True, timeout=30).stdout
        data = (json.loads(out or "{}").get("data") or {})
    except Exception as exc:  # noqa: BLE001
        return {"namespace": namespace, "name": name, "error": repr(exc)}
    return {"namespace": namespace, "name": name,
            "data_sha256": {k: hashlib.sha256(str(v).encode("utf-8")).hexdigest() for k, v in sorted(data.items())}}


def write_run_manifest(
    model_dir: Path,
    *,
    model: str,
    config: Mapping[str, Any],
    registry_path: Optional[Path],
    repo: Path,
    model_pods: Sequence[Mapping[str, Any]] = (),
    control_namespace: Optional[str] = None,
    registry_configmap: Optional[str] = None,
    run: Callable[..., Any] = subprocess.run,
) -> Optional[Path]:
    """``<model dir>/manifest.json`` - written once, by the first cell of the run
    (exclusive create: a second writer leaves it alone and gets None).

    Records what the run's data can only be interpreted against: the capture layout
    version, the code (commit, branch, dirty), the registry (the driver's file and the
    live ConfigMap the controller reads, both hashed), the model
    pods' and the control plane's images, the driver configuration, and the hashes of
    the campaign's own plan documents present at that moment (``plan.json``,
    ``fit_plan.json``, ``run_manifest.json``)."""
    path = run_manifest_path(model_dir)
    if path.exists():
        return None
    plan_docs = {}
    for name in ("plan.json", "fit_plan.json", "run_manifest.json"):
        p = Path(model_dir) / name
        if p.exists():
            plan_docs[name] = {"path": name, "sha256": file_sha256(p)}
    doc = {
        "kind": "tre.calibration_run_manifest",
        "layout_version": CAPTURE_LAYOUT_VERSION,
        "layout": {
            "cells_dir": CELLS_DIRNAME, "cell_meta": CELL_META,
            "vllm_metrics": f"{CELLS_DIRNAME}/<stem>/{VLLM_METRICS_DIRNAME}/<ns>_<pod>.jsonl",
            "gateway_redis_dump": f"{CELLS_DIRNAME}/<stem>/{GATEWAY_DUMP_DIRNAME}/{{hist,inst}}/<ns>_<pod>.jsonl",
            "controller_ticks": f"{CELLS_DIRNAME}/<stem>/{CONTROLLER_TICKS}",
            "unchanged": ["<stem>.csv", "raw/<stem>/<cell_id>.jsonl", "raw/<stem>/<cell_id>.instant.jsonl",
                          "raw/<stem>/<cell_id>.guard.json", "prompts/", "schedules/"],
        },
        "written_at_utc": utc_iso(),
        "writer": {"host": socket.gethostname(), "pid": os.getpid()},
        "model": model,
        "code": git_state(repo, run=run),
        "registry": {"path": None if registry_path is None else str(registry_path),
                     "sha256": None if registry_path is None else file_sha256(Path(registry_path)),
                     "live_configmap": (live_configmap(control_namespace, registry_configmap, run=run)
                                        if control_namespace and registry_configmap else None)},
        "images": {
            "model_pods": [
                {"pod": p.get("key"), "node": p.get("node"), "containers": p.get("containers")}
                for p in model_pods
            ],
            "control_plane": (
                {"namespace": control_namespace, "deployments": control_plane_images(control_namespace, run=run)}
                if control_namespace else None
            ),
        },
        "campaign_documents": plan_docs,
        "config": {k: v for k, v in sorted(config.items())},
    }
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    except FileExistsError:
        return None
    return path


def write_cell_meta(layout: CellLayout, meta: Mapping[str, Any]) -> Path:
    layout.cell_dir.mkdir(parents=True, exist_ok=True)
    tmp = layout.meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(layout.meta_path)
    return layout.meta_path


BACKFILL_MARKER = "BACKFILL_PENDING"
#: How long ``finalize_run`` waits, at most, for the controller to have processed the
#: last cell's tail windows before the final backfill.
DEFAULT_FINAL_BACKFILL_WAIT_S = 90.0
#: Cells whose marker is older than this are left alone by the per-cell backfill (the
#: gateway docs are gone after 30 min anyway; the CLI can still be run by hand).
DEFAULT_BACKFILL_MAX_AGE_S = 6 * 3600.0


def dump_range_ms(start_ms: int, end_ms: int, window_ms: int, shift_ms: int = 0) -> tuple[int, int]:
    """``[lo, hi]`` of a redis dump of a cell, in the source's clock (``shift_ms`` = source
    minus driver, :func:`source_clock`): from one window plus one gateway round before the
    cell (the first window's read reaches that far back) to one window plus one round
    after it (the last controller window that still holds part of the cell)."""
    return (int(start_ms) + int(shift_ms) - int(window_ms) - SCRAPE_INTERVAL_MS,
            int(end_ms) + int(shift_ms) + int(window_ms) + SCRAPE_INTERVAL_MS)


def tail_ms(end_ms: int, window_ms: int, shift_ms: int = 0) -> int:
    """End of the last window on the 10 s grid that still overlaps the cell, in the
    source's clock - a dump is *complete* once it reaches it (controller ticks:
    ``window_end_ms``; gateway docs: round stamp)."""
    end = int(end_ms) + int(shift_ms) + int(window_ms) - 1
    return (end // SCRAPE_INTERVAL_MS) * SCRAPE_INTERVAL_MS


def _shift_of(meta: Mapping[str, Any], source: str) -> int:
    return int(((meta.get("clock") or {}).get(source) or {}).get("shift_ms") or 0)


def _mark_completeness(meta: dict) -> bool:
    """Set ``complete`` (and ``tail_ms``) on the redis dumps of ``meta``; True when
    something is still incomplete (a backfill is due)."""
    pending = False
    for name, source, last_key in (("gateway_redis_dump", "gateway", "last_round_ms"),
                                   ("controller_ticks", "controller", "last_window_end_ms")):
        dump = meta.get(name)
        if dump is None:
            continue
        shift = _shift_of(meta, source)
        # a shift is an estimate good to about half a round: then wait one round longer
        tail = tail_ms(meta["end_ms"], meta["window_ms"], shift) + (SCRAPE_INTERVAL_MS if shift else 0)
        dump["tail_ms"] = tail
        dump["complete"] = dump.get(last_key) is not None and dump[last_key] >= tail
        pending |= not dump["complete"]
    return pending


def _write_meta_and_marker(layout: CellLayout, meta: dict) -> None:
    pending = _mark_completeness(meta)
    write_cell_meta(layout, meta)
    marker = layout.cell_dir / BACKFILL_MARKER
    if pending:
        marker.write_text(utc_iso() + "\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def capture_after_cell(
    layout: CellLayout,
    *,
    model: str,
    start_ms: int,
    end_ms: int,
    window_ms: int,
    redis_client: Any = None,
    targets: Sequence[Mapping[str, Any]] = (),
    recorder: Optional[VllmMetricsRecorder] = None,
    extra_legacy: Optional[Mapping[str, Optional[Path]]] = None,
    gateway_dump: bool = True,
    controller_ticks: bool = True,
    flush_wait_s: float = DEFAULT_GATEWAY_FLUSH_WAIT_S,
    phase_cache_s: float = DEFAULT_PHASE_CACHE_S,
    poll_s: float = DEFAULT_GATEWAY_POLL_S,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    info: Optional[Mapping[str, Any]] = None,
) -> dict:
    """After a cell (its load drained): measure the clocks, dump the gateway docs and the
    controller ticks over :func:`dump_range_ms`, write ``cell_meta.json``.

    Clocks: ``start_ms`` / ``end_ms`` are the driver's clock, the gateway stamps its
    rounds and the controller its windows with their own. :func:`source_clock` estimates
    each source's offset (redis TIME bracketed by the driver's clock, the source's newest
    stamp, the gateway's write phase) and shifts that source's range when the offset
    exceeds :data:`CLOCK_SKEW_TOLERANCE_MS`; everything is recorded under ``clock``.

    The gateway's write phase is measured by waiting (at most ``flush_wait_s``) for its
    next round - once per ``phase_cache_s`` and gateway instance set, not per cell.

    Right after the cell the controller has not yet processed the windows that end up to
    one window later, and the gateway has written at most one round past the end; such a
    dump is marked ``complete: false`` and the cell directory gets a ``BACKFILL_PENDING``
    marker. :func:`backfill_pending` - run by the next cell's driver and by the
    campaign's finalize - re-dumps it once the data exists (a re-dump only ever widens a
    dump). Never raises: a failed step leaves its error in ``cell_meta.json["errors"]``."""
    errors: list[str] = []
    meta: dict[str, Any] = {
        "kind": "tre.calibration_cell_meta",
        "layout_version": CAPTURE_LAYOUT_VERSION,
        "model": model,
        "stem": layout.stem,
        "cell_id": layout.cell_id,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "window_ms": int(window_ms),
        "written_at_utc": utc_iso(),
        "legacy": {},
        "pods": [dict(t) for t in targets],
    }
    legacy = dict(layout.legacy_paths())
    for key, path in (extra_legacy or {}).items():
        legacy[key] = path
    meta["legacy"] = {k: _rel(p, layout.cell_dir) for k, p in sorted(legacy.items()) if p is not None}
    if info:
        meta.update({k: v for k, v in info.items() if k not in meta})
    if recorder is not None:
        try:
            recorder.close()
            meta["vllm_metrics"] = recorder.summary(layout.cell_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vllm_metrics: {exc!r}")
    if redis_client is not None and (gateway_dump or controller_ticks):
        clock: dict[str, Any] = {}
        meta["clock"] = clock
        rml: Optional[float] = None
        try:
            probe = clock_probe(redis_client, now_ms)
            clock["probe_before"] = probe
            rml = probe["redis_minus_local_ms"]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"redis clock: {exc!r}")
        if gateway_dump:
            try:
                in_set = model_pod_keys(redis_client, model)
                # widened by the largest plausible skew: a skewed gateway's stamps may lie
                # before the unshifted range, and a pod must not be dropped for that
                pods = active_pods(redis_client, in_set,
                                   dump_range_ms(start_ms, end_ms, window_ms)[0] - ACTIVE_POD_MARGIN_MS)
                instances = gateway_instances(redis_client)
                cached = cached_phase(layout.cell_dir.parent, instances, now_ms=int(now_ms()),
                                      max_age_s=phase_cache_s)
                if cached is not None:
                    wait = {"skipped": "write phase cached for this gateway instance set",
                            "write_phase_ms": cached["write_phase_ms"],
                            "measured_at_ms": cached["measured_at_ms"], "new_round_seen": False}
                else:
                    wait = wait_for_gateway_write(redis_client, pods, timeout_s=flush_wait_s, poll_s=poll_s,
                                                  sleep=sleep, monotonic=monotonic)
                    try:
                        store_phase(layout.cell_dir.parent, instances, wait, now_ms=int(now_ms()))
                    except OSError as exc:
                        errors.append(f"gateway phase cache: {exc!r}")
                clock["gateway"] = source_clock(
                    _latest_inst_score(redis_client, pods), redis_time_ms(redis_client),
                    redis_minus_local_ms=rml, write_phase_ms=wait.get("write_phase_ms"),
                )
                lo, hi = dump_range_ms(start_ms, end_ms, window_ms, clock["gateway"]["shift_ms"])
                summary = dump_gateway_docs(redis_client, layout, pods, lo_ms=lo, hi_ms=hi)
                summary["flush_wait"] = wait
                summary["pods_in_set"] = len(in_set)
                summary["gateway_instances"] = instances
                meta["gateway_redis_dump"] = summary
            except Exception as exc:  # noqa: BLE001
                errors.append(f"gateway_redis_dump: {exc!r}")
        if controller_ticks:
            try:
                clock["controller"] = source_clock(
                    _latest_score(redis_client, decision_hist_key(model)), redis_time_ms(redis_client),
                    redis_minus_local_ms=rml, max_lag_ms=CONTROLLER_MAX_LAG_MS,
                    expected_lag_ms=CONTROLLER_EXPECTED_LAG_MS,
                )
                lo, hi = dump_range_ms(start_ms, end_ms, window_ms, clock["controller"]["shift_ms"])
                meta["controller_ticks"] = dump_controller_ticks(redis_client, layout, model, lo_ms=lo, hi_ms=hi)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"controller_ticks: {exc!r}")
        try:
            clock["probe_after"] = clock_probe(redis_client, now_ms)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"redis clock: {exc!r}")
        suspects = {k: v.get("suspect") for k, v in clock.items() if isinstance(v, dict) and v.get("suspect")}
        if suspects:
            meta["clock_skew_suspected"] = suspects
    elif gateway_dump or controller_ticks:
        errors.append("no redis client: gateway docs and controller ticks not captured")
    meta["errors"] = errors
    try:
        _write_meta_and_marker(layout, meta)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cell_meta: {exc!r}")
    return meta


def backfill_cell(
    cell_dir: Path, redis_client: Any, *, now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> Optional[dict]:
    """Re-dump the incomplete redis dumps of one cell over the same ranges (the clock
    shifts measured after the cell are reused); returns the backfill record (also
    appended to ``cell_meta.json["backfills"]``), None when the cell was already complete.
    A re-dump that would lose a row of the existing file (retention) keeps the old file
    and says so."""
    cell_dir = Path(cell_dir)
    meta = json.loads((cell_dir / CELL_META).read_text(encoding="utf-8"))
    layout = CellLayout(cell_dir.parent.parent, meta["stem"], meta["cell_id"])
    _mark_completeness(meta)
    gd, ct = meta.get("gateway_redis_dump"), meta.get("controller_ticks")
    todo_gw = gd is not None and not gd.get("complete")
    todo_ct = ct is not None and not ct.get("complete")
    if not (todo_gw or todo_ct):
        _write_meta_and_marker(layout, meta)
        return None
    record: dict[str, Any] = {"at_utc": utc_iso(), "redis_clock": clock_probe(redis_client, now_ms)}
    if todo_gw:
        lo, hi = dump_range_ms(meta["start_ms"], meta["end_ms"], meta["window_ms"], _shift_of(meta, "gateway"))
        try:
            new = dump_gateway_docs(redis_client, layout, gd.get("pods") or [], lo_ms=lo, hi_ms=hi, previous=gd)
            for key in ("flush_wait", "pods_in_set", "gateway_instances"):
                if key in gd:
                    new[key] = gd[key]
            record["gateway_docs"] = {"docs_before": sum(sum(v.values()) for v in (gd.get("docs") or {}).values()),
                                      "docs_after": sum(sum(v.values()) for v in new["docs"].values()),
                                      "last_round_before_ms": gd.get("last_round_ms"),
                                      "last_round_after_ms": new.get("last_round_ms")}
            meta["gateway_redis_dump"] = new
        except NotASuperset as exc:
            record["gateway_docs_kept"] = str(exc)
    if todo_ct:
        lo, hi = dump_range_ms(meta["start_ms"], meta["end_ms"], meta["window_ms"], _shift_of(meta, "controller"))
        try:
            new = dump_controller_ticks(redis_client, layout, meta["model"], lo_ms=lo, hi_ms=hi, previous=ct)
            record["controller_ticks"] = {"members_before": ct.get("members"), "members_after": new["members"]}
            meta["controller_ticks"] = new
        except NotASuperset as exc:
            record["controller_ticks_kept"] = str(exc)
    meta.setdefault("backfills", []).append(record)
    _write_meta_and_marker(layout, meta)
    record["complete"] = not (layout.cell_dir / BACKFILL_MARKER).exists()
    return record


def pending_cells(root: Path, *, max_age_s: Optional[float] = DEFAULT_BACKFILL_MAX_AGE_S,
                  now: Callable[[], float] = time.time) -> list[Path]:
    """Cell directories under ``root`` (a model directory, or a run directory holding
    model directories) whose ``BACKFILL_PENDING`` marker says a backfill is due, oldest
    first; markers older than ``max_age_s`` are skipped (None = no limit)."""
    root = Path(root)
    markers = list(root.glob(f"{CELLS_DIRNAME}/*/{BACKFILL_MARKER}")) + list(
        root.glob(f"*/{CELLS_DIRNAME}/*/{BACKFILL_MARKER}"))
    found = []
    for marker in markers:
        try:
            mtime = marker.stat().st_mtime
        except OSError:
            continue
        if max_age_s is None or now() - mtime <= max_age_s:
            found.append((mtime, marker.parent))
    return [cell for _, cell in sorted(found)]


def backfill_pending(
    root: Path,
    redis_client: Any,
    *,
    wait_s: float = 0.0,
    max_age_s: Optional[float] = DEFAULT_BACKFILL_MAX_AGE_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> list[dict]:
    """Backfill every pending cell under ``root``. With ``wait_s`` > 0 it first waits
    (bounded) until redis time is two rounds past the latest pending tail, so the
    controller has processed the last windows - what the campaign's finalize does after
    its last cell. Returns one record per backfilled cell (``{"cell", "error"}`` on a
    failure, which never stops the others)."""
    cells = pending_cells(root, max_age_s=max_age_s)
    if not cells:
        return []
    if wait_s > 0:
        tails = []
        for cell in cells:
            try:
                m = json.loads((cell / CELL_META).read_text(encoding="utf-8"))
                rml = (((m.get("clock") or {}).get("probe_before") or {}).get("redis_minus_local_ms") or 0.0)
                # the cell's tail in redis's clock (the driver's end plus redis's offset)
                tails.append(tail_ms(m["end_ms"], m["window_ms"], int(round(rml))))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        target = (max(tails) + 2 * SCRAPE_INTERVAL_MS) if tails else None
        deadline = monotonic() + wait_s
        while target is not None:
            server = redis_time_ms(redis_client)
            left = deadline - monotonic()
            if server is None or server >= target or left <= 0:
                break
            sleep(max(0.05, min((target - server) / 1000.0, left, 5.0)))
    out = []
    for cell in cells:
        try:
            rec = backfill_cell(cell, redis_client, now_ms=now_ms)
            if rec is not None:
                out.append({"cell": str(cell), **rec})
        except Exception as exc:  # noqa: BLE001 - one bad cell never stops the others
            out.append({"cell": str(cell), "error": repr(exc)})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m scripts.calibration_capture backfill <run or model dir> --redis-url URL``:
    complete the redis dumps of cells still marked ``BACKFILL_PENDING`` (the gateway keeps
    its docs 30 min, the controller its history ~24 h)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    bf = sub.add_parser("backfill")
    bf.add_argument("root", type=Path)
    bf.add_argument("--redis-url", required=True)
    bf.add_argument("--wait-s", type=float, default=0.0)
    bf.add_argument("--max-age-s", type=float, default=None,
                    help="skip markers older than this (default: no limit)")
    args = ap.parse_args(argv)
    import redis  # type: ignore[import-not-found]

    client = redis.Redis.from_url(args.redis_url)
    records = backfill_pending(args.root, client, wait_s=args.wait_s, max_age_s=args.max_age_s)
    for rec in records:
        print(json.dumps(rec, sort_keys=True, default=str))
    left = pending_cells(args.root, max_age_s=None)
    print(f"{len(records)} cell(s) backfilled; {len(left)} still pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
