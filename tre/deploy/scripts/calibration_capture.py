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

Clocks: every redis dump is placed in *redis time* (redis ``TIME`` read when the cell
starts and ends), never in the driver's clock, and nothing is shifted. The gateway's
round stamps and the controller's window ends are checked to be in redis's time domain
before a run (:func:`require_clock_domains`: a failure refuses the run) and around every
cell (:func:`cell_clock_mark`: a failure marks the cell's dumps ``clock_domain_mismatch``
- kept, never complete). See :class:`ClockDomainConfig`.
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

#: How long to wait for the gateway's next Redis write (its ticker has an arbitrary
#: phase against the 10 s round, so up to one interval plus the write itself).
DEFAULT_GATEWAY_FLUSH_WAIT_S = SCRAPE_INTERVAL_MS / 1000.0 + 2.0
#: ... after a cell, before the dump: none by default - the end mark's clock check has
#: just waited for a gateway round (:func:`cell_clock_mark`).
DEFAULT_CAPTURE_FLUSH_WAIT_S = 0.0
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
    """Redis TIME bracketed by the local clock. Only ``redis_time_ms`` is used (as the
    cell's time base); the local readings and ``redis_minus_local_ms`` are audit fields."""
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


def _latest_gateway_score(redis_client: Any, pods: Sequence[str]) -> Optional[int]:
    best: Optional[int] = None
    for pod in pods:
        for key in (inst_key(pod), hist_key(pod)):
            s = _latest_score(redis_client, key)
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


# ------------------------------------------------------------ the clock: redis TIME
#
# Every redis stamp this module exports (gateway round stamps, controller window ends)
# is read in *redis time*: the driver reads redis ``TIME`` when a cell starts and when it
# ends, and dumps ``[redis_start - margin, redis_end + margin]``. The driver's own clock
# never enters a dump range, so a skewed driver cannot misplace one. Nothing is shifted:
# instead the gateway and the controller are *checked* to stamp in redis time (before
# the run - a failure refuses the run - and around every cell - a failure marks the
# cell's dumps ``clock_domain_mismatch``, never complete). The labels and the fit use the
# client's own records only; what a mismatch costs is this supplementary evidence.

#: Host clocks agree to NTP accuracy, not better: a source may be this much outside its
#: normal range against redis TIME and still be in redis's time domain.
DEFAULT_CLOCK_TOLERANCE_MS = 2_000
#: Gateway docs are trimmed this long after their stamp (``treV2RetentionMS``,
#: ``pkg/cache/cache_tre_redis.go``).
GATEWAY_RETENTION_MS = 30 * 60 * 1000
#: The phase-aligned controller reads window ``B`` (a gateway round stamp: the window
#: is only published once the gateway has written its tick ``B``) at ``B + offset``; the
#: offset starts at ``TRE_METRICS_PHASE_OFFSET_MS`` (2 s) and adapts up to
#: ``period - retry`` = 9.5 s when the gateway writes late in its round
#: (``tre_controller.loops.metrics_task.PhaseAlignedSampler``).
DEFAULT_CONTROLLER_READ_OFFSET_MS = SCRAPE_INTERVAL_MS - 500
#: ... and the window reaches the decision history on the next tick of a loop that
#: writes it: the rescue loop (``TRE_RESCUE_INTERVAL_SECONDS``, 5 s) or, when the fast loop
#: is disabled, the fairness loop (``TRE_FAIRNESS_INTERVAL_SECONDS``, 10 s). The bound
#: takes the slower one.
DEFAULT_CONTROLLER_TICK_MS = 10_000
#: A failed check is repeated (a gateway round or a controller tick may be late once).
DEFAULT_CLOCK_CHECK_ATTEMPTS = 3
DEFAULT_CLOCK_CHECK_RETRY_S = 4.0
CLOCK_DOMAIN_OK = "ok"
CLOCK_DOMAIN_MISMATCH = "clock_domain_mismatch"


@dataclass(frozen=True)
class ClockDomainConfig:
    """The dump margin and the bounds of the clock-domain check.

    Gateway: its ticker starts wherever the process started, so it writes round ``S`` at
    ``S + d`` with a fixed *write delay* ``d`` in ``[0, period)``. The check waits for the
    next round and measures ``d`` = redis TIME at first sight minus the new stamp (the
    newest stamp alone trails redis by ``[d, d + period)``, up to two rounds, so it cannot
    tell a late ticker from a slow clock). In redis time ``d`` lies in ``[0, period]``;
    ``[-tol, period + tol]`` is accepted.

    Controller: the phase-aligned controller's ``window_end_ms`` is a gateway round (on
    the ``period`` grid; a free-running controller stamps its own clock and is refused),
    trailing redis TIME by ``[0, period + read offset + tick]`` (2-17 s at the base read
    offset and the 5 s rescue loop; up to 29.5 s with an adapted offset and the 10 s
    fairness loop); ``tolerance_ms`` is added on both sides.

    The gateway check cannot see an offset of up to one round plus the tolerance
    (:attr:`blind_spot_ms`, 12 s: a gateway that far ahead looks like one with a late
    ticker, and as far behind like one with an early ticker). The default ``margin_ms`` is
    therefore the reach of the first and last controller window over the cell (one
    window plus one round, 40 s at 30 s windows) plus that blind spot: 52 s; a smaller
    margin is refused, and the tail (:func:`tail_ms`) includes the blind spot too."""

    window_ms: int = 30_000
    period_ms: int = SCRAPE_INTERVAL_MS
    tolerance_ms: int = DEFAULT_CLOCK_TOLERANCE_MS
    controller_read_offset_ms: int = DEFAULT_CONTROLLER_READ_OFFSET_MS
    controller_tick_ms: int = DEFAULT_CONTROLLER_TICK_MS
    margin_ms: Optional[int] = None
    attempts: int = DEFAULT_CLOCK_CHECK_ATTEMPTS
    retry_s: float = DEFAULT_CLOCK_CHECK_RETRY_S
    poll_s: float = DEFAULT_GATEWAY_POLL_S

    def __post_init__(self) -> None:
        if min(self.window_ms, self.period_ms) <= 0 or min(
                self.tolerance_ms, self.controller_read_offset_ms, self.controller_tick_ms) < 0:
            raise ValueError(f"invalid clock-domain config: {self}")
        least = self.window_ms + self.period_ms + self.blind_spot_ms
        if self.margin_ms is not None and self.margin_ms < least:
            raise ValueError(f"capture margin {self.margin_ms} ms is below one window plus one gateway "
                             f"round plus the clock check's blind spot ({least} ms): the first and last "
                             "controller windows over the cell could be cut")

    @property
    def blind_spot_ms(self) -> int:
        return int(self.period_ms + self.tolerance_ms)

    @property
    def margin(self) -> int:
        return int(self.margin_ms if self.margin_ms is not None
                   else self.window_ms + self.period_ms + self.blind_spot_ms)

    @property
    def gateway_write_delay_bounds(self) -> tuple[int, int]:
        return -int(self.tolerance_ms), int(self.period_ms + self.tolerance_ms)

    @property
    def gateway_wait_s(self) -> float:
        """How long the check waits for the gateway's next round."""
        return (self.period_ms + self.tolerance_ms) / 1000.0

    @property
    def controller_lag_bounds(self) -> tuple[int, int]:
        return -int(self.tolerance_ms), int(self.period_ms + self.controller_read_offset_ms
                                            + self.controller_tick_ms + self.tolerance_ms)

    def as_dict(self) -> dict:
        return {"basis": "redis TIME", "window_ms": self.window_ms, "period_ms": self.period_ms,
                "tolerance_ms": self.tolerance_ms, "controller_read_offset_ms": self.controller_read_offset_ms,
                "controller_tick_ms": self.controller_tick_ms, "margin_ms": self.margin,
                "blind_spot_ms": self.blind_spot_ms,
                "gateway_write_delay_bounds_ms": list(self.gateway_write_delay_bounds),
                "controller_lag_bounds_ms": list(self.controller_lag_bounds),
                "attempts": self.attempts, "retry_s": self.retry_s, "poll_s": self.poll_s}


def dump_range_ms(redis_start_ms: int, redis_end_ms: int, margin_ms: int) -> tuple[int, int]:
    """``[lo, hi]`` of a cell's redis dumps, in redis time."""
    return int(redis_start_ms) - int(margin_ms), int(redis_end_ms) + int(margin_ms)


def tail_ms(redis_end_ms: int, window_ms: int, blind_spot_ms: int, period_ms: int = SCRAPE_INTERVAL_MS) -> int:
    """The last round stamp / window end a dump must hold to be *complete*: the last grid
    window end ``B`` whose window ``(B - window, B]`` can hold data of the cell, a gateway
    offset inside the blind spot included (data up to ``redis_end + blind`` in the
    gateway's stamps): ``redis_end + 32 .. 41 s`` at 30 s windows."""
    return (int(redis_end_ms) + int(window_ms) + int(blind_spot_ms) - 1) // int(period_ms) * int(period_ms)


def _check_once(redis_client: Any, model: str, cfg: ClockDomainConfig,
                sleep: Callable[[float], None]) -> dict:
    """One clock-domain check of ``model``: the controller's newest window end against
    redis TIME (read after it), then the gateway's write delay, measured by waiting (at
    most :attr:`ClockDomainConfig.gateway_wait_s`) for its next round."""
    reasons: list[str] = []
    newest = redis_client.zrevrangebyscore(decision_hist_key(model), "+inf", "-inf", start=0, num=8,
                                           withscores=True) or ()
    ends = sorted({int(float(score)) for _m, score in newest}, reverse=True)[:2]
    c_stamp = ends[0] if ends else None
    c_now = redis_time_ms(redis_client)
    c_lag = None if c_stamp is None or c_now is None else c_now - c_stamp
    lo_c, hi_c = cfg.controller_lag_bounds
    # phase-aligned: window ends are gateway rounds, one round apart (free_running stamps
    # its own clock: off the grid when sliding, one window apart when tumbling)
    on_grid = (len(ends) == 2 and ends[0] % cfg.period_ms == 0 and ends[0] - ends[1] == cfg.period_ms)
    c_ok = c_lag is not None and on_grid and lo_c <= c_lag <= hi_c
    if c_lag is None:
        reasons.append("controller: no decision history for the model or no redis TIME")
    elif not on_grid:
        reasons.append(f"controller: newest window ends {ends} are not consecutive {cfg.period_ms} ms gateway "
                       "rounds (not phase-aligned: it stamps its own clock)")
    elif not c_ok:
        reasons.append(f"controller: newest window_end_ms is {c_lag} ms behind redis TIME, outside "
                       f"[{lo_c}, {hi_c}]")
    pods = model_pod_keys(redis_client, model)
    now = redis_time_ms(redis_client)
    # pods with a doc in the last 10 min (wide enough for a badly skewed gateway to be
    # measured rather than just "silent"; the pod set also lists pods long gone)
    live = active_pods(redis_client, pods, now - 600_000) if now is not None else []
    g_stamp = _latest_gateway_score(redis_client, live)
    elapsed = [0.0]

    def _sleep(dt: float) -> None:
        sleep(dt)
        elapsed[0] += dt

    wait = wait_for_gateway_write(redis_client, live, timeout_s=cfg.gateway_wait_s, poll_s=cfg.poll_s,
                                  sleep=_sleep, monotonic=lambda: elapsed[0])
    delay = wait.get("write_phase_ms")
    lo, hi = cfg.gateway_write_delay_bounds
    g_ok = delay is not None and lo <= delay <= hi
    if not live:
        reasons.append("gateway: no doc of the model's pods in the last 10 min")
    elif delay is None:
        reasons.append(f"gateway: no new round within {cfg.gateway_wait_s:g} s")
    elif not g_ok:
        reasons.append(f"gateway: round {wait.get('new_round_ms')} first seen {delay} ms after its stamp "
                       f"(redis TIME), outside [{lo}, {hi}]")
    return {
        "ok": g_ok and c_ok,
        "gateway": {"ok": g_ok, "write_delay_ms": delay, "new_round_ms": wait.get("new_round_ms"),
                    "waited_s": wait.get("waited_s"), "newest_stamp_before_ms": g_stamp,
                    "lag_before_ms": None if g_stamp is None or now is None else now - g_stamp,
                    "pods_live": len(live), "pods_in_set": len(pods)},
        "controller": {"ok": c_ok, "newest_window_end_ms": c_stamp, "previous_window_end_ms":
                       ends[1] if len(ends) > 1 else None, "redis_time_ms": c_now, "lag_ms": c_lag},
        "reasons": reasons,
    }


def check_clock_domains(redis_client: Any, model: str, cfg: ClockDomainConfig, *,
                        sleep: Optional[Callable[[float], None]] = None) -> dict:
    """Whether the gateway's round stamps and the controller's window ends of ``model``
    are in redis's time domain (:class:`ClockDomainConfig` has the bounds). A failed check
    is repeated up to ``cfg.attempts`` times, ``cfg.retry_s`` apart; the last verdict is
    returned with ``attempts``. Never raises (a redis error is a failed check)."""
    verdict: dict[str, Any] = {}
    attempts = max(1, int(cfg.attempts))
    for attempt in range(1, attempts + 1):
        try:
            verdict = _check_once(redis_client, model, cfg, sleep or time.sleep)
        except Exception as exc:  # noqa: BLE001
            verdict = {"ok": False, "gateway": {"ok": False}, "controller": {"ok": False},
                       "reasons": [f"redis error: {exc!r}"]}
        verdict["attempts"] = attempt
        if verdict["ok"] or attempt == attempts:
            break
        (sleep or time.sleep)(cfg.retry_s)
    return verdict


class ClockDomainMismatch(RuntimeError):
    """The run-level check failed: the run must not start."""


def require_clock_domains(redis_client: Any, models: Iterable[str], cfg: ClockDomainConfig, *,
                          sleep: Optional[Callable[[float], None]] = None) -> dict[str, dict]:
    """The run-level (pre-flight) check: every model's gateway stamps and controller
    window ends are in redis time, else :class:`ClockDomainMismatch` with the offsets."""
    verdicts = {m: check_clock_domains(redis_client, m, cfg, sleep=sleep) for m in models}
    bad = {m: v["reasons"] for m, v in verdicts.items() if not v["ok"]}
    if bad:
        raise ClockDomainMismatch(
            "refusing to run: the capture's redis sources are not in redis's time domain "
            f"({json.dumps(bad, sort_keys=True)}); fix the node clocks (NTP), point --redis-url at the "
            "redis the gateway and the controller write to (a redis error above), or run with "
            "--no-capture-extras")
    return verdicts


def _doc_key(kind: str, pod: str, score: Any, doc: Any) -> str:
    """Identity of one gateway doc: ``kind|pod|score|sha1(canonical JSON)`` - the same for
    a doc read from redis (parsed) and for a row of a dump file."""
    canon = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}|{pod}|{int(float(score))}|{hashlib.sha1(canon.encode('utf-8')).hexdigest()[:16]}"


def _gateway_member_digests(redis_client: Any, pods: Sequence[str], lo_ms: int, hi_ms: int) -> set[str]:
    """:func:`_doc_key` of every hist / inst doc of ``pods`` with score in [lo, hi]."""
    out: set[str] = set()
    for pod in pods:
        for kind in GATEWAY_KINDS:
            key = hist_key(pod) if kind == "hist" else inst_key(pod)
            for member, score in redis_client.zrangebyscore(key, lo_ms, hi_ms, withscores=True) or ():
                out.add(_doc_key(kind, pod, score, _parse_member(member)))
    return out


def _late_docs(fetched: Mapping[str, Mapping[str, list]], known: set, known_upto_ms: int,
               known_pods: Iterable[str]) -> list[str]:
    """Docs of ``fetched`` stamped at or before ``known_upto_ms`` that ``known`` (the docs
    seen then, of ``known_pods``) lacks: written more than one round after their stamp, in
    redis time. Pods the earlier observation did not list are not judged (a pod whose
    first doc landed in between is not a late write)."""
    known_pods = set(known_pods)
    late = []
    for kind, per_pod in fetched.items():
        for pod, members in per_pod.items():
            if pod not in known_pods:
                continue
            for member, score in members:
                if int(float(score)) <= known_upto_ms:
                    k = _doc_key(kind, pod, score, _parse_member(member))
                    if k not in known:
                        late.append(k)
    return sorted(late)


def cell_clock_mark(
    redis_client: Any,
    model: str,
    cfg: ClockDomainConfig,
    *,
    start: Optional[Mapping[str, Any]] = None,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Redis TIME at a cell boundary plus the clock-domain check there. Call it right
    before the load (``start=None``) and right after it drains (``start`` = the start
    mark); redis TIME is read first (the late-write fingerprint right after it, the
    check - which waits for a gateway round and may retry - last), so the start mark is
    never after the load's first request, and the end mark never before its last
    response.

    Late writes: every mark fingerprints the gateway docs stamped in ``[redis_start -
    margin, its own redis TIME - blind spot]``. A doc that shows up there later was written
    more than one round (+ tol) after its stamp - a writer behind redis (e.g. a second
    gateway instance on a slow node) that the newest-stamp check cannot see. The end mark
    compares against the start mark, the capture's dump against the end mark
    (:func:`capture_after_cell`), each backfill against the previous dump file."""
    try:
        probe = clock_probe(redis_client, now_ms)
    except Exception as exc:  # noqa: BLE001
        probe = {"redis_time_ms": None, "error": repr(exc)}
    mark: dict[str, Any] = {"probe": probe}
    late_reasons: list[str] = []
    late_count: Optional[int] = None
    try:
        if start is not None:
            rng = start.get("late_write_range_ms")
            if not rng:
                raise ValueError("the start mark has no late-write baseline")
            if not start.get("late_write_pods"):
                raise ValueError("the start mark listed no gateway pods")
            late = (_gateway_member_digests(redis_client, start["late_write_pods"], *rng)
                    - set(start.get("late_write_baseline") or ()))
            late_count = len(late)
            if late:
                late_reasons.append(
                    f"gateway: {len(late)} doc(s) stamped in {rng} (more than one round before the cell "
                    f"started, in redis time) were written during the cell, e.g. {sorted(late)[0]}")
        rt = probe.get("redis_time_ms")
        if rt is not None:
            lo = int(start["late_write_range_ms"][0]) if start is not None else int(rt) - cfg.margin
            rng = [lo, int(rt) - cfg.blind_spot_ms]
            pods = active_pods(redis_client, model_pod_keys(redis_client, model), lo)
            mark["late_write_range_ms"] = rng
            mark["late_write_pods"] = pods
            mark["late_write_baseline"] = sorted(_gateway_member_digests(redis_client, pods, *rng))
    except Exception as exc:  # noqa: BLE001 - an unverifiable cell is a mismatch
        late_reasons.append(f"gateway: late-write check failed: {exc!r}")
    check = check_clock_domains(redis_client, model, cfg, sleep=sleep)
    mark["check"] = check
    check.setdefault("gateway", {})["late_writes"] = late_count
    if late_reasons:
        check["gateway"]["ok"] = False
        check["ok"] = False
        check.setdefault("reasons", []).extend(late_reasons)
    return mark


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
    the *write phase* / write delay (resolution: ``poll_s``; the clock-domain check,
    :func:`_check_once`, judges it). ``timeout_s <= 0`` skips the wait."""
    out: dict[str, Any] = {"timeout_s": timeout_s, "poll_s": poll_s, "waited_s": 0.0,
                           "new_round_seen": False}
    if not pods or timeout_s <= 0:
        out["reason"] = "no pods" if not pods else "wait disabled"
        return out
    start = monotonic()
    first = _latest_gateway_score(redis_client, pods)
    out["latest_round_before_ms"] = first
    while True:
        latest = _latest_gateway_score(redis_client, pods)
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


def gateway_instances(redis_client: Any) -> list[str]:
    """The live gateway plugin instances (``tre:v2:gw:instances``; audit)."""
    from tre_common.rediskeys import GW_INSTANCES_KEY

    try:
        return sorted(_text(m) for m in (redis_client.zrange(GW_INSTANCES_KEY, 0, -1) or ()))
    except Exception:  # noqa: BLE001
        return []


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
    known: Optional[set] = None,
    known_upto_ms: Optional[int] = None,
    known_pods: Optional[Iterable[str]] = None,
) -> dict:
    """``ZRANGEBYSCORE [lo_ms, hi_ms]`` of every pod's hist and inst key, one JSONL per
    pod and kind: ``{"score": round stamp ms, "doc": parsed JSON}``. Pods with no doc in
    the range get no file. ``observed_at_redis_ms`` is redis TIME just before the read.

    ``previous`` (the summary of an earlier dump of the same cell): every row of the
    existing files must still be in the new dump, else :class:`NotASuperset` is raised
    before anything is written - a backfill only ever widens a dump.

    ``known`` / ``known_upto_ms`` / ``known_pods``: the docs (:func:`_doc_key`) of
    ``known_pods`` already seen at an earlier observation and the stamp up to which that
    observation was final (its redis TIME minus the blind spot); ``late_writes`` counts
    the dumped docs stamped up to there that were not seen then (:func:`_late_docs`)."""
    observed_at = redis_time_ms(redis_client)
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
    late = (_late_docs(fetched, known, int(known_upto_ms), known_pods)
            if known is not None and known_upto_ms is not None and known_pods else None)
    return {
        "observed_at_redis_ms": observed_at,
        "late_writes": None if late is None else len(late),
        "late_write_example": late[0] if late else None,
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


def _config_from_clock(clock: Mapping[str, Any]) -> ClockDomainConfig:
    """The :class:`ClockDomainConfig` a cell was captured with (``cell_meta["clock"]``)."""
    c = clock.get("config") or {}
    kw = {k: c[k] for k in ("window_ms", "period_ms", "tolerance_ms", "controller_read_offset_ms",
                            "controller_tick_ms", "margin_ms", "attempts", "retry_s", "poll_s")
          if c.get(k) is not None}
    return ClockDomainConfig(**kw)


def _mark_summary(mark: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if mark is None:
        return None
    out = {k: v for k, v in mark.items() if k != "late_write_baseline"}
    if "late_write_baseline" in mark:
        out["late_write_baseline_docs"] = len(mark["late_write_baseline"] or ())
    return out


def clock_record(cfg: ClockDomainConfig, start: Optional[Mapping[str, Any]],
                 end: Optional[Mapping[str, Any]]) -> dict:
    """``cell_meta["clock"]``: the redis-time span of the cell, the dump range and tail
    derived from it, both marks (:func:`cell_clock_mark`) and the per-source verdict.

    The gateway is in redis's domain when both marks say so; the controller when both
    marks say so for it *and* for the gateway (its windows are built from the gateway's
    docs, so a skewed gateway makes them wrong whatever their stamps)."""
    rs = ((start or {}).get("probe") or {}).get("redis_time_ms")
    re_ = ((end or {}).get("probe") or {}).get("redis_time_ms")
    rec: dict[str, Any] = {"config": cfg.as_dict(), "cell_start": _mark_summary(start),
                           "cell_end": _mark_summary(end), "redis_start_ms": rs, "redis_end_ms": re_,
                           "range_ms": None, "tail_ms": None}
    reasons: list[str] = []
    if rs is None or re_ is None:
        reasons.append("no redis TIME at the cell " + ("start" if rs is None else "end"))
    else:
        lo, hi = dump_range_ms(rs, re_, cfg.margin)
        rec.update({"range_ms": [lo, hi], "tail_ms": tail_ms(re_, cfg.window_ms, cfg.blind_spot_ms, cfg.period_ms)})
    marks = [("cell start", start), ("cell end", end)]
    gw_ok = rs is not None and re_ is not None and all(
        m is not None and ((m.get("check") or {}).get("gateway") or {}).get("ok") for _, m in marks)
    ctl_ok = gw_ok and all(
        m is not None and ((m.get("check") or {}).get("controller") or {}).get("ok") for _, m in marks)
    for where, m in marks:
        reasons.extend(f"{where}: {r}" for r in (((m or {}).get("check") or {}).get("reasons") or ()))
    rec["domain"] = {"gateway": CLOCK_DOMAIN_OK if gw_ok else CLOCK_DOMAIN_MISMATCH,
                     "controller": CLOCK_DOMAIN_OK if ctl_ok else CLOCK_DOMAIN_MISMATCH}
    rec["mismatch_reasons"] = reasons if not (gw_ok and ctl_ok) else []
    return rec


def _gateway_late(clock: dict, where: str, summary: dict, *, cell_level: bool) -> None:
    """A gateway dump holding late-written docs (or one that could not be checked) is
    ``clock_domain_mismatch``, and so is the controller (its windows are built from those
    docs). ``cell_level`` (the capture right after the cell) also sets the cell's verdict
    ``clock.domain``; a backfill only marks the dumps it touches (the per-dump
    ``clock_domain`` is authoritative) and adds its reason."""
    reason = (f"{where}: gateway docs not verifiable against the previous observation"
              if summary.get("late_writes") is None else
              f"{where}: {summary['late_writes']} gateway doc(s) written more than one round after their "
              f"stamp, e.g. {summary.get('late_write_example')}")
    summary["clock_domain"] = CLOCK_DOMAIN_MISMATCH
    if cell_level:
        clock.setdefault("domain", {})["gateway"] = CLOCK_DOMAIN_MISMATCH
        clock["domain"]["controller"] = CLOCK_DOMAIN_MISMATCH
    clock.setdefault("mismatch_reasons", []).append(reason)


def _mark_completeness(meta: dict) -> bool:
    """Set ``tail_ms`` / ``reached_tail`` / ``complete`` on the redis dumps of ``meta``;
    True when a dump has not reached its tail yet (a backfill is due). ``complete`` also
    needs the dump's ``clock_domain`` to be ``ok`` (a ``clock_domain_mismatch`` dump is
    kept and backfilled like any other, but never complete) and, for the gateway docs,
    the head of the range to have been read before the gateway's 30 min retention could
    trim it (``head_within_retention``)."""
    tail = (meta.get("clock") or {}).get("tail_ms")
    pending = False
    for name, last_key in (("gateway_redis_dump", "last_round_ms"), ("controller_ticks", "last_window_end_ms")):
        dump = meta.get(name)
        if dump is None:
            continue
        reached = tail is not None and dump.get(last_key) is not None and dump[last_key] >= tail
        dump["tail_ms"] = tail
        dump["reached_tail"] = reached
        dump["complete"] = (reached and dump.get("clock_domain") == CLOCK_DOMAIN_OK
                            and (name != "gateway_redis_dump" or dump.get("head_within_retention") is True))
        pending |= tail is not None and not reached
    mismatched = sorted(name for name in ("gateway_redis_dump", "controller_ticks")
                        if meta.get(name) is not None and meta[name].get("clock_domain") != CLOCK_DOMAIN_OK)
    if mismatched:
        meta["clock_domain_mismatch"] = {"dumps": mismatched,
                                         "reasons": (meta.get("clock") or {}).get("mismatch_reasons") or []}
    else:
        meta.pop("clock_domain_mismatch", None)
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
    clock_start: Optional[Mapping[str, Any]] = None,
    clock_end: Optional[Mapping[str, Any]] = None,
    clock_config: Optional[ClockDomainConfig] = None,
    flush_wait_s: float = DEFAULT_CAPTURE_FLUSH_WAIT_S,
    poll_s: float = DEFAULT_GATEWAY_POLL_S,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    info: Optional[Mapping[str, Any]] = None,
) -> dict:
    """After a cell (its load drained): dump the gateway docs and the controller ticks
    over the cell's redis-time range, write ``cell_meta.json``.

    Clocks: ``start_ms`` / ``end_ms`` are the driver's clock and are recorded only. The
    dump range is ``[redis_start - margin, redis_end + margin]`` from ``clock_start`` /
    ``clock_end`` (:func:`cell_clock_mark` right before and right after the load; the end
    mark is taken here when not given). Without a start mark the cell has no redis time and
    nothing is dumped from redis. A source that failed the clock-domain check at either
    mark gets ``clock_domain: clock_domain_mismatch`` on its dump - dumped unshifted,
    never ``complete`` - and ``cell_meta["clock_domain_mismatch"]`` says why.

    Right after the cell the controller has not yet processed the tail windows, so the
    dumps are usually short of ``tail_ms``: the cell directory gets a ``BACKFILL_PENDING``
    marker and :func:`backfill_pending` - run by the next cell's driver and by the
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
        cfg = clock_config or ClockDomainConfig(window_ms=int(window_ms))
        if clock_end is None and clock_start is not None:
            clock_end = cell_clock_mark(redis_client, model, cfg, start=clock_start, now_ms=now_ms, sleep=sleep)
        clock = clock_record(cfg, clock_start, clock_end)
        meta["clock"] = clock
        if clock["range_ms"] is None:
            errors.append("redis clock: the cell has no redis-time span (" + "; ".join(clock["mismatch_reasons"])
                          + "): gateway docs and controller ticks not captured")
        else:
            lo, hi = clock["range_ms"]
            if gateway_dump:
                try:
                    wait = wait_for_gateway_write(redis_client,
                                                  active_pods(redis_client, model_pod_keys(redis_client, model), lo),
                                                  timeout_s=flush_wait_s, poll_s=poll_s,
                                                  sleep=sleep, monotonic=monotonic)
                    in_set = model_pod_keys(redis_client, model)  # after the wait: a pod that just wrote too
                    pods = active_pods(redis_client, in_set, lo)
                    end_rng = (clock_end or {}).get("late_write_range_ms")
                    summary = dump_gateway_docs(
                        redis_client, layout, pods, lo_ms=lo, hi_ms=hi,
                        known=set((clock_end or {}).get("late_write_baseline") or ()) if end_rng else None,
                        known_upto_ms=end_rng[1] if end_rng else None,
                        known_pods=(clock_end or {}).get("late_write_pods") or ())
                    summary["flush_wait"] = wait
                    summary["pods_in_set"] = len(in_set)
                    summary["gateway_instances"] = gateway_instances(redis_client)
                    seen = summary.get("observed_at_redis_ms")
                    summary["head_within_retention"] = (
                        seen is not None and seen + cfg.blind_spot_ms < lo + GATEWAY_RETENTION_MS)
                    summary["clock_domain"] = clock["domain"]["gateway"]
                    if not end_rng or summary["late_writes"]:
                        _gateway_late(clock, "capture dump", summary, cell_level=True)
                    meta["gateway_redis_dump"] = summary
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"gateway_redis_dump: {exc!r}")
            if controller_ticks:
                try:
                    summary = dump_controller_ticks(redis_client, layout, model, lo_ms=lo, hi_ms=hi)
                    summary["clock_domain"] = clock["domain"]["controller"]
                    meta["controller_ticks"] = summary
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"controller_ticks: {exc!r}")
    elif gateway_dump or controller_ticks:
        errors.append("no redis client: gateway docs and controller ticks not captured")
    meta["errors"] = errors
    try:
        _write_meta_and_marker(layout, meta)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cell_meta: {exc!r}")
    return meta


def backfill_cell(
    cell_dir: Path,
    redis_client: Any,
    *,
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    sleep: Callable[[float], None] = time.sleep,
    checks: Optional[dict] = None,
) -> Optional[dict]:
    """Re-dump the redis dumps of one cell that have not reached their tail, over the
    cell's recorded redis-time range; returns the backfill record (also appended to
    ``cell_meta.json["backfills"]``), None when nothing was due.

    The gateway pods are listed again (every pod of the model still writing in the
    range, plus the first dump's), so a pod that came up in the tail is included. The
    clock-domain check is run again first (``checks`` caches it per model): the rows added
    now were written since the cell, so a failed check marks the re-dumped source
    ``clock_domain_mismatch`` (per dump; ``clock.domain`` stays the verdict at the cell),
    as does a late-written gateway doc against the previous dump file. A re-dump that
    would lose a row of the existing file (retention) keeps the old file and says so."""
    cell_dir = Path(cell_dir)
    meta = json.loads((cell_dir / CELL_META).read_text(encoding="utf-8"))
    layout = CellLayout(cell_dir.parent.parent, meta["stem"], meta["cell_id"])
    _mark_completeness(meta)
    gd, ct = meta.get("gateway_redis_dump"), meta.get("controller_ticks")
    todo_gw = gd is not None and not gd.get("reached_tail")
    todo_ct = ct is not None and not ct.get("reached_tail")
    clock = meta.get("clock") or {}
    if not (todo_gw or todo_ct) or clock.get("range_ms") is None:
        _write_meta_and_marker(layout, meta)
        return None
    lo, hi = clock["range_ms"]
    model = meta["model"]
    cache = checks if checks is not None else {}
    if model not in cache:
        cache[model] = check_clock_domains(redis_client, model, _config_from_clock(clock), sleep=sleep)
    check = cache[model]
    record: dict[str, Any] = {"at_utc": utc_iso(), "redis_clock": clock_probe(redis_client, now_ms),
                              "clock_check": check}
    gw_ok = bool((check.get("gateway") or {}).get("ok"))
    ctl_ok = gw_ok and bool((check.get("controller") or {}).get("ok"))
    if not (gw_ok and ctl_ok):
        clock.setdefault("mismatch_reasons", []).extend(f"backfill: {r}" for r in check.get("reasons") or ())
    if todo_gw:
        try:
            pods = sorted(set(gd.get("pods") or ()) | set(active_pods(redis_client, model_pod_keys(redis_client, model), lo)))
            seen_at = gd.get("observed_at_redis_ms")
            new = dump_gateway_docs(
                redis_client, layout, pods, lo_ms=lo, hi_ms=hi, previous=gd,
                known=_dumped_doc_keys(layout, gd) if seen_at is not None else None,
                known_upto_ms=(int(seen_at) - _config_from_clock(clock).blind_spot_ms) if seen_at is not None else None,
                known_pods=gd.get("pods") or ())
            for key in ("flush_wait", "pods_in_set", "gateway_instances", "head_within_retention"):
                if key in gd:
                    new[key] = gd[key]
            new["clock_domain"] = gd.get("clock_domain") if gw_ok else CLOCK_DOMAIN_MISMATCH
            if new["late_writes"] is None or new["late_writes"]:
                _gateway_late(clock, "backfill", new, cell_level=False)
                ctl_ok = False
                if ct is not None and not todo_ct:
                    ct["clock_domain"] = CLOCK_DOMAIN_MISMATCH
            record["gateway_docs"] = {"docs_before": sum(sum(v.values()) for v in (gd.get("docs") or {}).values()),
                                      "docs_after": sum(sum(v.values()) for v in new["docs"].values()),
                                      "pods_before": len(gd.get("pods") or ()), "pods_after": len(pods),
                                      "last_round_before_ms": gd.get("last_round_ms"),
                                      "last_round_after_ms": new.get("last_round_ms")}
            meta["gateway_redis_dump"] = new
        except NotASuperset as exc:
            record["gateway_docs_kept"] = str(exc)
    if todo_ct:
        try:
            new = dump_controller_ticks(redis_client, layout, model, lo_ms=lo, hi_ms=hi, previous=ct)
            new["clock_domain"] = ct.get("clock_domain") if ctl_ok else CLOCK_DOMAIN_MISMATCH
            record["controller_ticks"] = {"members_before": ct.get("members"), "members_after": new["members"]}
            meta["controller_ticks"] = new
        except NotASuperset as exc:
            record["controller_ticks_kept"] = str(exc)
    meta.setdefault("backfills", []).append(record)
    _write_meta_and_marker(layout, meta)
    record["complete"] = all((meta.get(n) or {}).get("complete", True)
                             for n in ("gateway_redis_dump", "controller_ticks"))
    record["pending"] = (layout.cell_dir / BACKFILL_MARKER).exists()
    return record


def _dumped_doc_keys(layout: CellLayout, dump: Mapping[str, Any]) -> set[str]:
    """:func:`_doc_key` of every row of a gateway dump's files."""
    out: set[str] = set()
    for kind, per_pod in (dump.get("files") or {}).items():
        for pod, rel in per_pod.items():
            with (layout.cell_dir / rel).open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("kind") != "header" and "score" in row:
                        out.add(_doc_key(kind, pod, row["score"], row.get("doc")))
    return out


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
    (bounded) until redis time is past the latest pending tail plus the controller's
    largest normal lag, so the controller has processed the last windows - what the
    campaign's finalize does after its last cell. Returns one record per backfilled cell
    (``{"cell", "error"}`` on a failure, which never stops the others)."""
    cells = pending_cells(root, max_age_s=max_age_s)
    if not cells:
        return []
    if wait_s > 0:
        targets = []
        for cell in cells:
            try:
                clock = json.loads((cell / CELL_META).read_text(encoding="utf-8")).get("clock") or {}
                if clock.get("tail_ms") is not None:
                    targets.append(int(clock["tail_ms"]) + _config_from_clock(clock).controller_lag_bounds[1])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        target = max(targets) if targets else None
        deadline = monotonic() + wait_s
        while target is not None:
            server = redis_time_ms(redis_client)
            left = deadline - monotonic()
            if server is None or server >= target or left <= 0:
                break
            sleep(max(0.05, min((target - server) / 1000.0, left, 5.0)))
    out = []
    checks: dict = {}
    for cell in cells:
        try:
            rec = backfill_cell(cell, redis_client, now_ms=now_ms, sleep=sleep, checks=checks)
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
