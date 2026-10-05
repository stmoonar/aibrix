"""Where a tick's snapshot comes from.

* **Which pods are serving** - the service manager's ``GET /v2/state`` is the truth for
  "awake and not hidden" (``bindings[].awake`` / ``.hidden``; ``serve_id`` is the pod
  name, and ``models[m].awake`` the count). The SM also flips the pod label
  ``tre.aibrix.io/routable`` that the gateway routes on, but ``/v2/state`` carries no pod
  IP, so the address comes from a read-only pod list of the model namespace filtered on
  that label. A binding the SM calls awake whose pod is missing, not Running or not
  labelled routable is reported in ``ModelSnapshot.unscraped``.
* **Engine metrics** - each serving pod's ``/metrics``, scraped directly and in parallel
  with a short timeout. vLLM listens on the pod loopback only (``--host 127.0.0.1``); the
  reissue sidecar on the pod port (label ``model.aibrix.ai/port``) proxies ``/metrics``
  to it. A failed scrape leaves the pod out and is counted.
* **Request events** - ``XREAD`` of ``tre:v2:bl:req:<model>`` from a cursor kept across
  ticks; the first tick starts at the Redis clock "now" (the ``$`` semantics, which also
  works for a stream that does not exist yet). The reader keeps, per model, since when
  the stream has been read without a gap (``events_since_ms``: reset at start, when the
  stream was trimmed past the cursor or recreated, and withheld for a tick that left a
  backlog) and which requests are in flight on which pod (``arr`` without ``done``,
  :class:`InflightTracker`).
* **Event coverage** - after a gap (shell start, trimmed / recreated stream) every pod is
  unverified: requests already running there (the cohort) never had an ``arr`` this shell
  read. A pod becomes verified at the first scrape whose running + waiting is covered by
  the requests the stream shows there (the cohort has drained; state, not a timer) and
  stays verified until the next gap (``PodSnapshot.events_cover``). A pod whose engine
  reports nothing in flight is verified at once and its tracked requests that arrived
  before the scrape are dropped (their ``done`` was lost).
* **Clock** - Redis ``TIME``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import ssl
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.request import Request, urlopen

from tre_common.vllm_metrics import VLLM_METRICS

from tre_baselines.keys import REPLAY_T0_KEY, req_stream_key
from tre_baselines.snapshot import (
    ClusterSnapshot,
    ModelSnapshot,
    PodSnapshot,
    ReplayInfo,
    RequestEvent,
)

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------ metric name table

_HIST = ("_sum", "_count")

#: Standardized counter key -> (vLLM family candidates newest first, sample suffix).
#: Latency histograms reuse ``tre_common.vllm_metrics.VLLM_METRICS`` (old/new names);
#: the token and request counters are not in that table (nothing else in TRE reads them).
#: Verified against a live vLLM 0.30 fork pod (tests/fixtures/vllm030_metrics.txt).
COUNTER_SOURCES: Mapping[str, tuple[tuple[str, ...], str]] = {
    "gen_tokens": (("vllm:generation_tokens_total",), ""),
    "prompt_tokens": (("vllm:prompt_tokens_total",), ""),
    "itl_sum": (VLLM_METRICS["inter_token_latency_seconds"], "_sum"),
    "itl_count": (VLLM_METRICS["inter_token_latency_seconds"], "_count"),
    "ttft_sum": (VLLM_METRICS["time_to_first_token_seconds"], "_sum"),
    "ttft_count": (VLLM_METRICS["time_to_first_token_seconds"], "_count"),
    "e2e_sum": (VLLM_METRICS["e2e_request_latency_seconds"], "_sum"),
    "e2e_count": (VLLM_METRICS["e2e_request_latency_seconds"], "_count"),
    # One sample per finished_reason (stop / length / abort / error / ...): summed.
    "req_success": (("vllm:request_success_total",), ""),
}
GAUGE_SOURCES: Mapping[str, tuple[str, ...]] = {
    "running": VLLM_METRICS["num_requests_running"],
    "waiting": VLLM_METRICS["num_requests_waiting"],
    "kv_usage": VLLM_METRICS["kv_cache_usage_perc"],
}
#: Info gauge whose labels carry ``num_gpu_blocks`` and ``block_size``.
CACHE_CONFIG_INFO = "vllm:cache_config_info"

# ------------------------------------------------------------------ Prometheus text

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: Mapping[str, str]
    value: float


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def parse_prometheus_text(text: str) -> list[Sample]:
    """Samples of a Prometheus text exposition (comments skipped; bad lines ignored)."""
    out: list[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels: dict[str, str] = {}
        if "{" in line:
            name, rest = line.split("{", 1)
            body, sep, tail = rest.rpartition("}")
            if not sep:
                continue
            labels = {k: _unescape(v) for k, v in _LABEL_RE.findall(body)}
        else:
            name, _, tail = line.partition(" ")
        parts = tail.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        out.append(Sample(name.strip(), labels, value))
    return out


def _by_name(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.name, []).append(sample)
    return grouped


def _first_present(grouped: Mapping[str, list[Sample]], names: Iterable[str], suffix: str = "") -> Optional[list[Sample]]:
    for name in names:
        found = grouped.get(name + suffix)
        if found:
            return found
    return None


def _finite_sum(samples: list[Sample]) -> Optional[float]:
    values = [s.value for s in samples if math.isfinite(s.value)]
    return sum(values) if values else None


def _int_label(samples: Optional[list[Sample]], key: str) -> Optional[int]:
    for sample in samples or ():
        raw = sample.labels.get(key)
        if raw is None or raw in {"None", ""}:
            continue
        try:
            return int(float(raw))
        except ValueError:
            continue
    return None


def pod_snapshot_from_metrics(
    text: str, *, pod: str, model: str, node: Optional[str], gpu_ids: Iterable[int], scraped_at_ms: int
) -> PodSnapshot:
    """Map one ``/metrics`` body onto a :class:`PodSnapshot`. Samples of several engines
    (data-parallel ``engine`` label) are summed; ``kv_usage`` is their mean."""
    grouped = _by_name(parse_prometheus_text(text))
    counters: dict[str, float] = {}
    for key, (names, suffix) in COUNTER_SOURCES.items():
        found = _first_present(grouped, names, suffix)
        value = _finite_sum(found) if found else None
        if value is not None:
            counters[key] = value
    running = _first_present(grouped, GAUGE_SOURCES["running"])
    waiting = _first_present(grouped, GAUGE_SOURCES["waiting"])
    kv = _first_present(grouped, GAUGE_SOURCES["kv_usage"])
    kv_values = [s.value for s in kv or () if math.isfinite(s.value)]
    info = grouped.get(CACHE_CONFIG_INFO)
    return PodSnapshot(
        pod=pod,
        model=model,
        node=node,
        gpu_ids=tuple(int(g) for g in gpu_ids),
        # A missing or non-finite gauge is unknown (None), never 0: HTTP 200 alone does
        # not prove the pod is idle.
        running=_finite_sum(running) if running else None,
        waiting=_finite_sum(waiting) if waiting else None,
        kv_usage=(sum(kv_values) / len(kv_values)) if kv_values else None,
        counters=counters,
        num_gpu_blocks=_int_label(info, "num_gpu_blocks"),
        block_size=_int_label(info, "block_size"),
        scraped_at_ms=int(scraped_at_ms),
    )


# ------------------------------------------------------------------ request events

EVENT_KINDS = ("arr", "ft", "done")


def entry_ts_ms(entry_id: str) -> int:
    """Millisecond part of a stream entry ID ``<ms>-<seq>`` (the Redis server clock)."""
    return int(str(entry_id).split("-", 1)[0])


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _positive(value: Optional[int]) -> Optional[int]:
    """in_tokens <= 0 means "unknown" (an empty or failed count), not a real prompt."""
    return value if value is not None and value > 0 else None


def _non_negative(value: Optional[int]) -> Optional[int]:
    """out_tokens < 0 (the gateway writes -1 when the response carried no usage) = unknown."""
    return value if value is not None and value >= 0 else None


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _opt_bool(value: Any) -> Optional[bool]:
    text = str(_decode(value)).strip().lower() if value is not None else ""
    return True if text in {"true", "1"} else False if text in {"false", "0"} else None


def _decode(value: Any) -> Any:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def parse_event(model: str, entry_id: Any, fields: Mapping[Any, Any]) -> Optional[RequestEvent]:
    """One stream entry -> :class:`RequestEvent`; None for an unknown ``kind``. Missing
    fields become None (``reissue``: "none")."""
    entry_id = str(_decode(entry_id))
    data = {str(_decode(k)): _decode(v) for k, v in (fields or {}).items()}
    kind = str(data.get("kind") or "").strip()
    if kind not in EVENT_KINDS:
        return None
    reissue = str(data.get("reissue") or "").strip().lower() or "none"
    return RequestEvent(
        kind=kind,  # type: ignore[arg-type]
        model=model,
        pod=_opt_str(data.get("pod")),
        req_id=str(data.get("req_id") or ""),
        ts_ms=entry_ts_ms(entry_id),
        in_tokens=_positive(_opt_int(data.get("in_tokens"))),
        in_src=_opt_str(data.get("in_src")),
        max_tokens=_opt_int(data.get("max_tokens")),
        out_tokens=_non_negative(_opt_int(data.get("out_tokens"))),
        status=_opt_str(data.get("status")),
        reissue=reissue,
        entry_id=entry_id,
        stream=_opt_bool(data.get("stream")),
    )


class InflightTracker:
    """Requests per pod according to the event stream: ``arr`` (which names the target pod)
    adds a request, ``done`` (exactly once per request that had ``arr``) stamps its end.

    ``count(pod, since_ms)`` = requests that may have been in the engine at any time from
    ``since_ms`` on (no ``done`` yet, or ``done`` at/after ``since_ms``): compared with a
    scrape that started at ``since_ms`` it over- rather than under-counts, so a request
    finishing during the scrape never looks like an unknown one. Ended requests are pruned
    once they are older than the scrape that is compared (:meth:`prune`).
    """

    def __init__(self) -> None:
        self._where: dict[str, str] = {}
        #: pod -> req_id -> [arr ts, done ts or None]
        self._pods: dict[str, dict[str, list]] = {}

    def apply(self, event: RequestEvent) -> None:
        if event.kind == "arr" and event.pod:
            old = self._where.get(event.req_id)
            if old is not None:
                self._pods.get(old, {}).pop(event.req_id, None)
            self._where[event.req_id] = event.pod
            self._pods.setdefault(event.pod, {})[event.req_id] = [int(event.ts_ms), None]
        elif event.kind == "done":
            pod = self._where.pop(event.req_id, None)
            entry = self._pods.get(pod, {}).get(event.req_id) if pod is not None else None
            if entry is not None:
                entry[1] = int(event.ts_ms)

    def count(self, pod: str, since_ms: int) -> int:
        return sum(1 for _arr, done in self._pods.get(pod, {}).values() if done is None or done >= since_ms)

    def prune(self, before_ms: int) -> None:
        for reqs in self._pods.values():
            for req_id in [r for r, (_a, done) in reqs.items() if done is not None and done < before_ms]:
                del reqs[req_id]

    def engine_idle(self, pod: str, at_ms: int) -> int:
        """The engine of ``pod`` reported nothing running or waiting in a scrape that
        started at ``at_ms``: every request tracked there that arrived by then is over
        (its ``done`` was lost). Returns how many were dropped."""
        reqs = self._pods.get(pod)
        if not reqs:
            return 0
        gone = [r for r, (arr, done) in reqs.items() if done is None and arr <= at_ms]
        for req_id in gone:
            reqs.pop(req_id, None)
            self._where.pop(req_id, None)
        return len(gone)


def _id_tuple(entry_id: str) -> tuple[int, int]:
    ms, _, seq = str(entry_id).partition("-")
    return int(ms), int(seq or 0)


class EventReader:
    """XREAD of every model's stream from cursors kept across ticks, with gap detection
    and per-pod in-flight tracking (see the module docstring)."""

    def __init__(self, redis: Any, models: Iterable[str], *, batch: int = 1000, max_rounds: int = 20) -> None:
        self._redis = redis
        self._models = list(models)
        self._batch = int(batch)
        self._max_rounds = int(max_rounds)
        self.cursors: dict[str, str] = {}
        self.parse_errors = 0
        self.events_total = 0
        self.last_event_ms: dict[str, int] = {}
        #: Redis ms since which each model's stream has been read without a known gap.
        self.since_ms: dict[str, int] = {}
        #: Models whose last read left entries unread (the snapshot withholds history).
        self.behind: set[str] = set()
        #: Detected gaps per model (trimmed / recreated stream).
        self.gaps: dict[str, int] = {m: 0 for m in self._models}
        self._stream_seen: dict[str, bool] = {}
        self.inflight: dict[str, InflightTracker] = {m: InflightTracker() for m in self._models}
        #: Pods whose in-flight requests the stream covers since the model's last gap.
        self.verified: dict[str, set[str]] = {m: set() for m in self._models}

    def _gap(self, model: str, now_ms: int) -> None:
        self.since_ms[model] = int(now_ms)
        self.verified[model].clear()
        # Pre-gap entries may have lost their done in the gap: counting them would let a
        # pod look covered while the new cohort still runs. Only post-gap arrivals count.
        self.inflight[model] = InflightTracker()
        self.gaps[model] += 1

    def start(self, now_ms: int) -> None:
        for model in self._models:
            if model not in self.cursors:
                self.cursors[model] = f"{int(now_ms)}-0"
                self.since_ms[model] = int(now_ms)

    def history_since_ms(self, model: str) -> Optional[int]:
        """Since when the stream is gap-free; None for a tick that left a backlog."""
        return None if model in self.behind else self.since_ms.get(model)

    def _check_trimmed(self, model: str, now_ms: int) -> None:
        """A stream we saw before whose first entry is now past our cursor was trimmed
        (more than MAXLEN entries since the last read) or deleted and recreated: the
        events in between are lost, so the history restarts now."""
        try:
            first = self._redis.xrange(req_stream_key(model), "-", "+", count=1) or []
        except Exception as exc:  # unknown -> treat as a gap (fail closed)
            LOG.warning("xrange of %s failed: %s", req_stream_key(model), exc)
            self._stream_seen[model] = False
            self._gap(model, now_ms)
            return
        seen_before = self._stream_seen.get(model, False)
        exists = bool(first)
        self._stream_seen[model] = exists
        if not exists:
            if seen_before:  # the stream vanished: whatever was in it is gone
                self._gap(model, now_ms)
            return
        first_id = str(_decode(first[0][0]))
        if seen_before and _id_tuple(first_id) > _id_tuple(self.cursors[model]):
            LOG.warning("event stream %s trimmed past the cursor (%s > %s): history restarts",
                        req_stream_key(model), first_id, self.cursors[model])
            self._gap(model, now_ms)

    def read(self, now_ms: int) -> dict[str, tuple[RequestEvent, ...]]:
        self.start(now_ms)
        for model in self._models:
            self._check_trimmed(model, now_ms)
        out: dict[str, list[RequestEvent]] = {model: [] for model in self._models}
        key_to_model = {req_stream_key(m): m for m in self._models}
        pending = set(self._models)
        for _ in range(self._max_rounds):
            if not pending:
                break
            streams = {req_stream_key(m): self.cursors[m] for m in sorted(pending)}
            reply = self._redis.xread(streams, count=self._batch) or []
            full: set[str] = set()
            for key, entries in reply:
                model = key_to_model.get(str(_decode(key)))
                if model is None:
                    continue
                for entry_id, fields in entries:
                    entry_id = str(_decode(entry_id))
                    self.cursors[model] = entry_id
                    try:
                        event = parse_event(model, entry_id, fields)
                    except (TypeError, ValueError):
                        event = None
                    if event is None:
                        self.parse_errors += 1
                        continue
                    out[model].append(event)
                    self.inflight[model].apply(event)
                    self.events_total += 1
                    self.last_event_ms[model] = event.ts_ms
                if len(entries) >= self._batch:
                    full.add(model)
            pending = full
        self.behind = set(pending)
        return {model: tuple(events) for model, events in out.items()}


def read_replay_info(redis: Any) -> Optional[ReplayInfo]:
    raw = redis.get(REPLAY_T0_KEY)
    if raw is None:
        return None
    try:
        doc = json.loads(_decode(raw))
        seed = doc.get("seed")
        return ReplayInfo(t0_ms=int(doc["t0_ms"]), trace_path=str(doc.get("trace_path") or ""),
                          seed=None if seed is None or isinstance(seed, bool) else int(seed))
    except (ValueError, KeyError, TypeError):
        LOG.warning("malformed %s: %r", REPLAY_T0_KEY, raw)
        return None


def redis_now_ms(redis: Any) -> int:
    seconds, micros = redis.time()
    return int(seconds) * 1000 + int(micros) // 1000


# ------------------------------------------------------------------ pod discovery

ROUTABLE_LABEL = "tre.aibrix.io/routable"
MODEL_LABEL = "model.aibrix.ai/name"
PORT_LABEL = "model.aibrix.ai/port"
DEFAULT_POD_PORT = 8000

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


@dataclass(frozen=True)
class PodEndpoint:
    name: str
    model: Optional[str]
    ip: Optional[str]
    port: int
    ready: bool


def endpoints_from_pod_list(doc: Mapping[str, Any], *, port_override: Optional[int] = None) -> list[PodEndpoint]:
    out: list[PodEndpoint] = []
    for item in doc.get("items") or ():
        meta = item.get("metadata") or {}
        labels = meta.get("labels") or {}
        status = item.get("status") or {}
        port = port_override
        if port is None:
            try:
                port = int(labels.get(PORT_LABEL) or DEFAULT_POD_PORT)
            except ValueError:
                port = DEFAULT_POD_PORT
        ready = (
            status.get("phase") == "Running"
            and bool(status.get("podIP"))
            and not meta.get("deletionTimestamp")
            and str(labels.get(ROUTABLE_LABEL, "")).lower() == "true"
        )
        out.append(PodEndpoint(
            name=str(meta.get("name") or ""), model=labels.get(MODEL_LABEL),
            ip=status.get("podIP"), port=int(port), ready=ready,
        ))
    return out


class K8sPodLister:
    """Read-only in-cluster pod list (stdlib; ServiceAccount token re-read per call)."""

    def __init__(self, namespace: str, *, port_override: Optional[int] = None, timeout_s: float = 5.0,
                 api_base: Optional[str] = None, token_path: str = f"{_SA_DIR}/token",
                 ca_path: str = f"{_SA_DIR}/ca.crt") -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self._base = (api_base or f"https://{host}:{port}").rstrip("/")
        self._namespace = namespace
        self._port_override = port_override
        self._timeout_s = timeout_s
        self._token_path = token_path
        self._ctx = ssl.create_default_context(cafile=ca_path) if os.path.exists(ca_path) else None

    def list_routable(self) -> list[PodEndpoint]:
        query = urllib.parse.urlencode({"labelSelector": f"{ROUTABLE_LABEL}=true"})
        url = f"{self._base}/api/v1/namespaces/{self._namespace}/pods?{query}"
        headers = {"accept": "application/json"}
        if os.path.exists(self._token_path):
            with open(self._token_path, encoding="utf-8") as fh:
                headers["authorization"] = f"Bearer {fh.read().strip()}"
        with urlopen(Request(url, headers=headers), timeout=self._timeout_s, context=self._ctx) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        return endpoints_from_pod_list(doc, port_override=self._port_override)


def http_get_text(url: str, timeout_s: float) -> str:
    with urlopen(Request(url, headers={"accept": "text/plain"}), timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", "replace")


# ------------------------------------------------------------------ the live source


@dataclass(frozen=True)
class ServingBinding:
    pod: str
    model: str
    node: Optional[str]
    gpu_ids: tuple[int, ...]


def serving_bindings(state: Mapping[str, Any], models: Iterable[str]) -> dict[str, list[ServingBinding]]:
    """Awake, not hidden bindings of ``state`` (``GET /v2/state``) per model, by pod name."""
    wanted = set(models)
    out: dict[str, list[ServingBinding]] = {m: [] for m in wanted}
    for binding in state.get("bindings") or ():
        model = binding.get("model")
        if model not in wanted or not binding.get("awake") or binding.get("hidden"):
            continue
        pod = binding.get("serve_id")
        if not pod:
            continue
        gpu_ids = tuple(int(g) for g in (binding.get("gpu_ids") or ()))
        out[model].append(ServingBinding(str(pod), model, binding.get("node"), gpu_ids))
    for model in out:
        out[model].sort(key=lambda b: b.pod)
    return out


def wakeable_slots(state: Mapping[str, Any], model: str) -> Optional[int]:
    """How many sleeping bindings of ``model`` could be woken now without a donor: every
    GPU of the binding is ``wakeable`` in ``/v2/state`` ``gpus[]`` (no awake binding, no
    sleep draining, no load or wake in flight, no unexplained memory use), counted
    GPU-disjointly. None when the state has no ``gpus[]``."""
    gpus = state.get("gpus")
    if not isinstance(gpus, list):
        return None
    free = {(g.get("node"), int(g.get("gpu"))) for g in gpus
            if isinstance(g, Mapping) and g.get("wakeable") and g.get("gpu") is not None}
    used: set = set()
    count = 0
    for binding in state.get("bindings") or ():
        if binding.get("model") != model or binding.get("awake"):
            continue
        slot = {(binding.get("node"), int(g)) for g in (binding.get("gpu_ids") or ())}
        if slot and slot <= free and not slot & used:
            used |= slot
            count += 1
    return count


def awake_counts(state: Mapping[str, Any], models: Iterable[str]) -> dict[str, int]:
    counts = state.get("models") or {}
    out: dict[str, int] = {}
    for model in models:
        entry = counts.get(model)
        if not isinstance(entry, dict):
            raise ValueError(f"/v2/state has no counts for {model}")
        out[model] = int(entry.get("awake", 0))
    return out


class LiveSource:
    """Builds one :class:`ClusterSnapshot` per tick from SM + pods + Redis."""

    def __init__(
        self,
        config: Any,
        redis: Any,
        get_state: Callable[[], Mapping[str, Any]],
        list_pods: Callable[[], list[PodEndpoint]],
        *,
        fetch_text: Callable[[str, float], str] = http_get_text,
        max_scrape_workers: int = 16,
    ) -> None:
        self._config = config
        self._redis = redis
        self._get_state = get_state
        self._list_pods = list_pods
        self._fetch = fetch_text
        self._models = sorted(config.models)
        self.events = EventReader(redis, self._models)
        self._pool = ThreadPoolExecutor(max_workers=max_scrape_workers, thread_name_prefix="bl-scrape")
        self.scrape_failures = 0
        self.scrape_total = 0

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _scrape(self, target: tuple[ServingBinding, PodEndpoint], now_ms: int) -> Optional[PodSnapshot]:
        binding, endpoint = target
        url = f"http://{endpoint.ip}:{endpoint.port}/metrics"
        try:
            text = self._fetch(url, float(self._config.scrape_timeout_s))
            return pod_snapshot_from_metrics(
                text, pod=binding.pod, model=binding.model, node=binding.node,
                gpu_ids=binding.gpu_ids, scraped_at_ms=now_ms,
            )
        except Exception as exc:  # any failure = pod left out and counted
            LOG.warning("scrape of %s failed: %s", binding.pod, exc)
            return None

    def gather(self, tick: int = 0) -> ClusterSnapshot:
        now_ms = redis_now_ms(self._redis)
        state = self._get_state()
        awake = awake_counts(state, self._models)
        serving = serving_bindings(state, self._models)
        endpoints = {ep.name: ep for ep in self._list_pods()}

        targets: list[tuple[ServingBinding, PodEndpoint]] = []
        unscraped: dict[str, list[str]] = {m: [] for m in self._models}
        for model in self._models:
            for binding in serving[model]:
                ep = endpoints.get(binding.pod)
                if ep is None or not ep.ready or not ep.ip:
                    unscraped[model].append(binding.pod)
                else:
                    targets.append((binding, ep))
        results = list(self._pool.map(lambda t: self._scrape(t, now_ms), targets))
        failed = sum(len(v) for v in unscraped.values()) + sum(1 for r in results if r is None)
        self.scrape_total += len(targets)
        self.scrape_failures += failed

        # Events after the scrape: an arrival the scrape already shows is in the stream.
        events = self.events.read(now_ms)
        pods: dict[str, list[PodSnapshot]] = {m: [] for m in self._models}
        for (binding, _ep), snap in zip(targets, results):
            if snap is None:
                unscraped[binding.model].append(binding.pod)
                continue
            tracker = self.events.inflight[binding.model]
            verified = self.events.verified[binding.model]
            if snap.queued == 0:
                tracker.engine_idle(binding.pod, snap.scraped_at_ms)
            tracked = tracker.count(binding.pod, snap.scraped_at_ms)
            if snap.queued is not None and snap.queued <= tracked:
                verified.add(binding.pod)  # the cohort in flight at the last gap is gone
            pods[binding.model].append(replace(
                snap, tracked_inflight=tracked, events_cover=binding.pod in verified))
        for model in self._models:
            self.events.inflight[model].prune(now_ms)
        replay = read_replay_info(self._redis)
        lag = {
            m: max(0.0, (now_ms - self.events.last_event_ms[m]) / 1000.0)
            for m in self._models if m in self.events.last_event_ms
        }
        models: dict[str, ModelSnapshot] = {}
        for model in self._models:
            lim = self._config.models[model]
            models[model] = ModelSnapshot(
                model=model,
                awake=awake[model],
                min_replicas=lim.min_replicas,
                max_replicas=lim.max_replicas,
                gpus_per_replica=lim.gpus_per_replica,
                ttft_slo_ms=lim.ttft_slo_ms,
                tpot_slo_ms=lim.tpot_slo_ms,
                max_num_seqs=lim.max_num_seqs,
                pods=tuple(sorted(pods[model], key=lambda p: p.pod)),
                events=events.get(model, ()),
                unscraped=tuple(sorted(unscraped[model])),
                slo=lim.slo,
                events_since_ms=self.events.history_since_ms(model),
                wakeable_slots=wakeable_slots(state, model),
            )
        return ClusterSnapshot(
            now_ms=now_ms,
            tick_s=float(self._config.tick_s),
            models=models,
            replay=replay,
            tick=tick,
            extra={"event_lag_s": lag, "scrape_failed": failed, "sm_state_version": state.get("version")},
        )
