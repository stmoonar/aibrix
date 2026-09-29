"""SafeScale direct evidence (2026-09-29, plan B+D): the controller scrapes the vLLM
``/metrics`` of the probe's remaining pods itself instead of waiting for the gateway's
10 s Redis docs.

* **baseline** - right after the SM confirmed the hide (ActionQueue ``on_hide_done`` ->
  :meth:`DirectEvidenceCollector.on_hide_done`), every remaining pod (the model's awake
  pods minus the probe pods) is scraped once, concurrently; the cumulative TTFT /
  inter-token-latency histograms and the prompt-token sum / count are kept in the probe
  record (compact: only the bucket steps, see :func:`compact_hist`), so a restarted
  controller continues from it;
* **poll** - every ``evidence_poll_s`` (one SafeScale tick) the same pods are scraped
  again; the difference to the baseline is the evidence of ``[baseline, now]``: TTFT /
  TPOT p95 per pod (percentile mode and per-pod minimum samples of the metrics store),
  model p95 = max over the pods, ``n`` = TTFT count, mean prompt length ``L``, and the
  KV-cache fill = mean ``kv_cache_usage_perc`` of the pods of the latest scrape (the
  KV gate's aggregation, ``remaining_pods_kv_cache``);
* **failures** - a pod whose scrape fails is left out of that tick (recorded); a pod
  whose counters went backwards (restart) is dropped for good; a tick in which no
  remaining pod yields evidence switches the probe to the Redis evidence path
  (``redis_fallback``, sticky).

Scrapes run in a dedicated thread pool (stdlib ``urllib``, as the controller's other
HTTP reads; no new dependency) and are awaited with ``asyncio.wait_for``: the event loop
and the fast loop never block on them, and a tick waits at most ``scrape_timeout_s``.

Metric names (vLLM 0.30, verified on a live pod): ``vllm:time_to_first_token_seconds``,
``vllm:inter_token_latency_seconds`` (old name ``vllm:time_per_output_token_seconds``;
the gateway's TPOT is the same per-token histogram), ``vllm:request_prompt_tokens``,
``vllm:kv_cache_usage_perc`` (old ``vllm:gpu_cache_usage_perc``), all with an
``engine`` label (summed / averaged over engines), and ``vllm:engine_sleep_state``.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

from tre_common.percentile import histogram_percentile
from tre_common.vllm_metrics import vllm_candidates

LOG = logging.getLogger("tre_controller.safescale")

#: evidence_source_used values (audit / summary).
SOURCE_DIRECT = "direct"
SOURCE_REDIS_FALLBACK = "redis_fallback"
SOURCE_REDIS = "redis"

_TTFT = "time_to_first_token_seconds"
_TPOT = "inter_token_latency_seconds"
_PROMPT = "request_prompt_tokens"
_KV = "kv_cache_usage_perc"
_SLEEP_STATE = "vllm:engine_sleep_state"
#: Scrape timestamps kept in the probe record (audit), newest last.
MAX_SCRAPE_TS = 64

_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# ------------------------------------------------------------------ parsing
@dataclass(frozen=True)
class Hist:
    """A cumulative Prometheus histogram: ``count`` and ``((le, cumulative), ...)``
    sorted by le, ``+Inf`` included."""

    count: float
    buckets: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PodScrape:
    """What one ``/metrics`` read of a pod yields for the evidence."""

    ts_ms: int
    ttft: Hist | None = None
    tpot: Hist | None = None
    #: request_prompt_tokens (sum, count).
    prompt: tuple[float, float] | None = None
    kv_cache: float | None = None
    #: False when vllm:engine_sleep_state reports the engine not awake; None = unknown.
    awake: bool | None = None
    #: The raw TPOT family found (0.30: inter_token_latency_seconds).
    tpot_name: str | None = None


def parse_vllm_metrics(text: str, *, ts_ms: int) -> PodScrape:
    """The evidence families of a vLLM ``/metrics`` body (Prometheus text format).
    Label sets (engines) are summed; the KV gauge is averaged over engines."""
    wanted: dict[str, tuple[str, str]] = {}
    families: dict[str, tuple[str, ...]] = {}
    for canonical in (_TTFT, _TPOT, _PROMPT):
        families[canonical] = vllm_candidates(canonical)
        for raw in families[canonical]:
            for suffix in ("_bucket", "_count", "_sum"):
                wanted[raw + suffix] = (raw, suffix)
    kv_names = vllm_candidates(_KV)
    for raw in kv_names:
        wanted[raw] = (raw, "")
    wanted[_SLEEP_STATE] = (_SLEEP_STATE, "")

    buckets: dict[str, dict[float, float]] = {}
    counts: dict[str, float] = {}
    sums: dict[str, float] = {}
    gauges: dict[str, list[float]] = {}
    awake_values: list[float] = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        brace = line.find("{")
        space = line.find(" ")
        name_end = brace if 0 <= brace < space or (brace >= 0 and space < 0) else space
        if name_end <= 0:
            continue
        name = line[:name_end]
        target = wanted.get(name)
        if target is None:
            continue
        if brace == name_end:
            close = line.rfind("}")
            if close < 0:
                continue
            labels = dict(_LABEL_RE.findall(line[brace + 1 : close]))
            rest = line[close + 1 :]
        else:
            labels = {}
            rest = line[name_end:]
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        family, suffix = target
        if family == _SLEEP_STATE:
            if labels.get("sleep_state") == "awake":
                awake_values.append(value)
            continue
        if suffix == "_bucket":
            le_raw = labels.get("le")
            if le_raw is None:
                continue
            le = math.inf if le_raw in ("+Inf", "Inf", "inf") else float(le_raw)
            per = buckets.setdefault(family, {})
            per[le] = per.get(le, 0.0) + value
        elif suffix == "_count":
            counts[family] = counts.get(family, 0.0) + value
        elif suffix == "_sum":
            sums[family] = sums.get(family, 0.0) + value
        else:
            gauges.setdefault(family, []).append(value)

    def hist(canonical: str) -> tuple[Hist | None, str | None]:
        for raw in families[canonical]:  # newest name first
            if raw in counts and raw in buckets:
                return Hist(counts[raw], tuple(sorted(buckets[raw].items()))), raw
        return None, None

    ttft, _ = hist(_TTFT)
    tpot, tpot_name = hist(_TPOT)
    prompt = None
    for raw in families[_PROMPT]:
        if raw in counts and raw in sums:
            prompt = (sums[raw], counts[raw])
            break
    kv = None
    for raw in kv_names:
        values = gauges.get(raw)
        if values:
            kv = sum(values) / len(values)
            break
    awake = None if not awake_values else all(value >= 1.0 for value in awake_values)
    return PodScrape(ts_ms=int(ts_ms), ttft=ttft, tpot=tpot, prompt=prompt, kv_cache=kv, awake=awake,
                     tpot_name=tpot_name)


# ------------------------------------------------------------------ compact baseline
def compact_hist(hist: Hist) -> list[Any]:
    """``[count, [[le, cumulative], ...]]`` with only the finite les where the cumulative
    count changes (the step points): the step function - and so any delta against it -
    is unchanged (``cumulative(x)`` = the last step at or below x, 0 before the first,
    ``count`` at +Inf). Keeps the baseline small and JSON-safe (no Infinity)."""
    steps: list[list[float]] = []
    previous = 0.0
    for le, cumulative in hist.buckets:
        if math.isinf(le):
            continue
        if cumulative != previous:
            steps.append([le, cumulative])
            previous = cumulative
    return [hist.count, steps]


def expand_hist(raw: Any) -> Hist | None:
    try:
        count = float(raw[0])
        steps = [(float(le), float(cumulative)) for le, cumulative in raw[1]]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return Hist(count, tuple(sorted(steps)) + ((math.inf, count),))


def _cumulative_at(buckets: Iterable[tuple[float, float]], upper: float) -> float:
    """Cumulative count at ``upper`` of a (possibly compacted) cumulative histogram."""
    best = 0.0
    for le, cumulative in buckets:
        if le <= upper and cumulative > best:
            best = cumulative
    return best


def hist_delta(base: Hist, current: Hist) -> tuple[float, tuple[tuple[float, float], ...]] | None:
    """(count delta, cumulative bucket delta) of ``current`` since ``base``; None when a
    counter went backwards (the pod restarted: its baseline is void). Bucket delta rule =
    the metrics store's ``_bucket_delta`` (union of les, running max)."""
    if current.count < base.count:
        return None
    les = sorted({le for le, _ in base.buckets} | {le for le, _ in current.buckets})
    delta: list[tuple[float, float]] = []
    running = 0.0
    for le in les:
        now, then = _cumulative_at(current.buckets, le), _cumulative_at(base.buckets, le)
        if now < then:
            return None
        running = max(running, now - then)
        delta.append((le, running))
    return current.count - base.count, tuple(delta)


# ------------------------------------------------------------------ probe state
@dataclass(frozen=True)
class PodBaseline:
    ts_ms: int
    ttft: Hist | None
    tpot: Hist | None
    prompt: tuple[float, float] | None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"ts_ms": self.ts_ms}
        if self.ttft is not None:
            record["ttft"] = compact_hist(self.ttft)
        if self.tpot is not None:
            record["tpot"] = compact_hist(self.tpot)
        if self.prompt is not None:
            record["prompt"] = [self.prompt[0], self.prompt[1]]
        return record

    @classmethod
    def from_record(cls, raw: Any) -> "PodBaseline | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            ts_ms = int(float(raw["ts_ms"]))
        except (KeyError, TypeError, ValueError):
            return None
        prompt = None
        if isinstance(raw.get("prompt"), (list, tuple)) and len(raw["prompt"]) == 2:
            try:
                prompt = (float(raw["prompt"][0]), float(raw["prompt"][1]))
            except (TypeError, ValueError):
                prompt = None
        return cls(
            ts_ms=ts_ms,
            ttft=expand_hist(raw["ttft"]) if raw.get("ttft") is not None else None,
            tpot=expand_hist(raw["tpot"]) if raw.get("tpot") is not None else None,
            prompt=prompt,
        )

    @classmethod
    def from_scrape(cls, scrape: PodScrape) -> "PodBaseline":
        return cls(ts_ms=scrape.ts_ms, ttft=scrape.ttft, tpot=scrape.tpot, prompt=scrape.prompt)


@dataclass(frozen=True)
class DirectWindow:
    """The remaining pods' evidence over ``[start_ms, end_ms]`` (baseline -> poll)."""

    start_ms: int
    end_ms: int
    pods: tuple[str, ...]
    #: pod -> why it is not in this window (scrape failure this tick, counter reset, ...).
    excluded: dict[str, str]
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    ttft_count: float
    judged_count: float
    prompt_tokens: float | None
    prompt_count: float | None
    kv_cache: float | None
    #: pod -> {"n", "ttft_p95_ms", "tpot_p95_ms", "ts_ms"} (audit).
    per_pod: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def mean_prompt_tokens(self) -> float | None:
        if self.prompt_tokens is None or not self.prompt_count or self.prompt_count <= 0:
            return None
        return float(self.prompt_tokens) / float(self.prompt_count)

    @property
    def p95_available(self) -> bool:
        return self.ttft_p95_ms is not None or self.tpot_p95_ms is not None

    def audit(self) -> dict[str, Any]:
        return {
            "direct_window_start_ms": self.start_ms,
            "direct_window_end_ms": self.end_ms,
            "direct_pods": {pod: dict(values) for pod, values in sorted(self.per_pod.items())},
            "direct_excluded_pods": dict(sorted(self.excluded.items())),
        }


@dataclass(frozen=True)
class DirectState:
    """Per-probe direct-evidence state, persisted in the probe record
    (``direct_evidence``)."""

    #: pod -> baseline; None = not taken yet.
    baseline: dict[str, PodBaseline] | None = None
    baseline_ts_ms: int | None = None
    #: pod -> {"reason", "ts_ms"}: out of the evidence for good (baseline failed,
    #: counters went backwards).
    dropped: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Wall-clock ms of every poll (newest last, capped at MAX_SCRAPE_TS).
    scrapes: tuple[int, ...] = ()
    #: Sticky switch to the Redis path: {"reason", "ts_ms"}.
    fallback: dict[str, Any] | None = None
    #: The latest window's audit / verdict inputs (not needed to resume: recomputed).
    last: DirectWindow | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        if self.baseline is not None:
            record["baseline"] = {pod: base.as_record() for pod, base in sorted(self.baseline.items())}
            record["baseline_ts_ms"] = self.baseline_ts_ms
        if self.dropped:
            record["dropped"] = {pod: dict(value) for pod, value in sorted(self.dropped.items())}
        if self.scrapes:
            record["scrapes"] = list(self.scrapes)
        if self.fallback is not None:
            record["fallback"] = dict(self.fallback)
        return record

    @classmethod
    def from_record(cls, raw: Any) -> "DirectState | None":
        if not isinstance(raw, Mapping):
            return None
        baseline = None
        if isinstance(raw.get("baseline"), Mapping):
            baseline = {}
            for pod, value in raw["baseline"].items():
                parsed = PodBaseline.from_record(value)
                if parsed is not None:
                    baseline[str(pod)] = parsed
        try:
            baseline_ts = int(float(raw["baseline_ts_ms"])) if raw.get("baseline_ts_ms") is not None else None
        except (TypeError, ValueError):
            baseline_ts = None
        dropped = {str(pod): dict(value) for pod, value in (raw.get("dropped") or {}).items()
                   if isinstance(value, Mapping)}
        scrapes: tuple[int, ...] = ()
        if isinstance(raw.get("scrapes"), list):
            scrapes = tuple(int(float(ts)) for ts in raw["scrapes"] if isinstance(ts, (int, float)))
        fallback = dict(raw["fallback"]) if isinstance(raw.get("fallback"), Mapping) else None
        return cls(baseline=baseline, baseline_ts_ms=baseline_ts, dropped=dropped, scrapes=scrapes,
                   fallback=fallback)

    def live_pods(self) -> tuple[str, ...]:
        """Baseline pods still in the evidence (polled every tick)."""
        if not self.baseline:
            return ()
        return tuple(sorted(pod for pod in self.baseline if pod not in self.dropped))


@dataclass(frozen=True)
class DirectPoll:
    """One tick's scrape of a probe's baseline pods: pod -> PodScrape, or an error text."""

    request_id: str
    ts_ms: int
    results: Mapping[str, "PodScrape | str"]


def take_baseline(results: Mapping[str, "PodScrape | str"], *, ts_ms: int) -> DirectState:
    """The probe's direct state from its baseline scrape: pods that answered (awake,
    with a TTFT histogram) form the baseline, the others are dropped with the reason."""
    baseline: dict[str, PodBaseline] = {}
    dropped: dict[str, dict[str, Any]] = {}
    for pod, result in sorted(results.items()):
        reason = _unusable(result)
        if reason is not None:
            dropped[pod] = {"reason": f"baseline_{reason}", "ts_ms": int(ts_ms)}
            continue
        baseline[pod] = PodBaseline.from_scrape(result)  # type: ignore[arg-type]
    start = min((base.ts_ms for base in baseline.values()), default=int(ts_ms))
    return DirectState(baseline=baseline, baseline_ts_ms=start, dropped=dropped)


def _unusable(result: "PodScrape | str") -> str | None:
    if isinstance(result, str):
        return result or "scrape_failed"
    if result.awake is False:
        return "asleep"
    if result.ttft is None:
        return "no_ttft_histogram"
    return None


def evaluate_poll(
    state: DirectState,
    poll: DirectPoll,
    *,
    percentile_mode: str,
    min_latency_samples: int,
) -> tuple[DirectState, DirectWindow | None]:
    """Difference one poll against the baseline. Returns the updated state (new drops,
    scrape log) and the window, or None when no remaining pod yields evidence."""
    base = state.baseline or {}
    dropped = dict(state.dropped)
    excluded: dict[str, str] = {}
    per_pod: dict[str, dict[str, Any]] = {}
    ttft_p95: list[float] = []
    tpot_p95: list[float] = []
    n = judged = 0.0
    prompt_sum: list[float] = []
    prompt_count: list[float] = []
    kv_values: list[float] = []
    for pod in state.live_pods():
        result = poll.results.get(pod, "not_polled")
        reason = _unusable(result)
        if reason is not None:
            excluded[pod] = reason
            continue
        assert isinstance(result, PodScrape)
        pod_base = base[pod]
        ttft = hist_delta(pod_base.ttft, result.ttft) if pod_base.ttft is not None and result.ttft else None
        tpot = (
            hist_delta(pod_base.tpot, result.tpot)
            if pod_base.tpot is not None and result.tpot is not None
            else (0.0, ())
        )
        prompt_delta = None
        if pod_base.prompt is not None and result.prompt is not None:
            prompt_delta = (result.prompt[0] - pod_base.prompt[0], result.prompt[1] - pod_base.prompt[1])
        if ttft is None or tpot is None or (prompt_delta is not None and min(prompt_delta) < 0):
            dropped[pod] = {"reason": "counter_reset", "ts_ms": int(poll.ts_ms)}
            excluded[pod] = "counter_reset"
            continue
        ttft_n, ttft_buckets = ttft
        tpot_n, tpot_buckets = tpot
        pod_ttft = _p95_ms(ttft_buckets, ttft_n, percentile_mode, min_latency_samples)
        pod_tpot = _p95_ms(tpot_buckets, tpot_n, percentile_mode, min_latency_samples)
        n += ttft_n
        if pod_ttft is not None or pod_tpot is not None:
            judged += ttft_n
        if pod_ttft is not None:
            ttft_p95.append(pod_ttft)
        if pod_tpot is not None:
            tpot_p95.append(pod_tpot)
        if prompt_delta is not None:
            prompt_sum.append(prompt_delta[0])
            prompt_count.append(prompt_delta[1])
        if result.kv_cache is not None:
            kv_values.append(float(result.kv_cache))
        per_pod[pod] = {"n": ttft_n, "ttft_p95_ms": pod_ttft, "tpot_p95_ms": pod_tpot, "ts_ms": result.ts_ms}
    scrapes = (state.scrapes + (int(poll.ts_ms),))[-MAX_SCRAPE_TS:]
    updated = replace(state, dropped=dropped, scrapes=scrapes)
    if not per_pod:
        return replace(updated, last=None), None
    window = DirectWindow(
        start_ms=int(state.baseline_ts_ms if state.baseline_ts_ms is not None else poll.ts_ms),
        end_ms=int(poll.ts_ms),
        pods=tuple(sorted(per_pod)),
        excluded={**{pod: str(value.get("reason")) for pod, value in dropped.items()}, **excluded},
        ttft_p95_ms=max(ttft_p95) if ttft_p95 else None,
        tpot_p95_ms=max(tpot_p95) if tpot_p95 else None,
        ttft_count=n,
        judged_count=judged,
        prompt_tokens=sum(prompt_sum) if prompt_sum else None,
        prompt_count=sum(prompt_count) if prompt_count else None,
        kv_cache=(sum(kv_values) / len(kv_values)) if kv_values else None,
        per_pod=per_pod,
    )
    return replace(updated, last=window), window


def _p95_ms(buckets, count: float, mode: str, min_samples: int) -> float | None:
    """Per-pod p95 (ms) with the metrics store's rule: None below ``min_samples``."""
    if count <= 0 or (min_samples > 0 and count < min_samples):
        return None
    value = histogram_percentile(buckets, 0.95, mode=mode)
    return None if value is None else float(value) * 1000.0


# ------------------------------------------------------------------ scraping
def http_get_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - cluster-internal
        return response.read().decode("utf-8", errors="replace")


def metrics_url(ip: str, port: int) -> str:
    host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
    return f"http://{host}:{int(port)}/metrics"


class PodMetricsScraper:
    """Concurrent, bounded ``/metrics`` reads: each in a dedicated thread pool, awaited
    with ``asyncio.wait_for(timeout)``, so a slow / dead pod costs the caller at most
    ``timeout_s`` and never blocks the event loop."""

    def __init__(
        self,
        *,
        timeout_s: float,
        fetch: Callable[[str, float], str] = http_get_text,
        clock_ms: Callable[[], int] | None = None,
        max_workers: int = 16,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._fetch = fetch
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="safescale-scrape")

    def _read(self, url: str) -> PodScrape:
        text = self._fetch(url, self._timeout_s)
        return parse_vllm_metrics(text, ts_ms=int(self._clock_ms()))

    async def scrape(self, targets: Mapping[str, str | None]) -> dict[str, "PodScrape | str"]:
        loop = asyncio.get_running_loop()

        async def one(pod: str, url: str | None) -> tuple[str, "PodScrape | str"]:
            if not url:
                return pod, "no_endpoint"
            try:
                future = loop.run_in_executor(self._executor, self._read, url)
                return pod, await asyncio.wait_for(future, timeout=self._timeout_s)
            except asyncio.TimeoutError:
                return pod, "timeout"
            except Exception as exc:  # noqa: BLE001 - recorded per pod
                return pod, f"error:{type(exc).__name__}"

        pairs = await asyncio.gather(*(one(pod, url) for pod, url in sorted(targets.items())))
        return dict(pairs)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class DirectEvidenceCollector:
    """Takes the baselines (at the hide confirmation, else at the next tick) and the
    per-tick polls of the probes on the direct path.

    ``targets(model, exclude)`` -> ``{pod: url | None}`` of the model's remaining pods
    (awake, not in ``exclude``); the app builds it from the SM cluster view."""

    def __init__(
        self,
        safescale: Any,
        scraper: PodMetricsScraper,
        targets: Callable[[str, tuple[str, ...]], Mapping[str, str | None]],
        *,
        poll_ms: float,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._safescale = safescale
        self._scraper = scraper
        self._targets = targets
        self._poll_ms = float(poll_ms)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._inflight: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    def on_hide_done(self, model: str, pods: tuple[str, ...]) -> bool:
        """ActionQueue ``on_hide_done``: anchor the probe (``mark_hidden``) and start its
        baseline scrape at once (asynchronously, on the running loop)."""
        anchored = self._safescale.mark_hidden(model, pods=pods)
        if anchored:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return anchored  # no loop (sync callers / tests): the next tick takes it
            task = loop.create_task(self.take_baselines(only_model=model))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return anchored

    async def take_baselines(self, *, only_model: str | None = None) -> int:
        taken = 0
        for probe in self._safescale.direct_baseline_due():
            if only_model is not None and probe.model != only_model:
                continue
            if probe.request_id in self._inflight:
                continue
            self._inflight.add(probe.request_id)
            try:
                targets = dict(self._targets(probe.model, tuple(probe.pods)))
                results = await self._scraper.scrape(targets) if targets else {}
                ts_ms = int(self._clock_ms())
                if self._safescale.set_direct_baseline(probe.model, request_id=probe.request_id,
                                                       results=results, ts_ms=ts_ms):
                    taken += 1
            except Exception:  # noqa: BLE001 - the next tick retries; the deadline falls back
                LOG.exception("safescale direct baseline of %s failed", probe.model)
            finally:
                self._inflight.discard(probe.request_id)
        return taken

    async def poll(self) -> dict[str, DirectPoll]:
        """One tick: missing baselines first, then one scrape of every probe's baseline
        pods whose last poll (or baseline) is at least ~one poll period old."""
        await self.take_baselines()
        polls: dict[str, DirectPoll] = {}
        now = int(self._clock_ms())
        jobs = []
        for probe in self._safescale.direct_poll_due():
            state = probe.direct
            last = max([state.baseline_ts_ms or 0, *state.scrapes]) if state is not None else 0
            if now - last < 0.5 * self._poll_ms:
                continue  # a baseline / poll this young adds nothing (and n ~ 0)
            pods = state.live_pods()
            targets = self._targets(probe.model, tuple(probe.pods))
            jobs.append((probe, {pod: targets.get(pod) for pod in pods}))
        if not jobs:
            return polls
        results = await asyncio.gather(*(self._scraper.scrape(targets) for _, targets in jobs))
        ts_ms = int(self._clock_ms())
        for (probe, _), result in zip(jobs, results):
            polls[probe.model] = DirectPoll(request_id=probe.request_id, ts_ms=ts_ms, results=result)
        return polls


def cluster_view_targets(view_getter: Callable[[], Any], *, port: int):
    """``targets(model, exclude)`` over the SM cluster view: the model's awake bindings
    minus ``exclude`` (the probe pods), URL from the fleet state's pod IP (None when the
    view has no IP for the pod)."""

    def targets(model: str, exclude: tuple[str, ...]) -> dict[str, str | None]:
        view = view_getter()
        if view is None:
            return {}
        ips = getattr(view, "pod_ips", None) or {}
        skip = set(exclude)
        out: dict[str, str | None] = {}
        for binding in getattr(view, "bindings", ()) or ():
            if binding.model != model or not binding.awake or binding.serve_id in skip:
                continue
            ip = ips.get(binding.serve_id)
            out[binding.serve_id] = metrics_url(ip, port) if ip else None
        return out

    return targets
