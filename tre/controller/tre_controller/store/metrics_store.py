
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, Observed, PodWindowMetrics
from tre_common.persistence import fresh
from tre_common.gateway_inflight import instance_max_age_ms, live_gateway_instances, pod_inflight
from tre_common.percentile import histogram_percentile
from tre_common.rediskeys import GATEWAY_CADENCE_KEY, hist_key, inst_key, pods_key
from tre_controller.gateway_cadence import cadence_problem, last_hist_ms
from tre_common.tss import window_is_idle
from tre_common.vllm_metrics import doc_lookup
from tre_common.window_pods import aggregate_pods

LOG = logging.getLogger("tre_controller.store.metrics_store")

# Gateway doc identifiers (what the gateway writes today). Every read resolves them through
# tre_common.vllm_metrics.GATEWAY_DOC_KEYS, so the vLLM 0.30 identifiers
# (inter_token_latency_seconds, kv_cache_usage_perc) are accepted as well, newest first.
HISTOGRAM_METRICS = {
    "prompt_tokens": "request_prompt_tokens",
    "generation_tokens": "request_generation_tokens",
    "ttft": "time_to_first_token_seconds",
    "tpot": "time_per_output_token_seconds",
    "e2e": "e2e_request_latency_seconds",
}

INSTANT_METRICS = {
    "waiting": "num_requests_waiting",
    "running": "num_requests_running",
    "swapping": "num_requests_swapped",
    "kv_hit": "kv_cache_hit_rate",
    # KV-cache fill (0..1) for the SafeScale KV-cache guard (A12); optional per pod.
    "gpu_cache": "gpu_cache_usage_perc",
}

#: A token span (``scraped_ms`` of the baseline to the end histogram doc) outside
#: [MIN, MAX] x window flags the pod (a lost baseline / end doc, no baseline in the
#: lookback, a stalled gateway); its rate is still computed over the real span.
MIN_TOKEN_SPAN_FACTOR = 0.67
MAX_TOKEN_SPAN_FACTOR = 1.5
#: A pod's newest instant sample is current when it was observed at most this many
#: instant sample periods before the window end (round 4/5: I3 current pods and the
#: saturation sample; the producer's cadence, not the evidence period).
CURRENT_SAMPLE_PERIODS = 4

LEGACY_HIST_PREFIX = "aibrix:pod_histogram_metrics_"
LEGACY_INST_PREFIX = "aibrix:pod_instant_metrics_"


class MetricsStore:
    def __init__(
        self,
        redis_client: Any,
        registry: Any,
        *,
        instant_sample_interval_ms: int,
        evidence_period_ms: int | None = None,
        min_sample_ratio: float = 0.0,
        read_instant: bool = True,
        expected_cadence: tuple[int, int] | None = None,
        cadence_enforce: bool = True,
        percentile_mode: str = "bucket_upper",
        schema: str = "v2",
        histogram_lookback_ms: int = 90_000,
        min_latency_samples: int = 0,
        suffix_period_ms: int = 0,
        suffix_breakpoint: Callable[[str], int | None] | None = None,
        suffix_tail: Callable[[str], int | None] | None = None,
    ) -> None:
        if instant_sample_interval_ms <= 0:
            raise ValueError("instant_sample_interval_ms must be positive")
        if evidence_period_ms is not None and evidence_period_ms <= 0:
            raise ValueError("evidence_period_ms must be positive")
        if not 0.0 <= float(min_sample_ratio) <= 1.0:
            raise ValueError("min_sample_ratio must be in [0, 1]")
        if suffix_period_ms < 0:
            raise ValueError("suffix_period_ms must be non-negative")
        if histogram_lookback_ms < 0:
            raise ValueError("histogram_lookback_ms must be non-negative")
        if min_latency_samples < 0:
            raise ValueError("min_latency_samples must be non-negative")
        if schema not in {"v1", "v2"}:
            raise ValueError("schema must be v1 or v2")
        self._redis = redis_client
        self._registry = registry
        # 2026-10-25 (design docs/design/20261025-instant-sampling.md): the gateway's
        # instant-doc period (the coverage grid; round 4: also the freshness bound of
        # scrape_current_pods). ``evidence_period_ms`` (the histogram-doc period) is
        # validated for the callers; the store's suffix grid is ``suffix_period_ms``.
        self._instant_sample_interval_ms = int(instant_sample_interval_ms)
        # Per-pod coverage (review round 1): distinct scraped_ms in the window / (window /
        # instant period) below this flags the pod (sample_incomplete_pods: a donor-gate
        # flag); the pod stays in the window. 0 = no coverage flag.
        self._min_sample_ratio = float(min_sample_ratio)
        # M5: the SafeScale evidence store reads latency histograms only (False): no
        # instant docs, no coverage / token-span rules (raw counter deltas); the I3
        # rule still leaves frozen pods out.
        self._read_instant = bool(read_instant)
        # Declared gateway cadence (instant_ms, evidence_ms) checked on every snapshot;
        # None = no check (offline tools, tests). A problem marks the snapshot stale
        # when enforced, else is only logged.
        self._expected_cadence = expected_cadence
        self._cadence_enforce = bool(cadence_enforce)
        self._cadence_logged: str | None = None
        self._percentile_mode = percentile_mode
        self._schema = schema
        self._histogram_lookback_ms = histogram_lookback_ms
        # N1: below this many latency observations in a window, a p95 estimate is too
        # noisy to decide on (short windows + low QPS can have single-digit samples).
        # 0 disables the guard (default; the live controller sets it from config).
        self._min_latency_samples = min_latency_samples
        # O1 (breakpoint-aware window): > 0 also builds every model window's grid-aligned
        # suffixes (ModelWindowMetrics.suffix_windows) from the docs already read - no
        # extra redis round trip. 0 = off (the SafeScale evidence store, older callers).
        self._suffix_period_ms = int(suffix_period_ms)
        # Round 2 (cheap 2 s build): with a breakpoint source, only the one suffix O1
        # decides on is built - the one starting at the first grid boundary at or after
        # the model's breakpoint, when that lies inside the window (none without a
        # breakpoint). None = every grid-aligned suffix (offline tools, tests). A
        # breakpoint first seen after the read gets its suffix with the next snapshot.
        self._suffix_breakpoint = suffix_breakpoint
        # Round 5 (the SafeScale commit gate's tail, one value over the post-hide span):
        # model -> its probe's post-hide start (gateway doc domain; None: no probe). The
        # suffix starting at the first boundary at or after it is built too (a start at
        # or before the window start: the whole window is post-hide, no suffix needed).
        self._suffix_tail = suffix_tail
        # P3-2: gateway plugin instance liveness (the registry's SM instance staleness).
        self._gateway_instance_max_age_ms = instance_max_age_ms(registry)
        self._window_cache: dict[tuple[str, str, int, int], ModelWindowMetrics] = {}

    @property
    def redis_client(self) -> Any:
        """The metrics redis client (read-only use: the gateway-clock check)."""
        return self._redis

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def histogram_lookback_ms(self) -> int:
        return self._histogram_lookback_ms

    @property
    def p95_rule(self) -> tuple[str, int]:
        """(percentile mode, per-pod minimum latency samples) of this store."""
        return (self._percentile_mode, int(self._min_latency_samples))

    def read_snapshot(
        self,
        window_start_ms: int,
        window_end_ms: int,
        *,
        use_cache: bool = True,
        start_exclusive: bool = False,
    ) -> MetricsSnapshot:
        names = [spec.name for spec in self._registry.models()]
        read_start_ms = int(window_start_ms) + 1 if start_exclusive else int(window_start_ms)
        prefetched: dict[str, dict] | None = None
        cadence_raw: Any = None
        if self._schema == "v2":
            # Two round trips for the whole snapshot (pod sets + cadence, then every
            # pod's docs), not one per key.
            prefetched, cadence_raw = self._fetch_v2(
                names, read_start_ms, int(window_end_ms), with_cadence=self._expected_cadence is not None
            )
        models = {
            name: self.read_model_window(
                name, window_start_ms, window_end_ms, use_cache=use_cache, start_exclusive=start_exclusive,
                _docs=None if prefetched is None else prefetched[name],
            )
            for name in names
        }
        problem = self._cadence_problem(cadence_raw)
        return MetricsSnapshot(
            ts_ms=int(window_end_ms),
            models=models,
            stale=bool(problem) and self._cadence_enforce,
            stale_reason=problem,
            gateway_last_hist_ms=last_hist_ms(cadence_raw) if self._expected_cadence is not None else None,
        )

    def _fetch_v2(
        self, models: list[str], read_start_ms: int, window_end_ms: int, *, with_cadence: bool = False
    ) -> tuple[dict[str, dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]], Any]:
        """model -> pod key -> (histogram docs: the baseline - the newest at or before
        the window start, within the lookback, one ZREVRANGEBYSCORE ... LIMIT 0 1 - then
        the window's; the window's instant docs), and the raw cadence hash. Two
        pipelined round trips."""
        first = _Batch(self._redis)
        for model in models:
            first.add("smembers", pods_key(model))
        if with_cadence:
            first.add("hgetall", GATEWAY_CADENCE_KEY)
        replies = first.run()
        cadence_raw = replies[len(models)] if with_cadence else None
        pods_by_model = {
            model: sorted(_decode_text(pod) for pod in (replies[index] or ())) for index, model in enumerate(models)
        }
        second = _Batch(self._redis)
        layout: list[tuple[str, str]] = []
        lookback = self._histogram_lookback_ms
        for model, pods in pods_by_model.items():
            for pod in pods:
                layout.append((model, pod))
                if lookback:
                    second.add_baseline(hist_key(pod), read_start_ms - 1, max(0, read_start_ms - lookback))
                second.add("zrangebyscore", hist_key(pod), read_start_ms, window_end_ms)
                if self._read_instant:
                    second.add("zrangebyscore", inst_key(pod), read_start_ms, window_end_ms)
        replies = iter(second.run())
        out: dict[str, dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {m: {} for m in models}
        for model, pod in layout:
            baseline = _parse_docs(next(replies)) if lookback else []
            hist = _parse_docs(next(replies))
            inst = _parse_docs(next(replies)) if self._read_instant else []
            out[model][pod] = (baseline[-1:] + hist, inst)
        return out, cadence_raw

    def _cadence_problem(self, raw: Any = None) -> str | None:
        """The gateway's declared cadence (``raw``: the hash read with the snapshot)
        against ``expected_cadence`` (None: no check or it is fine). Logged as an error
        when it changes. Round 4: the hash's freshness is the sampler's rule
        (``last_hist_ms`` >= the window end), not a declaration problem."""
        if self._expected_cadence is None:
            return None
        instant_ms, evidence_ms = self._expected_cadence
        problem = cadence_problem(raw, instant_ms=instant_ms, evidence_ms=evidence_ms)
        if problem != self._cadence_logged:
            if problem:
                LOG.error(
                    "GATEWAY CADENCE: %s - snapshots %s", problem,
                    "marked stale" if self._cadence_enforce else "not marked stale (warn mode)",
                )
            else:
                LOG.info("gateway cadence ok: instant %d ms, evidence %d ms", instant_ms, evidence_ms)
            self._cadence_logged = problem
        return problem

    def read_model_window(
        self,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        *,
        use_cache: bool = True,
        start_exclusive: bool = False,
        _docs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    ) -> ModelWindowMetrics:
        """One model's window. ``start_exclusive`` reads the half-open ``(start, end]``.

        The phase-aligned sampler ends every window exactly on a gateway write boundary,
        and the gateway stamps its samples with that boundary, so a closed ``[start, end]``
        window would hold one instant tick too many (both ends) and the histogram
        baseline would sit one evidence period before ``start``. Half-open gives exactly
        ``window_ms / instant period`` instant ticks and a token delta between the
        histogram docs at ``start`` and ``end``, i.e. exactly one window.

        2026-10-25 (design docs/design/20261025-instant-sampling.md, review round 1):

        * Pod set: every pod with a doc in the window, except I3 pods (newest
          ``scraped_ms`` of its window docs before the read start: frozen metrics) -
          left out entirely, listed in ``scrape_stale_pods``. Queue and tokens always
          come from this one pod set; the O1 suffixes use the same set.
        * Queue gauges: the mean over all the pod's instant docs in the window (a
          repeated ``scraped_ms`` is a zero-order hold of the last observation).
        * Tokens: (end - baseline counter sum) / (their ``scraped_ms`` span) x window;
          the baseline is the newest histogram doc at or before the window start, else
          the first one inside it.
        * Flags (``sample_incomplete_pods``; a donor-gate flag only): coverage =
          distinct ``scraped_ms`` in the window / (window / instant period) below
          ``min_sample_ratio``; a token span outside [0.67, 1.5] x window; a missing or
          invalid ``scraped_ms`` (then the unscaled delta, fail closed into the flag).
        * The same rules per O1 suffix.
        """
        cache_key = (self._schema, model, int(window_start_ms), int(window_end_ms), bool(start_exclusive))
        # Sliding windows (S1.1) pass use_cache=False: every window is unique, so the
        # per-window cache never hits and would grow without bound. Only tumbling reads
        # (repeated identical [start, end] within a block) benefit from caching.
        if use_cache and cache_key in self._window_cache:
            return self._window_cache[cache_key]

        # Timestamps are integer ms, so (start, end] == [start + 1, end].
        read_start_ms = int(window_start_ms) + 1 if start_exclusive else int(window_start_ms)
        suffix_starts = self._suffix_starts(model, int(window_start_ms), int(window_end_ms))
        suffix_pods: dict[int, dict[str, PodWindowMetrics]] = {start: {} for start in suffix_starts}
        # I3: pod name -> newest ``scraped_ms`` of its docs in the window (None: no doc
        # carries the field); only pods with a doc in the window.
        scraped: dict[str, int | None] = {}
        # 2026-10-25: flagged pods (a donor-gate flag; they stay in the window) and I3
        # pods (left out; frozen-scrape rescue evidence).
        incomplete: set[str] = set()
        stale: set[str] = set()
        suffix_incomplete: dict[int, set[str]] = {start: set() for start in suffix_starts}
        all_ticks: set[int] = set()
        if self._schema == "v1":
            # O1 suffixes are built for the v2 schema only (v1 = legacy fallback: the
            # controller then waits for a whole clean window after a breakpoint).
            suffix_starts = ()
            per_pod = self._read_v1_model_window(model, read_start_ms, window_end_ms, scraped=scraped)
        else:
            docs_by_pod = _docs if _docs is not None else self._fetch_v2([model], read_start_ms, int(window_end_ms))[0][model]
            per_pod = {}
            for pod_key, (hist_docs, inst_docs) in docs_by_pod.items():
                pod_name = _pod_name(hist_docs, inst_docs, pod_key)
                _note_scraped(scraped, pod_name, hist_docs, inst_docs, read_start_ms)
                all_ticks.update(_doc_ticks(inst_docs))
                if pod_name not in scraped:
                    continue  # no doc in the window (at most the histogram baseline)
                newest = scraped[pod_name]
                if newest is not None and newest < read_start_ms:
                    stale.add(pod_name)  # I3: frozen metrics - left out entirely
                    continue
                pod_metrics, flag = self._window_pod(
                    model, pod_key, hist_docs, inst_docs, read_start_ms, window_start_ms, window_end_ms
                )
                if flag or newest is None:
                    incomplete.add(pod_name)
                if pod_metrics is None:
                    # Review round 3 (M2): no usable token delta (no histogram doc in the
                    # window, a counter reset without a later doc) or no instant doc - out
                    # of queue AND tokens, flagged. Its O1 suffixes cannot be better.
                    for start in suffix_starts:
                        suffix_incomplete[start].add(pod_name)
                    continue
                per_pod[pod_metrics.pod] = pod_metrics
                for start in suffix_starts:
                    # The suffix (start, end] from the same docs and the same pod: the
                    # histogram baseline is the newest doc at or before the suffix start -
                    # exactly what a read of that shorter window returns.
                    suffix_read = start + 1 if start_exclusive else start
                    suffix_metrics, suffix_flag = self._window_pod(
                        model,
                        pod_key,
                        _with_baseline_doc(hist_docs, suffix_read) if hist_docs else [],
                        [doc for doc in inst_docs if _number(doc.get("timestamp"), 0.0) >= suffix_read],
                        suffix_read,
                        start,
                        window_end_ms,
                    )
                    if suffix_metrics is None:
                        suffix_incomplete[start].add(pod_name)
                        continue
                    suffix_pods[start][suffix_metrics.pod] = suffix_metrics
                    if suffix_flag or newest is None:
                        suffix_incomplete[start].add(pod_name)

        if self._schema == "v1":
            per_pod, v1_stale = _scrape_valid(per_pod, scraped, read_start_ms)
            stale.update(v1_stale)
            incomplete.update(name for name in per_pod if name in scraped and scraped[name] is None)
            all_ticks.update(tick for pod in per_pod.values() for tick in pod.instant_ticks_ms)
        # Ticks are gateway-wide provenance (the sampler waits for the window's end
        # tick): every pod's docs as written, flagged or left-out ones included.
        stale_pods = tuple(sorted(stale))
        incomplete_pods = tuple(sorted(incomplete))
        model_metrics = self._aggregate_model(model, window_start_ms, window_end_ms, per_pod)
        scrape_fresh = any(value is not None and value >= read_start_ms for value in scraped.values())
        model_metrics = replace(
            model_metrics,
            instant_ticks_ms=tuple(sorted(all_ticks)),
            sample_incomplete_pods=incomplete_pods,
            scrape_fresh=scrape_fresh,
        )
        if stale_pods or incomplete_pods:
            # The gateway in-flight count of every pod whose data is doubtful (I3 or
            # flagged): frozen-scrape rescue evidence (review round 1, item 7).
            model_metrics = replace(
                model_metrics,
                scrape_stale_pods=stale_pods,
                scrape_stale_inflight=self._stale_inflight(tuple(sorted(set(stale_pods) | set(incomplete_pods)))),
            )
        # P2-2 (2026-10-06): an idle window needs every serving pod scraped recently and
        # nothing in flight on it at the gateway (the tick decides on both). Round 4:
        # "recently" = at most 4 instant sample periods before the window end (the
        # producer's cadence, not the evidence period).
        current_max_age_ms = self._current_max_age_ms()
        model_metrics = replace(
            model_metrics,
            scrape_current_pods=frozenset(
                pod for pod, value in scraped.items() if fresh(int(window_end_ms), value, current_max_age_ms)
            ),
        )
        if (
            window_is_idle(model_metrics.prompt_tokens, model_metrics.generation_tokens)
            and float(model_metrics.avg_running or 0.0) + float(model_metrics.avg_waiting or 0.0) <= 1e-9
        ):
            model_metrics = replace(
                model_metrics,
                gateway_inflight=self._pods_inflight(tuple(sorted(set(scraped) | set(per_pod)))),
            )
        if suffix_starts:
            model_metrics = replace(
                model_metrics,
                suffix_windows=tuple(
                    _suffix_with_flags(
                        self._aggregate_model(model, start, window_end_ms, suffix_pods[start]),
                        suffix_incomplete[start],
                    )
                    for start in suffix_starts
                ),
            )
        if use_cache:
            self._window_cache[cache_key] = model_metrics
        return model_metrics

    def _stale_inflight(self, pods: tuple[str, ...]) -> dict[str, int]:
        """The gateway's in-flight count of each scrape-stale pod. Read only when pods
        are stale; an unknown count is 0 here (no evidence: it never invents
        frozen-scrape rescue demand)."""
        counts = self._pods_inflight(pods) or {}
        return {pod: count or 0 for pod, count in counts.items()}

    def _pods_inflight(self, pods: tuple[str, ...]) -> dict[str, int | None] | None:
        """pod -> routed, unfinished requests at the gateway, summed over the live
        gateway instances (heartbeat at most the registry's
        ``service_manager.sleep.instance_staleness_s`` old); a pod's count is None when a
        live instance's field is unreadable. None = unknown: the gateway coordination
        keys cannot be read, or no gateway instance is live."""
        try:
            live = live_gateway_instances(self._redis, max_age_ms=self._gateway_instance_max_age_ms)
            if not live:
                return None
            return {pod: pod_inflight(self._redis, pod, live) for pod in pods}
        except Exception:  # noqa: BLE001 - unknown, never zero
            return None

    def _suffix_starts(self, model: str, window_start_ms: int, window_end_ms: int) -> tuple[int, ...]:
        """Gateway boundaries strictly inside an aligned window (O1 suffix starts) - with
        a breakpoint source only the one O1 would decide on (the first boundary at or
        after the breakpoint) plus, with a tail source, the SafeScale tail's (round 5: the
        first boundary at or after the probe's post-hide start, gateway doc domain); ()
        when suffixes are off or the window is not on the suffix grid."""
        period = self._suffix_period_ms
        if period <= 0 or window_start_ms % period or window_end_ms % period:
            return ()
        if self._suffix_breakpoint is None:
            return tuple(range(window_start_ms + period, window_end_ms, period))
        starts: set[int] = set()
        try:
            point = self._suffix_breakpoint(model)
        except Exception:  # noqa: BLE001 - no suffix: O1 waits for a whole window
            LOG.warning("O1 suffix: breakpoint of %s unreadable", model, exc_info=True)
            point = None
        if point is not None:
            starts.add(-(-int(point) // period) * period)  # first grid boundary >= the breakpoint
        try:
            tail = self._suffix_tail(model) if self._suffix_tail is not None else None
        except Exception:  # noqa: BLE001 - no tail suffix: the gate finds no tail evidence
            LOG.warning("SafeScale tail: post-hide start of %s unreadable", model, exc_info=True)
            tail = None
        if tail is not None:
            starts.add(-(-int(tail) // period) * period)  # first grid boundary >= the post-hide start
        return tuple(sorted(start for start in starts if window_start_ms < start < window_end_ms))

    def read_latest_instant(
        self, model: str, now_ms: int, lookback_ms: int | None = None
    ) -> dict[str, float] | None:
        """Latest instant queue snapshot (waiting/running/swapping), summed across pods,
        plus ``age_ms``: how old the oldest pod's sample is (``now`` minus its
        ``scraped_ms``, the doc stamp without one).

        Unlike ``read_model_window`` this returns the single most-recent scrape value per
        pod (not a windowed mean). Round 4: v2 reads each pod's newest doc at or before
        ``now`` (``ZREVRANGEBYSCORE ... LIMIT 0 1``) whatever the gateway's cadence - no
        lookback to size (the former "2 x the period" lookback missed a doc of a late
        round, r3 SMOKE_FINDINGS defect 1) - and reports its age instead of guessing
        freshness. ``lookback_ms``: v1 legacy keys only (default 20 s). Used by the
        offline r3 sidecar sampler. None when no pod has a doc (review round 3, L6:
        unknown, not an empty queue).
        """
        if self._schema == "v1":
            start_ms = max(0, now_ms - int(lookback_ms if lookback_ms is not None else 20_000))
            inst_by_pod = self._read_legacy_docs(LEGACY_INST_PREFIX, model, start_ms, now_ms)
            latest = {pod: docs[-1] for pod, docs in inst_by_pod.items() if docs}
        else:
            pods = sorted(_decode_text(pod) for pod in (self._redis.smembers(pods_key(model)) or ()))
            batch = _Batch(self._redis)
            for pod in pods:
                batch.add_baseline(inst_key(pod), int(now_ms), 0)
            latest = {}
            for pod, raw in zip(pods, batch.run()):
                docs = _parse_docs(raw)
                if docs:
                    latest[pod] = docs[-1]
        if not latest:
            return None
        totals = {"waiting": 0.0, "running": 0.0, "swapping": 0.0}
        ages: list[int] = []
        for doc in latest.values():
            observed = _scraped_at(doc)
            stamp = observed if observed is not None and observed >= 0 else int(_number(doc.get("timestamp"), 0.0))
            ages.append(int(now_ms) - int(stamp))
            metrics = doc.get("model_metrics")
            if not isinstance(metrics, dict):
                continue
            for out_key, metric in (
                ("waiting", INSTANT_METRICS["waiting"]),
                ("running", INSTANT_METRICS["running"]),
                ("swapping", INSTANT_METRICS["swapping"]),
            ):
                totals[out_key] += _number(doc_lookup(metrics, model, metric), 0.0)
        totals["age_ms"] = float(max(ages))
        return totals

    def _read_v1_model_window(
        self,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        *,
        scraped: dict[str, int | None] | None = None,
    ) -> dict[str, PodWindowMetrics]:
        hist_by_pod = self._read_legacy_docs(
            LEGACY_HIST_PREFIX,
            model,
            window_start_ms,
            window_end_ms,
            lookback_ms=self._histogram_lookback_ms,
        )
        inst_by_pod = self._read_legacy_docs(LEGACY_INST_PREFIX, model, window_start_ms, window_end_ms)
        per_pod: dict[str, PodWindowMetrics] = {}
        for pod_key in sorted(set(hist_by_pod) | set(inst_by_pod)):
            pod_metrics = self._aggregate_pod(
                model, pod_key, hist_by_pod.get(pod_key, []), inst_by_pod.get(pod_key, []), window_start_ms
            )
            if pod_metrics is not None:
                per_pod[pod_metrics.pod] = pod_metrics
                if scraped is not None:
                    _note_scraped(
                        scraped, pod_metrics.pod, hist_by_pod.get(pod_key, []), inst_by_pod.get(pod_key, []),
                        window_start_ms,
                    )
        return per_pod

    def _read_legacy_docs(
        self,
        prefix: str,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        *,
        lookback_ms: int = 0,
    ) -> dict[str, list[dict[str, Any]]]:
        keys: list[str] = []
        for raw_key in self._redis.scan_iter(prefix + "*"):
            key = _decode_text(raw_key)
            parsed = _parse_legacy_key(prefix, key)
            if parsed is None:
                continue
            _, ts_ms = parsed
            if max(0, window_start_ms - lookback_ms) <= ts_ms <= window_end_ms:
                keys.append(key)
        values = self._redis.mget(keys) if keys else []
        docs_by_pod: dict[str, list[dict[str, Any]]] = {}
        for key, raw_value in zip(keys, values):
            if raw_value is None:
                continue
            parsed = _parse_legacy_key(prefix, key)
            if parsed is None:
                continue
            pod_key, ts_ms = parsed
            try:
                doc = json.loads(_decode_text(raw_value))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(doc, dict) or not _doc_has_model(doc, model):
                continue
            doc.setdefault("timestamp", ts_ms)
            docs_by_pod.setdefault(pod_key, []).append(doc)
        for docs in docs_by_pod.values():
            docs.sort(key=lambda doc: _number(doc.get("timestamp"), 0.0))
        if lookback_ms:
            return {pod: _with_baseline_doc(docs, window_start_ms) for pod, docs in docs_by_pod.items()}
        return docs_by_pod

    def _aggregate_pod(
        self,
        model: str,
        pod_key: str,
        hist_docs: list[dict[str, Any]],
        inst_docs: list[dict[str, Any]],
        window_start_ms: int,
        *,
        instant_ticks: tuple[int, ...] | None = None,
    ) -> PodWindowMetrics | None:
        # window_start_ms is the first *readable* timestamp (start + 1 for a half-open
        # read); the instant gauges are means over ``inst_docs`` (the samples, 2026-10-25).
        # ``instant_ticks``: the ticks to report as provenance (default: those of
        # ``inst_docs``).
        if not hist_docs and not inst_docs:
            return None
        pod_name = _pod_name(hist_docs, inst_docs, pod_key)

        prompt_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs, window_start_ms)
        generation_tokens = self._hist_sum_delta(model, HISTOGRAM_METRICS["generation_tokens"], hist_docs, window_start_ms)
        ttft_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms)
        tpot_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms)
        e2e_avg_s = self._hist_avg_delta(model, HISTOGRAM_METRICS["e2e"], hist_docs, window_start_ms)

        ttft_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms)
        tpot_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms)
        e2e_p95_s = self._hist_percentile(model, HISTOGRAM_METRICS["e2e"], hist_docs, window_start_ms)
        # Ungated window histogram: the model e2e p95 merges the pods' histograms
        # first and gates the merged count (tre_common.window_pods.aggregate_pods).
        e2e_hist, e2e_hist_count = self._hist_window_buckets(
            model, HISTOGRAM_METRICS["e2e"], hist_docs, window_start_ms
        )
        ttft_hist, ttft_hist_count = self._hist_window_buckets(
            model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms
        )
        tpot_hist, tpot_hist_count = self._hist_window_buckets(
            model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms
        )

        return PodWindowMetrics(
            pod=pod_name,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            avg_waiting=self._instant_avg(model, INSTANT_METRICS["waiting"], inst_docs),
            avg_running=self._instant_avg(model, INSTANT_METRICS["running"], inst_docs),
            avg_swapping=self._instant_avg(model, INSTANT_METRICS["swapping"], inst_docs),
            kv_cache_hit_rate=self._instant_avg(model, INSTANT_METRICS["kv_hit"], inst_docs),
            ttft_p95_ms=_seconds_to_ms(ttft_p95_s),
            tpot_p95_ms=_seconds_to_ms(tpot_p95_s),
            e2e_p95_ms=_seconds_to_ms(e2e_p95_s),
            request_count=self._hist_count_delta(
                model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs, window_start_ms
            ),
            request_count_observed=self._hist_count_delta(
                model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs, window_start_ms
            ),
            token_counter_reset=(
                self._hist_counter_reset(
                    model, HISTOGRAM_METRICS["prompt_tokens"], hist_docs
                )
                or self._hist_counter_reset(
                    model, HISTOGRAM_METRICS["generation_tokens"], hist_docs
                )
            ),
            instant_ticks_ms=_doc_ticks(inst_docs) if instant_ticks is None else instant_ticks,
            gpu_cache_usage=self._instant_avg_optional(model, INSTANT_METRICS["gpu_cache"], inst_docs),
            ttft_avg_ms=_seconds_to_ms(ttft_avg_s),
            ttft_count=self._hist_count_delta(model, HISTOGRAM_METRICS["ttft"], hist_docs, window_start_ms),
            tpot_avg_ms=_seconds_to_ms(tpot_avg_s),
            tpot_count=self._hist_count_delta(model, HISTOGRAM_METRICS["tpot"], hist_docs, window_start_ms),
            e2e_hist=e2e_hist,
            e2e_hist_count=e2e_hist_count,
            hist_first_ts_ms=_first_metric_doc_ts(model, HISTOGRAM_METRICS["ttft"], hist_docs),
            hist_last_ts_ms=_last_metric_doc_ts(model, HISTOGRAM_METRICS["ttft"], hist_docs),
            ttft_hist=ttft_hist,
            ttft_hist_count=ttft_hist_count,
            tpot_hist=tpot_hist,
            tpot_hist_count=tpot_hist_count,
            **self._latest_instant(model, inst_docs),
        )

    def _latest_instant(self, model: str, inst_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """The newest instant doc's queue gauges and KV-cache fill (onset saturation
        rescue: the latest grid, not the window average). Docs are sorted ascending."""
        for doc in reversed(inst_docs):
            metrics = doc.get("model_metrics")
            if not isinstance(metrics, dict):
                continue
            kv = doc_lookup(metrics, model, INSTANT_METRICS["gpu_cache"])
            return {
                "latest_waiting": _number(doc_lookup(metrics, model, INSTANT_METRICS["waiting"]), 0.0),
                "latest_running": _number(doc_lookup(metrics, model, INSTANT_METRICS["running"]), 0.0),
                "latest_gpu_cache": None if kv is None else _number(kv, 0.0),
                # Round 4: when the sample was observed (scraped_ms), the doc stamp
                # only for a doc without one.
                "latest_instant_ms": (
                    _scraped_at(doc) if _scraped_at(doc) is not None else int(_number(doc.get("timestamp"), 0.0))
                ),
            }
        return {}

    def _aggregate_model(
        self,
        model: str,
        window_start_ms: int,
        window_end_ms: int,
        per_pod: dict[str, PodWindowMetrics],
    ) -> ModelWindowMetrics:
        # One aggregation rule (tre_common.window_pods), shared with the controller's
        # restriction to the awake pods. NOTE: per_pod / routable_pods here count every
        # pod with a doc in the window, sleeping ones included (the gateway writes docs
        # for them too); the decision path replaces them with the fleet state's view.
        # The e2e p95 is formed from the pods' merged histograms and gated on the
        # merged count (not per pod: at low load each pod alone is below the gate).
        return aggregate_pods(
            model, window_start_ms, window_end_ms, per_pod,
            p95_rule=(self._percentile_mode, int(self._min_latency_samples)),
        )

    def _hist_sum_delta(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_hist_doc(docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return 0.0
        delta = _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0)
        return delta if delta >= 0.0 else None

    def _hist_counter_reset(
        self, model: str, metric: str, docs: list[dict[str, Any]]
    ) -> bool:
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return False
        return _number(last.get("sum"), 0.0) < _number(first.get("sum"), 0.0)

    def _hist_count_delta(
        self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int
    ) -> float | None:
        if not _has_window_hist_doc(docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return 0.0
        return max(
            0.0,
            _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0),
        )

    def _hist_avg_delta(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_metric(model, metric, docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        sum_delta = max(0.0, _number(last.get("sum"), 0.0) - _number(first.get("sum"), 0.0))
        count_delta = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
        if count_delta <= 0.0:
            return None
        return sum_delta / count_delta

    def _hist_percentile(self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> float | None:
        if not _has_window_metric(model, metric, docs, window_start_ms):
            return None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None
        if self._min_latency_samples > 0:
            count_delta = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
            if count_delta < self._min_latency_samples:
                # N1: too few observations for a stable p95 -> None (not 0), so the
                # signal/safescale layer treats the latency metric as unavailable rather
                # than deciding on noise.
                return None
        first_buckets = _normal_buckets(first.get("buckets"))
        last_buckets = _normal_buckets(last.get("buckets"))
        if not first_buckets or not last_buckets:
            return None
        delta = _bucket_delta(first_buckets, last_buckets)
        return histogram_percentile(delta.items(), 0.95, mode=self._percentile_mode)

    def _hist_window_buckets(
        self, model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int
    ) -> tuple[tuple[tuple[float, float], ...] | None, float | None]:
        """The window's cumulative bucket delta ``((upper_s, count), ...)`` and its
        observation count, without the minimum-samples gate; (None, None) when the
        window has no such histogram."""
        if not _has_window_metric(model, metric, docs, window_start_ms):
            return None, None
        first, last = _first_last_metric(model, metric, docs)
        if first is None or last is None:
            return None, None
        first_buckets = _normal_buckets(first.get("buckets"))
        last_buckets = _normal_buckets(last.get("buckets"))
        if not first_buckets or not last_buckets:
            return None, None
        count = max(0.0, _number(last.get("count"), 0.0) - _number(first.get("count"), 0.0))
        delta = _bucket_delta(first_buckets, last_buckets)
        return tuple(sorted(delta.items())), count

    def _instant_avg(self, model: str, metric: str, docs: list[dict[str, Any]]) -> float:
        """Mean of the gauge over the given instant samples; an absent value reads 0.
        2026-10-25: the divisor is the number of samples, not the window's expected
        sample count - a missing sample no longer pulls the queue average down (v1 lost
        ticks under CPU throttling: Q too low, Z too high, false HIGH). A pod with too few
        samples is flagged by the coverage rule instead."""
        values = [
            _number(doc_lookup(doc["model_metrics"], model, metric), 0.0)
            for doc in docs
            if isinstance(doc.get("model_metrics"), dict)
        ]
        return sum(values) / len(values) if values else 0.0

    def _window_pod(
        self,
        model: str,
        pod_key: str,
        hist_docs: list[dict[str, Any]],
        inst_docs: list[dict[str, Any]],
        read_start_ms: int,
        span_start_ms: int,
        window_end_ms: int,
    ) -> tuple[PodWindowMetrics | None, bool]:
        """(one pod's metrics over ``[read_start_ms, window_end_ms]``, flagged).

        Queue gauges: the mean over all ``inst_docs``. Tokens and request count: scaled
        to the window length by the ``scraped_ms`` span of the token baseline and end
        docs. Flagged: coverage below ``min_sample_ratio``, a span outside
        [0.67, 1.5] x window, or no valid span (then the unscaled delta). Without
        instant reads (the SafeScale evidence store): raw deltas, never flagged.

        Review round 3 (M2): queue and tokens always come from the same pods. A token
        counter that went down (the engine restarted) restarts the delta at the first
        doc after the reset - its ``scraped_ms`` starts the span; flagged. A pod with no
        usable delta (no histogram doc in the window, or nothing after the reset) is left
        out - (None, True) - of both. (A pod with histogram docs but no instant doc has
        coverage 0: flagged by the coverage rule.)"""
        reset = False
        if self._read_instant:
            hist_docs, reset = _after_counter_reset(model, hist_docs)
            if not _has_window_hist_doc(hist_docs, read_start_ms) or (
                reset and _token_span_ms(hist_docs, read_start_ms) is None
            ):
                return None, True
        metrics = self._aggregate_pod(model, pod_key, hist_docs, inst_docs, read_start_ms)
        if metrics is None:
            return None, False
        if not self._read_instant:
            return replace(metrics, observed=_pod_observed(hist_docs, inst_docs, read_start_ms)), False
        if reset:
            metrics = replace(metrics, token_counter_reset=True)
        window_ms = int(window_end_ms) - int(span_start_ms)
        if window_ms <= 0:
            return metrics, True
        flagged = False
        span = _token_span_ms(hist_docs, read_start_ms)
        if span is None:
            flagged = metrics.generation_tokens is not None or metrics.prompt_tokens is not None
        else:
            scale = window_ms / span
            metrics = replace(
                metrics,
                prompt_tokens=_scaled(metrics.prompt_tokens, scale),
                generation_tokens=_scaled(metrics.generation_tokens, scale),
                request_count=_scaled(metrics.request_count, scale),
            )
            flagged = not MIN_TOKEN_SPAN_FACTOR * window_ms <= span <= MAX_TOKEN_SPAN_FACTOR * window_ms
        observed = {
            int(_number(doc.get("scraped_ms"), -1.0))
            for doc in inst_docs
            if doc.get("scraped_ms") is not None
        }
        coverage = sum(1 for value in observed if read_start_ms <= value <= window_end_ms)
        if coverage + 1e-9 < self._min_sample_ratio * (window_ms / self._instant_sample_interval_ms):
            flagged = True
        metrics = replace(
            metrics,
            observed=_pod_observed(hist_docs, inst_docs, read_start_ms),
            # Round 5 (M2): the newest instant sample is current only within the
            # producer's cadence of the window end - the scrape_current_pods rule.
            latest_current=fresh(int(window_end_ms), metrics.latest_instant_ms, self._current_max_age_ms()),
        )
        return metrics, flagged or reset

    def _current_max_age_ms(self) -> int:
        """How old a pod's newest sample may be and still be current (scrape_current_pods,
        the saturation sample): ``CURRENT_SAMPLE_PERIODS`` instant periods - the
        producer's cadence, never the evidence period."""
        return CURRENT_SAMPLE_PERIODS * int(self._instant_sample_interval_ms)

    def _instant_avg_optional(self, model: str, metric: str, docs: list[dict[str, Any]]) -> float | None:
        """Mean of the gauge over the samples that actually carry it; None when none
        does (an absent metric is not a zero)."""
        values = [
            _number(value, 0.0)
            for value in (
                doc_lookup(doc["model_metrics"], model, metric)
                for doc in docs
                if isinstance(doc.get("model_metrics"), dict)
            )
            if value is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)


def _note_scraped(
    scraped: dict[str, int | None],
    pod: str,
    hist_docs: list[dict[str, Any]],
    inst_docs: list[dict[str, Any]],
    read_start_ms: int,
) -> None:
    """``scraped[pod]`` = the newest gateway ``scraped_ms`` (wall-clock ms of the pod's
    last successful /metrics fetch) of the pod's docs inside the window, None when none
    carries it. A pod without a doc in the window (only the histogram baseline before
    it) is not recorded: it contributes no window data either way."""
    in_window = False
    newest: int | None = None
    for doc in (*hist_docs, *inst_docs):
        if _number(doc.get("timestamp"), 0.0) < read_start_ms:
            continue
        in_window = True
        raw = doc.get("scraped_ms")
        if raw is None:
            continue
        value = int(_number(raw, -1.0))
        if value >= 0 and (newest is None or value > newest):
            newest = value
    if in_window:
        scraped[pod] = newest


def _scrape_valid(
    per_pod: dict[str, PodWindowMetrics], scraped: dict[str, int | None], read_start_ms: int
) -> tuple[dict[str, PodWindowMetrics], tuple[str, ...]]:
    """I3 (2026-10-04): the pods whose data is valid for a window, and the pods left out.

    The gateway keeps a pod's last metrics when a /metrics fetch fails and still writes
    them every round: frozen counters read as a zero-token window with a frozen queue.
    A pod is valid for the window ``[read_start_ms, end]`` only if its last successful
    scrape (``scraped_ms``) is at or after ``read_start_ms`` - compared with the doc
    timestamps' clock (both are the gateway's wall clock; the window bounds are doc
    timestamps), never with the controller's clock. A pod left out contributes nothing
    (tokens, queue, latency); with no valid pod left the model's tokens are None
    (unknown), never zero.

    v1 schema only since 2026-10-25 (the v2 read applies the same rule inline). A pod
    without ``scraped_ms`` is kept (the caller flags it: fail closed into the donor-gate
    flag, never into unknown). A pod with no doc in the window is kept as before."""
    kept: dict[str, PodWindowMetrics] = {}
    stale: list[str] = []
    for name, pod in per_pod.items():
        if name not in scraped:
            kept[name] = pod
            continue
        value = scraped[name]
        if value is None or value >= read_start_ms:
            kept[name] = pod
        else:
            stale.append(name)
    return kept, tuple(sorted(stale))


class _Batch:
    """Redis reads sent in one pipelined round trip (``transaction=False``); clients
    without ``pipeline`` (test fakes) run them one by one, same results."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._calls: list[tuple[str, tuple, dict]] = []

    def add(self, name: str, *args: Any, **kwargs: Any) -> None:
        self._calls.append((name, args, kwargs))

    def add_baseline(self, key: str, max_score: int, min_score: int) -> None:
        """The newest member scored in ``[min_score, max_score]`` (a list of 0 or 1)."""
        self._calls.append(("zrevrangebyscore", (key, max_score, min_score), {"start": 0, "num": 1}))

    def run(self) -> list[Any]:
        if not self._calls:
            return []
        make = getattr(self._client, "pipeline", None)
        if callable(make):
            try:
                pipe = make(transaction=False)
            except TypeError:
                pipe = None
            if pipe is not None and all(callable(getattr(pipe, name, None)) for name, _a, _k in self._calls):
                for name, args, kwargs in self._calls:
                    getattr(pipe, name)(*args, **kwargs)
                return list(pipe.execute())
        return [self._one(name, args, kwargs) for name, args, kwargs in self._calls]

    def _one(self, name: str, args: tuple, kwargs: dict) -> Any:
        if name == "zrevrangebyscore" and not callable(getattr(self._client, name, None)):
            key, max_score, min_score = args
            members = list(self._client.zrangebyscore(key, min_score, max_score))
            members.sort(key=lambda raw: _number(_doc_or_empty(raw).get("timestamp"), 0.0))
            return members[-1:]
        return getattr(self._client, name)(*args, **kwargs)


def _doc_or_empty(raw: Any) -> dict[str, Any]:
    try:
        doc = json.loads(_decode_text(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _parse_docs(raw_members: Any) -> list[dict[str, Any]]:
    """Decoded docs of a ZRANGE reply, ascending by ``timestamp``."""
    docs = [doc for doc in (_doc_or_empty(raw) for raw in (raw_members or ())) if doc]
    docs.sort(key=lambda doc: _number(doc.get("timestamp"), 0.0))
    return docs


def _scraped_at(doc: dict[str, Any]) -> int | None:
    raw = doc.get("scraped_ms")
    return None if raw is None else int(_number(raw, -1.0))


def _token_span_ms(hist_docs: list[dict[str, Any]], read_start_ms: int) -> int | None:
    """The observation span (ms) of the token delta: ``scraped_ms`` of the newest
    histogram doc minus that of the baseline (the first of ``hist_docs``: the newest at
    or before the window start, else the first in the window). None without a doc in
    the window, a missing ``scraped_ms``, or no positive span (a single doc)."""
    docs = [doc for doc in hist_docs if isinstance(doc.get("model_histogram_metrics"), dict)]
    if not docs or _number(docs[-1].get("timestamp"), 0.0) < read_start_ms:
        return None
    first, last = _scraped_at(docs[0]), _scraped_at(docs[-1])
    if first is None or last is None or last <= first:
        return None
    return last - first


def _suffix_with_flags(metrics: ModelWindowMetrics, pods: set[str]) -> ModelWindowMetrics:
    return replace(metrics, sample_incomplete_pods=tuple(sorted(pods)))


def _pod_observed(hist_docs: list[dict[str, Any]], inst_docs: list[dict[str, Any]], read_start_ms: int) -> Observed:
    """One pod's :class:`Observed` (round 4): from the observation of the token
    baseline (the first histogram doc the deltas start from; else the first
    observation in the window) to the newest one."""
    hist = [doc for doc in hist_docs if isinstance(doc.get("model_histogram_metrics"), dict)]
    in_window = [
        doc for doc in (*hist_docs, *inst_docs) if _number(doc.get("timestamp"), 0.0) >= read_start_ms
    ]
    seen = sorted({value for value in (_scraped_at(doc) for doc in in_window) if value is not None and value >= 0})
    first = _scraped_at(hist[0]) if hist else None
    if first is None or first < 0:
        first = seen[0] if seen else None
    return Observed(t_first_ms=first, t_last_ms=seen[-1] if seen else None)


#: The token counters whose decrease marks an engine restart (counter reset).
_TOKEN_COUNTERS = (HISTOGRAM_METRICS["prompt_tokens"], HISTOGRAM_METRICS["generation_tokens"])


def _after_counter_reset(model: str, hist_docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """``hist_docs`` from the first doc after the last token-counter reset (a ``sum``
    lower than the previous doc's: the engine restarted), and whether there was one.
    That doc is the new baseline of every delta."""
    cut = 0
    for metric in _TOKEN_COUNTERS:
        previous: float | None = None
        for index, doc in enumerate(hist_docs):
            entry = _metric_entry(model, metric, doc)
            if entry is None:
                continue
            value = _number(entry.get("sum"), 0.0)
            if previous is not None and value < previous:
                cut = max(cut, index)
            previous = value
    return (hist_docs[cut:], True) if cut else (hist_docs, False)


def _scaled(value: float | None, scale: float) -> float | None:
    return None if value is None else value * scale


def _doc_ticks(docs: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(sorted({int(_number(doc.get("timestamp"), 0.0)) for doc in docs if doc.get("timestamp") is not None}))


def _parse_legacy_key(prefix: str, key: str) -> tuple[str, int] | None:
    if not key.startswith(prefix):
        return None
    body = key[len(prefix):]
    try:
        pod_key, raw_ts = body.rsplit("_", 1)
        return pod_key, _timestamp_to_ms(int(raw_ts))
    except (ValueError, TypeError):
        return None


def _timestamp_to_ms(raw: int) -> int:
    if raw > 1_000_000_000_000_000:
        return raw // 1_000_000
    if raw < 100_000_000_000:
        return raw * 1000
    return raw


def _doc_has_model(doc: dict[str, Any], model: str) -> bool:
    prefix = model + "/"
    for field in ("model_histogram_metrics", "model_metrics"):
        metrics = doc.get(field)
        if isinstance(metrics, dict) and any(isinstance(key, str) and key.startswith(prefix) for key in metrics):
            return True
    return False


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _number(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pod_name(hist_docs: list[dict[str, Any]], inst_docs: list[dict[str, Any]], pod_key: str) -> str:
    for doc in hist_docs + inst_docs:
        name = doc.get("pod_name")
        if isinstance(name, str) and name:
            return name
    return pod_key


def _metric_entry(model: str, metric: str, doc: dict[str, Any]) -> dict[str, Any] | None:
    metrics = doc.get("model_histogram_metrics")
    if not isinstance(metrics, dict):
        return None
    entry = doc_lookup(metrics, model, metric)
    return entry if isinstance(entry, dict) else None


def _has_window_metric(model: str, metric: str, docs: list[dict[str, Any]], window_start_ms: int) -> bool:
    for doc in docs:
        if _number(doc.get("timestamp"), 0.0) < window_start_ms:
            continue
        if _metric_entry(model, metric, doc) is not None:
            return True
    return False


def _has_window_hist_doc(docs: list[dict[str, Any]], window_start_ms: int) -> bool:
    return any(
        _number(doc.get("timestamp"), 0.0) >= window_start_ms
        and isinstance(doc.get("model_histogram_metrics"), dict)
        for doc in docs
    )


def _first_last_metric(model: str, metric: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entries = [_metric_entry(model, metric, doc) for doc in docs]
    entries = [entry for entry in entries if entry is not None]
    if not entries:
        return None, None
    return entries[0], entries[-1]


def _first_metric_doc_ts(model: str, metric: str, docs: list[dict[str, Any]]) -> int | None:
    """Timestamp of the first doc carrying ``metric`` - the doc ``_first_last_metric``
    takes the delta's baseline from (docs are sorted, baseline doc first)."""
    for doc in docs:
        if _metric_entry(model, metric, doc) is not None and doc.get("timestamp") is not None:
            return int(_number(doc.get("timestamp"), 0.0))
    return None


def _last_metric_doc_ts(model: str, metric: str, docs: list[dict[str, Any]]) -> int | None:
    """Timestamp of the last doc carrying ``metric`` (the end of the delta)."""
    for doc in reversed(docs):
        if _metric_entry(model, metric, doc) is not None and doc.get("timestamp") is not None:
            return int(_number(doc.get("timestamp"), 0.0))
    return None


def _with_baseline_doc(docs: list[dict[str, Any]], window_start_ms: int) -> list[dict[str, Any]]:
    baseline = None
    window_docs: list[dict[str, Any]] = []
    for doc in docs:
        if _number(doc.get("timestamp"), 0.0) < window_start_ms:
            baseline = doc
        else:
            window_docs.append(doc)
    if baseline is None:
        return window_docs
    return [baseline, *window_docs]


def _normal_buckets(raw: Any) -> dict[float, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[float, float] = {}
    for key, value in raw.items():
        try:
            upper = float("inf") if str(key) in {"+Inf", "Inf", "inf"} else float(key)
            out[upper] = _number(value, 0.0)
        except ValueError:
            continue
    return out


def _cumulative_at(buckets: dict[float, float], upper: float) -> float:
    candidates = [count for bucket_upper, count in buckets.items() if bucket_upper <= upper]
    return max(candidates) if candidates else 0.0


def _bucket_delta(first: dict[float, float], last: dict[float, float]) -> dict[float, float]:
    result: dict[float, float] = {}
    running = 0.0
    for upper in sorted(set(first) | set(last)):
        delta = max(0.0, _cumulative_at(last, upper) - _cumulative_at(first, upper))
        running = max(running, delta)
        result[upper] = running
    return result


def _seconds_to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0



def _sum_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None
