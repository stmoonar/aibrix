from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tre_controller.config import SafeScaleConfig

ProbeStatus = Literal["none", "probing", "commit", "rollback"]
CommandKind = Literal["hide", "unhide", "scale_down", "scale_up"]


class ProbeStore(Protocol):
    def save_probe(self, request_id: str, record: dict[str, Any]) -> None: ...

    def delete_probe(self, request_id: str) -> None: ...

    def list_unresolved_probes(self) -> list[dict[str, Any]]: ...

    def append_probe_journal(self, request_id: str, record: dict[str, Any]) -> None: ...

    def load_probe_journal(self, request_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ProbeObservation:
    ts_ms: int
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    z_m: float | None = None
    q_ctl: float | None = None
    has_traffic: bool = False
    avg_gpu_cache_norm: float | None = None


@dataclass(frozen=True)
class SafeScaleCommand:
    kind: CommandKind
    model: str
    pods: tuple[str, ...] = ()
    delta: int = 0
    reason: str = ""


@dataclass(frozen=True)
class SafeScaleDecision:
    status: ProbeStatus
    reason: str
    commands: tuple[SafeScaleCommand, ...] = ()


@dataclass(frozen=True)
class SafeScaleProbe:
    model: str
    pods: tuple[str, ...]
    start_ms: int
    deadline_ms: int
    request_id: str
    status: Literal["probing"] = "probing"
    pending_upscales: dict[str, int] = field(default_factory=dict)
    observations: tuple[ProbeObservation, ...] = ()


@dataclass(frozen=True)
class ProbeTailSummary:
    latency_ok: bool
    z_min: float | None
    has_traffic: bool
    sample_count: int
    tail_count: int
    gpu_cache_max: float | None = None


class SafeScaleStateMachine:
    def __init__(self, *, config: SafeScaleConfig, store: ProbeStore | None = None) -> None:
        self._config = config
        self._store = store
        self._probes: dict[str, SafeScaleProbe] = {}

    def active_probe(self, model: str) -> SafeScaleProbe | None:
        return self._probes.get(model)

    def start_probe(
        self,
        *,
        model: str,
        pods: tuple[str, ...],
        now_ms: int,
        pending_upscales: dict[str, int] | None = None,
    ) -> SafeScaleDecision:
        if model in self._probes:
            return SafeScaleDecision(status="probing", reason="probe_already_active")
        if not pods:
            return SafeScaleDecision(status="none", reason="no_pods_to_probe")

        normalized_pending = _normalize_pending_upscales(pending_upscales)
        probe = SafeScaleProbe(
            model=model,
            pods=tuple(pods),
            start_ms=int(now_ms),
            deadline_ms=int(now_ms + self._config.default_window_ms),
            request_id=f"{model}-{int(now_ms)}",
            pending_upscales=normalized_pending,
        )
        self._probes[model] = probe
        self._persist_probe(probe)
        return SafeScaleDecision(
            status="probing",
            reason="probe_started",
            commands=(SafeScaleCommand(kind="hide", model=model, pods=probe.pods, reason="probe_started"),),
        )

    def observe(self, model: str, observation: ProbeObservation, *, now_ms: int) -> SafeScaleDecision:
        probe = self._probes.get(model)
        if probe is None:
            return SafeScaleDecision(status="none", reason="probe_not_found")

        updated = _replace_observations(probe, probe.observations + (observation,))
        self._probes[model] = updated
        self._persist_observation(updated, observation)

        if self._violates_slo(observation):
            return self._rollback(updated, reason="slo_violation")

        if now_ms < updated.deadline_ms:
            self._persist_probe(updated)
            return SafeScaleDecision(status="probing", reason="probe_pending")

        summary = _summarize_tail(
            updated,
            hq=self._config.hq,
            ttft_p95_slo_ms=self._config.ttft_p95_slo_ms,
            tpot_p95_slo_ms=self._config.tpot_p95_slo_ms,
        )
        if _tail_allows_commit(summary, tau_low=self._config.tau_low):
            return self._commit(updated, reason="formal_commit_gate_passed")
        return self._rollback(updated, reason="formal_commit_gate_failed")

    def restore(self) -> int:
        if self._store is None:
            return 0
        restored = 0
        for row in self._store.list_unresolved_probes():
            probe = _probe_from_record(row, self._store)
            if probe is None or probe.model in self._probes:
                continue
            self._probes[probe.model] = probe
            restored += 1
        return restored

    def _commit(self, probe: SafeScaleProbe, *, reason: str) -> SafeScaleDecision:
        self._probes.pop(probe.model, None)
        commands: list[SafeScaleCommand] = [
            SafeScaleCommand(kind="scale_down", model=probe.model, delta=-len(probe.pods), reason=reason)
        ]
        for model, delta in sorted(probe.pending_upscales.items()):
            commands.append(
                SafeScaleCommand(kind="scale_up", model=model, delta=delta, reason="safescale_followup_upscale")
            )
        self._finish_probe(probe, status="commit", reason=reason)
        return SafeScaleDecision(status="commit", reason=reason, commands=tuple(commands))

    def _rollback(self, probe: SafeScaleProbe, *, reason: str) -> SafeScaleDecision:
        self._probes.pop(probe.model, None)
        self._finish_probe(probe, status="rollback", reason=reason)
        return SafeScaleDecision(
            status="rollback",
            reason=reason,
            commands=(SafeScaleCommand(kind="unhide", model=probe.model, pods=probe.pods, reason=reason),),
        )

    def _violates_slo(self, observation: ProbeObservation) -> bool:
        return (
            observation.ttft_p95_ms is not None
            and observation.ttft_p95_ms > self._config.ttft_p95_slo_ms
        ) or (
            observation.tpot_p95_ms is not None
            and observation.tpot_p95_ms > self._config.tpot_p95_slo_ms
        )

    def _persist_probe(self, probe: SafeScaleProbe, *, terminal_reason: str | None = None) -> None:
        if self._store is None:
            return
        self._store.save_probe(probe.request_id, _probe_record(probe, terminal_reason=terminal_reason))

    def _persist_observation(self, probe: SafeScaleProbe, observation: ProbeObservation) -> None:
        if self._store is None:
            return
        record = _probe_record(probe, terminal_reason=None)
        record["last_observation"] = _observation_record(observation)
        self._store.append_probe_journal(probe.request_id, record)

    def _finish_probe(self, probe: SafeScaleProbe, *, status: Literal["commit", "rollback"], reason: str) -> None:
        if self._store is None:
            return
        self._store.save_probe(probe.request_id, _probe_record(probe, terminal_reason=reason, status=status))
        self._store.delete_probe(probe.request_id)


def _replace_observations(probe: SafeScaleProbe, observations: tuple[ProbeObservation, ...]) -> SafeScaleProbe:
    return SafeScaleProbe(
        model=probe.model,
        pods=probe.pods,
        start_ms=probe.start_ms,
        deadline_ms=probe.deadline_ms,
        request_id=probe.request_id,
        pending_upscales=dict(probe.pending_upscales),
        observations=observations,
    )


def _summarize_tail(
    probe: SafeScaleProbe,
    *,
    hq: float,
    ttft_p95_slo_ms: float,
    tpot_p95_slo_ms: float,
) -> ProbeTailSummary:
    observations = probe.observations
    if not observations:
        return ProbeTailSummary(latency_ok=True, z_min=None, has_traffic=False, sample_count=0, tail_count=0)

    sample_count = len(observations)
    hq_value = hq if hq > 0 else 0.25
    if hq_value < 1.0:
        desired_tail = max(2, int(math.ceil(sample_count * hq_value)))
    else:
        desired_tail = max(2, int(hq_value))
    tail = observations[-min(sample_count, desired_tail) :]

    latency_ok = True
    has_traffic = False
    z_values: list[float] = []
    gpu_cache_values: list[float] = []
    for observation in tail:
        if observation.has_traffic:
            has_traffic = True
        if observation.ttft_p95_ms is not None and observation.ttft_p95_ms > ttft_p95_slo_ms:
            latency_ok = False
        if observation.tpot_p95_ms is not None and observation.tpot_p95_ms > tpot_p95_slo_ms:
            latency_ok = False
        if observation.z_m is not None:
            z_values.append(float(observation.z_m))
        if observation.avg_gpu_cache_norm is not None:
            gpu_cache_values.append(float(observation.avg_gpu_cache_norm))

    z_min = min(z_values) if z_values else None
    if z_min is None and not has_traffic:
        z_min = float("inf")
    return ProbeTailSummary(
        latency_ok=latency_ok,
        z_min=z_min,
        has_traffic=has_traffic,
        sample_count=sample_count,
        tail_count=len(tail),
        gpu_cache_max=max(gpu_cache_values) if gpu_cache_values else None,
    )


def _tail_allows_commit(summary: ProbeTailSummary, *, tau_low: float) -> bool:
    if not summary.latency_ok:
        return False
    if summary.z_min is None:
        return not summary.has_traffic
    if summary.z_min < tau_low:
        return False
    if summary.gpu_cache_max is not None and summary.gpu_cache_max > 0.8:
        return False
    return True


def _probe_record(
    probe: SafeScaleProbe,
    *,
    terminal_reason: str | None = None,
    status: str = "probing",
) -> dict[str, Any]:
    return {
        "model": probe.model,
        "request_id": probe.request_id,
        "pods": list(probe.pods),
        "hidden_count": len(probe.pods),
        "start_ms": probe.start_ms,
        "deadline_ms": probe.deadline_ms,
        "status": status,
        "pending_upscales": dict(probe.pending_upscales),
        "terminal_reason": terminal_reason,
    }


def _observation_record(observation: ProbeObservation) -> dict[str, Any]:
    return {
        "ts_ms": observation.ts_ms,
        "ttft_p95_ms": observation.ttft_p95_ms,
        "tpot_p95_ms": observation.tpot_p95_ms,
        "z_m": observation.z_m,
        "q_ctl": observation.q_ctl,
        "has_traffic": observation.has_traffic,
        "avg_gpu_cache_norm": observation.avg_gpu_cache_norm,
    }


def _probe_from_record(row: dict[str, Any], store: ProbeStore) -> SafeScaleProbe | None:
    model = str(row.get("model", "")).strip()
    request_id = str(row.get("request_id") or row.get("probe_id") or "").strip()
    pods = _normalize_pods(row)
    if not model or not request_id or not pods:
        return None
    try:
        start_ms = _time_ms(row, "start_ms", "start_ts")
        deadline_ms = _time_ms(row, "deadline_ms", "deadline_ts")
    except (TypeError, ValueError):
        return None

    observations: list[ProbeObservation] = []
    for entry in store.load_probe_journal(request_id):
        if not isinstance(entry, dict):
            continue
        raw = entry.get("last_observation")
        if isinstance(raw, dict):
            observation = _observation_from_record(raw)
            if observation is not None:
                observations.append(observation)

    return SafeScaleProbe(
        model=model,
        pods=pods,
        start_ms=start_ms,
        deadline_ms=deadline_ms,
        request_id=request_id,
        pending_upscales=_normalize_pending_upscales(row.get("pending_upscales")),
        observations=tuple(observations),
    )


def _normalize_pods(row: dict[str, Any]) -> tuple[str, ...]:
    raw_pods = row.get("pods")
    if isinstance(raw_pods, (list, tuple)):
        pods = tuple(str(item).strip() for item in raw_pods if str(item).strip())
        if pods:
            return pods
    targets = row.get("target_instances")
    if isinstance(targets, list):
        pods = []
        for item in targets:
            if not isinstance(item, dict):
                continue
            pod = str(item.get("pod_name") or item.get("serve_name") or "").strip()
            if pod:
                pods.append(pod)
        if pods:
            return tuple(pods)
    hidden_count = row.get("hidden_count")
    try:
        count = int(hidden_count)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return tuple(f"hidden-{idx}" for idx in range(count))
    return ()


def _time_ms(row: dict[str, Any], ms_key: str, seconds_key: str) -> int:
    if row.get(ms_key) is not None:
        return int(float(row[ms_key]))
    return int(float(row[seconds_key]) * 1000)


def _observation_from_record(raw: dict[str, Any]) -> ProbeObservation | None:
    try:
        ts_ms = int(float(raw.get("ts_ms", raw.get("ts", 0))))
    except (TypeError, ValueError):
        return None
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=_optional_float(raw.get("ttft_p95_ms")),
        tpot_p95_ms=_optional_float(raw.get("tpot_p95_ms")),
        z_m=_optional_float(raw.get("z_m")),
        q_ctl=_optional_float(raw.get("q_ctl", raw.get("Q_ctl"))),
        has_traffic=bool(raw.get("has_traffic", False)),
        avg_gpu_cache_norm=_optional_float(raw.get("avg_gpu_cache_norm")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_pending_upscales(raw: dict[str, int] | None | Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for model, value in raw.items():
        try:
            delta = int(value)
        except (TypeError, ValueError):
            continue
        if model and delta > 0:
            normalized[str(model)] = delta
    return normalized
