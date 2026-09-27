from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SloSpec:
    ttft_p95_ms: float
    tpot_p95_ms: float
    e2e_p95_ms: float
    #: Idle (isolated-request) TTFT fit ``c + b * prompt_tokens`` that the calibration
    #: label's slowdown TTFT SLO scales (plan 2026-09-21 6.9h, D6). Optional: absent in
    #: older registries, and nothing online reads it.
    ttft_idle_c_ms: float | None = None
    ttft_idle_b_ms_per_token: float | None = None
    #: The calibration label built from it (plan 2026-09-21 6.11 D6'): mode, slowdown
    #: factor k and floor. Offline only; ``tre_common.slo_labels`` falls back to its
    #: module defaults (slowdown, 5, 500 ms) when absent.
    ttft_slo_mode: str | None = None
    ttft_slowdown_k: float | None = None
    ttft_floor_ms: float | None = None


ALT_THRESHOLD_DIRECTIONS = {"higher_is_healthier", "lower_is_healthier"}
EXPECTED_SIGNAL_DIRECTIONS = {
    "queue_len": "lower_is_healthier",
    "decode_tps": "lower_is_healthier",
    "prefill_tps": "lower_is_healthier",
}


#: Band margins an alternative signal falls back to when its registry entry carries none:
#: the plan defaults, the same numbers classify_all_models uses for an unfitted TSS.
ALT_DEFAULT_DELTA_CRIT = 0.2
ALT_DEFAULT_DELTA_HIGH = 0.25


@dataclass(frozen=True)
class AltThreshold:
    theta: float
    direction: str
    #: Per-signal band margins around tau_low = 1 (plan §6.9 item 3): the classifier runs
    #: an alternative signal on its OWN fitted bands, not on the TSS tau_crit/tau_high.
    #: ``None`` = not fitted; :meth:`bands` then falls back to the plan defaults.
    delta_crit: float | None = None
    delta_high: float | None = None

    def bands(self) -> tuple[float, float, bool]:
        """(delta_crit, delta_high, defaulted) - ``defaulted`` is True when either margin
        is missing and the plan default was used."""
        defaulted = self.delta_crit is None or self.delta_high is None
        return (
            float(self.delta_crit) if self.delta_crit is not None else ALT_DEFAULT_DELTA_CRIT,
            float(self.delta_high) if self.delta_high is not None else ALT_DEFAULT_DELTA_HIGH,
            defaulted,
        )


@dataclass(frozen=True)
class TrsParams:
    w_p: float
    w_d: float
    lambda_wait: float
    qmin: float
    #: DEPRECATED - only the legacy fixed-alpha EMA (ema_tau_ms unset) reads it; kept because
    #: the golden parity tests and every registry/params payload still carry it.
    ema_alpha: float
    theta_m: float
    tau_crit: float
    tau_low: float
    tau_high: float
    # DEPRECATED (ADR-0014): the saturation-segment concept was removed; scaling and
    # fairness receiver eligibility are decided solely by z_m threshold bands. These
    # fields are retained only for backward-compatible registry.yaml parsing and are no
    # longer fitted by R3. `qsat`/`epsat`/`hsat` are now inert; queue_len uses the
    # model-specific alt_thresholds entry.
    qsat: float
    epsat: float
    hsat: int
    ema_tau_ms: float | None = None


@dataclass(frozen=True)
class NodeSpec:
    name: str
    gpus: int
    two_gpu_slots: tuple[tuple[int, int], ...]
    gpu_uuids: tuple[str, ...] = ()


#: Default for ``cluster.max_bound_per_gpu``: at most this many bindings (sleeping
#: or awake) may share one physical GPU (each sleeping resident keeps a small
#: CUDA context on the GPU).
DEFAULT_MAX_BOUND_PER_GPU = 3


@dataclass(frozen=True)
class ClusterTopology:
    nodes: tuple[NodeSpec, ...]
    #: Registry ``cluster.max_bound_per_gpu``.
    max_bound_per_gpu: int = DEFAULT_MAX_BOUND_PER_GPU


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights_path: str
    tp_size: int
    min_replicas: int
    #: GPU layout: how many bindings (pods / slots) the model gets. ``make manifests``
    #: renders ``feasible_slots[:max_replicas]`` deployments; NOT the scaling cap.
    max_replicas: int
    vllm_image: str
    slo: SloSpec
    trs: TrsParams
    vllm_extra_args: tuple[str, ...] = ()
    alt_thresholds: dict[str, AltThreshold] = field(default_factory=dict)
    #: Scaling cap: the most bindings that may be awake at once (controller planner,
    #: service-manager target / binding wake, console). ``None`` = ``max_replicas``.
    #: Registry key models[].max_awake_replicas (v1/paper alignment A1: 4 for TRE and APA).
    max_awake_replicas: int | None = None
    #: Optional vLLM fork features the model's image supports (``VLLM_FEATURE_FLAGS``);
    #: their flags are rendered only while the reissue sidecar is enabled.
    vllm_features: tuple[str, ...] = ()

    @property
    def scale_max_replicas(self) -> int:
        return scale_max_replicas(self)


def scale_max_replicas(spec: Any) -> int:
    """The scaling cap of a model spec: ``max_awake_replicas`` when set, else the layout
    size ``max_replicas`` (duck-typed so lightweight test specs without the field work)."""
    cap = getattr(spec, "max_awake_replicas", None)
    return int(spec.max_replicas) if cap is None else int(cap)


#: Sleep paths (plan 2026-09-27 D1): every caller that puts a binding to sleep names
#: one; the service-manager looks up its soft drain budget in
#: ``service_manager.sleep.budgets_s``.
SLEEP_PATHS = (
    "safescale_commit",  # SafeScale commit of a hidden probe pod
    "urgent",  # controller *_immediate donor paths (fast loop)
    "scale_down",  # ordinary scale-down (TRE slow loop and APA alike)
    "defrag",  # buddy defragmentation migration
    "repair",  # fleet repair quarantine / resident pool rebuild
    "startup",  # startup admission / startup convergence
    "default",  # any caller that names no path
)

#: Soft drain budgets (s). ``None`` = only the hard cap. Every drain is additionally
#: capped by ``hard_cap_s``; requests the gateway marks non-continuable are always
#: waited for up to the hard cap (never aborted at the soft budget).
DEFAULT_SLEEP_BUDGETS_S: dict[str, float | None] = {
    # The caller passes the probe window as drain_budget_s; without one, hard cap.
    "safescale_commit": None,
    "urgent": 30.0,
    # No deadline pressure: wait for in-flight requests up to the hard cap. The same
    # budget applies to TRE and APA scale-downs (both arms drain identically).
    "scale_down": None,
    "defrag": 30.0,
    "repair": 30.0,
    "startup": 30.0,
    "default": 30.0,
}

#: Default ``gateway.route_timeout_s``: the gateway's per-request route timeout. The
#: single source for the model HTTPRoute timeout (deploy/gen_model_manifests.py) and
#: the service-manager drain hard cap (no request can outlive it anyway).
DEFAULT_ROUTE_TIMEOUT_S = 150.0

#: ``service_manager.sleep.vllm_sleep_mode_param`` values.
SLEEP_MODE_PARAM_CHOICES = ("auto", "true", "false")


#: vLLM fork features a model image may declare (``models[].vllm_features``) and the
#: engine flags they enable (fork branch tre/transparent-sleep). Both only matter to the
#: reissue sidecar, so the flags are rendered only while ``reissue.enabled``.
VLLM_FEATURE_FLAGS: dict[str, tuple[str, ...]] = {
    # 503 + Retry-After + {"error": {"type": "EngineSleeping"}} for new requests while
    # the engine sleeps / is paused (the sidecar retries them elsewhere).
    "sleep_reject_new": ("--sleep-reject-new",),
    # abort outputs carry prompt_token_ids + generated_token_ids (token-exact continuation).
    "abort_return_token_ids": ("--abort-return-token-ids",),
}

#: Stable in-cluster Service of the tre-v2 Envoy proxy (overlays/tre-v2/gateway-service.yaml).
#: Envoy Gateway's own proxy Service carries a generated hash suffix, so nothing in TRE
#: may name it; this ClusterIP selects the same proxy pods by their owning-gateway labels.
DEFAULT_GATEWAY_SERVICE_NAME = "tre-gateway"
#: Namespace of the Envoy proxy pods (Envoy Gateway's controller namespace by default); a
#: Service can only select pods of its own namespace.
DEFAULT_GATEWAY_SERVICE_NAMESPACE = "envoy-gateway-system"
DEFAULT_GATEWAY_SERVICE_PORT = 80


def gateway_service_url(name: str, namespace: str, port: int) -> str:
    return f"http://{name}.{namespace}.svc.cluster.local:{int(port)}"


DEFAULT_REISSUE_GATEWAY_URL = gateway_service_url(
    DEFAULT_GATEWAY_SERVICE_NAME, DEFAULT_GATEWAY_SERVICE_NAMESPACE, DEFAULT_GATEWAY_SERVICE_PORT
)
#: The pod's serving port (Service targetPort, model.aibrix.ai/port, gateway target-pod,
#: service-manager, probes). With the sidecar enabled the sidecar listens here.
POD_SERVING_PORT = 8000


@dataclass(frozen=True)
class ReissueConfig:
    """Registry ``reissue:`` section: the retry / continuation sidecar in every model pod
    (tre/docs/design/20260927-reissue-sidecar-v2.md). Read by ``make manifests`` AND by
    the service-manager when it creates a Deployment at runtime, so both render the same
    pod. Disabled, the model pods are rendered exactly as without the sidecar (vLLM on
    the serving port, no fork flags)."""

    enabled: bool = True
    #: TRE gateway reached from inside the cluster (DNS name, never an IP / NodePort).
    #: None = the gateway: section's stable Service (``GatewayConfig.internal_url``).
    gateway_url: str | None = None
    #: vLLM's internal port (127.0.0.1) behind the sidecar.
    vllm_port: int = 8001
    #: Max retry / continuation hops of one request.
    max_depth: int = 3
    #: Gateway attempts per retry / continuation.
    retry_attempts: int = 4
    #: None = the model's vllm_image (it ships python3 + aiohttp; the script comes from
    #: a ConfigMap, so no image build is needed).
    image: str | None = None
    configmap: str = "tre-reissue-sidecar"
    #: Namespace of the model Deployments (and of the script ConfigMap).
    namespace: str = "default"
    cpu_request: str = "50m"
    cpu_limit: str = "500m"
    memory_request: str = "64Mi"
    memory_limit: str = "256Mi"
    #: Extra TRE_REISSUE_* environment for the sidecar (field / header / path names).
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GatewayConfig:
    """Registry ``gateway:`` section."""

    #: Request timeout of every model route (HTTPRoute ``timeouts.request``).
    route_timeout_s: float = DEFAULT_ROUTE_TIMEOUT_S
    #: Stable ClusterIP Service in front of the tre-v2 Envoy proxy pods (rendered by the
    #: tre-v2 overlay, kustomize param ``tre-gateway-service-params``): how in-cluster
    #: clients (the reissue sidecar) reach the gateway.
    service_name: str = DEFAULT_GATEWAY_SERVICE_NAME
    service_namespace: str = DEFAULT_GATEWAY_SERVICE_NAMESPACE
    service_port: int = DEFAULT_GATEWAY_SERVICE_PORT

    @property
    def internal_url(self) -> str:
        return gateway_service_url(self.service_name, self.service_namespace, self.service_port)


@dataclass(frozen=True)
class SleepPolicy:
    """How the service-manager puts a vLLM pod to sleep (plan D1-D4).

    hide (routable=false + route-gen bump) -> gateway ack -> drain -> /sleep.
    """

    #: Max wait for every live gateway plugin instance to ack the hide.
    ack_timeout_s: float = 10.0
    #: A plugin instance is live while its heartbeat score keeps ADVANCING: the SM
    #: saw the score change within this many seconds of its own monotonic clock.
    #: The score is never compared with any wall clock (clock-skew proof).
    instance_staleness_s: float = 10.0
    #: The hide converges only with at least this many live plugin instances. An
    #: empty live set never passes: an old plugin image (or a misconfigured
    #: coordination Redis) routes traffic without ever heartbeating, and "every
    #: live instance acked" would then be vacuously true.
    gateway_min_instances: int = 1
    #: Opt-in: with NO live plugin instance at all, fall back to waiting for the
    #: k8s label plus ``no_plugin_grace_s`` (no in-flight view from the gateway).
    #: Default off: no live instance = not converged = ack timeout = rollback.
    fallback_no_plugin: bool = False
    #: Grace delay of the opt-in no-plugin fallback, for informers to catch up.
    no_plugin_grace_s: float = 5.0
    poll_interval_s: float = 0.5
    #: HTTP timeout of one /sleep call (weight offload). A mode=wait call that
    #: fails is retried once with mode=abort, so a sleep spends up to 2x this.
    sleep_call_timeout_s: float = 45.0
    #: HTTP timeout of every other vLLM probe of a sleep (``GET /metrics``,
    #: ``/version``, ``/is_sleeping``); part of the worst-case call duration.
    probe_timeout_s: float = 5.0
    #: Allowance for the Redis / Kubernetes calls of one sleep (patches, journal,
    #: reservation renewals) in the worst-case call duration.
    io_margin_s: float = 5.0
    #: After /sleep returned, wait this long for /is_sleeping to report true.
    physical_confirm_timeout_s: float = 15.0
    #: ``auto``: probe the pod's ``GET /version`` (cached per pod) and send
    #: ``mode=wait|abort`` on /sleep only to vLLM versions that accept it;
    #: ``true`` / ``false`` force it. Without the mode parameter the SM drains
    #: fully before a plain /sleep.
    vllm_sleep_mode_param: str = "auto"
    #: Absolute drain cap; defaults to (and may not exceed) gateway.route_timeout_s.
    hard_cap_s: float = DEFAULT_ROUTE_TIMEOUT_S
    #: TTL of the per-binding sleep reservation that fences a draining binding
    #: (and its GPUs) while the drain runs outside the writer lock. Renewed every
    #: poll; the reservation of a dead owner expires after this long.
    reservation_ttl_s: float = 30.0
    #: Gateway plugin pods that must ack, besides advancing heartbeats: a plugin
    #: with Redis trouble may still route while its heartbeat stalls, so Ready pods
    #: matching this selector count as live too. None = heartbeats only.
    plugin_namespace: str = "tre-v2"
    plugin_label_selector: str | None = "app=tre-gateway-plugins"
    budgets_s: dict[str, float | None] = field(
        default_factory=lambda: dict(DEFAULT_SLEEP_BUDGETS_S)
    )

    def soft_budget_s(self, path: str, requested_s: float | None = None) -> float:
        """Soft drain budget of one sleep: the caller's budget, else the path's."""
        if requested_s is not None:
            budget = float(requested_s)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError(f"drain budget must be a finite number >= 0, got {requested_s!r}")
        else:
            key = path if path in self.budgets_s else "default"
            raw = self.budgets_s.get(key)
            budget = self.hard_cap_s if raw is None else float(raw)
        return min(budget, self.hard_cap_s)


@dataclass(frozen=True)
class ServiceManagerConfig:
    sleep: SleepPolicy = field(default_factory=SleepPolicy)
    #: Wake fails closed unless every target GPU's used memory (gpu-truth) is at
    #: most this fraction of the GPU's total memory: sleeping residents keep only
    #: a small footprint; an awake resident (or a leak) is far above it.
    wake_max_used_fraction: float = 0.2
    #: Optional absolute override (MiB) of the wake threshold; None = the fraction.
    wake_max_used_mib: int | None = None
    #: gpu-truth lags a just-finished sleep; re-read it for up to this long.
    wake_truth_wait_s: float = 5.0
    #: Startup check of the local clock against Redis TIME.
    clock_skew_warn_s: float = 1.0
    #: Refuse to start above this skew (None = warn only).
    clock_skew_fail_s: float | None = None
    #: Only nodes in cluster.nodes can block a cold start with node pressure.
    pressure_registry_nodes_only: bool = True
    #: A request that needs the SM writer lock waits up to this long for it (the
    #: lock is held only for short phases; a drain runs outside it). Waiters are
    #: served first-come first-served.
    writer_lock_wait_s: float = 10.0
    #: How long the commit phase of a drained sleep waits for the writer lock
    #: (None = writer_lock_wait_s). The sleep reservation must outlive it.
    commit_lock_wait_s: float | None = None
    #: Timeout clients (the controller) use for slow SM calls (scale / binding
    #: power / defrag). Must exceed :meth:`worst_case_sleep_call_s`; the controller
    #: uses it unless TRE_SM_SLOW_TIMEOUT_SECONDS overrides it (validated too).
    api_call_timeout_s: float = 360.0

    @property
    def commit_wait_s(self) -> float:
        return self.writer_lock_wait_s if self.commit_lock_wait_s is None else self.commit_lock_wait_s

    def worst_case_commit_s(self) -> float:
        """The commit phase once it holds the lock (targets are committed in
        parallel - sends, confirmation rounds and rollback probes alike - so this
        does not grow with the number of targets):

        * send, per target: ``/version`` probe + /sleep mode=wait +
          ``/is_sleeping`` + ``/metrics`` re-read + /sleep mode=abort +
          ``/is_sleeping``, and a failed send's rollback re-probes
          ``/is_sleeping`` once (5 probes + 2 sleeps);
        * confirmation: ``physical_confirm_timeout_s``, overshot by one poll
          interval and one probe round, then the rollback of a pod that never
          converged re-probes it once (review 3 P3: the rollback probe and the
          final-round overshoot were not counted before)."""
        sleep = self.sleep
        send = 5 * sleep.probe_timeout_s + 2 * sleep.sleep_call_timeout_s
        confirm = (
            sleep.physical_confirm_timeout_s
            + sleep.poll_interval_s
            + 2 * sleep.probe_timeout_s
        )
        return send + confirm

    def worst_case_drain_s(self) -> float:
        """Gateway ack + drain up to the hard cap + the last poll round (engine
        metrics of every target read in parallel, so one probe timeout)."""
        sleep = self.sleep
        return sleep.ack_timeout_s + sleep.hard_cap_s + sleep.probe_timeout_s

    def worst_case_sleep_call_s(self) -> float:
        """Upper bound of one sleeping SM call, for any number of targets:
        writer-lock wait (hide phase) + drain + commit-lock wait + commit +
        the Redis / Kubernetes allowance."""
        return (
            self.writer_lock_wait_s
            + self.worst_case_drain_s()
            + self.commit_wait_s
            + self.worst_case_commit_s()
            + self.sleep.io_margin_s
        )

    def shutdown_timeout_s(self) -> float:
        """How long SIGTERM waits for sleeps in progress: a drain rolls back at its
        next poll (after at most one poll round, or once its commit-lock wait
        ends), a commit already past /sleep finishes."""
        sleep = self.sleep
        return (
            self.commit_wait_s
            + self.worst_case_commit_s()
            + sleep.poll_interval_s
            + sleep.probe_timeout_s
            + sleep.io_margin_s
        )

    def wake_limit_mib(self, total_mib: int | None) -> int | None:
        """Max used MiB for a wake on a GPU of ``total_mib`` (None = unknown)."""
        if self.wake_max_used_mib is not None:
            return int(self.wake_max_used_mib)
        if total_mib is None or total_mib <= 0:
            return None
        return int(total_mib * self.wake_max_used_fraction)


class Registry:
    def __init__(
        self,
        topology: ClusterTopology,
        models: list[ModelSpec],
        service_manager: ServiceManagerConfig | None = None,
        gateway: GatewayConfig | None = None,
        reissue: ReissueConfig | None = None,
    ) -> None:
        self._topology = topology
        self._models = tuple(models)
        self._service_manager = service_manager or ServiceManagerConfig()
        self._gateway = gateway or GatewayConfig()
        self._reissue = reissue or ReissueConfig()
        self._model_index: dict[str, ModelSpec] = {}
        for model in models:
            self._model_index.setdefault(model.name, model)

    def service_manager(self) -> ServiceManagerConfig:
        return self._service_manager

    def gateway(self) -> GatewayConfig:
        return self._gateway

    def reissue(self) -> ReissueConfig:
        return self._reissue

    def model(self, name: str) -> ModelSpec:
        try:
            return self._model_index[name]
        except KeyError as exc:
            raise KeyError(f"unknown model: {name}") from exc

    def models(self) -> list[ModelSpec]:
        return list(self._models)

    def topology(self) -> ClusterTopology:
        return self._topology

    def validate_service_manager(self) -> list[str]:
        """Only the service_manager: / gateway: checks (the SM refuses to start on
        these; model / topology errors are the manifest generator's concern)."""
        return _validate_service_manager(self._service_manager, self._gateway)

    def validate(self) -> list[str]:
        errors: list[str] = []
        seen_models: set[str] = set()
        for model in self._models:
            if model.name in seen_models:
                errors.append(f"duplicate model: {model.name}")
            seen_models.add(model.name)
            if model.tp_size not in (1, 2):
                errors.append(f"model {model.name}: unsupported tp_size {model.tp_size}")
            if model.min_replicas < 0:
                errors.append(f"model {model.name}: min_replicas must be non-negative")
            if model.max_replicas < model.min_replicas:
                errors.append(f"model {model.name}: max_replicas below min_replicas")
            unknown = sorted(set(model.vllm_features) - set(VLLM_FEATURE_FLAGS))
            if unknown:
                errors.append(
                    f"model {model.name}: unknown vllm_features {unknown} "
                    f"(known: {', '.join(sorted(VLLM_FEATURE_FLAGS))})"
                )
            if model.max_awake_replicas is not None and not (
                model.min_replicas <= model.max_awake_replicas <= model.max_replicas
            ):
                errors.append(
                    f"model {model.name}: max_awake_replicas must be within [min_replicas, max_replicas]"
                )
            for signal, threshold in model.alt_thresholds.items():
                if not math.isfinite(threshold.theta) or threshold.theta <= 0.0:
                    errors.append(f"model {model.name}: alt_thresholds.{signal}.theta must be positive")
                if threshold.direction not in ALT_THRESHOLD_DIRECTIONS:
                    errors.append(
                        f"model {model.name}: alt_thresholds.{signal}.direction must be one of "
                        f"{sorted(ALT_THRESHOLD_DIRECTIONS)}"
                    )
                expected = EXPECTED_SIGNAL_DIRECTIONS.get(signal)
                if expected is not None and threshold.direction != expected:
                    errors.append(
                        f"model {model.name}: alt_thresholds.{signal}.direction must be {expected}"
                    )
                if threshold.delta_crit is not None and not (
                    math.isfinite(threshold.delta_crit) and 0.0 <= threshold.delta_crit < 1.0
                ):
                    errors.append(f"model {model.name}: alt_thresholds.{signal}.delta_crit must be in [0, 1)")
                if threshold.delta_high is not None and not (
                    math.isfinite(threshold.delta_high) and threshold.delta_high >= 0.0
                ):
                    errors.append(f"model {model.name}: alt_thresholds.{signal}.delta_high must be >= 0")

        seen_nodes: set[str] = set()
        for node in self._topology.nodes:
            if node.name in seen_nodes:
                errors.append(f"duplicate node: {node.name}")
            seen_nodes.add(node.name)
            if node.gpus <= 0:
                errors.append(f"node {node.name}: gpus must be positive")
            if len(node.gpu_uuids) != node.gpus:
                errors.append(
                    f"node {node.name}: gpu_uuids length {len(node.gpu_uuids)} does not match gpus {node.gpus}"
                )
            if len(set(node.gpu_uuids)) != len(node.gpu_uuids):
                errors.append(f"node {node.name}: duplicate gpu_uuids")
            for slot in node.two_gpu_slots:
                if len(slot) != 2:
                    errors.append(f"node {node.name}: two_gpu_slot {slot} must contain two GPUs")
                    continue
                if slot[0] == slot[1]:
                    errors.append(f"node {node.name}: two_gpu_slot {slot} duplicates a GPU")
                for gpu in slot:
                    if gpu < 0 or gpu >= node.gpus:
                        errors.append(f"node {node.name}: gpu {gpu} outside gpu range 0..{node.gpus - 1}")
        if self._topology.max_bound_per_gpu < 1:
            errors.append("cluster.max_bound_per_gpu must be >= 1")
        errors.extend(_validate_reissue(self._reissue))
        errors.extend(_validate_service_manager(self._service_manager, self._gateway))
        return errors


def load_registry(path: str | None = None) -> Registry:
    registry_path = Path(path) if path else Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return _parse_registry(raw)


def _parse_registry(raw: dict[str, Any]) -> Registry:
    cluster = raw.get("cluster") or {}
    nodes = tuple(_parse_node(item) for item in cluster.get("nodes", []))
    models = [_parse_model(item) for item in raw.get("models", [])]
    max_bound = cluster.get("max_bound_per_gpu")
    return Registry(
        ClusterTopology(
            nodes=nodes,
            max_bound_per_gpu=(
                DEFAULT_MAX_BOUND_PER_GPU if max_bound is None else int(max_bound)
            ),
        ),
        models,
        service_manager=parse_service_manager_config(
            raw.get("service_manager"), gateway=raw.get("gateway")
        ),
        gateway=parse_gateway_config(raw.get("gateway")),
        reissue=parse_reissue_config(raw.get("reissue")),
    )


def parse_reissue_config(raw: Any) -> ReissueConfig:
    """Parse the optional ``reissue:`` registry section (absent = defaults, enabled)."""
    if raw is None:
        return ReissueConfig()
    if not isinstance(raw, dict):
        raise ValueError("reissue must be a mapping")
    known = {f for f in ReissueConfig.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"reissue: unknown keys {unknown} (known: {', '.join(sorted(known))})")
    defaults = ReissueConfig()
    extra_env = raw.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        raise ValueError("reissue.extra_env must be a mapping")
    return ReissueConfig(
        enabled=_parse_bool(raw.get("enabled", defaults.enabled)),
        gateway_url=(str(raw["gateway_url"]).rstrip("/") if raw.get("gateway_url") else None),
        vllm_port=int(raw.get("vllm_port", defaults.vllm_port)),
        max_depth=int(raw.get("max_depth", defaults.max_depth)),
        retry_attempts=int(raw.get("retry_attempts", defaults.retry_attempts)),
        image=(str(raw["image"]) if raw.get("image") else None),
        configmap=str(raw.get("configmap", defaults.configmap)),
        namespace=str(raw.get("namespace", defaults.namespace)),
        cpu_request=str(raw.get("cpu_request", defaults.cpu_request)),
        cpu_limit=str(raw.get("cpu_limit", defaults.cpu_limit)),
        memory_request=str(raw.get("memory_request", defaults.memory_request)),
        memory_limit=str(raw.get("memory_limit", defaults.memory_limit)),
        extra_env={str(k): str(v) for k, v in extra_env.items()},
    )


def _validate_reissue(reissue: ReissueConfig) -> list[str]:
    errors: list[str] = []
    if not reissue.enabled:
        return errors
    if reissue.gateway_url is not None and not reissue.gateway_url.startswith(("http://", "https://")):
        errors.append("reissue.gateway_url must be an http(s) URL (in-cluster DNS name)")
    if not 1 <= reissue.vllm_port <= 65535 or reissue.vllm_port == POD_SERVING_PORT:
        errors.append(f"reissue.vllm_port must be a valid port other than {POD_SERVING_PORT}")
    if reissue.max_depth < 0:
        errors.append("reissue.max_depth must be non-negative")
    if reissue.retry_attempts < 1:
        errors.append("reissue.retry_attempts must be >= 1")
    for key in reissue.extra_env:
        if not key.startswith("TRE_"):
            errors.append(f"reissue.extra_env: {key} is not a TRE_* variable")
    return errors


def parse_gateway_config(raw: dict[str, Any] | None) -> GatewayConfig:
    """Parse the optional ``gateway:`` registry section."""
    raw = raw or {}
    timeout = raw.get("route_timeout_s")
    return GatewayConfig(
        route_timeout_s=float(DEFAULT_ROUTE_TIMEOUT_S if timeout is None else timeout),
        service_name=str(raw.get("service_name") or DEFAULT_GATEWAY_SERVICE_NAME),
        service_namespace=str(raw.get("service_namespace") or DEFAULT_GATEWAY_SERVICE_NAMESPACE),
        service_port=int(raw.get("service_port") or DEFAULT_GATEWAY_SERVICE_PORT),
    )


def parse_service_manager_config(
    raw: dict[str, Any] | None, *, gateway: dict[str, Any] | None = None
) -> ServiceManagerConfig:
    """Parse the optional ``service_manager:`` registry section (all keys optional).

    ``sleep.hard_cap_s`` defaults to ``gateway.route_timeout_s`` (the gateway's
    request timeout: no request can outlive it anyway), else 150 s.
    """
    raw = raw or {}
    sleep_raw = raw.get("sleep") or {}
    wake_raw = raw.get("wake") or {}
    skew_raw = raw.get("clock_skew") or {}
    pressure_raw = raw.get("node_pressure") or {}
    defaults = SleepPolicy()
    plugin_pods_raw = sleep_raw.get("gateway_plugin_pods") or {}
    hard_cap = sleep_raw.get("hard_cap_s")
    if hard_cap is None:
        hard_cap = (gateway or {}).get("route_timeout_s")
    budgets = dict(DEFAULT_SLEEP_BUDGETS_S)
    for path, value in (sleep_raw.get("budgets_s") or {}).items():
        if str(path) not in SLEEP_PATHS:
            raise ValueError(
                f"service_manager.sleep.budgets_s: unknown sleep path {path!r} "
                f"(known: {', '.join(SLEEP_PATHS)})"
            )
        budgets[str(path)] = None if value is None else float(value)
    sleep = SleepPolicy(
        ack_timeout_s=_num(sleep_raw, "ack_timeout_s", defaults.ack_timeout_s),
        instance_staleness_s=_num(sleep_raw, "instance_staleness_s", defaults.instance_staleness_s),
        gateway_min_instances=int(
            _num(sleep_raw, "gateway_min_instances", defaults.gateway_min_instances)
        ),
        fallback_no_plugin=_parse_bool(
            sleep_raw.get("fallback_no_plugin", defaults.fallback_no_plugin)
        ),
        no_plugin_grace_s=_num(sleep_raw, "no_plugin_grace_s", defaults.no_plugin_grace_s),
        poll_interval_s=_num(sleep_raw, "poll_interval_s", defaults.poll_interval_s),
        sleep_call_timeout_s=_num(
            sleep_raw, "sleep_call_timeout_s", defaults.sleep_call_timeout_s
        ),
        physical_confirm_timeout_s=_num(
            sleep_raw, "physical_confirm_timeout_s", defaults.physical_confirm_timeout_s
        ),
        probe_timeout_s=_num(sleep_raw, "probe_timeout_s", defaults.probe_timeout_s),
        io_margin_s=_num(sleep_raw, "io_margin_s", defaults.io_margin_s),
        vllm_sleep_mode_param=parse_sleep_mode_param(
            sleep_raw.get("vllm_sleep_mode_param", defaults.vllm_sleep_mode_param)
        ),
        hard_cap_s=float(DEFAULT_ROUTE_TIMEOUT_S if hard_cap is None else hard_cap),
        reservation_ttl_s=_num(sleep_raw, "reservation_ttl_s", defaults.reservation_ttl_s),
        budgets_s=budgets,
        plugin_namespace=str(plugin_pods_raw.get("namespace", defaults.plugin_namespace)),
        plugin_label_selector=plugin_pods_raw.get(
            "label_selector", defaults.plugin_label_selector
        ),
    )
    base = ServiceManagerConfig()
    fail_s = skew_raw.get("fail_s", base.clock_skew_fail_s)
    max_used_mib = wake_raw.get("max_used_mib", base.wake_max_used_mib)
    return ServiceManagerConfig(
        sleep=sleep,
        wake_max_used_fraction=_num(wake_raw, "max_used_fraction", base.wake_max_used_fraction),
        wake_max_used_mib=None if max_used_mib is None else int(max_used_mib),
        wake_truth_wait_s=_num(wake_raw, "truth_wait_s", base.wake_truth_wait_s),
        clock_skew_warn_s=_num(skew_raw, "warn_s", base.clock_skew_warn_s),
        clock_skew_fail_s=None if fail_s is None else float(fail_s),
        pressure_registry_nodes_only=_parse_bool(
            pressure_raw.get("registry_nodes_only", base.pressure_registry_nodes_only)
        ),
        writer_lock_wait_s=_num(raw, "writer_lock_wait_s", base.writer_lock_wait_s),
        commit_lock_wait_s=(
            None if raw.get("commit_lock_wait_s") is None else float(raw["commit_lock_wait_s"])
        ),
        api_call_timeout_s=_num(raw, "api_call_timeout_s", base.api_call_timeout_s),
    )


def parse_sleep_mode_param(value: Any) -> str:
    """``auto`` | ``true`` | ``false`` (YAML booleans accepted)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return "true" if _parse_bool(text) else "false"


def _num(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key)
    return float(default if value is None else value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _validate_service_manager(
    config: ServiceManagerConfig, gateway: GatewayConfig | None = None
) -> list[str]:
    errors: list[str] = []
    sleep = config.sleep
    for name in (
        "ack_timeout_s",
        "instance_staleness_s",
        "poll_interval_s",
        "sleep_call_timeout_s",
        "physical_confirm_timeout_s",
        "hard_cap_s",
        "reservation_ttl_s",
        "probe_timeout_s",
    ):
        value = float(getattr(sleep, name))
        if not math.isfinite(value) or value <= 0:
            errors.append(f"service_manager.sleep.{name} must be positive")
    if not math.isfinite(sleep.io_margin_s) or sleep.io_margin_s < 0:
        errors.append("service_manager.sleep.io_margin_s must be >= 0")
    if not math.isfinite(sleep.no_plugin_grace_s) or sleep.no_plugin_grace_s < 0:
        errors.append("service_manager.sleep.no_plugin_grace_s must be >= 0")
    if sleep.gateway_min_instances < 1:
        errors.append("service_manager.sleep.gateway_min_instances must be >= 1")
    if sleep.vllm_sleep_mode_param not in SLEEP_MODE_PARAM_CHOICES:
        errors.append(
            "service_manager.sleep.vllm_sleep_mode_param must be one of "
            f"{', '.join(SLEEP_MODE_PARAM_CHOICES)}"
        )
    if sleep.reservation_ttl_s <= 2 * sleep.poll_interval_s:
        errors.append(
            "service_manager.sleep.reservation_ttl_s must exceed 2 x poll_interval_s "
            "(the reservation is renewed once per poll)"
        )
    # Review 2 P2-1: the longest gap between two renewals is the last drain poll
    # round (every target's engine metrics, read in parallel) followed by the wait
    # for the commit-phase writer lock; the reservation must survive it, or the
    # commit finds it lost and rolls back.
    renew_gap = (
        config.commit_wait_s + sleep.poll_interval_s + sleep.probe_timeout_s + sleep.io_margin_s
    )
    if sleep.reservation_ttl_s <= renew_gap:
        errors.append(
            f"service_manager.sleep.reservation_ttl_s ({sleep.reservation_ttl_s:g}) must "
            f"exceed the longest renewal gap {renew_gap:g}s (commit-lock wait + "
            "poll_interval_s + probe_timeout_s + io_margin_s)"
        )
    for path, value in sleep.budgets_s.items():
        if path not in SLEEP_PATHS:
            errors.append(f"service_manager.sleep.budgets_s: unknown sleep path {path}")
        elif value is not None and (not math.isfinite(value) or value < 0):
            errors.append(f"service_manager.sleep.budgets_s.{path} must be >= 0 or null")
    if gateway is not None:
        if not math.isfinite(gateway.route_timeout_s) or gateway.route_timeout_s <= 0:
            errors.append("gateway.route_timeout_s must be positive")
        elif sleep.hard_cap_s > gateway.route_timeout_s:
            errors.append(
                f"service_manager.sleep.hard_cap_s ({sleep.hard_cap_s:g}) must not exceed "
                f"gateway.route_timeout_s ({gateway.route_timeout_s:g}): no request "
                "outlives the route timeout"
            )
    if not (0.0 < config.wake_max_used_fraction <= 1.0):
        errors.append("service_manager.wake.max_used_fraction must be in (0, 1]")
    if config.wake_max_used_mib is not None and config.wake_max_used_mib <= 0:
        errors.append("service_manager.wake.max_used_mib must be positive or null")
    if config.wake_truth_wait_s < 0:
        errors.append("service_manager.wake.truth_wait_s must be >= 0")
    if config.clock_skew_warn_s <= 0:
        errors.append("service_manager.clock_skew.warn_s must be positive")
    if config.clock_skew_fail_s is not None and config.clock_skew_fail_s <= 0:
        errors.append("service_manager.clock_skew.fail_s must be positive or null")
    if config.writer_lock_wait_s < 0:
        errors.append("service_manager.writer_lock_wait_s must be >= 0")
    if config.commit_lock_wait_s is not None and config.commit_lock_wait_s < 0:
        errors.append("service_manager.commit_lock_wait_s must be >= 0 or null")
    errors.extend(sleep_call_timeout_errors(config, config.api_call_timeout_s))
    return errors


def sleep_call_timeout_errors(
    config: ServiceManagerConfig, call_timeout_s: float, *, name: str = "api_call_timeout_s"
) -> list[str]:
    """Empty when a client timeout of ``call_timeout_s`` outlasts the worst-case
    sleeping SM call; used by the registry validation and the controller."""
    worst = config.worst_case_sleep_call_s()
    if worst < call_timeout_s:
        return []
    return [
        f"worst-case sleeping service-manager call is {worst:g}s (writer_lock_wait_s + "
        "sleep.ack_timeout_s + sleep.hard_cap_s + commit-lock wait + 2 x "
        "sleep.sleep_call_timeout_s + 8 x sleep.probe_timeout_s + "
        "sleep.physical_confirm_timeout_s + sleep.poll_interval_s + "
        "sleep.io_margin_s), not below "
        f"{name} = {call_timeout_s:g}s: the caller would time out mid-drain"
    ]


def _parse_node(raw: dict[str, Any]) -> NodeSpec:
    slots = tuple(tuple(int(gpu) for gpu in slot) for slot in raw.get("two_gpu_slots", []))
    gpu_uuids = tuple(str(uuid) for uuid in raw.get("gpu_uuids", []))
    return NodeSpec(name=str(raw["name"]), gpus=int(raw["gpus"]), two_gpu_slots=slots, gpu_uuids=gpu_uuids)  # type: ignore[arg-type]


LOG = logging.getLogger(__name__)


def _parse_model(raw: dict[str, Any]) -> ModelSpec:
    slo = raw.get("slo") or {}
    trs = raw.get("trs") or {}
    if float(trs.get("w_d", 1.0)) != 1.0:
        # Schema-compatible, but the unified TSS (tre_common.tss) fixes w_d = 1.
        LOG.warning(
            "model %s: trs.w_d=%s is ignored; the unified TSS fixes w_d = 1",
            raw.get("name"), trs.get("w_d"),
        )
    return ModelSpec(
        name=str(raw["name"]),
        weights_path=str(raw["weights_path"]),
        tp_size=int(raw["tp_size"]),
        min_replicas=int(raw["min_replicas"]),
        max_replicas=int(raw["max_replicas"]),
        max_awake_replicas=(
            int(raw["max_awake_replicas"]) if raw.get("max_awake_replicas") is not None else None
        ),
        vllm_image=str(raw["vllm_image"]),
        slo=SloSpec(
            ttft_p95_ms=float(slo["ttft_p95_ms"]),
            tpot_p95_ms=float(slo["tpot_p95_ms"]),
            e2e_p95_ms=float(slo["e2e_p95_ms"]),
            ttft_idle_c_ms=(float(slo["ttft_idle_c_ms"]) if slo.get("ttft_idle_c_ms") is not None else None),
            ttft_idle_b_ms_per_token=(
                float(slo["ttft_idle_b_ms_per_token"])
                if slo.get("ttft_idle_b_ms_per_token") is not None
                else None
            ),
            ttft_slo_mode=(str(slo["ttft_slo_mode"]) if slo.get("ttft_slo_mode") is not None else None),
            ttft_slowdown_k=(float(slo["ttft_slowdown_k"]) if slo.get("ttft_slowdown_k") is not None else None),
            ttft_floor_ms=(float(slo["ttft_floor_ms"]) if slo.get("ttft_floor_ms") is not None else None),
        ),
        trs=TrsParams(
            w_p=float(trs["w_p"]),
            w_d=float(trs["w_d"]),
            lambda_wait=float(trs["lambda_wait"]),
            qmin=float(trs["qmin"]),
            ema_alpha=float(trs["ema_alpha"]),
            theta_m=float(trs.get("theta_m", 0.0)),
            tau_crit=float(trs["tau_crit"]),
            tau_low=float(trs["tau_low"]),
            tau_high=float(trs["tau_high"]),
            qsat=float(trs["qsat"]),
            epsat=float(trs["epsat"]),
            hsat=int(trs["hsat"]),
            ema_tau_ms=(float(trs["ema_tau_ms"]) if trs.get("ema_tau_ms") is not None else None),
        ),
        vllm_extra_args=tuple(str(arg) for arg in raw.get("vllm_extra_args", [])),
        vllm_features=tuple(str(feature) for feature in raw.get("vllm_features") or ()),
        alt_thresholds={
            str(signal): AltThreshold(
                theta=float(values["theta"]),
                direction=str(values["direction"]),
                delta_crit=(float(values["delta_crit"]) if values.get("delta_crit") is not None else None),
                delta_high=(float(values["delta_high"]) if values.get("delta_high") is not None else None),
            )
            for signal, values in (raw.get("alt_thresholds") or {}).items()
        },
    )
