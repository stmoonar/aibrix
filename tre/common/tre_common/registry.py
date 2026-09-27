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


@dataclass(frozen=True)
class ClusterTopology:
    nodes: tuple[NodeSpec, ...]


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

#: Fallback hard cap when neither sleep.hard_cap_s nor gateway.route_timeout_s is set.
DEFAULT_ROUTE_TIMEOUT_S = 150.0


@dataclass(frozen=True)
class SleepPolicy:
    """How the service-manager puts a vLLM pod to sleep (plan D1-D4).

    hide (routable=false + route-gen bump) -> gateway ack -> drain -> /sleep.
    """

    #: Max wait for every live gateway plugin instance to ack the hide.
    ack_timeout_s: float = 10.0
    #: A plugin instance whose heartbeat is older than this is not live.
    instance_staleness_s: float = 10.0
    #: No live plugin instance (plugin absent / fallback routing): after the k8s
    #: label is off, wait this long for informers to catch up, and log a warning.
    no_plugin_grace_s: float = 5.0
    poll_interval_s: float = 0.5
    #: HTTP timeout of the /sleep call itself (weight offload), on top of the drain.
    sleep_call_timeout_s: float = 60.0
    #: Send ``mode=wait|abort`` on /sleep (vLLM >= 0.30). False for images whose
    #: /sleep takes no mode: the SM then drains and calls a plain /sleep.
    vllm_sleep_mode_param: bool = True
    #: Absolute drain cap, normally the gateway route timeout.
    hard_cap_s: float = DEFAULT_ROUTE_TIMEOUT_S
    #: Gateway plugin pods that must ack, besides fresh heartbeats: a plugin with
    #: Redis trouble may still route while its heartbeat goes stale, so Ready pods
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
    #: most this: sleeping residents keep only a small footprint; an awake
    #: resident (or a leak) is far above it.
    wake_max_used_mib: int = 8192
    #: gpu-truth lags a just-finished sleep; re-read it for up to this long.
    wake_truth_wait_s: float = 5.0
    #: Startup check of the local clock against Redis TIME.
    clock_skew_warn_s: float = 1.0
    #: Refuse to start above this skew (None = warn only).
    clock_skew_fail_s: float | None = None
    #: Only nodes in cluster.nodes can block a cold start with node pressure.
    pressure_registry_nodes_only: bool = True


class Registry:
    def __init__(
        self,
        topology: ClusterTopology,
        models: list[ModelSpec],
        service_manager: ServiceManagerConfig | None = None,
    ) -> None:
        self._topology = topology
        self._models = tuple(models)
        self._service_manager = service_manager or ServiceManagerConfig()
        self._model_index: dict[str, ModelSpec] = {}
        for model in models:
            self._model_index.setdefault(model.name, model)

    def service_manager(self) -> ServiceManagerConfig:
        return self._service_manager

    def model(self, name: str) -> ModelSpec:
        try:
            return self._model_index[name]
        except KeyError as exc:
            raise KeyError(f"unknown model: {name}") from exc

    def models(self) -> list[ModelSpec]:
        return list(self._models)

    def topology(self) -> ClusterTopology:
        return self._topology

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
        errors.extend(_validate_service_manager(self._service_manager))
        return errors


def load_registry(path: str | None = None) -> Registry:
    registry_path = Path(path) if path else Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return _parse_registry(raw)


def _parse_registry(raw: dict[str, Any]) -> Registry:
    cluster = raw.get("cluster") or {}
    nodes = tuple(_parse_node(item) for item in cluster.get("nodes", []))
    models = [_parse_model(item) for item in raw.get("models", [])]
    return Registry(
        ClusterTopology(nodes=nodes),
        models,
        service_manager=parse_service_manager_config(
            raw.get("service_manager"), gateway=raw.get("gateway")
        ),
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
        ack_timeout_s=float(sleep_raw.get("ack_timeout_s", defaults.ack_timeout_s)),
        instance_staleness_s=float(
            sleep_raw.get("instance_staleness_s", defaults.instance_staleness_s)
        ),
        no_plugin_grace_s=float(sleep_raw.get("no_plugin_grace_s", defaults.no_plugin_grace_s)),
        poll_interval_s=float(sleep_raw.get("poll_interval_s", defaults.poll_interval_s)),
        sleep_call_timeout_s=float(
            sleep_raw.get("sleep_call_timeout_s", defaults.sleep_call_timeout_s)
        ),
        vllm_sleep_mode_param=_parse_bool(
            sleep_raw.get("vllm_sleep_mode_param", defaults.vllm_sleep_mode_param)
        ),
        hard_cap_s=float(DEFAULT_ROUTE_TIMEOUT_S if hard_cap is None else hard_cap),
        budgets_s=budgets,
        plugin_namespace=str(plugin_pods_raw.get("namespace", defaults.plugin_namespace)),
        plugin_label_selector=plugin_pods_raw.get(
            "label_selector", defaults.plugin_label_selector
        ),
    )
    base = ServiceManagerConfig()
    fail_s = skew_raw.get("fail_s", base.clock_skew_fail_s)
    return ServiceManagerConfig(
        sleep=sleep,
        wake_max_used_mib=int(wake_raw.get("max_used_mib", base.wake_max_used_mib)),
        wake_truth_wait_s=float(wake_raw.get("truth_wait_s", base.wake_truth_wait_s)),
        clock_skew_warn_s=float(skew_raw.get("warn_s", base.clock_skew_warn_s)),
        clock_skew_fail_s=None if fail_s is None else float(fail_s),
        pressure_registry_nodes_only=_parse_bool(
            pressure_raw.get("registry_nodes_only", base.pressure_registry_nodes_only)
        ),
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _validate_service_manager(config: ServiceManagerConfig) -> list[str]:
    errors: list[str] = []
    sleep = config.sleep
    for name in ("ack_timeout_s", "instance_staleness_s", "poll_interval_s", "sleep_call_timeout_s", "hard_cap_s"):
        value = float(getattr(sleep, name))
        if not math.isfinite(value) or value <= 0:
            errors.append(f"service_manager.sleep.{name} must be positive")
    if not math.isfinite(sleep.no_plugin_grace_s) or sleep.no_plugin_grace_s < 0:
        errors.append("service_manager.sleep.no_plugin_grace_s must be >= 0")
    for path, value in sleep.budgets_s.items():
        if path not in SLEEP_PATHS:
            errors.append(f"service_manager.sleep.budgets_s: unknown sleep path {path}")
        elif value is not None and (not math.isfinite(value) or value < 0):
            errors.append(f"service_manager.sleep.budgets_s.{path} must be >= 0 or null")
    if config.wake_max_used_mib <= 0:
        errors.append("service_manager.wake.max_used_mib must be positive")
    if config.wake_truth_wait_s < 0:
        errors.append("service_manager.wake.truth_wait_s must be >= 0")
    if config.clock_skew_warn_s <= 0:
        errors.append("service_manager.clock_skew.warn_s must be positive")
    if config.clock_skew_fail_s is not None and config.clock_skew_fail_s <= 0:
        errors.append("service_manager.clock_skew.fail_s must be positive or null")
    return errors


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
