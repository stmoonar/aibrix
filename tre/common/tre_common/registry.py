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
    max_replicas: int
    vllm_image: str
    slo: SloSpec
    trs: TrsParams
    vllm_extra_args: tuple[str, ...] = ()
    alt_thresholds: dict[str, AltThreshold] = field(default_factory=dict)
    # Utilisation-gated scale-down (TRE_UTIL_SCALE_DOWN): max per-replica in-flight load
    # (avg_running + avg_waiting) after removing one replica. Optional; None -> controller
    # default. Registry key: models[].scale_down_q_per_replica.
    scale_down_q_per_replica: float | None = None


class Registry:
    def __init__(self, topology: ClusterTopology, models: list[ModelSpec]) -> None:
        self._topology = topology
        self._models = tuple(models)
        self._model_index: dict[str, ModelSpec] = {}
        for model in models:
            self._model_index.setdefault(model.name, model)

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
            if model.scale_down_q_per_replica is not None and not (
                math.isfinite(model.scale_down_q_per_replica) and model.scale_down_q_per_replica > 0.0
            ):
                errors.append(f"model {model.name}: scale_down_q_per_replica must be positive")
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
        return errors


def load_registry(path: str | None = None) -> Registry:
    registry_path = Path(path) if path else Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return _parse_registry(raw)


def _parse_registry(raw: dict[str, Any]) -> Registry:
    cluster = raw.get("cluster") or {}
    nodes = tuple(_parse_node(item) for item in cluster.get("nodes", []))
    models = [_parse_model(item) for item in raw.get("models", [])]
    return Registry(ClusterTopology(nodes=nodes), models)


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
        scale_down_q_per_replica=(
            float(raw["scale_down_q_per_replica"])
            if raw.get("scale_down_q_per_replica") is not None
            else None
        ),
    )
