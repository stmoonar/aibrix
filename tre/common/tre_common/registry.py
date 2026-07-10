from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SloSpec:
    ttft_p95_ms: float
    tpot_p95_ms: float
    e2e_p95_ms: float


ALT_THRESHOLD_DIRECTIONS = {"higher_is_healthier", "lower_is_healthier"}
EXPECTED_SIGNAL_DIRECTIONS = {
    "queue_len": "lower_is_healthier",
    "decode_tps": "lower_is_healthier",
    "prefill_tps": "lower_is_healthier",
}


@dataclass(frozen=True)
class AltThreshold:
    theta: float
    direction: str


@dataclass(frozen=True)
class TrsParams:
    w_p: float
    w_d: float
    lambda_wait: float
    qmin: float
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


def _parse_model(raw: dict[str, Any]) -> ModelSpec:
    slo = raw.get("slo") or {}
    trs = raw.get("trs") or {}
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
            )
            for signal, values in (raw.get("alt_thresholds") or {}).items()
        },
    )
