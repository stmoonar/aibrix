"""Shell configuration: environment + policy ConfigMap + the shared TRE registry.

Every endpoint and environment-specific value comes from the environment (nothing about
the cluster is written in code). Per-model bounds and SLOs come from the registry
(``TRE_REGISTRY_PATH``, the ``tre-v2-registry`` ConfigMap mounted at ``/etc/tre``) through
the existing readers in ``tre_common``:

* ``min_replicas`` / scaling cap: ``ModelSpec.min_replicas`` and
  ``ModelSpec.scale_max_replicas`` (``max_awake_replicas`` else ``max_replicas``, the same
  cap the service manager enforces on ``PUT /v2/models/{m}/target``);
* GPUs per replica: ``ModelSpec.tp_size``;
* SLO: ``tre_common.slo_labels.label_def_for_model`` (the one SLO definition calibration
  labels use) fed with the registry's fixed ``slo.ttft_p95_ms`` / ``slo.tpot_p95_ms``;
* ``--max-num-seqs``: ``scripts.admission_cap.max_num_seqs_from_args`` over the model's
  engine arguments (the reader the admission-cap tooling uses).

Environment:

==========================  ==========================================================
``TRE_SM_URL``              service manager base URL (required)
``TRE_REDIS_URL``           Redis URL (required)
``TRE_BL_POLICY``           policy name, a key of ``tre_baselines.policies.POLICIES``
``TRE_BL_DRY_RUN``          ``true`` (default): decide and log, never call the SM
``TRE_BL_TICK_S``           tick period in seconds (default 2)
``TRE_BL_LOG_DIR``          decision JSONL directory (default ``./bl-logs``)
``TRE_BL_POLICY_CONFIG``    YAML file with the policy's parameters (optional)
``TRE_REGISTRY_PATH``       registry YAML (default ``/etc/tre/registry.yaml``)
``TRE_BL_WRITE_REDIS``      write ``tre:v2:bl:decision:<model>`` (default true)
``TRE_BL_DECISION_STREAM``  also XADD every decision line to ``tre:v2:bl:decisions``
                            (MAXLEN ~ 100000; default true)
``TRE_BL_MODELS``           comma list restricting the models acted on (default: all)
``TRE_MODEL_NAMESPACE``     namespace of the model pods (default ``default``)
``TRE_BL_METRICS_PORT``     pod port serving ``/metrics``; unset = the pod's
                            ``model.aibrix.ai/port`` label, else 8000
``TRE_BL_SCRAPE_TIMEOUT_S`` per-pod scrape timeout (default 2.5)
``TRE_BL_SM_TIMEOUT_S``     SM target-call timeout (default 300, the controller's)
``TRE_BL_SM_STATE_TIMEOUT_S`` SM ``/v2/state`` timeout (default 5)
``TRE_BL_ABORT_SLEEP_PATH`` ``sleep_path`` of scale-downs (default ``urgent``): must be one
                            of the registry's ``service_manager.sleep.no_drain_paths`` -
                            every arm sleeps the same way (hide, gateway ack, ``/sleep
                            mode=abort``, sidecar continuation; no drain). The old
                            ``TRE_BL_SLEEP_PATH`` / ``TRE_BL_DRAIN_BUDGET_S`` are refused.
``TRE_BL_MAX_TICK_FAILURES`` consecutive failed ticks before /healthz is 503 (default 5)
``TRE_BL_LOCK_TTL_S``       owner-lock TTL (default 30)
``TRE_BL_BACKOFF_MAX_S``    cap of the per-model backoff after SM refusals (default 10; a
                            refusal is retried at once when the SM state version changes)
``TRE_BL_LIVENESS_STALL_S`` ``/livez`` fails when the loop has not ticked for this long
                            (default 120)
``TRE_BL_HTTP_PORT``        /healthz + /metrics port (default 8080)
``TRE_BL_SEED``             seed handed to policies (default 0)
==========================  ==========================================================
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from tre_common.registry import Registry, load_registry
from tre_common.slo_labels import TTFT_SLO_MODE_FIXED, label_def_for_model

from tre_baselines.sm_client import DEFAULT_ABORT_SLEEP_PATH

LOG = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "/etc/tre/registry.yaml"
#: Sleep paths the service manager accepts from an HTTP caller
#: (``tre_sm.api.v2.EXTERNAL_SLEEP_PATHS``; not imported: the shell must not depend on
#: the service-manager package).
EXTERNAL_SLEEP_PATHS = ("scale_down", "urgent", "safescale_commit", "apa")
#: Environment names of the old drain semantics; refused so a stale manifest cannot
#: silently ask for a drain the shell no longer claims.
RETIRED_ENV = ("TRE_BL_SLEEP_PATH", "TRE_BL_DRAIN_BUDGET_S")


@dataclass(frozen=True)
class ModelLimits:
    name: str
    min_replicas: int
    max_replicas: int
    gpus_per_replica: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    max_num_seqs: Optional[int]
    #: ``tre_common.slo_labels.LabelDefinition`` of the model (per-request TTFT SLO).
    slo: Any = None


@dataclass(frozen=True)
class Config:
    sm_url: str
    redis_url: str
    policy: str
    dry_run: bool = True
    tick_s: float = 2.0
    log_dir: str = "./bl-logs"
    policy_config_path: Optional[str] = None
    policy_params: Mapping[str, Any] = field(default_factory=dict)
    registry_path: str = DEFAULT_REGISTRY_PATH
    write_redis: bool = True
    decision_stream: bool = True
    models: Mapping[str, ModelLimits] = field(default_factory=dict)
    model_namespace: str = "default"
    metrics_port: Optional[int] = None
    scrape_timeout_s: float = 2.5
    sm_timeout_s: float = 300.0
    sm_state_timeout_s: float = 5.0
    #: Scale-down sleep path: a no-drain (abort) path (see ``sm_client``).
    abort_sleep_path: str = DEFAULT_ABORT_SLEEP_PATH
    max_tick_failures: int = 5
    lock_ttl_s: float = 30.0
    backoff_max_s: float = 10.0
    liveness_stall_s: float = 120.0
    http_port: int = 8080
    seed: int = 0


def parse_bool(value: Any, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _max_num_seqs(args: tuple[str, ...]) -> Optional[int]:
    from scripts.admission_cap import max_num_seqs_from_args

    return max_num_seqs_from_args(args)


def _slo_definition(model: str, spec: Any, registry: Registry) -> Any:
    """The model's SLO definition via the shared label reader; falls back to the fixed arm
    when the registry has no idle-TTFT fit (the slowdown arm needs one)."""
    try:
        return label_def_for_model(
            model,
            ttft_p95_ms=spec.slo.ttft_p95_ms,
            tpot_p95_ms=spec.slo.tpot_p95_ms,
            registry=registry,
        )
    except (SystemExit, ValueError) as exc:
        LOG.warning("model %s: slowdown SLO unavailable (%s); using the fixed arm", model, exc)
        return label_def_for_model(
            model,
            ttft_p95_ms=spec.slo.ttft_p95_ms,
            tpot_p95_ms=spec.slo.tpot_p95_ms,
            mode=TTFT_SLO_MODE_FIXED,
            registry=registry,
        )


def model_limits(registry: Registry, only: Optional[set[str]] = None) -> dict[str, ModelLimits]:
    out: dict[str, ModelLimits] = {}
    for spec in registry.models():
        if only and spec.name not in only:
            continue
        slo = _slo_definition(spec.name, spec, registry)
        out[spec.name] = ModelLimits(
            name=spec.name,
            min_replicas=int(spec.min_replicas),
            max_replicas=int(spec.scale_max_replicas),
            gpus_per_replica=int(spec.tp_size),
            ttft_slo_ms=float(slo.ttft_p95_ms),
            tpot_slo_ms=float(slo.tpot_p95_ms),
            max_num_seqs=_max_num_seqs(tuple(spec.vllm_args)),
            slo=slo,
        )
    if only:
        unknown = only - set(out)
        if unknown:
            raise ValueError(f"TRE_BL_MODELS names models not in the registry: {sorted(unknown)}")
    return out


def load_policy_params(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOG.warning("policy config %s does not exist; policy gets no parameters", path)
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"policy config {path} must be a mapping")
    return raw


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def load_config(env: Optional[Mapping[str, str]] = None, registry: Optional[Registry] = None) -> Config:
    env = dict(os.environ if env is None else env)
    registry_path = env.get("TRE_REGISTRY_PATH", "").strip() or DEFAULT_REGISTRY_PATH
    if registry is None:
        registry = load_registry(registry_path)
    only = {m.strip() for m in env.get("TRE_BL_MODELS", "").split(",") if m.strip()} or None
    retired = [k for k in RETIRED_ENV if env.get(k, "").strip()]
    if retired:
        raise ValueError(f"{retired} are retired (the shell never drains): use TRE_BL_ABORT_SLEEP_PATH")
    abort_path = env.get("TRE_BL_ABORT_SLEEP_PATH", "").strip() or DEFAULT_ABORT_SLEEP_PATH
    if abort_path not in EXTERNAL_SLEEP_PATHS:
        raise ValueError(f"TRE_BL_ABORT_SLEEP_PATH {abort_path!r} not in {EXTERNAL_SLEEP_PATHS}")
    if not registry.service_manager().sleep.no_drain(abort_path):
        raise ValueError(f"TRE_BL_ABORT_SLEEP_PATH {abort_path!r} drains on this registry "
                         f"(service_manager.sleep.no_drain_paths="
                         f"{list(registry.service_manager().sleep.no_drain_paths)})")
    tick_s = float(env.get("TRE_BL_TICK_S", "").strip() or 2.0)
    if tick_s <= 0:
        raise ValueError("TRE_BL_TICK_S must be > 0")
    port = env.get("TRE_BL_METRICS_PORT", "").strip()
    policy_config_path = env.get("TRE_BL_POLICY_CONFIG", "").strip() or None
    return Config(
        sm_url=_required(env, "TRE_SM_URL").rstrip("/"),
        redis_url=_required(env, "TRE_REDIS_URL"),
        policy=_required(env, "TRE_BL_POLICY"),
        dry_run=parse_bool(env.get("TRE_BL_DRY_RUN"), True),
        tick_s=tick_s,
        log_dir=env.get("TRE_BL_LOG_DIR", "").strip() or "./bl-logs",
        policy_config_path=policy_config_path,
        policy_params=load_policy_params(policy_config_path),
        registry_path=registry_path,
        write_redis=parse_bool(env.get("TRE_BL_WRITE_REDIS"), True),
        decision_stream=parse_bool(env.get("TRE_BL_DECISION_STREAM"), True),
        models=model_limits(registry, only),
        model_namespace=env.get("TRE_MODEL_NAMESPACE", "").strip() or "default",
        metrics_port=int(port) if port else None,
        scrape_timeout_s=float(env.get("TRE_BL_SCRAPE_TIMEOUT_S", "").strip() or 2.5),
        sm_timeout_s=float(env.get("TRE_BL_SM_TIMEOUT_S", "").strip() or 300.0),
        sm_state_timeout_s=float(env.get("TRE_BL_SM_STATE_TIMEOUT_S", "").strip() or 5.0),
        abort_sleep_path=abort_path,
        max_tick_failures=int(env.get("TRE_BL_MAX_TICK_FAILURES", "").strip() or 5),
        lock_ttl_s=float(env.get("TRE_BL_LOCK_TTL_S", "").strip() or 30.0),
        backoff_max_s=float(env.get("TRE_BL_BACKOFF_MAX_S", "").strip() or 10.0),
        liveness_stall_s=float(env.get("TRE_BL_LIVENESS_STALL_S", "").strip() or 120.0),
        http_port=int(env.get("TRE_BL_HTTP_PORT", "").strip() or 8080),
        seed=int(env.get("TRE_BL_SEED", "").strip() or 0),
    )
