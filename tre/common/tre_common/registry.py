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

#: Widest tensor-parallel size the binding layout can place: a model binds single
#: GPUs (tp 1) or a node's declared ``two_gpu_slots`` (tp 2) - see
#: ``tre_common.bindings.feasible_slots`` and the SM ``SlotAllocator``. The buddy
#: placement (``tre_common.gpu_placement``) handles any power of two; widening this
#: needs a wider slot declaration in the topology first.
MAX_SUPPORTED_TP_SIZE = 2


def tp_size_error(
    tp_size: object,
    *,
    widest_node_gpus: int | None = None,
    max_tp_size: int | None = MAX_SUPPORTED_TP_SIZE,
) -> str | None:
    """Why ``tp_size`` is not a valid model tp_size, or None. The single rule the
    registry (load + validate), the SM allocator, the manifest generator and the
    placement policy of the controller and the SM all apply: an int, a power of
    two >= 1, not wider than the widest node (tensor parallelism cannot span
    nodes) and at most :data:`MAX_SUPPORTED_TP_SIZE` (``max_tp_size=None``: the
    generic buddy rule only, for the placement library that ranks wider blocks
    too)."""
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        return f"tp_size must be an int, got {tp_size!r}"
    if tp_size < 1 or tp_size & (tp_size - 1):
        return f"tp_size {tp_size} must be a power of two >= 1"
    if widest_node_gpus is not None and tp_size > widest_node_gpus:
        return (
            f"tp_size {tp_size} exceeds the widest node ({widest_node_gpus} gpus); "
            "tensor parallelism cannot span nodes"
        )
    if max_tp_size is not None and tp_size > max_tp_size:
        return (
            f"tp_size {tp_size} is not supported by the binding layout (max "
            f"{max_tp_size}: single GPUs or cluster.nodes[].two_gpu_slots)"
        )
    return None


def _widest_node_gpus(nodes) -> int | None:
    widths = [int(node.gpus) for node in nodes]
    return max(widths) if widths else None


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
    #: ``--max-model-len`` (models[].max_model_len); None = the model's own maximum.
    max_model_len: int | None = None
    #: ``--sleep-mode-backend`` (models[].sleep_mode_backend, ``SLEEP_MODE_BACKENDS``);
    #: None = no flag, i.e. vLLM's default (cumem).
    sleep_mode_backend: str | None = None
    #: Per-model vLLM container environment (models[].vllm_env), merged over the
    #: registry-wide ``vllm.env`` (see ``Registry.vllm_env_for``).
    vllm_env: dict[str, str] = field(default_factory=dict)

    @property
    def scale_max_replicas(self) -> int:
        return scale_max_replicas(self)

    @property
    def gpu_memory_utilization(self) -> float:
        """The engine's ``--gpu-memory-utilization`` (see :func:`gpu_memory_utilization`)."""
        return gpu_memory_utilization(self)

    @property
    def vllm_args(self) -> tuple[str, ...]:
        """Every engine argument the registry gives this model beyond the fixed serve
        arguments: ``vllm_extra_args`` + ``--max-model-len`` + ``--sleep-mode-backend``."""
        args = list(self.vllm_extra_args)
        if self.max_model_len is not None:
            args.extend(["--max-model-len", str(int(self.max_model_len))])
        if self.sleep_mode_backend is not None:
            args.extend(["--sleep-mode-backend", self.sleep_mode_backend])
        return tuple(args)


def scale_max_replicas(spec: Any) -> int:
    """The scaling cap of a model spec: ``max_awake_replicas`` when set, else the layout
    size ``max_replicas`` (duck-typed so lightweight test specs without the field work)."""
    cap = getattr(spec, "max_awake_replicas", None)
    return int(spec.max_replicas) if cap is None else int(cap)


#: vLLM's own ``--gpu-memory-utilization`` default (a model whose vllm_extra_args
#: do not set the flag starts with it).
VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
_GPU_MEMORY_UTILIZATION_FLAG = "--gpu-memory-utilization"


def gpu_memory_utilization(spec: Any) -> float:
    """The ``--gpu-memory-utilization`` a model's engine starts with: the value in its
    ``vllm_extra_args`` (``--gpu-memory-utilization X`` or ``--gpu-memory-utilization=X``;
    the last one wins, as in vLLM's argparse), else vLLM's default 0.9. Duck-typed
    (a spec without ``vllm_extra_args`` gets the default). A malformed value raises
    ValueError; the registry validation reports it."""
    flag = _GPU_MEMORY_UTILIZATION_FLAG
    args = tuple(getattr(spec, "vllm_extra_args", ()) or ())
    value: str | None = None
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 >= len(args):
                raise ValueError(f"{flag} has no value")
            value = str(args[index + 1])
        elif str(arg).startswith(flag + "="):
            value = str(arg)[len(flag) + 1:]
    if value is None:
        return VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
    try:
        util = float(value)
    except ValueError:
        raise ValueError(f"{flag} {value!r} is not a number") from None
    if not (math.isfinite(util) and 0.0 < util <= 1.0):
        raise ValueError(f"{flag} {value!r} must be in (0, 1]")
    return util


#: Sleep paths (plan 2026-09-27 D1): every caller that puts a binding to sleep names
#: one; the service-manager looks up its soft drain budget in
#: ``service_manager.sleep.budgets_s``.
SLEEP_PATHS = (
    "safescale_commit",  # SafeScale commit of a hidden probe pod
    "urgent",  # controller *_immediate donor paths (fast loop)
    "scale_down",  # ordinary scale-down (manual / console, TRE without SafeScale)
    "apa",  # APA scale-down through the v1-compatible /scale_service
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
    # APA through /scale_service (was "scale_down" before 2026-09-29): with "apa" in
    # no_drain_paths (default) the budget is unused; without it, the old hard-cap drain.
    "apa": None,
    "defrag": 30.0,
    "repair": 30.0,
    "startup": 30.0,
    "default": 30.0,
}

#: Sleep paths that do NOT drain (v1 / paper semantics, 2026-09-29): hide -> gateway ack
#: -> /sleep ``mode=abort`` at once. Continuable requests are continued by the reissue
#: sidecar; non-continuable ones (and requests whose state the SM cannot read) are cut
#: off, never waited for and never a reason to roll back - the outcome counts them.
#: The caller's drain_budget_s is ignored on these paths. The SafeScale probe window
#: (the pod is already hidden while it runs) is the only drain of a TRE scale-down;
#: the fast-loop donors (urgent) and APA scale-downs release at once. Maintenance
#: paths (defrag, repair, startup, scale_down = manual binding power) keep draining.
#: ``service_manager.sleep.no_drain_paths: []`` restores the draining behaviour.
DEFAULT_NO_DRAIN_PATHS: tuple[str, ...] = ("safescale_commit", "urgent", "apa")

#: Default ``gateway.route_timeout_s``: the gateway's per-request route timeout. The
#: single source for the model HTTPRoute timeout (deploy/gen_model_manifests.py) and
#: the service-manager drain hard cap (no request can outlive it anyway).
DEFAULT_ROUTE_TIMEOUT_S = 150.0

#: Default ``gateway.upstream_idle_timeout_s``: how long the tre-v2 Envoy keeps an idle
#: connection to a model pod (Envoy's own default is 1 h). It must stay BELOW the server
#: keep-alive of whatever answers on the pod's serving port (the reissue sidecar's
#: ``server_keepalive_s``, else vLLM's ``VLLM_HTTP_TIMEOUT_KEEP_ALIVE``): the side that
#: sends requests closes idle connections first, otherwise Envoy reuses a connection the
#: server is closing (503 UC / reset).
DEFAULT_GATEWAY_UPSTREAM_IDLE_S = 60.0
#: Minimum margin (s) between a client's idle limit and the server's keep-alive.
KEEPALIVE_MARGIN_S = 1.0

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

#: ``models[].sleep_mode_backend`` values (vLLM fork, ``--sleep-mode-backend``). Unset =
#: no flag = vLLM's default ``cumem``; ``pinned_weights`` keeps a pinned host copy of the
#: weights for the process lifetime (fast sleep, level 1 only).
SLEEP_MODE_BACKENDS = ("cumem", "pinned_weights")

#: vLLM container environment every model pod gets unless the registry overrides it
#: (``vllm.env`` / ``models[].vllm_env`` merge over it). vLLM >= 0.2x mounts /sleep,
#: /wake_up and /is_sleeping only in dev mode, and the service-manager needs them, so it
#: is a default and may not be switched off.
DEFAULT_VLLM_ENV: dict[str, str] = {"VLLM_SERVER_DEV_MODE": "1"}
#: Set per binding by the manifest generator; the registry may not set them.
RESERVED_VLLM_ENV = frozenset({"NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"})
#: vLLM's HTTP keep-alive (uvicorn ``timeout_keep_alive``): an idle client connection is
#: closed by vLLM after this many seconds; vLLM's default when the variable is unset.
VLLM_KEEP_ALIVE_ENV = "VLLM_HTTP_TIMEOUT_KEEP_ALIVE"
VLLM_DEFAULT_KEEP_ALIVE_S = 5.0


def vllm_keep_alive_s(env: dict[str, str]) -> float:
    """vLLM's effective keep-alive for a container environment. vLLM parses the variable
    with ``int()``, so anything else is a ValueError here too."""
    raw = env.get(VLLM_KEEP_ALIVE_ENV)
    return VLLM_DEFAULT_KEEP_ALIVE_S if raw in (None, "") else float(int(raw))


@dataclass(frozen=True)
class VllmConfig:
    """Registry ``vllm:`` section: settings shared by every model pod."""

    #: Extra vLLM container environment (merged over ``DEFAULT_VLLM_ENV``; a model's
    #: ``vllm_env`` merges over this).
    env: dict[str, str] = field(default_factory=dict)


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
    #: Idle keep-alive (s) of the sidecar's pooled connections to the local vLLM. Must be
    #: below vLLM's own keep-alive (``VLLM_HTTP_TIMEOUT_KEEP_ALIVE`` of every model, default
    #: 5 s), else the sidecar reuses connections vLLM is closing (502s before 2026-09-30).
    upstream_keepalive_s: float = 2.0
    #: Fresh-connection re-sends after a pooled connection to the local vLLM failed before
    #: the first response byte (0 = off).
    local_reconnect_attempts: int = 1
    #: The re-send above happens only when the failure came within this many seconds of the
    #: pooled connection being handed to the request (the keep-alive race fails at once; a
    #: later failure may have run the request).
    local_reconnect_window_s: float = 1.0
    #: Keep-alive (s) of the sidecar's own HTTP server, i.e. how long it keeps an idle
    #: connection from Envoy. Must be at least 1 s ABOVE ``gateway.upstream_idle_timeout_s``
    #: (Envoy, the sender, closes first). aiohttp's default is 75.
    server_keepalive_s: float = 75.0
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
class PlacementConfig:
    """Registry ``placement:`` section (design note
    tre/docs/design/20260928-placement-node-balance.md). The placement policy itself
    (``tre_common.gpu_placement.placement_policy_from_registry``) is derived from
    these keys plus the models' ``tp_size``."""

    #: Keep this many fully free aligned blocks of the widest model's size (TP2 pairs
    #: today) when placing narrower replicas. Soft: it ranks placements, never blocks
    #: one; a no-op when every model is tp_size 1.
    reserve_tp_pairs: int = 1
    #: Automatic defragmentation (controller ``critical_tp_defrag``). Off by default,
    #: as in v1. The manual service-manager ``POST /v2/defrag`` also refuses while it
    #: is off unless the request carries ``force: true``.
    defrag_enabled: bool = False
    #: ``placement.placement_penalty``: node name -> extra load (a fraction of the
    #: node, >= 0) added to the node's GPU load when ranking placements (S5,
    #: 2026-09-30): a node that also runs the load generator or the control plane
    #: fills later and is released earlier. Empty = no node is penalised. Keyed by
    #: the registry's own node names (cluster.nodes), never by a built-in name.
    node_penalty: dict[str, float] = field(default_factory=dict)
    #: ``placement.wake_cooldown``: after the service-manager refused a wake on a GPU
    #: (409 wake_conflict / wake_failed), the controller keeps its planner off that
    #: GPU for ``gpu_s`` seconds; a node-scope refusal (no gpu-truth for the node
    #: and a resident unverifiable) keeps it off the whole node for ``node_s``. The
    #: service-manager reports the same values as ``retry_after_s``.
    wake_cooldown_gpu_s: float = 30.0
    wake_cooldown_node_s: float = 60.0


@dataclass(frozen=True)
class ScalingRegistryConfig:
    """Registry ``scaling:`` section (controller only; every key optional; read at
    controller start, restart-to-apply). Older images ignore the whole section
    (the registry loader only reads the sections it knows).

    C1 (2026-10-01, design 20261001-c1-deficit-scaleup): the fast-loop rescue of a
    CRITICAL receiver asks for its whole deficit at once,
    ``desired = min(max(n + 1, ceil(n * tau_crit / Z)), max(n + 1, floor(ratio * n)),
    scaling cap, capacity found)`` with ``n`` its routable replicas, instead of one
    ``ceil(0.1 * n)`` step per window."""

    #: ``ratio`` above: the rescue target is at most ``ratio x n`` (never below n + 1).
    #: 0 = the legacy one-step rescue (``ceil(0.1 * n)`` per decision window).
    rescue_max_step_ratio: float = 2.0
    #: Hold a CRITICAL receiver's next scale-up until a metrics window starting after
    #: its last scale-up completed (review F4 cooldown, scale-up direction of the
    #: fast loop). Off by default under C1: the rescue target bookkeeping already keeps
    #: a not-yet-reflected scale-up from being repeated. Scale-down holds, the slow
    #: loop and the LOW receivers keep the cooldown (TRE_ACTION_COOLDOWN).
    scale_up_cooldown_enabled: bool = False
    #: The rescue target may also reach ``n + rescue_max_step_pods`` (HPA's default
    #: scale-up policy shape, "max(100%, +4 pods)"): cap = max(n + 1,
    #: floor(ratio * n), n + pods). 0 = the ratio alone.
    rescue_max_step_pods: int = 0
    #: An immediate IDLE / HIGH donor of a CRITICAL receiver gives its whole surplus in
    #: one tick (IDLE down to its floor, HIGH down to its tau_high level). Off: one step
    #: per tick, as before C1 (scale-down stays cautious).
    donor_surplus_release: bool = False
    #: A rescue target counts as reflected once the model's decision window starts
    #: ``k * trs.ema_tau_ms`` after the scale-up completed (the EMA'd Z lags the raw
    #: window by about its time constant). 0 = the window start alone (F4 rule).
    rescue_settle_ema_k: float = 2.0
    #: O1 (2026-10-01, design 20261001-o1-breakpoint-window): decide on the part of the
    #: metrics window after the model's last breakpoint (traffic onset or routable-count
    #: change), complete gateway grids only, the TSS numerator normalised to a whole
    #: window and the EMA restarted at the breakpoint. Replaces the onset warmup guard.
    breakpoint_window: bool = True
    #: The ADR-0013 onset warmup guard (TRE_SIGNAL_WARMUP_MS) on top of O1. With
    #: ``breakpoint_window: false`` the guard always applies (= pre-O1), whatever this says.
    onset_warmup_guard: bool = False
    #: O1: complete gateway grids after the breakpoint before the model's signal decides
    #: (scale-ups; scale-downs always need a whole clean window). 2 = 20 s on the 10 s grid.
    min_evidence_grids: int = 2
    #: O1: also this many completed requests in the post-breakpoint window (0 = off).
    #: Tokens count at request completion: with one short request done and long ones
    #: still running, a 20 s suffix can read Z ~ 5 % (review P2-1).
    min_evidence_requests: int = 3
    #: O1: added to a routable-count change time before rounding up to the gateway grid
    #: (the gateway applies the SM's routable label through its pod informer).
    breakpoint_margin_ms: int = 1000
    #: O1 (review P2-1, evidence-gated): a C1 rescue decided on a partial
    #: (post-breakpoint) window with fewer than ``breakpoint_lowevidence_requests``
    #: completed requests adds at most ``breakpoint_partial_max_step`` replicas; with at
    #: least that many it asks for the whole deficit (ratio / step_pods caps apply).
    #: ``breakpoint_partial_max_step: 0`` = no cap at all.
    breakpoint_partial_max_step: int = 1
    breakpoint_lowevidence_requests: int = 10
    #: O1: after this many consecutive held metrics windows (10 s each) a receiver
    #: decides on the whole window again (donors still need a clean one) - a model whose
    #: routable count keeps changing is not starved (review P2-2). 0 = never.
    breakpoint_hold_max_windows: int = 6
    #: O1 same-clock check (gateway doc stamps vs the controller clock, review P2-3):
    #: tolerance and period (s, 0 = off). A violation suspends O1 (pre-O1 behaviour).
    gateway_clock_tolerance_ms: int = 2000
    gateway_clock_check_s: int = 60


SCALING_KEYS = frozenset({
    "rescue_max_step_ratio", "scale_up_cooldown_enabled", "rescue_max_step_pods",
    "donor_surplus_release", "rescue_settle_ema_k",
    "breakpoint_window", "onset_warmup_guard", "min_evidence_grids", "min_evidence_requests",
    "breakpoint_margin_ms", "breakpoint_partial_max_step", "breakpoint_lowevidence_requests",
    "breakpoint_hold_max_windows",
    "gateway_clock_tolerance_ms", "gateway_clock_check_s",
})


def _scaling_count(raw: dict[str, Any], key: str, default: int, minimum: int) -> int:
    value = raw.get(key)
    if value is None:
        return default
    try:
        valid = not isinstance(value, bool) and float(value) == int(float(value)) and int(float(value)) >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"scaling.{key} must be an integer >= {minimum}, got {value!r}")
    return int(float(value))


def _scaling_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"scaling.{key} must be true or false, got {value!r}")


def parse_scaling_config(raw: dict[str, Any] | None) -> ScalingRegistryConfig:
    """Parse the optional ``scaling:`` registry section; raise ValueError on bad values
    (unknown keys are ignored with a warning, like ``safescale:``)."""
    if raw is None:
        return ScalingRegistryConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"scaling must be a mapping, got {raw!r}")
    unknown = sorted(str(key) for key in set(raw) - SCALING_KEYS)
    if unknown:
        LOG.warning("registry scaling: ignoring unknown keys %s (known: %s)", unknown, sorted(SCALING_KEYS))
    defaults = ScalingRegistryConfig()
    ratio_raw = raw.get("rescue_max_step_ratio")
    if isinstance(ratio_raw, bool):
        raise ValueError(f"scaling.rescue_max_step_ratio must be a number, got {ratio_raw!r}")
    try:
        ratio = float(defaults.rescue_max_step_ratio if ratio_raw is None else ratio_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"scaling.rescue_max_step_ratio must be a number, got {ratio_raw!r}") from exc
    if not math.isfinite(ratio) or ratio < 0 or 0 < ratio < 1:
        raise ValueError(
            f"scaling.rescue_max_step_ratio must be 0 (legacy step) or at least 1, got {ratio_raw!r}"
        )
    pods_raw = raw.get("rescue_max_step_pods")
    pods = defaults.rescue_max_step_pods if pods_raw is None else pods_raw
    try:
        valid_pods = not isinstance(pods, bool) and float(pods) == int(float(pods)) and int(float(pods)) >= 0
    except (TypeError, ValueError, OverflowError):
        valid_pods = False
    if not valid_pods:
        raise ValueError(f"scaling.rescue_max_step_pods must be a non-negative integer, got {pods_raw!r}")
    k_raw = raw.get("rescue_settle_ema_k")
    if isinstance(k_raw, bool):
        raise ValueError(f"scaling.rescue_settle_ema_k must be a number, got {k_raw!r}")
    try:
        settle_k = float(defaults.rescue_settle_ema_k if k_raw is None else k_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"scaling.rescue_settle_ema_k must be a number, got {k_raw!r}") from exc
    if not math.isfinite(settle_k) or settle_k < 0:
        raise ValueError(f"scaling.rescue_settle_ema_k must be a non-negative number, got {k_raw!r}")
    return ScalingRegistryConfig(
        rescue_max_step_ratio=ratio,
        scale_up_cooldown_enabled=_scaling_bool(raw, "scale_up_cooldown_enabled", defaults.scale_up_cooldown_enabled),
        rescue_max_step_pods=int(float(pods)),
        donor_surplus_release=_scaling_bool(raw, "donor_surplus_release", defaults.donor_surplus_release),
        rescue_settle_ema_k=settle_k,
        breakpoint_window=_scaling_bool(raw, "breakpoint_window", defaults.breakpoint_window),
        onset_warmup_guard=_scaling_bool(raw, "onset_warmup_guard", defaults.onset_warmup_guard),
        min_evidence_grids=_scaling_count(raw, "min_evidence_grids", defaults.min_evidence_grids, 1),
        min_evidence_requests=_scaling_count(
            raw, "min_evidence_requests", defaults.min_evidence_requests, 0
        ),
        breakpoint_margin_ms=_scaling_count(raw, "breakpoint_margin_ms", defaults.breakpoint_margin_ms, 0),
        breakpoint_partial_max_step=_scaling_count(
            raw, "breakpoint_partial_max_step", defaults.breakpoint_partial_max_step, 0
        ),
        breakpoint_lowevidence_requests=_scaling_count(
            raw, "breakpoint_lowevidence_requests", defaults.breakpoint_lowevidence_requests, 0
        ),
        breakpoint_hold_max_windows=_scaling_count(
            raw, "breakpoint_hold_max_windows", defaults.breakpoint_hold_max_windows, 0
        ),
        gateway_clock_tolerance_ms=_scaling_count(
            raw, "gateway_clock_tolerance_ms", defaults.gateway_clock_tolerance_ms, 0
        ),
        gateway_clock_check_s=_scaling_count(raw, "gateway_clock_check_s", defaults.gateway_clock_check_s, 0),
    )


#: ``safescale.slo_mode``: where the SafeScale probe's latency thresholds come from.
#: ``labels`` - the calibration label's rule (``tre_common.slo_labels.label_def_for_model``:
#: TPOT 75 ms, TTFT = max(floor, k * (c + b * L)) with the mean prompt length L of the
#: evidence window); ``fixed`` - the model's ``models[].slo`` ttft_p95_ms / tpot_p95_ms.
SAFESCALE_SLO_MODES = ("labels", "fixed")


@dataclass(frozen=True)
class SafeScaleRegistryConfig:
    """Registry ``safescale:`` section (controller only; every key optional).

    Read at controller start (restart-to-apply). Controller images before this
    section existed ignore it."""

    slo_mode: str = "labels"
    #: Upper bound (s) of the probe window W, deadline extensions included:
    #: W = min(max(multiplier * p95_e2e, floor), window_ceiling_s).
    window_ceiling_s: float = 60.0
    #: Completed requests (TTFT count of the remaining pods over the evidence window)
    #: needed before the latency part of the commit gate is judged.
    min_commit_samples: int = 20
    #: The first gateway doc of the evidence window must be stamped within
    #: [hide, hide + this]; otherwise the probe rolls back (clock / missing-tick guard).
    #: Redis evidence path only (``evidence_source: redis`` or the direct path's fallback).
    evidence_clock_tolerance_s: float = 20.0
    #: Where the probe's latency / KV evidence comes from (2026-09-29 B+D):
    #: ``direct`` - the controller scrapes the remaining pods' vLLM ``/metrics`` itself
    #: (baseline at the SM's hide confirmation, then every ``evidence_poll_s``), falling
    #: back to ``redis`` for a probe when every remaining pod fails; ``redis`` - the
    #: gateway's histogram docs in Redis only (the 10 s doc grid).
    evidence_source: str = "direct"
    #: Direct path: scrape period (s) of a probe's remaining pods; also the deadline
    #: extension step while the evidence is short.
    evidence_poll_s: float = 2.0
    #: Direct path: timeout (s) of one pod scrape (the scrapes of a tick run concurrently).
    scrape_timeout_s: float = 1.0
    #: Direct path: the baseline scrape runs this long (ms) after the SM confirmed the
    #: hide, so the gateway has applied it (pod watch) - a baseline taken while requests
    #: still go to the hidden pods under-counts the remaining pods' load. The deadline
    #: still counts from the confirmation.
    baseline_delay_ms: float = 1000.0
    #: Direct path: the port of a model pod serving ``GET /metrics`` (the pod's serving
    #: port; with the reissue sidecar the sidecar forwards it to vLLM).
    metrics_port: int = POD_SERVING_PORT


SAFESCALE_KEYS = frozenset({
    "slo_mode", "window_ceiling_s", "min_commit_samples", "evidence_clock_tolerance_s",
    "evidence_source", "evidence_poll_s", "scrape_timeout_s", "metrics_port", "baseline_delay_ms",
})
SAFESCALE_EVIDENCE_SOURCES = ("direct", "redis")


def _safescale_num(section: dict[str, Any], key: str, default: float) -> float:
    """A number of the safescale section; anything unparsable is a ValueError."""
    value = section.get(key)
    if isinstance(value, bool):
        raise ValueError(f"safescale.{key} must be a number, got {value!r}")
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"safescale.{key} must be a number, got {value!r}") from exc


def parse_safescale_config(raw: dict[str, Any] | None) -> SafeScaleRegistryConfig:
    """Parse the optional ``safescale:`` registry section; raise ValueError on bad values.

    Unknown keys are ignored with a warning (forward compatibility: a registry written
    for a newer controller must not stop the service-manager / UI / an older controller
    that parse this section too); invalid values of known keys refuse the start."""
    if raw is None:
        return SafeScaleRegistryConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"safescale must be a mapping, got {raw!r}")
    unknown = sorted(str(key) for key in set(raw) - SAFESCALE_KEYS)
    if unknown:
        LOG.warning("registry safescale: ignoring unknown keys %s (known: %s)", unknown, sorted(SAFESCALE_KEYS))
    defaults = SafeScaleRegistryConfig()
    source = str(raw.get("evidence_source") or defaults.evidence_source).strip().lower()
    if source not in SAFESCALE_EVIDENCE_SOURCES:
        raise ValueError(f"safescale.evidence_source must be one of {SAFESCALE_EVIDENCE_SOURCES}, got {source!r}")
    poll = _safescale_num(raw, "evidence_poll_s", defaults.evidence_poll_s)
    scrape_timeout = _safescale_num(raw, "scrape_timeout_s", defaults.scrape_timeout_s)
    for name, value in (("evidence_poll_s", poll), ("scrape_timeout_s", scrape_timeout)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"safescale.{name} must be a positive number, got {value!r}")
    if scrape_timeout >= poll:
        raise ValueError(
            f"safescale.scrape_timeout_s ({scrape_timeout}) must be below evidence_poll_s ({poll}): "
            "a tick's scrapes must finish before the next one"
        )
    baseline_delay = _safescale_num(raw, "baseline_delay_ms", defaults.baseline_delay_ms)
    if not math.isfinite(baseline_delay) or baseline_delay < 0:
        raise ValueError(f"safescale.baseline_delay_ms must be a non-negative number, got {baseline_delay!r}")
    port_raw = raw.get("metrics_port")
    port = defaults.metrics_port if port_raw is None else port_raw
    try:
        valid_port = (
            not isinstance(port, bool) and float(port) == int(float(port)) and 1 <= int(float(port)) <= 65535
        )
    except (TypeError, ValueError, OverflowError):
        valid_port = False
    if not valid_port:
        raise ValueError(f"safescale.metrics_port must be a port number, got {port!r}")
    mode = str(raw.get("slo_mode") or defaults.slo_mode).strip().lower()
    if mode not in SAFESCALE_SLO_MODES:
        raise ValueError(f"safescale.slo_mode must be one of {SAFESCALE_SLO_MODES}, got {mode!r}")
    ceiling = _safescale_num(raw, "window_ceiling_s", defaults.window_ceiling_s)
    tolerance = _safescale_num(raw, "evidence_clock_tolerance_s", defaults.evidence_clock_tolerance_s)
    for name, value in (("window_ceiling_s", ceiling), ("evidence_clock_tolerance_s", tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"safescale.{name} must be a positive number, got {value!r}")
    if baseline_delay >= 1000.0 * ceiling:
        raise ValueError(
            f"safescale.baseline_delay_ms ({baseline_delay}) must be below window_ceiling_s ({ceiling}) in ms"
        )
    samples_raw = raw.get("min_commit_samples")
    samples = defaults.min_commit_samples if samples_raw is None else samples_raw
    if isinstance(samples, bool) or float(samples) != int(float(samples)) or int(float(samples)) < 0:
        raise ValueError(f"safescale.min_commit_samples must be a non-negative integer, got {samples!r}")
    return SafeScaleRegistryConfig(
        slo_mode=mode,
        window_ceiling_s=float(ceiling),
        min_commit_samples=int(float(samples)),
        evidence_clock_tolerance_s=float(tolerance),
        evidence_source=source,
        evidence_poll_s=float(poll),
        scrape_timeout_s=float(scrape_timeout),
        metrics_port=int(float(port)),
        baseline_delay_ms=float(baseline_delay),
    )


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
    #: Idle timeout (s) of Envoy's upstream connections to the model pods (see
    #: ``DEFAULT_GATEWAY_UPSTREAM_IDLE_S``). The single source for the hand-written
    #: ``connection_idle``/``idle_timeout`` of the ORIGINAL_DST clusters and the
    #: BackendTrafficPolicy ``connectionIdleTimeout`` (a guard test keeps them equal).
    upstream_idle_timeout_s: float = DEFAULT_GATEWAY_UPSTREAM_IDLE_S

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
    #: Paths that never drain (see DEFAULT_NO_DRAIN_PATHS); () = every path drains.
    no_drain_paths: tuple[str, ...] = DEFAULT_NO_DRAIN_PATHS

    def no_drain(self, path: str) -> bool:
        """True when a sleep on ``path`` aborts everything in flight right after the
        gateway ack (no drain, no rollback over in-flight requests)."""
        return path in self.no_drain_paths

    def soft_budget_s(self, path: str, requested_s: float | None = None) -> float:
        """Soft drain budget of one sleep: 0 on a no-drain path (the caller's
        budget is ignored), else the caller's budget, else the path's."""
        if requested_s is not None:
            budget = float(requested_s)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError(f"drain budget must be a finite number >= 0, got {requested_s!r}")
        if self.no_drain(path):
            return 0.0
        if requested_s is None:
            key = path if path in self.budgets_s else "default"
            raw = self.budgets_s.get(key)
            budget = self.hard_cap_s if raw is None else float(raw)
        return min(budget, self.hard_cap_s)


#: Accepted values of ``service_manager.log_level`` (and TRE_SM_LOG_LEVEL).
LOG_LEVEL_NAMES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


@dataclass(frozen=True)
class ServiceManagerConfig:
    sleep: SleepPolicy = field(default_factory=SleepPolicy)
    #: A wake with a TRUSTED gpu-truth sample (within its TTL and not older than
    #: the SM's last power change on the GPU) fails unless every target GPU's used
    #: memory is at most this fraction of the GPU's total memory: sleeping
    #: residents keep only a small footprint; an awake resident (or a leak) is far
    #: above it. Without a trusted sample the GPU's residents are probed instead.
    wake_max_used_fraction: float = 0.2
    #: Optional absolute override (MiB) of the wake threshold; None = the fraction.
    wake_max_used_mib: int | None = None
    #: How long the cold-start headroom gate waits for a gpu-truth sample taken
    #: AFTER it asked for one (the agent's on-demand refresh,
    #: ``tre:gpu_truth_refresh:<node>``); with an agent that does not answer
    #: refreshes it re-reads the periodic sample for up to this long instead. The
    #: wake gate never waits (2026-09-30).
    wake_truth_wait_s: float = 10.0
    #: Cold start (create) headroom (B9). vLLM refuses to start an engine unless the
    #: GPU's free memory is at least gpu_memory_utilization x total, so a create is
    #: allowed while every target GPU's used memory (gpu-truth) is at most
    #: total x (1 - gpu_memory_utilization of the model being created) - this margin
    #: (the new process' own CUDA context and allocator slack). Sleeping neighbours
    #: of a full layout fit under it; an awake one does not.
    create_margin_mib: int = 512
    #: Optional absolute override (MiB) of the create limit; None = derived per GPU
    #: and model. The TRE_CREATE_MAX_USED_MIB env var (set only on purpose)
    #: overrides both: env > create.max_used_mib > derived.
    create_max_used_mib: int | None = None
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
    #: Level of the tre_sm / tre_common loggers (a logging level name); the
    #: TRE_SM_LOG_LEVEL environment variable overrides it.
    log_level: str = "INFO"
    #: Replica floor (``service_manager.replica_floor.enforce``, 2026-09-29): no hide
    #: or sleep may leave a model with fewer routable replicas than its registry
    #: ``min_replicas`` (hide / urgent / scale_down / defrag refused with 409
    #: floor_violation, APA targets clamped, startup wakes another replica first
    #: best effort and is otherwise exempt, fleet repair exempt; every exemption is
    #: recorded). false = the previous behaviour.
    replica_floor_enforce: bool = True
    #: ``service_manager.replica_floor.log_interval_s``: the WARNING log of a floor
    #: event is emitted at most once per (outcome, path, model) in this many seconds
    #: (the counters always count; 0 = log every event).
    replica_floor_log_interval_s: float = 60.0
    #: ``service_manager.startup_admission.gate_seen_s``: a Pod counts as waiting in
    #: its startup gate while its gate asked for admission within this many seconds
    #: (the gate polls every 2 s, each call answering within its 15 s timeout).
    startup_gate_seen_s: float = 30.0
    #: ``service_manager.startup_admission.drift_grace_s``: a Pod waiting in its
    #: startup gate is reported as ``startup_admission_pending`` (informational, no
    #: fleet repair) instead of fleet drift for at most this long since its first
    #: admission request; a Pod waiting longer is reported as drift again (a stuck
    #: gate is not masked). 0 = never exempt.
    startup_gate_drift_grace_s: float = 600.0
    #: ``service_manager.wake.recovery_unknown_attempts``: a wake the journal
    #: recovery cannot read (/is_sleeping unknown) is kept this many supervisor
    #: passes, then rolled back with an alert (at once when its pod is not Ready).
    wake_recovery_unknown_attempts: int = 12
    #: ``service_manager.wake.transport_recheck_s``: a /wake_up that raised (timeout,
    #: transport error) may still wake the engine: its waking lease and journal
    #: entry are kept and the recovery rechecks it after this long.
    wake_transport_recheck_s: float = 30.0
    #: ``service_manager.startup_admission.placeholder_max_s``: a Pod admitted at its
    #: startup gate (or whose engine container restarted) holds its GPUs (the
    #: ``starting`` lease) until it converged or is gone. The lease is released only
    #: while the engine container is not running and does not read awake; an engine
    #: running but not Ready past this long is alerted, the lease kept.
    startup_placeholder_max_s: float = 900.0
    #: ``service_manager.test_hooks``: honour the fault-injection keys
    #: ``tre:v2:sm:fault:<refuse_wake|fail_wake>:<node>/<gpu>`` (acceptance tests
    #: only). Off by default: the keys are then never read.
    test_hooks: bool = False
    #: ``service_manager.operations.max_records``: the operation journal
    #: (``tre:v2:sm:operations``) keeps at most this many finished records (oldest
    #: dropped first; running ones are never dropped).
    operations_max_records: int = 20000

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

    def create_limit_mib(self, total_mib: int | None, gpu_memory_utilization: float) -> int | None:
        """Max used MiB for a cold start of a model with ``gpu_memory_utilization`` on
        a GPU of ``total_mib`` (None = unknown total and no absolute override)."""
        if self.create_max_used_mib is not None:
            return int(self.create_max_used_mib)
        if total_mib is None or total_mib <= 0:
            return None
        # vLLM needs util x total free; floor the rest (the epsilon absorbs float
        # noise such as 40960 x 0.15 = 6143.999...).
        startup_free = math.floor(total_mib * (1.0 - gpu_memory_utilization) + 1e-6)
        return int(startup_free) - int(self.create_margin_mib)


class Registry:
    def __init__(
        self,
        topology: ClusterTopology,
        models: list[ModelSpec],
        service_manager: ServiceManagerConfig | None = None,
        gateway: GatewayConfig | None = None,
        reissue: ReissueConfig | None = None,
        vllm: VllmConfig | None = None,
        placement: PlacementConfig | None = None,
        safescale: SafeScaleRegistryConfig | None = None,
        scaling: ScalingRegistryConfig | None = None,
    ) -> None:
        self._safescale = safescale or SafeScaleRegistryConfig()
        self._scaling = scaling or ScalingRegistryConfig()
        self._topology = topology
        self._placement = placement or PlacementConfig()
        self._models = tuple(models)
        self._service_manager = service_manager or ServiceManagerConfig()
        self._gateway = gateway or GatewayConfig()
        self._reissue = reissue or ReissueConfig()
        self._vllm = vllm or VllmConfig()
        self._model_index: dict[str, ModelSpec] = {}
        for model in models:
            self._model_index.setdefault(model.name, model)

    def service_manager(self) -> ServiceManagerConfig:
        return self._service_manager

    def gateway(self) -> GatewayConfig:
        return self._gateway

    def reissue(self) -> ReissueConfig:
        return self._reissue

    def vllm(self) -> VllmConfig:
        return self._vllm

    def placement(self) -> PlacementConfig:
        return self._placement

    def safescale(self) -> SafeScaleRegistryConfig:
        return self._safescale

    def scaling(self) -> ScalingRegistryConfig:
        return self._scaling

    def vllm_env_for(self, model: ModelSpec) -> dict[str, str]:
        """The vLLM container environment of ``model``'s pods (besides the per-binding
        GPU variables): ``DEFAULT_VLLM_ENV``, then ``vllm.env``, then the model's
        ``vllm_env``; later keys win."""
        return {**DEFAULT_VLLM_ENV, **self._vllm.env, **getattr(model, "vllm_env", {})}

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
            problem = tp_size_error(
                model.tp_size, widest_node_gpus=_widest_node_gpus(self._topology.nodes)
            )
            if problem:
                errors.append(f"model {model.name}: unsupported {problem}")
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
            errors.extend(_validate_model_vllm(model))
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
        if self._reissue.enabled:
            for model in self._models:
                try:
                    server_keep_alive = vllm_keep_alive_s(self.vllm_env_for(model))
                except ValueError:
                    errors.append(f"model {model.name}: {VLLM_KEEP_ALIVE_ENV} must be an integer number of seconds")
                    continue
                # >= 1 s margin: the sidecar's pool clock lags vLLM's idle clock under
                # CPU throttling (same rule as the sidecar's startup check).
                if self._reissue.upstream_keepalive_s > server_keep_alive - 1.0:
                    errors.append(
                        f"model {model.name}: reissue.upstream_keepalive_s ({self._reissue.upstream_keepalive_s}) "
                        f"must be at least 1 s below vLLM's {VLLM_KEEP_ALIVE_ENV} ({server_keep_alive})"
                    )
        # Envoy -> model pod: the connection is closed by the side that sends requests, so
        # Envoy's idle timeout must be below the keep-alive of whatever serves port 8000
        # (the sidecar when enabled, else vLLM itself).
        idle = self._gateway.upstream_idle_timeout_s
        if not math.isfinite(idle) or idle <= 0:
            errors.append("gateway.upstream_idle_timeout_s must be positive")
        elif self._reissue.enabled:
            if self._reissue.server_keepalive_s < idle + KEEPALIVE_MARGIN_S:
                errors.append(
                    f"reissue.server_keepalive_s ({self._reissue.server_keepalive_s:g}) must be at least "
                    f"{KEEPALIVE_MARGIN_S:g} s above gateway.upstream_idle_timeout_s ({idle:g})"
                )
        else:
            for model in self._models:
                try:
                    server_keep_alive = vllm_keep_alive_s(self.vllm_env_for(model))
                except ValueError:
                    continue  # reported above / by the env validation
                if server_keep_alive < idle + KEEPALIVE_MARGIN_S:
                    errors.append(
                        f"model {model.name}: vLLM's {VLLM_KEEP_ALIVE_ENV} ({server_keep_alive:g}) must be at "
                        f"least {KEEPALIVE_MARGIN_S:g} s above gateway.upstream_idle_timeout_s ({idle:g})"
                    )
        if self._placement.reserve_tp_pairs < 0:
            errors.append("placement.reserve_tp_pairs must be >= 0")
        node_names = {node.name for node in self._topology.nodes}
        for node_name in sorted(set(self._placement.node_penalty) - node_names):
            errors.append(f"placement.placement_penalty: unknown node {node_name!r} (not in cluster.nodes)")
        errors.extend(_validate_vllm_env("vllm.env", self._vllm.env))
        errors.extend(_validate_service_manager(self._service_manager, self._gateway))
        return errors


def _arg_present(args: tuple[str, ...], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in args)


def _validate_model_vllm(model: ModelSpec) -> list[str]:
    errors: list[str] = []
    prefix = f"model {model.name}"
    if model.max_model_len is not None:
        if model.max_model_len <= 0:
            errors.append(f"{prefix}: max_model_len must be positive")
        if _arg_present(model.vllm_extra_args, "--max-model-len"):
            errors.append(f"{prefix}: set max_model_len OR --max-model-len in vllm_extra_args, not both")
    if model.sleep_mode_backend is not None:
        if model.sleep_mode_backend not in SLEEP_MODE_BACKENDS:
            errors.append(
                f"{prefix}: sleep_mode_backend must be one of {list(SLEEP_MODE_BACKENDS)} or null"
            )
        if _arg_present(model.vllm_extra_args, "--sleep-mode-backend"):
            errors.append(f"{prefix}: set sleep_mode_backend OR --sleep-mode-backend in vllm_extra_args, not both")
    try:
        gpu_memory_utilization(model)
    except ValueError as exc:
        errors.append(f"{prefix}: vllm_extra_args {exc}")
    errors.extend(_validate_vllm_env(f"{prefix}: vllm_env", getattr(model, "vllm_env", {})))
    return errors


def _validate_vllm_env(where: str, env: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for key, value in env.items():
        if key in RESERVED_VLLM_ENV:
            errors.append(f"{where}: {key} is set per binding by the manifest generator")
        elif key in DEFAULT_VLLM_ENV and value != DEFAULT_VLLM_ENV[key]:
            errors.append(f"{where}: {key} must be {DEFAULT_VLLM_ENV[key]!r} (TRE needs it)")
    return errors


def load_registry(path: str | None = None) -> Registry:
    registry_path = Path(path) if path else Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return _parse_registry(raw)


def _parse_registry(raw: dict[str, Any]) -> Registry:
    cluster = raw.get("cluster") or {}
    nodes = tuple(_parse_node(item) for item in cluster.get("nodes", []))
    models = [_parse_model(item) for item in raw.get("models", [])]
    # Refused at load (not only by validate()): the SM, the controller and the
    # UI all load through here, so none of them runs with a tp_size another
    # component would reject or silently place differently.
    widest = _widest_node_gpus(nodes)
    tp_errors = [
        f"model {model.name}: unsupported {problem}"
        for model in models
        if (problem := tp_size_error(model.tp_size, widest_node_gpus=widest))
    ]
    if tp_errors:
        raise ValueError("invalid registry: " + "; ".join(tp_errors))
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
        vllm=parse_vllm_config(raw.get("vllm")),
        placement=parse_placement_config(raw.get("placement")),
        safescale=parse_safescale_config(raw.get("safescale")),
        scaling=parse_scaling_config(raw.get("scaling")),
    )


def _parse_env(raw: Any, where: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    env: dict[str, str] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"{where}.{key} must be a scalar")
        # YAML turns 1 / true into int / bool; the container env wants the text.
        env[str(key)] = str(value).lower() if isinstance(value, bool) else str(value)
    return env


def parse_vllm_config(raw: Any) -> VllmConfig:
    """Parse the optional ``vllm:`` registry section (absent = defaults)."""
    if raw is None:
        return VllmConfig()
    if not isinstance(raw, dict):
        raise ValueError("vllm must be a mapping")
    known = {f for f in VllmConfig.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"vllm: unknown keys {unknown} (known: {', '.join(sorted(known))})")
    return VllmConfig(env=_parse_env(raw.get("env"), "vllm.env"))


def parse_placement_config(raw: Any) -> PlacementConfig:
    """Parse the optional ``placement:`` registry section (absent = defaults:
    reserve_tp_pairs 1, defrag disabled)."""
    if raw is None:
        return PlacementConfig()
    if not isinstance(raw, dict):
        raise ValueError("placement must be a mapping")
    known = {"reserve_tp_pairs", "defrag", "placement_penalty", "wake_cooldown"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"placement: unknown keys {unknown} (known: {', '.join(sorted(known))})")
    penalty_raw = raw.get("placement_penalty")
    if penalty_raw is None:
        penalty_raw = {}
    if not isinstance(penalty_raw, dict):
        raise ValueError("placement.placement_penalty must be a mapping node name -> number")
    node_penalty: dict[str, float] = {}
    for node_name, value in penalty_raw.items():
        if isinstance(value, bool):
            raise ValueError(f"placement.placement_penalty[{node_name!r}] must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"placement.placement_penalty[{node_name!r}] must be a number") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"placement.placement_penalty[{node_name!r}] must be a finite number >= 0")
        node_penalty[str(node_name)] = number
    cooldown_raw = raw.get("wake_cooldown")
    if cooldown_raw is None:
        cooldown_raw = {}
    if not isinstance(cooldown_raw, dict):
        raise ValueError("placement.wake_cooldown must be a mapping (gpu_s, node_s)")
    unknown = sorted(set(cooldown_raw) - {"gpu_s", "node_s"})
    if unknown:
        raise ValueError(f"placement.wake_cooldown: unknown keys {unknown} (known: gpu_s, node_s)")
    cooldowns = {
        key: _nonneg_num(cooldown_raw, key, default, f"placement.wake_cooldown.{key}")
        for key, default in (("gpu_s", PlacementConfig.wake_cooldown_gpu_s), ("node_s", PlacementConfig.wake_cooldown_node_s))
    }
    defaults = PlacementConfig()
    defrag = raw.get("defrag")
    if defrag is None:
        defrag = {}
    if not isinstance(defrag, dict):
        raise ValueError("placement.defrag must be a mapping")
    unknown = sorted(set(defrag) - {"enabled"})
    if unknown:
        raise ValueError(f"placement.defrag: unknown keys {unknown} (known: enabled)")
    reserve = raw.get("reserve_tp_pairs")
    if isinstance(reserve, bool):
        raise ValueError("placement.reserve_tp_pairs must be an integer")
    return PlacementConfig(
        reserve_tp_pairs=defaults.reserve_tp_pairs if reserve is None else int(reserve),
        defrag_enabled=(
            defaults.defrag_enabled
            if defrag.get("enabled") is None
            else _parse_bool(defrag["enabled"])
        ),
        node_penalty=node_penalty,
        wake_cooldown_gpu_s=cooldowns["gpu_s"],
        wake_cooldown_node_s=cooldowns["node_s"],
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
        upstream_keepalive_s=float(raw.get("upstream_keepalive_s", defaults.upstream_keepalive_s)),
        local_reconnect_attempts=int(raw.get("local_reconnect_attempts", defaults.local_reconnect_attempts)),
        local_reconnect_window_s=float(raw.get("local_reconnect_window_s", defaults.local_reconnect_window_s)),
        server_keepalive_s=float(raw.get("server_keepalive_s", defaults.server_keepalive_s)),
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
    if not reissue.upstream_keepalive_s > 0:
        errors.append("reissue.upstream_keepalive_s must be > 0")
    if reissue.local_reconnect_attempts < 0:
        errors.append("reissue.local_reconnect_attempts must be >= 0")
    if not reissue.local_reconnect_window_s >= 0:
        errors.append("reissue.local_reconnect_window_s must be >= 0")
    if not reissue.server_keepalive_s > 0:
        errors.append("reissue.server_keepalive_s must be > 0")
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
    idle = raw.get("upstream_idle_timeout_s")
    return GatewayConfig(
        route_timeout_s=float(DEFAULT_ROUTE_TIMEOUT_S if timeout is None else timeout),
        upstream_idle_timeout_s=float(DEFAULT_GATEWAY_UPSTREAM_IDLE_S if idle is None else idle),
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
    create_raw = raw.get("create") or {}
    skew_raw = raw.get("clock_skew") or {}
    pressure_raw = raw.get("node_pressure") or {}
    floor_raw = raw.get("replica_floor") or {}
    if not isinstance(floor_raw, dict):
        raise ValueError(
            f"service_manager.replica_floor must be a mapping (enforce: bool), got {floor_raw!r}"
        )
    startup_raw = raw.get("startup_admission") or {}
    if not isinstance(startup_raw, dict):
        raise ValueError(
            "service_manager.startup_admission must be a mapping "
            f"(gate_seen_s, drift_grace_s), got {startup_raw!r}"
        )
    operations_raw = raw.get("operations") or {}
    if not isinstance(operations_raw, dict):
        raise ValueError(
            f"service_manager.operations must be a mapping (max_records), got {operations_raw!r}"
        )
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
    no_drain_raw = sleep_raw.get("no_drain_paths", DEFAULT_NO_DRAIN_PATHS)
    if no_drain_raw is None:
        no_drain_raw = ()
    if isinstance(no_drain_raw, str) or not isinstance(no_drain_raw, (list, tuple)):
        raise ValueError(
            "service_manager.sleep.no_drain_paths must be a list of sleep paths "
            f"(known: {', '.join(SLEEP_PATHS)}), got {no_drain_raw!r}"
        )
    for path in no_drain_raw:
        if str(path) not in SLEEP_PATHS:
            raise ValueError(
                f"service_manager.sleep.no_drain_paths: unknown sleep path {path!r} "
                f"(known: {', '.join(SLEEP_PATHS)})"
            )
    no_drain_paths = tuple(dict.fromkeys(str(path) for path in no_drain_raw))
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
        no_drain_paths=no_drain_paths,
        plugin_namespace=str(plugin_pods_raw.get("namespace", defaults.plugin_namespace)),
        plugin_label_selector=plugin_pods_raw.get(
            "label_selector", defaults.plugin_label_selector
        ),
    )
    base = ServiceManagerConfig()
    fail_s = skew_raw.get("fail_s", base.clock_skew_fail_s)
    max_used_mib = wake_raw.get("max_used_mib", base.wake_max_used_mib)
    create_max_used_mib = create_raw.get("max_used_mib", base.create_max_used_mib)
    return ServiceManagerConfig(
        sleep=sleep,
        wake_max_used_fraction=_num(wake_raw, "max_used_fraction", base.wake_max_used_fraction),
        wake_max_used_mib=None if max_used_mib is None else int(max_used_mib),
        wake_truth_wait_s=_num(wake_raw, "truth_wait_s", base.wake_truth_wait_s),
        create_margin_mib=int(_num(create_raw, "margin_mib", base.create_margin_mib)),
        create_max_used_mib=None if create_max_used_mib is None else int(create_max_used_mib),
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
        log_level=(
            base.log_level if raw.get("log_level") is None else str(raw["log_level"]).strip().upper()
        ),
        replica_floor_enforce=_parse_bool(
            floor_raw.get("enforce", base.replica_floor_enforce)
        ),
        replica_floor_log_interval_s=_nonneg_num(
            floor_raw, "log_interval_s", base.replica_floor_log_interval_s,
            "service_manager.replica_floor.log_interval_s",
        ),
        startup_gate_seen_s=_nonneg_num(
            startup_raw, "gate_seen_s", base.startup_gate_seen_s,
            "service_manager.startup_admission.gate_seen_s",
        ),
        startup_gate_drift_grace_s=_nonneg_num(
            startup_raw, "drift_grace_s", base.startup_gate_drift_grace_s,
            "service_manager.startup_admission.drift_grace_s",
        ),
        test_hooks=_parse_bool(raw.get("test_hooks", base.test_hooks)),
        wake_recovery_unknown_attempts=int(
            _num(wake_raw, "recovery_unknown_attempts", base.wake_recovery_unknown_attempts)
        ),
        wake_transport_recheck_s=_nonneg_num(
            wake_raw, "transport_recheck_s", base.wake_transport_recheck_s,
            "service_manager.wake.transport_recheck_s",
        ),
        startup_placeholder_max_s=_nonneg_num(
            startup_raw, "placeholder_max_s", base.startup_placeholder_max_s,
            "service_manager.startup_admission.placeholder_max_s",
        ),
        operations_max_records=int(
            _num(operations_raw, "max_records", base.operations_max_records)
        ),
    )


def _nonneg_num(section: dict[str, Any], key: str, default: float, name: str) -> float:
    value = _num(section, key, default)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {section.get(key)!r}")
    return value


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
    if config.operations_max_records < 100:
        errors.append("service_manager.operations.max_records must be >= 100")
    if config.wake_recovery_unknown_attempts < 1:
        errors.append("service_manager.wake.recovery_unknown_attempts must be >= 1")
    if config.startup_placeholder_max_s < 60:
        errors.append("service_manager.startup_admission.placeholder_max_s must be >= 60")
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
    for path in sleep.no_drain_paths:
        if path not in SLEEP_PATHS:
            errors.append(f"service_manager.sleep.no_drain_paths: unknown sleep path {path}")
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
    if config.create_margin_mib < 0:
        errors.append("service_manager.create.margin_mib must be >= 0")
    if config.create_max_used_mib is not None and config.create_max_used_mib <= 0:
        errors.append("service_manager.create.max_used_mib must be positive or null")
    if config.clock_skew_warn_s <= 0:
        errors.append("service_manager.clock_skew.warn_s must be positive")
    if config.clock_skew_fail_s is not None and config.clock_skew_fail_s <= 0:
        errors.append("service_manager.clock_skew.fail_s must be positive or null")
    if config.writer_lock_wait_s < 0:
        errors.append("service_manager.writer_lock_wait_s must be >= 0")
    if config.commit_lock_wait_s is not None and config.commit_lock_wait_s < 0:
        errors.append("service_manager.commit_lock_wait_s must be >= 0 or null")
    if config.log_level not in LOG_LEVEL_NAMES:
        errors.append(
            f"service_manager.log_level must be one of {', '.join(LOG_LEVEL_NAMES)}, "
            f"got {config.log_level!r}"
        )
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


def _parse_tp_size(value: Any, name: Any) -> int:
    """An integral tp_size (1.5 or "two" is an error, never truncated)."""
    if isinstance(value, bool):
        raise ValueError(f"model {name}: tp_size must be an int, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"model {name}: tp_size must be an int, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"model {name}: tp_size must be an int, got {value!r}") from None


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
        tp_size=_parse_tp_size(raw["tp_size"], raw.get("name")),
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
        max_model_len=(int(raw["max_model_len"]) if raw.get("max_model_len") is not None else None),
        sleep_mode_backend=(
            str(raw["sleep_mode_backend"]) if raw.get("sleep_mode_backend") is not None else None
        ),
        vllm_env=_parse_env(raw.get("vllm_env"), f"model {raw.get('name')}: vllm_env"),
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
