RETENTION_MS = 30 * 60 * 1000
FALLBACK_TTL_SECONDS = 2 * 60 * 60

# SINGLE SOURCE OF TRUTH for the instant/histogram scrape cadence. The AIBrix gateway
# writes the redis inst/hist buckets (inst_key/hist_key, and the legacy
# aibrix:pod_instant_metrics_* keys) on a fixed, boundary-aligned ticker whose period is
# the Go constant `RequestTraceWriteInterval = 10 * time.Second`
# (aibrix pkg/cache/trace.go). That constant lives in aibrix-system and is not tunable
# from TRE, so the Python side must mirror it here rather than re-inventing a second magic
# number. `MetricsStore._instant_avg` uses this to count expected samples in a window, and
# the offline r3 sidecar sampler uses ~2x this as its freshness lookback. Keep in sync if
# the gateway's RequestTraceWriteInterval ever changes.
SCRAPE_INTERVAL_MS = 10_000

DECISION_LATEST_KEY = "tre:v2:decision:latest"
# S5.1: per-model decision time-series (score = window_end_ms). Backs the UI timelines
# and post-hoc experiment analysis. Retained ~24h with a TTL backstop.
DECISION_HIST_RETENTION_MS = 24 * 60 * 60 * 1000
DECISION_HIST_TTL_SECONDS = 25 * 60 * 60
SM_STATE_KEY = "tre:v2:sm:state"
SM_VERSION_KEY = "tre:v2:sm:version"
SM_WRITER_LOCK_KEY = "tre:v2:sm:writer_lock"
SM_FENCE_COUNTER_KEY = "tre:v2:sm:fence_counter"
# Fair writer lock (review 2 P2-4): FIFO waiter queue (zset ticket -> arrival
# sequence), each waiter's poll deadline (hash ticket -> Redis-TIME ms) and the
# arrival sequence counter.
SM_WRITER_QUEUE_KEY = "tre:v2:sm:writer_queue"
SM_WRITER_QUEUE_DEADLINES_KEY = "tre:v2:sm:writer_queue_deadlines"
SM_WRITER_QUEUE_SEQ_KEY = "tre:v2:sm:writer_queue_seq"
SM_OPERATIONS_KEY = "tre:v2:sm:operations"
SM_DESIRED_KEY = "tre:v2:sm:desired"
SM_DESIRED_VERSION_KEY = "tre:v2:sm:desired_version"
SM_OBSERVED_KEY = "tre:v2:sm:observed"
SM_OBSERVED_VERSION_KEY = "tre:v2:sm:observed_version"
SM_GPU_LEASES_KEY = "tre:v2:sm:gpu_leases"
#: Controller run mode, "active" | "observe" (observe = compute and record only).
#: Absent, not a mode, or unreadable before the first successful read = observe
#: (fail-closed). Independent of SM_ACTUATION_KEY (tre_common.run_mode).
CONTROLLER_MODE_KEY = "tre:v2:controller:mode"
#: Service-manager actuation switch, "active" | "observe": whether the SM
#: SUPERVISOR may change capacity or recreate / delete workloads on its own. The
#: SM HTTP write API is NOT gated by it (the APA arm and operators drive the SM
#: directly). Independent of CONTROLLER_MODE_KEY: absent or not a mode =
#: observe (fail-closed; never derived from the controller mode).
SM_ACTUATION_KEY = "tre:v2:sm:actuation"
#: SM maintenance lock (JSON {"operation_id", "kind", "owner", "since_ms"}) held
#: by a running fleet repair for its whole run; deleting it aborts the repair at
#: its next safety check. Replaces borrowing CONTROLLER_MODE_KEY as that lock.
SM_MAINTENANCE_KEY = "tre:v2:sm:maintenance"
#: LIST (newest first, capped) of supervisor actions the SM did NOT take because
#: its actuation is observe: JSON {"ts_ms", "action", "detail"}.
SM_SUPPRESSED_ACTIONS_KEY = "tre:v2:sm:actuation:suppressed"
SM_SUPPRESSED_ACTIONS_MAX = 200
CONTROLLER_SAFESCALE_PROBES_KEY = "tre:v2:controller:safescale:probes"
CONTROLLER_ORPHAN_WATCH_KEY = "tre:v2:controller:orphan_watch"
CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY = "tre:v2:controller:alerts:hidden_orphans"
CONTROLLER_SIGNAL_LOG_KEY = "tre:v2:controller:signal_log"


def controller_safescale_probe_journal_key(request_id: str) -> str:
    return f"tre:v2:controller:safescale:probe:{request_id}:journal"


def hist_key(pod: str) -> str:
    return f"tre:v2:hist:{pod}"


def inst_key(pod: str) -> str:
    return f"tre:v2:inst:{pod}"


def pods_key(model: str) -> str:
    return f"tre:v2:pods:{model}"


def decision_hist_key(model: str) -> str:
    return f"tre:v2:decision:hist:{model}"


# --- Transparent sleep: gateway plugin <-> service-manager contract (plan 2026-09-27) ---
# Written by the gateway plugin (Go constants must match), read by the service-manager.
# Keys are per pod NAME (no namespace): TRE model pods must have unique names across
# namespaces (the generated Deployments embed model, node and GPUs in the name).
#: ZSET, member = plugin instance id (pod name), score = heartbeat epoch ms (every 2 s).
#: The SM treats an instance as live while its score keeps CHANGING between SM reads
#: (SM monotonic clock), so neither side's wall clock matters.
GW_INSTANCES_KEY = "tre:v2:gw:instances"
#: k8s Pod annotation (integer) bumped by the SM in the same patch that changes the
#: tre.aibrix.io/routable label; plugins report the last generation they applied.
ROUTE_GEN_ANNOTATION = "tre.aibrix.io/route-gen"


def gw_seen_key(pod: str) -> str:
    """HASH field=instance id, value JSON {"gen":int,"routable":bool,"ts":ms}; TTL 300 s."""
    return f"tre:v2:gw:seen:{pod}"


def gw_inflight_key(pod: str) -> str:
    """HASH field=instance id, value JSON {"total":int,"non_continuable":int,"ts":ms}."""
    return f"tre:v2:gw:inflight:{pod}"


# Written by the service-manager sleep primitive.
#: HASH field=pod name, value JSON: the sleep in progress for that pod (crash evidence).
SM_SLEEP_OPS_KEY = "tre:v2:sm:sleep_ops"
#: HASH field=binding_id, value JSON: the sleep reservation fencing a draining binding
#: and its GPUs while the drain runs outside the writer lock (expiry = Redis TIME).
SM_SLEEP_RESERVATIONS_KEY = "tre:v2:sm:sleep_reservations"
#: HASH of integer counters (sleeps, forced aborts, ack fallbacks, rollbacks, ...).
SM_SLEEP_STATS_KEY = "tre:v2:sm:sleep_stats"
#: HASH field=binding_id, value JSON: a wake whose /wake_up runs outside the writer
#: lock (S6, 2026-09-30) - the pod, its GPUs and the desired power before the wake
#: (crash evidence; resolved by ServiceManagerV2.recover_wake_journal).
SM_WAKE_OPS_KEY = "tre:v2:sm:wake_ops"
#: HASH of wake counters (wake_done_total, wake_failed_total, wake_parallel_max,
#: wake_compensating_sleep_total, truth_fallback_total:<reason>, ...): GET /v2/wake.
SM_WAKE_STATS_KEY = "tre:v2:sm:wake_stats"
#: STRING (any value, with a TTL) per GPU: test hooks of the service-manager, read
#: only while registry service_manager.test_hooks is true. ``refuse_wake`` refuses
#: a wake on the GPU (409 gpu_busy); ``fail_wake`` makes the wake fail after
#: /wake_up (the engine is awake: exercises the compensating sleep).
SM_FAULT_KEY_PREFIX = "tre:v2:sm:fault:"
#: LIST of recent gateway-ack latencies (ms), newest first, capped.
SM_SLEEP_ACK_LATENCY_KEY = "tre:v2:sm:sleep_ack_latency_ms"
SM_SLEEP_ACK_LATENCY_MAX = 5000


# --- gpu-truth: DaemonSet agent <-> service-manager ---------------------------------
# The agent (deploy/scripts/gpu_truth_agent.py) runs standalone from a ConfigMap and
# repeats these literals; deploy/tests/test_gpu_truth_agent.py keeps them equal.
#: STRING (JSON) per node, SETEX by the agent: {"node", "timestamp", "gpus": [{"uuid",
#: "used_mib", "total_mib"}], "seq", "refresh_seq"}. Readers SCAN ``tre:gpu_truth:*``.
GPU_TRUTH_KEY_PREFIX = "tre:gpu_truth:"
#: STRING counter per node: the service-manager INCRs it to ask for a fresh sample
#: (reply N); the agent polls it and publishes a sample with ``refresh_seq`` >= N
#: taken after it read N. Deliberately NOT under ``tre:gpu_truth:`` (that prefix is
#: scanned for node payloads).
GPU_TRUTH_REFRESH_KEY_PREFIX = "tre:gpu_truth_refresh:"


def gpu_truth_key(node: str) -> str:
    return f"{GPU_TRUTH_KEY_PREFIX}{node}"


def gpu_truth_refresh_key(node: str) -> str:
    return f"{GPU_TRUTH_REFRESH_KEY_PREFIX}{node}"


def sm_fault_key(kind: str, node: str, gpu: int) -> str:
    """``tre:v2:sm:fault:<kind>:<node>/<gpu>`` (see SM_FAULT_KEY_PREFIX)."""
    return f"{SM_FAULT_KEY_PREFIX}{kind}:{node}/{int(gpu)}"
