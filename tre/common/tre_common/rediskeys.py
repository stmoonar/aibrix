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
CONTROLLER_SAFESCALE_PROBES_KEY = "tre:v2:controller:safescale:probes"
CONTROLLER_ORPHAN_WATCH_KEY = "tre:v2:controller:orphan_watch"
CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY = "tre:v2:controller:alerts:hidden_orphans"


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
