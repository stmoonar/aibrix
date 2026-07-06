RETENTION_MS = 30 * 60 * 1000
FALLBACK_TTL_SECONDS = 2 * 60 * 60

DECISION_LATEST_KEY = "tre:v2:decision:latest"
# S5.1: per-model decision time-series (score = window_end_ms). Backs the UI timelines
# and post-hoc experiment analysis. Retained ~24h with a TTL backstop.
DECISION_HIST_RETENTION_MS = 24 * 60 * 60 * 1000
DECISION_HIST_TTL_SECONDS = 25 * 60 * 60
SM_STATE_KEY = "tre:v2:sm:state"
SM_VERSION_KEY = "tre:v2:sm:version"
CONTROLLER_SAFESCALE_PROBES_KEY = "tre:v2:controller:safescale:probes"


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
