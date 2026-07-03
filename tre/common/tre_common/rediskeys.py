RETENTION_MS = 30 * 60 * 1000
FALLBACK_TTL_SECONDS = 2 * 60 * 60

DECISION_LATEST_KEY = "tre:v2:decision:latest"
SM_STATE_KEY = "tre:v2:sm:state"
SM_VERSION_KEY = "tre:v2:sm:version"


def hist_key(pod: str) -> str:
    return f"tre:v2:hist:{pod}"


def inst_key(pod: str) -> str:
    return f"tre:v2:inst:{pod}"


def pods_key(model: str) -> str:
    return f"tre:v2:pods:{model}"
