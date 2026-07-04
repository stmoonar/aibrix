from __future__ import annotations

TTFT_DEFINITION = "time from request send to first SSE content byte arrival"
TOKEN_CONTROL_FIELDS = ("prompt_tokens", "max_output_tokens")


def metric_semantics() -> dict[str, object]:
    return {
        "ttft": TTFT_DEFINITION,
        "token_controls": list(TOKEN_CONTROL_FIELDS),
    }
