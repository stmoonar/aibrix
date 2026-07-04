from __future__ import annotations

from tre_replayer.metrics import TTFT_DEFINITION, TOKEN_CONTROL_FIELDS, metric_semantics


def test_metric_semantics_documents_ttft_and_token_controls() -> None:
    semantics = metric_semantics()

    assert semantics["ttft"] == TTFT_DEFINITION
    assert "first SSE content byte" in semantics["ttft"]
    assert TOKEN_CONTROL_FIELDS == ("prompt_tokens", "max_output_tokens")
    assert semantics["token_controls"] == list(TOKEN_CONTROL_FIELDS)
