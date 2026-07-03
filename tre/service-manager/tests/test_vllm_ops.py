from tre_sm.ops.vllm_ops import VllmOps


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class FakeHttp:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, *, timeout):
        self.calls.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_vllm_ops_retries_sleep_and_passes_timeout():
    http = FakeHttp([TimeoutError("slow"), FakeResponse(200, "ok")])
    ops = VllmOps(http=http, timeout_s=1.5, max_attempts=2)

    result = ops.sleep("10.0.0.9")

    assert result.success is True
    assert result.attempts == 2
    assert http.calls == [
        ("http://10.0.0.9:8000/sleep", 1.5),
        ("http://10.0.0.9:8000/sleep", 1.5),
    ]


def test_vllm_ops_treats_conflict_as_idempotent_success_for_wake_up():
    http = FakeHttp([FakeResponse(409, "already awake")])
    ops = VllmOps(http=http, timeout_s=2.0, max_attempts=3)

    result = ops.wake_up("10.0.0.10", port=18000)

    assert result.success is True
    assert result.idempotent is True
    assert result.status_code == 409
    assert result.attempts == 1
    assert http.calls == [("http://10.0.0.10:18000/wake_up", 2.0)]


def test_vllm_ops_reports_failure_after_exhausting_attempts():
    http = FakeHttp([FakeResponse(503, "busy"), FakeResponse(503, "busy")])
    ops = VllmOps(http=http, timeout_s=1.0, max_attempts=2)

    result = ops.sleep("10.0.0.11")

    assert result.success is False
    assert result.status_code == 503
    assert result.attempts == 2
    assert result.message == "busy"
