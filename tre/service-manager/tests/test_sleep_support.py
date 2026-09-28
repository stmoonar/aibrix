"""VllmOps sleep modes / metrics, clock-skew check, wake headroom (plan 2026-09-27)."""

import logging

import pytest

from tre_sm.api.v2 import ServiceManagerV2, WakeConflict
from tre_sm.clock_check import ClockSkewError, check_clock_skew
from tre_sm.gpu_truth import NodeGpuTruth
from tre_sm.ops.k8s_ops import K8sOps
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.store import StateStore

from sm_test_fakes import FakeRuntime, FakeVllm, LegacyRedis, binding_of, pod, registry


class Response:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class Http:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.posts = []
        self.gets = []

    def post(self, url, *, timeout, headers=None):
        self.posts.append((url, timeout, headers))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get(self, url, *, timeout):
        self.gets.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_sleep_sends_mode_and_hidden_header_single_shot_with_budget_timeout():
    http = Http([TimeoutError("slow")])
    ops = VllmOps(http=http, timeout_s=5.0, max_attempts=3)

    result = ops.sleep("10.0.0.1", mode="wait", timeout_s=42.0, hidden=True)

    assert result.success is False
    # a timed-out mode=wait is never blindly retried
    assert http.posts == [("http://10.0.0.1:8000/sleep?mode=wait", 42.0, {"X-TRE-Hidden": "1"})]


def test_plain_sleep_keeps_the_legacy_url_and_retries():
    http = Http([TimeoutError("slow"), Response(200)])
    ops = VllmOps(http=http, timeout_s=5.0, max_attempts=3)

    assert ops.sleep("10.0.0.1").success is True
    assert [p[0] for p in http.posts] == ["http://10.0.0.1:8000/sleep"] * 2


def test_sleep_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VllmOps(http=Http([])).sleep("10.0.0.1", mode="drain")


def test_metrics_returns_text_or_none():
    ops = VllmOps(http=Http([Response(200, "vllm:num_requests_running 1\n"), Response(500), OSError()]))
    assert ops.metrics("10.0.0.1") == "vllm:num_requests_running 1\n"
    assert ops.metrics("10.0.0.1") is None
    assert ops.metrics("10.0.0.1") is None


class TimeRedis:
    def __init__(self, server_s):
        self.server_s = server_s

    def time(self):
        return (int(self.server_s), int((self.server_s % 1) * 1_000_000))


def test_clock_skew_warns_and_optionally_refuses(caplog):
    redis = TimeRedis(1000.0)
    with caplog.at_level(logging.WARNING):
        skew = check_clock_skew(redis, warn_s=1.0, wall=lambda: 1160.0)
    assert skew == pytest.approx(160.0)
    assert "off Redis TIME" in caplog.text

    assert check_clock_skew(redis, warn_s=1.0, wall=lambda: 1000.2) == pytest.approx(0.2)
    with pytest.raises(ClockSkewError):
        check_clock_skew(redis, warn_s=1.0, fail_s=30.0, wall=lambda: 1160.0)


def test_clock_skew_check_survives_redis_errors():
    class Broken:
        def time(self):
            raise ConnectionError("down")

    assert check_clock_skew(Broken(), warn_s=1.0, fail_s=1.0) is None


class Truth:
    def __init__(self, used, *, missing=False, total=None):
        self.used = list(used)  # successive payloads for GPU-0
        self.missing = missing
        self.total = total

    def node_truth(self, *, node):
        if self.missing:
            return None
        value = self.used.pop(0) if len(self.used) > 1 else self.used[0]
        totals = {} if self.total is None else {"GPU-0": self.total, "GPU-1": self.total}
        return NodeGpuTruth(
            node=node, used_by_uuid={"GPU-0": value, "GPU-1": 500}, total_by_uuid=totals
        )


def _wake_service(truth, *, require=True, **config):
    config = config or {"wake_max_used_mib": 8192}
    sleeping = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="sleeping")
    runtime = FakeRuntime([sleeping])
    vllm = FakeVllm()
    vllm.sleeping["10.0.0.1"] = True
    store = StateStore(LegacyRedis())
    store.save([binding_of(sleeping)], expected_version=0)
    service = ServiceManagerV2(
        registry(wake_truth_wait_s=5.0, **config),
        store,
        runtime_ops=runtime,
        vllm_ops=vllm,
        gpu_truth=truth,
        require_gpu_truth=require,
    )
    return service, vllm


def test_wake_proceeds_when_gpu_truth_shows_headroom():
    service, vllm = _wake_service(Truth([1200]))

    service.put_binding_power("pod-a", awake=True)

    assert ("wake_up", "10.0.0.1") in vllm.calls


def test_wake_fails_closed_when_another_resident_is_awake_on_the_gpu():
    service, vllm = _wake_service(Truth([33_000]))

    with pytest.raises(WakeConflict, match="insufficient wake headroom"):
        service.put_binding_power("pod-a", awake=True)
    assert not any(call[0] == "wake_up" for call in vllm.calls)


def test_wake_rereads_lagging_gpu_truth_after_a_sleep():
    # First read still shows the previous occupant; the next one is fresh.
    service, vllm = _wake_service(Truth([33_000, 900]))

    service.put_binding_power("pod-a", awake=True)

    assert ("wake_up", "10.0.0.1") in vllm.calls


def test_wake_fails_closed_without_gpu_truth_unless_explicitly_permissive():
    service, vllm = _wake_service(Truth([0], missing=True))
    with pytest.raises(WakeConflict, match="gpu truth unavailable"):
        service.put_binding_power("pod-a", awake=True)

    permissive, vllm2 = _wake_service(Truth([0], missing=True), require=False)
    permissive.put_binding_power("pod-a", awake=True)
    assert ("wake_up", "10.0.0.1") in vllm2.calls


def test_wake_threshold_is_a_fraction_of_the_gpu_total_by_default():
    # 40 GiB GPU, default fraction 0.2 -> limit 8192 MiB.
    ok, vllm = _wake_service(Truth([8000], total=40960), wake_max_used_fraction=0.2)
    ok.put_binding_power("pod-a", awake=True)
    assert ("wake_up", "10.0.0.1") in vllm.calls

    busy, vllm2 = _wake_service(Truth([8500], total=40960), wake_max_used_fraction=0.2)
    with pytest.raises(WakeConflict, match="wake limit 8192 MiB"):
        busy.put_binding_power("pod-a", awake=True)

    # An 80 GiB GPU gets a proportionally larger limit (16384 MiB).
    big, vllm3 = _wake_service(Truth([12000], total=81920), wake_max_used_fraction=0.2)
    big.put_binding_power("pod-a", awake=True)
    assert ("wake_up", "10.0.0.1") in vllm3.calls


def test_relative_wake_threshold_fails_closed_without_a_total():
    service, vllm = _wake_service(Truth([100]), wake_max_used_fraction=0.2)
    with pytest.raises(WakeConflict, match="reports no total memory"):
        service.put_binding_power("pod-a", awake=True)
    assert not any(call[0] == "wake_up" for call in vllm.calls)


def test_redis_gpu_truth_parses_total_memory():
    import json as _json

    from tre_sm.gpu_truth import RedisGpuTruth

    class R:
        def get(self, key):
            return _json.dumps(
                {"gpus": [{"uuid": "GPU-0", "used_mib": 10, "total_mib": 40960}, {"uuid": "GPU-1", "used_mib": 5}]}
            )

    truth = RedisGpuTruth(R()).node_truth(node="n")
    assert truth.total_mib("GPU-0") == 40960 and truth.total_mib("GPU-1") is None
    assert truth.used_mib("GPU-1") == 5


class PodApi:
    def __init__(self, pods):
        self.pods = pods
        self.selectors = []

    def list_namespaced_pod(self, *, namespace, label_selector=None):
        self.selectors.append((namespace, label_selector))
        return self.pods


def test_list_ready_pod_names_only_counts_running_ready_pods():
    def plugin(name, phase="Running", ready=True):
        return {
            "metadata": {"name": name},
            "status": {"phase": phase, "containerStatuses": [{"ready": ready, "restartCount": 0}]},
        }

    api = PodApi([plugin("gw-1"), plugin("gw-2", ready=False), plugin("gw-3", phase="Pending")])
    ops = K8sOps(api=api, namespace="default")

    assert ops.list_ready_pod_names(namespace="tre-v2", label_selector="app=gw") == {"gw-1"}
    assert api.selectors == [("tre-v2", "app=gw")]
