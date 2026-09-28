from __future__ import annotations

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.reconcile import PodRecord


class FakeOps:
    def list_pod_snapshots(self):
        return [
            K8sPodSnapshot(
                name="serve-a",
                model="dsqwen-7b",
                node="nscc-ds-4a100-node9",
                env={"CUDA_VISIBLE_DEVICES": "0"},
            )
        ]


def test_k8s_pod_client_from_ops_converts_snapshots_to_reconcile_records() -> None:
    topology = ClusterTopology(
        nodes=(NodeSpec(name="nscc-ds-4a100-node9", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )
    client = K8sPodClientFromOps(topology, FakeOps())

    assert client.list_pods() == [
        PodRecord(
            serve_id="serve-a",
            model="dsqwen-7b",
            node="nscc-ds-4a100-node9",
            cuda_visible_devices="0",
        )
    ]


def test_gpu_truth_required_from_env_defaults_to_fail_closed() -> None:
    from tre_sm.server import gpu_truth_required_from_env

    assert gpu_truth_required_from_env({}) is True


def test_gpu_truth_required_from_env_can_be_disabled_for_emergencies() -> None:
    from tre_sm.server import gpu_truth_required_from_env

    assert gpu_truth_required_from_env({"TRE_GPU_TRUTH_REQUIRED": "false"}) is False
    assert gpu_truth_required_from_env({"TRE_GPU_TRUTH_REQUIRED": "0"}) is False
    assert gpu_truth_required_from_env({"TRE_GPU_TRUTH_REQUIRED": "true"}) is True



def test_service_manager_refuses_an_invalid_sleep_configuration():
    import pytest

    from tre_common.registry import ClusterTopology, Registry, parse_service_manager_config
    from tre_sm.server import check_service_manager_config

    check_service_manager_config(Registry(ClusterTopology(nodes=()), []))
    slow = parse_service_manager_config({"sleep": {"sleep_call_timeout_s": 200}})
    with pytest.raises(RuntimeError, match="worst-case sleeping service-manager call"):
        check_service_manager_config(Registry(ClusterTopology(nodes=()), [], service_manager=slow))


# ------------------------------------------------------------------ B5 logging
import io
import logging

import pytest


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers = list(root.handlers)
    levels = {name: logging.getLogger(name).level for name in ("tre_sm", "tre_common", "uvicorn")}
    root_level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(root_level)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_log_level_env_overrides_registry_and_defaults_to_info():
    from tre_sm.server import resolve_log_level

    assert resolve_log_level({}) == logging.INFO
    assert resolve_log_level({}, "debug") == logging.DEBUG
    assert resolve_log_level({"TRE_SM_LOG_LEVEL": "warning"}, "DEBUG") == logging.WARNING
    assert resolve_log_level({"TRE_SM_LOG_LEVEL": ""}, "ERROR") == logging.ERROR
    # A typo in the env never keeps the SM from starting.
    assert resolve_log_level({"TRE_SM_LOG_LEVEL": "loud"}) == logging.INFO


def test_configure_logging_emits_sm_info_lines_once_with_timestamps(restore_logging):
    from tre_sm.server import _HANDLER_NAME, configure_logging

    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # what uvicorn leaves behind
    configure_logging(logging.INFO)
    configure_logging(logging.INFO)  # idempotent: one handler
    ours = [h for h in root.handlers if h.get_name() == _HANDLER_NAME]
    assert len(ours) == 1
    stream = io.StringIO()
    ours[0].setStream(stream)

    logging.getLogger("tre_sm.ops.sleep_primitive").info("sleep committed")
    logging.getLogger("tre_common.registry").info("registry loaded")
    logging.getLogger("kubernetes.client").info("noisy third-party line")

    out = stream.getvalue()
    assert "INFO tre_sm.ops.sleep_primitive: sleep committed" in out
    assert "INFO tre_common.registry: registry loaded" in out
    assert "noisy third-party line" not in out  # root stays at WARNING
    assert out.splitlines()[0][:4].isdigit()  # asctime prefix
    assert root.level == logging.WARNING


def test_configure_logging_leaves_uvicorn_loggers_alone(restore_logging):
    from tre_sm.server import configure_logging

    uvicorn = logging.getLogger("uvicorn")
    uvicorn.setLevel(logging.INFO)
    before = (list(uvicorn.handlers), uvicorn.propagate)
    configure_logging(logging.DEBUG)
    assert (list(uvicorn.handlers), uvicorn.propagate) == before
    assert uvicorn.level == logging.INFO
    assert logging.getLogger("tre_sm").level == logging.DEBUG


def test_registry_log_level_is_validated():
    from tre_common.registry import ClusterTopology, Registry, parse_service_manager_config
    from tre_sm.server import check_service_manager_config

    assert parse_service_manager_config({"log_level": "debug"}).log_level == "DEBUG"
    assert parse_service_manager_config({}).log_level == "INFO"
    assert parse_service_manager_config({"log_level": None}).log_level == "INFO"
    bad = parse_service_manager_config({"log_level": "chatty"})
    with pytest.raises(RuntimeError, match="service_manager.log_level"):
        check_service_manager_config(Registry(ClusterTopology(nodes=()), [], service_manager=bad))
