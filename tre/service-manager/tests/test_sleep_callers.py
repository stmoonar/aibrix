"""Every sleep path goes through the sleep primitive (plan 2026-09-27 D1/D2)."""

import ast
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tre_sm.allocator.slots import Binding, Migration, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.ops.sleep_primitive import SleepPrimitive
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.store import StateStore

from sm_test_fakes import (
    FakeCoordinator,
    FakeLeases,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    binding_of,
    deployment,
    fence,
    pod,
    registry,
    startup_pod,
)

TRE_SM = Path(__file__).resolve().parents[1] / "tre_sm"
PRIMITIVE = TRE_SM / "ops" / "sleep_primitive.py"


TRE_ROOT = TRE_SM.parents[1]
#: Every production Python tree of TRE (tests excluded).
GUARDED_TREES = [
    TRE_ROOT / "service-manager" / "tre_sm",
    TRE_ROOT / "controller" / "tre_controller",
    TRE_ROOT / "deploy",
    TRE_ROOT / "calibration",
    TRE_ROOT / "replayer",
    TRE_ROOT / "loadgen_v1",
    TRE_ROOT / "ui",
]
#: The only code allowed to put a vLLM engine to sleep directly (plan D2):
#: (file relative to tre/, enclosing function or None for the whole file, reason).
SLEEP_CALLER_ALLOWLIST = {
    ("service-manager/tre_sm/ops/sleep_primitive.py", None): "the sleep primitive itself",
    ("service-manager/tre_sm/ops/vllm_ops.py", "sleep"): "the HTTP client method the primitive calls",
    # Offline bootstrap of an empty, controller-paused fleet: each binding is cold
    # started on an idle GPU with no routes and no traffic (it is never routable
    # before it sleeps), so there is nothing to hide or drain. Kept as an explicit
    # exception; the online path is the service-manager fleet repair.
    ("deploy/scripts/staggered_model_fleet.py", "main"): "offline fleet bootstrap without traffic",
}
#: vLLM's /sleep; the service-manager's own read-only GET /v2/sleep is not it.
_SLEEP_URL = re.compile(r"(?<!/v2)/sleep(?![A-Za-z0-9_])")


def _python_files():
    for tree in GUARDED_TREES:
        for path in sorted(tree.rglob("*.py")):
            if "tests" in path.relative_to(tree).parts or path.name.startswith("test_"):
                continue
            yield path


def _docstring_nodes(tree):
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                nodes.add(id(body[0].value))
    return nodes


def _enclosing_functions(tree):
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owner.setdefault(id(child), node.name)
    return owner


def _vllm_ish(node) -> bool:
    return "vllm" in ast.unparse(node).lower()


def _sleep_call_sites(path):
    """(lineno, function, what) of every way to reach vLLM /sleep in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    owner = _enclosing_functions(tree)
    sites = []
    for node in ast.walk(tree):
        what = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if _SLEEP_URL.search(node.value):
                what = f"/sleep URL literal {node.value!r}"
        elif isinstance(node, ast.Attribute) and node.attr == "sleep" and _vllm_ish(node.value):
            what = f"vLLM ops .sleep ({ast.unparse(node)})"  # call or alias
        elif isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            string_args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if func == "getattr" and "sleep" in string_args and node.args and _vllm_ish(node.args[0]):
                what = f"getattr alias of vLLM sleep ({ast.unparse(node)})"
            elif "post" in func.lower() and "sleep" in string_args:
                what = f"POST of the sleep action ({ast.unparse(node)})"
        if what is not None:
            sites.append((node.lineno, owner.get(id(node)), what))
    return sites


def test_only_the_sleep_primitive_reaches_vllm_sleep():
    """Static guard (plan D2): no code but the primitive may POST /sleep, in any
    TRE tree (service-manager, controller, deploy scripts, ...)."""
    offenders = []
    used = set()
    for path in _python_files():
        rel = path.relative_to(TRE_ROOT).as_posix()
        for lineno, function, what in _sleep_call_sites(path):
            if (rel, None) in SLEEP_CALLER_ALLOWLIST:
                used.add((rel, None))
                continue
            if (rel, function) in SLEEP_CALLER_ALLOWLIST:
                used.add((rel, function))
                continue
            offenders.append(f"{rel}:{lineno} in {function}: {what}")
    assert offenders == []
    assert used == set(SLEEP_CALLER_ALLOWLIST), "stale allow-list entries"


def test_the_guard_detects_every_known_bypass(tmp_path):
    sample = tmp_path / "bypass.py"
    sample.write_text(
        "def a(vllm_ops):\n"
        "    vllm_ops.sleep('10.0.0.1')\n"
        "def b(self):\n"
        "    fn = self._vllm.sleep\n"
        "def c(ops):\n"
        "    return getattr(ops.vllm, 'sleep')\n"
        "def d(self, ip):\n"
        "    self._post(ip, 'sleep')\n"
        "def e(http, ip):\n"
        "    http.post(f'http://{ip}:8000/sleep?mode=abort')\n"
        "def f(time):\n"
        "    '''a docstring mentioning POST /sleep is fine'''\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    kinds = [what.split(" ")[0] for _line, _fn, what in _sleep_call_sites(sample)]
    lines = sorted(line for line, _fn, _what in _sleep_call_sites(sample))
    assert lines == [2, 4, 6, 8, 10]
    assert kinds.count("/sleep") == 1


def test_primitive_does_call_vllm_sleep():
    source = PRIMITIVE.read_text(encoding="utf-8")
    assert "self._vllm.sleep(" in source


def test_fleet_repair_cannot_be_built_without_the_sleep_primitive():
    with pytest.raises(TypeError, match="sleep_binding"):
        FleetRepairExecutor(runtime_ops=object(), vllm_ops=object(), safety_gate=object())


def spy_on(primitive, paths):
    """Record (path, budget, pods) of every sleep: every sleep starts with prepare()."""
    original = SleepPrimitive.prepare

    def spy(self, targets, *, path, drain_budget_s=None, **kwargs):
        paths.append((path, drain_budget_s, [t.binding.serve_id for t in targets]))
        return original(self, targets, path=path, drain_budget_s=drain_budget_s, **kwargs)

    primitive.prepare = spy.__get__(primitive)


class Harness:
    def __init__(self, snapshots, *, desired=None, deployments=()):
        self.redis = FakeRedis()
        self.runtime = FakeRuntime(snapshots, deployments)
        self.events = []
        self.runtime.events = self.events
        self.vllm = FakeVllm()
        self.vllm.events = self.events
        for snapshot in snapshots:
            self.vllm.sleeping[snapshot.pod_ip] = snapshot.annotations["tre.aibrix.io/state"] == "sleeping"
        self.store = StateStore(LegacyRedis())
        self.store.save([binding_of(s) for s in snapshots], expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        if desired is not None:
            with fence(self.redis):
                self.fleet.save_desired(desired, expected_version=0)
        self.coordinator = FakeCoordinator(self.redis)
        self.leases = FakeLeases()
        self.service = ServiceManagerV2(
            registry(),
            self.store,
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            operation_coordinator=self.coordinator,
            safety_gate=FakeSafety(),
            fleet_store=self.fleet,
            gpu_leases=self.leases,
        )
        self.paths = []
        spy_on(self.service._sleep_primitive, self.paths)

    def assert_hidden_before_every_sleep(self):
        hidden = set()
        sleeps = 0
        for event in self.events:
            if event[0] == "patch" and event[2] == "hidden":
                hidden.add(event[1])
            if event[:2] == ("vllm", "sleep"):
                sleeps += 1
                ip = event[2]
                owner = next(s.name for s in self.runtime.snapshots.values() if s.pod_ip == ip)
                assert owner in hidden, f"{owner} slept without a prior hide"
                assert event[4] is True  # X-TRE-Hidden
        assert sleeps > 0


def _desired(binding_id, model, gpus, power, *, hidden=False):
    now = datetime.now(timezone.utc).isoformat()
    return DesiredBinding(binding_id, model, "node-a", tuple(gpus), "resident", power, hidden, 1, now, "t", "t")


def _two_awake():
    return [pod("pod-a", "m1", (0,), ip="10.0.0.1"), pod("pod-b", "m1", (1,), ip="10.0.0.2")]


def _desired_two_awake():
    return [_desired("m1/node-a/0", "m1", (0,), "awake"), _desired("m1/node-a/1", "m1", (1,), "awake")]


def test_model_target_shrink_goes_through_primitive_with_request_path():
    h = Harness(_two_awake(), desired=_desired_two_awake())
    client = TestClient(create_app(h.service))

    response = client.put(
        "/v2/models/m1/target",
        json={"wake_replicas": 1, "sleep_path": "urgent", "drain_budget_s": 12.5},
    )

    assert response.status_code == 200, response.text
    assert h.paths == [("urgent", 12.5, ["pod-b"])]
    h.assert_hidden_before_every_sleep()


def test_model_target_default_path_is_scale_down_and_bad_path_is_rejected():
    h = Harness(_two_awake(), desired=_desired_two_awake())
    client = TestClient(create_app(h.service))

    assert client.put("/v2/models/m1/target", json={"wake_replicas": 1, "sleep_path": "yolo"}).status_code == 400
    assert client.put("/v2/models/m1/target", json={"wake_replicas": 1}).status_code == 200
    assert [p[0] for p in h.paths] == ["scale_down"]


def test_binding_power_sleep_goes_through_primitive():
    h = Harness(_two_awake(), desired=_desired_two_awake())
    client = TestClient(create_app(h.service))

    response = client.put(
        "/v2/bindings/pod-a/power",
        json={"awake": False, "sleep_path": "safescale_commit", "drain_budget_s": 40},
    )

    assert response.status_code == 200, response.text
    assert h.paths == [("safescale_commit", 40.0, ["pod-a"])]
    h.assert_hidden_before_every_sleep()
    assert ("release", "m1/node-a/0") in h.leases.calls


def test_v1_compat_scale_down_goes_through_primitive():
    h = Harness(_two_awake(), desired=_desired_two_awake())
    client = TestClient(create_app(h.service))

    response = client.post("/scale_service", params={"model_name": "m1", "scale_type": "down", "scale_value": 1})

    assert response.status_code == 200, response.text
    assert [p[0] for p in h.paths] == ["apa"]  # APA: its own (no-drain) sleep path
    h.assert_hidden_before_every_sleep()


def test_defrag_migration_goes_through_primitive():
    h = Harness([pod("pod-a", "m1", (0,), ip="10.0.0.1")])
    new_pod = pod("pod-a-new", "m1", (1,), ip="10.0.0.9")

    def create(model, slot):
        h.runtime.snapshots[new_pod.name] = new_pod
        h.vllm.sleeping[new_pod.pod_ip] = False
        return new_pod.name

    h.runtime.delete_model_deployment = lambda binding: binding.serve_id
    h.runtime.create_model_deployment = create
    h.runtime.wait_pod_deleted = lambda serve_id: None
    h.runtime.wait_pod_ready = lambda serve_id: h.runtime.snapshots[serve_id]
    binding = binding_of(h.runtime.snapshots["pod-a"])

    actions, moved = h.service._execute_runtime_defrag_migration(
        binding, Migration("pod-a", binding.slot, Slot("node-a", (1,)))
    )

    assert h.paths == [("defrag", None, ["pod-a"])]
    assert [a["action"] for a in actions][:2] == ["hide", "sleep"]
    h.assert_hidden_before_every_sleep()


def test_startup_admission_sleeps_overlapping_resident_through_primitive():
    resident = pod("pod-tp2", "tp2", (0, 1), ip="10.0.0.5")
    h = Harness(
        [resident],
        desired=[
            _desired("tp2/node-a/0,1", "tp2", (0, 1), "awake"),
            _desired("m1/node-a/0", "m1", (0,), "sleeping"),
        ],
    )
    h.runtime.get_startup_pod = lambda name: startup_pod(name, "m1", (0,), uid="new-uid")
    h.runtime.list_startup_resident_snapshots = lambda: h.runtime.list_pod_snapshots()
    h.runtime.admit_startup_pod = lambda name, **kwargs: None

    result = h.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert result["suspended_binding_ids"] == ["tp2/node-a/0,1"]
    assert h.paths == [("startup", None, ["pod-tp2"])]
    h.assert_hidden_before_every_sleep()


def test_startup_convergence_sleeps_through_primitive():
    fresh = pod("pod-a", "m1", (0,), ip="10.0.0.1", state="hidden", admitted=True)
    h = Harness([fresh], desired=[_desired("m1/node-a/0", "m1", (0,), "sleeping")])
    h.runtime.list_startup_resident_snapshots = lambda: h.runtime.list_pod_snapshots()
    h.runtime.clear_startup_admission = lambda name: None
    h.service._k8s_client = None
    h.service._reconcile_unlocked = lambda drop_missing=False: {}

    result = h.service.converge_startups()

    assert result["converged"] == ["pod-a"]
    assert h.paths == [("startup", None, ["pod-a"])]
    h.assert_hidden_before_every_sleep()


def test_fleet_repair_quarantine_sleeps_through_primitive():
    awake = pod("pod-a", "m1", (0,), ip="10.0.0.1")
    h = Harness([awake], deployments=[deployment("m1", (0,))])
    h.runtime.scale_model_deployment = lambda name, *, replicas: None
    h.runtime.wait_deployment_pods_deleted = lambda name: None
    h.runtime.wait_pod_ready = lambda name: h.runtime.snapshots["pod-a"]
    service = ServiceManagerV2(
        registry(),
        h.store,
        runtime_ops=h.runtime,
        vllm_ops=h.vllm,
        safety_gate=FakeSafety(),
    )
    assert service._fleet_repair is not None
    spy_on(service._sleep_primitive, h.paths)

    service._fleet_repair._quarantine_and_sleep_residents(
        type("Op", (), {"assert_active": lambda self: None})(),
        {"m1/node-a/0": deployment("m1", (0,))},
    )

    assert h.paths == [("repair", None, ["pod-a"])]
    h.assert_hidden_before_every_sleep()



def test_partial_multi_target_failure_releases_leases_and_records_only_what_slept():
    """Review P2-7: a scale-down of two pods where one /sleep fails."""
    h = Harness(_two_awake(), desired=_desired_two_awake())
    h.vllm.fail_sleep_for.add("10.0.0.2")  # pod-b's engine refuses to sleep
    client = TestClient(create_app(h.service))

    response = client.put("/v2/models/m1/target", json={"wake_replicas": 0})

    assert response.status_code == 409, response.text
    statuses = {o["serve_id"]: o["status"] for o in response.json()["outcomes"]}
    assert statuses == {"pod-a": "slept", "pod-b": "rolled_back"}
    # GPU lease released only for the pod that slept.
    assert [c for c in h.leases.calls if c[0] == "release"] == [("release", "m1/node-a/0")]
    # Legacy store: exactly the slept pod is asleep.
    by_serve = {b.serve_id: b for b in h.store.load().bindings}
    assert by_serve["pod-a"].awake is False and by_serve["pod-b"].awake is True
    # Desired: the rolled-back pod is awake again; the slept one stays sleeping.
    desired = {d.binding_id: d.power for d in h.fleet.load_desired().bindings}
    assert desired == {"m1/node-a/0": "sleeping", "m1/node-a/1": "awake"}
    # Routing: pod-b is routable again, pod-a sleeping.
    assert h.runtime.snapshots["pod-b"].annotations["tre.aibrix.io/state"] == "awake"
    assert h.runtime.snapshots["pod-a"].annotations["tre.aibrix.io/state"] == "sleeping"
