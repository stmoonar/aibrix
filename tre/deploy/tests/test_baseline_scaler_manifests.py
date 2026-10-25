"""Guards for the baseline-scaler manifests (deploy/baselines/tre) and its Dockerfile:
ships scaled to zero and in dry-run, read-only RBAC, nothing cluster-specific."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
TRE_ROOT = DEPLOY_ROOT.parent
BL_DIR = DEPLOY_ROOT / "baselines" / "tre"
POLICIES = ("chiron", "tokenscale", "preserve")
BASELINE_SCALER_TAG = "20261020-db24f36b"  # built from that commit (clean git archive of tre/)
IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _docs() -> list[dict]:
    kustomization = yaml.safe_load((BL_DIR / "kustomization.yaml").read_text(encoding="utf-8"))
    docs: list[dict] = []
    for name in kustomization["resources"]:
        docs.extend(d for d in yaml.safe_load_all((BL_DIR / name).read_text(encoding="utf-8")) if d)
    return docs


def _one(kind: str, name: str) -> dict:
    found = [d for d in _docs() if d["kind"] == kind and d["metadata"]["name"] == name]
    assert len(found) == 1, (kind, name)
    return found[0]


def _env(container: dict) -> dict:
    return {e["name"]: e.get("value") for e in container.get("env", [])}


def test_deployment_ships_off_and_dry_run() -> None:
    dep = _one("Deployment", "tre-v2-baseline-scaler")
    assert dep["metadata"]["namespace"] == "tre-v2"
    assert dep["spec"]["replicas"] == 0
    assert dep["spec"]["strategy"]["type"] == "Recreate"
    pod = dep["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "tre-v2-baseline-scaler"
    assert "nodeSelector" not in pod and "nodeName" not in pod
    (container,) = pod["containers"]
    env = _env(container)
    assert env["TRE_BL_DRY_RUN"] == "true"
    for key in ("TRE_SM_URL", "TRE_REDIS_URL", "TRE_BL_POLICY", "TRE_BL_TICK_S", "TRE_BL_LOG_DIR",
                "TRE_BL_POLICY_CONFIG", "TRE_REGISTRY_PATH"):
        assert env.get(key), key
    assert env["TRE_REGISTRY_PATH"] == "/etc/tre/registry.yaml"
    # no drain is ever asked for: the abort path, and none of the retired names
    assert env["TRE_BL_ABORT_SLEEP_PATH"] == "urgent"
    assert "TRE_BL_SLEEP_PATH" not in env and "TRE_BL_DRAIN_BUDGET_S" not in env
    assert env["TRE_BL_POLICY_CONFIG"] == "/etc/tre-baselines/$(TRE_BL_POLICY).yaml"
    names = [e["name"] for e in container["env"]]
    assert names.index("TRE_BL_POLICY") < names.index("TRE_BL_POLICY_CONFIG")  # $(VAR) needs it first
    image = container["image"]
    assert image.startswith("tre-v2-baseline-scaler:") and not image.endswith(":latest")
    # the deployed tag: bump it here, in baseline-scaler.yaml and in kustomization.yaml together
    kustomization = yaml.safe_load((BL_DIR / "kustomization.yaml").read_text(encoding="utf-8"))
    (img,) = [i for i in kustomization["images"] if i["name"] == "tre-v2-baseline-scaler"]
    assert image == f"tre-v2-baseline-scaler:{BASELINE_SCALER_TAG}" and img["newTag"] == BASELINE_SCALER_TAG
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts["registry"] == "/etc/tre" and mounts["policies"] == "/etc/tre-baselines"
    volumes = {v["name"]: v for v in pod["volumes"]}
    assert volumes["registry"]["configMap"]["name"] == "tre-v2-registry"
    projected = [s["configMap"]["name"] for s in volumes["policies"]["projected"]["sources"]]
    assert projected == [f"tre-v2-baseline-{p}" for p in POLICIES]


def test_probes_split_liveness_from_readiness() -> None:
    (container,) = _one("Deployment", "tre-v2-baseline-scaler")["spec"]["template"]["spec"]["containers"]
    # readiness follows the ticks (503 after repeated failures, e.g. SM down) ...
    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    # ... liveness only the loop: an SM / Redis outage must never restart the pod
    assert container["livenessProbe"]["httpGet"]["path"] == "/livez"


def test_trace_volume_is_a_patchable_empty_dir() -> None:
    pod = _one("Deployment", "tre-v2-baseline-scaler")["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    mounts = {m["name"]: m for m in container["volumeMounts"]}
    assert mounts["traces"]["mountPath"] == "/etc/tre-baselines-traces" and mounts["traces"]["readOnly"] is True
    volumes = {v["name"]: v for v in pod["volumes"]}
    assert volumes["traces"] == {"name": "traces", "emptyDir": {}}  # no hostPath / PVC baked in
    example = yaml.safe_load((TRE_ROOT / "baselines" / "examples" / "preserve.yaml").read_text(encoding="utf-8"))
    assert example["trace_path"].startswith("/etc/tre-baselines-traces/")


def test_policy_configmaps_hold_frozen_parameters() -> None:
    """The shipped ConfigMaps are the frozen baseline parameters: every policy starts on
    them for every registry model (the policies refuse missing / placeholder values). A block
    still marked ``calibration: pending`` must be refused (fail closed), never started."""
    from types import SimpleNamespace

    from tre_common.registry import load_registry

    from tre_baselines.config import Config, model_limits
    from tre_baselines.policies import build_policy
    from tre_baselines.policies.preserve import PreServePolicy

    models = model_limits(load_registry(str(DEPLOY_ROOT / "registry.yaml")), strict=False)
    assert models
    for policy in POLICIES:
        cm = _one("ConfigMap", f"tre-v2-baseline-{policy}")
        assert cm["metadata"]["namespace"] == "tre-v2"
        params = yaml.safe_load(cm["data"][f"{policy}.yaml"])
        cfg = Config(sm_url="x", redis_url="y", policy=policy, policy_params=params, models=models)
        if params.get("calibration") == "pending":
            with pytest.raises(ValueError, match="calibration pending"):
                build_policy(policy, cfg)
            continue
        if policy == "preserve":
            # Tier-1 reads the trace at start from the per-environment trace volume; the shipped
            # weights load (sha256 checked); the history comes with the trace (injected here)
            assert params["trace_path"].startswith("/etc/tre-baselines-traces/")
            oracle = SimpleNamespace(window_s=float(params["window_s"]), max_tokens_max={}, path=params["trace_path"],
                                     n_windows=24)
            hist = {m: [[1.0, 1.0]] * int(params["seq_len"]) for m in models}
            PreServePolicy(cfg, oracle=oracle, history=hist)
        else:
            build_policy(policy, cfg)


def test_pending_blocks_are_what_the_freeze_tool_renders() -> None:
    """The TokenScale / PreServe blocks are written by tools/freeze_params (never by hand): a
    pending block equals its pending rendering; the PreServe block names the shipped weights'
    sha256."""
    from tre_baselines.tools import freeze_params as fp

    for policy in ("tokenscale", "preserve"):
        text = _one("ConfigMap", f"tre-v2-baseline-{policy}")["data"][f"{policy}.yaml"]
        if yaml.safe_load(text).get("calibration") != "pending":
            continue
        want = fp.render_tokenscale(None, None, {}) if policy == "tokenscale" else \
            fp.render_preserve(None, {}, fp.weights_sha256())
        assert text == want, policy


def test_cluster_overlay_mounts_traces_read_only_and_stays_off() -> None:
    """The 75/76 overlay only swaps the trace volume to a host directory; the mount stays
    read-only (base) and the deployment ships at 0 replicas."""
    kust = yaml.safe_load((DEPLOY_ROOT / "baselines" / "tre-cluster-75-76" / "kustomization.yaml").read_text(encoding="utf-8"))
    assert kust["resources"] == ["../tre"]
    (patch,) = kust["patches"]
    doc = yaml.safe_load(patch["patch"])
    assert set(doc["spec"]) == {"template"}  # no replicas / env override
    (vol,) = doc["spec"]["template"]["spec"]["volumes"]
    assert vol["name"] == "traces" and vol["emptyDir"] is None and vol["hostPath"]["type"] == "Directory"


def test_rbac_is_read_only() -> None:
    sa = _one("ServiceAccount", "tre-v2-baseline-scaler")
    assert sa["metadata"]["namespace"] == "tre-v2"
    for doc in _docs():
        if doc["kind"] in {"Role", "ClusterRole"}:
            for rule in doc["rules"]:
                assert set(rule["verbs"]) <= {"get", "list", "watch"}, doc["metadata"]["name"]
        if doc["kind"] in {"RoleBinding", "ClusterRoleBinding"}:
            assert [s["name"] for s in doc["subjects"]] == ["tre-v2-baseline-scaler"]
    assert not any(d["kind"] in {"ClusterRole", "ClusterRoleBinding"} for d in _docs())


def test_nothing_cluster_specific() -> None:
    for path in sorted(BL_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert not IP_RE.search(text), path.name
        assert "nscc-" not in text and "/data/nfs_shared_data" not in text, path.name
        assert "nodePort" not in text and "hostname" not in text, path.name


def test_dockerfile_contract() -> None:
    dockerfile = (TRE_ROOT / "baselines" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile and "latest" not in dockerfile.lower()
    for directive in ("COPY common", "COPY deploy", "COPY replayer", "COPY baselines", "requirements-test.txt"):
        assert directive in dockerfile
    (pythonpath,) = [ln for ln in dockerfile.splitlines() if ln.startswith("ENV PYTHONPATH=")]
    entries = pythonpath.split("=", 1)[1].split(":")
    # trace_oracle imports tre_replayer (segment traces of PreServe Tier-1)
    for entry in ("/app/tre/common", "/app/tre/deploy", "/app/tre/replayer", "/app/tre/baselines"):
        assert entry in entries, entry
    assert 'CMD ["python", "-m", "tre_baselines.main"]' in dockerfile
    # PreServe mLSTM: numpy pinned, one BLAS thread (v1 D27)
    reqs = (TRE_ROOT / "baselines" / "requirements-baselines.txt").read_text(encoding="utf-8")
    assert re.search(r"^numpy==\d+\.\d+\.\d+$", reqs, re.M)
    assert "requirements-baselines.txt" in dockerfile and "ENV OPENBLAS_NUM_THREADS=1" in dockerfile
    for forbidden in ("COPY service-manager", "COPY controller", "COPY reissue"):
        assert forbidden not in dockerfile


SENSITIVITY_ROWS = (  # (file, policy, the only key that differs from the main ConfigMap)
    # the TokenScale / PreServe rows of 2026-10-06 belonged to the old single-shape design and went
    # with it (2026-10-26 v1 port); re-create them from the frozen blocks when a row is wanted
    ("chiron-alg1.yaml", "chiron", "batch_mode"),
)


@pytest.mark.parametrize("fname,policy,key", SENSITIVITY_ROWS)
def test_sensitivity_rows_differ_from_main_only_in_their_key(fname: str, policy: str, key: str) -> None:
    """A sensitivity row replaces the main ConfigMap of its policy while applied: same name,
    same parameters except its one key, and it starts the policy for every model."""
    from types import SimpleNamespace

    from tre_common.registry import load_registry

    from tre_baselines.config import Config, model_limits
    from tre_baselines.policies import build_policy
    from tre_baselines.policies.preserve import PreServePolicy

    text = (BL_DIR / "sensitivity" / fname).read_text(encoding="utf-8")
    assert not IP_RE.search(text) and "nscc-" not in text and "/data/nfs_shared_data" not in text
    (sens,) = [d for d in yaml.safe_load_all(text) if d]
    main = _one("ConfigMap", f"tre-v2-baseline-{policy}")
    assert sens["metadata"] == main["metadata"]
    sp, mp = (yaml.safe_load(d["data"][f"{policy}.yaml"]) for d in (sens, main))
    assert sp[key] != mp[key]
    assert {k: v for k, v in sp.items() if k != key} == {k: v for k, v in mp.items() if k != key}
    models = model_limits(load_registry(str(DEPLOY_ROOT / "registry.yaml")), strict=False)
    cfg = Config(sm_url="x", redis_url="y", policy=policy, policy_params=sp, models=models)
    if policy == "preserve":
        PreServePolicy(cfg, oracle=SimpleNamespace(window_s=float(sp["window_s"]), max_tokens_max={}))
    else:
        build_policy(policy, cfg)


def test_baseline_tick_equals_the_tre_fast_loop() -> None:
    """Every arm decides on the same cadence (2026-10-26, v1 D20): the shell's tick = the TRE
    controller's fast loop (rescue interval); the shell's code default is the same value."""
    (container,) = _one("Deployment", "tre-v2-baseline-scaler")["spec"]["template"]["spec"]["containers"]
    tick = float(_env(container)["TRE_BL_TICK_S"])
    ctrl = [d for d in yaml.safe_load_all((DEPLOY_ROOT / "overlays" / "tre-v2" / "controller.yaml")
                                          .read_text(encoding="utf-8")) if d and d.get("kind") == "Deployment"]
    rescue = {float(e["value"]) for d in ctrl for c in d["spec"]["template"]["spec"]["containers"]
              for e in c.get("env", []) if e["name"] == "TRE_RESCUE_INTERVAL_SECONDS"}
    assert rescue == {tick}
    from tre_baselines.config import Config  # noqa: PLC0415
    assert Config.__dataclass_fields__["tick_s"].default == tick
