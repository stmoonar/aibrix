"""Guards for the baseline-scaler manifests (deploy/baselines/tre) and its Dockerfile:
ships scaled to zero and in dry-run, read-only RBAC, nothing cluster-specific."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
TRE_ROOT = DEPLOY_ROOT.parent
BL_DIR = DEPLOY_ROOT / "baselines" / "tre"
POLICIES = ("chiron", "tokenscale", "preserve")
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
    assert env["TRE_BL_POLICY_CONFIG"] == "/etc/tre-baselines/$(TRE_BL_POLICY).yaml"
    names = [e["name"] for e in container["env"]]
    assert names.index("TRE_BL_POLICY") < names.index("TRE_BL_POLICY_CONFIG")  # $(VAR) needs it first
    image = container["image"]
    assert image.startswith("tre-v2-baseline-scaler:") and not image.endswith(":latest")
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


def test_policy_configmaps_present() -> None:
    for policy in POLICIES:
        cm = _one("ConfigMap", f"tre-v2-baseline-{policy}")
        assert cm["metadata"]["namespace"] == "tre-v2"
        assert yaml.safe_load(cm["data"][f"{policy}.yaml"]) == {}


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
    for forbidden in ("COPY service-manager", "COPY controller", "COPY reissue"):
        assert forbidden not in dockerfile
