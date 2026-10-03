"""The run provenance records the images a run is measured on (read-only kubectl), and
records null with a warning - never fails - when kubectl cannot say."""
from __future__ import annotations

import json
import subprocess

from scripts import calibration_campaign as campaign


def _pod(name: str, containers: list[tuple[str, str, str]]) -> dict:
    return {"metadata": {"name": name, "namespace": "ns"}, "spec": {
        "nodeName": "n1", "containers": [{"name": c, "image": i} for c, i, _ in containers]},
        "status": {"containerStatuses": [{"name": c, "imageID": d} for c, _, d in containers]}}


def _fake_kubectl(calls: list):
    def run(argv, **kw):
        calls.append(argv)
        assert argv[0] == "kubectl" and "get" in argv  # read-only
        if "deployment" in argv:
            doc = {"spec": {"selector": {"matchLabels": {"app": "gw"}},
                            "template": {"spec": {"containers": [{"name": "gp", "image": "gw:tag-nozmq2"}]}}}}
        elif "model.aibrix.ai/name=m7,tre.aibrix.io/routable=true" in argv:
            doc = {"items": [_pod("m7-a", [("vllm-openai", "vllm:ts-1", "sha256:aa"),
                                           ("tre-reissue-sidecar", "vllm:ts-1", "sha256:aa")])]}
        elif "app=gw" in argv:
            doc = {"items": [_pod("gw-0", [("gp", "gw:tag-nozmq2", "sha256:bb")])]}
        else:
            raise subprocess.CalledProcessError(1, argv, stderr="forbidden")
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(doc))
    return run


def test_images_of_model_pods_and_gateway_plugins_are_recorded(capsys) -> None:
    calls: list = []
    rec = campaign.image_provenance(["m7", "m8"], model_namespace="default", control_namespace="cp",
                                    gateway_deployment="gwp", run=_fake_kubectl(calls))
    pods = rec["model_pods"]["m7"]
    assert pods[0]["pod"] == "ns/m7-a"
    assert {(c["name"], c["image"], c["image_id"]) for c in pods[0]["containers"]} == {
        ("vllm-openai", "vllm:ts-1", "sha256:aa"), ("tre-reissue-sidecar", "vllm:ts-1", "sha256:aa")}
    gw = rec["gateway_plugins"]
    assert gw["deployment"] == "gwp" and gw["containers"] == [{"name": "gp", "image": "gw:tag-nozmq2"}]
    assert gw["pods"][0]["containers"][0]["image_id"] == "sha256:bb"
    # m8 could not be read: null, a warning, and the run goes on
    assert rec["model_pods"]["m8"] is None and any("m8" in w for w in rec["warnings"])
    assert "WARNING: run provenance" in capsys.readouterr().err


def test_no_kubectl_means_nulls_not_a_failed_run() -> None:
    def missing(argv, **kw):
        raise FileNotFoundError("kubectl")

    rec = campaign.image_provenance(["m7"], model_namespace="default", control_namespace="cp", run=missing)
    assert rec["model_pods"] == {"m7": None} and rec["gateway_plugins"] is None and len(rec["warnings"]) == 2
