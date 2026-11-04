"""campaign-bl-preserve-muold-20261104 (ablation: PreServe V1 with the old ladder mu, Alternating then Sinusoidal): the spec,
its driver and its trace list agree with the paper campaign it copies and carry no site path.

No cluster, no data root: the spec is loaded with a fake CAMPAIGN_DATA_ROOT. The invariants:
* 2 arms: preserve-v1-muold on Alternating, then on Sinusoidal; reports off; the paper campaign's registry pin;
* the arm = the paper campaign's arm preserve-v1 except mu (same base, same policy keys otherwise), and mu is not the
  ConfigMap's Alg. 1 mu; the values are the ones the spec header names (the driver checks them against the mu file);
* the traces are the paper campaign's files for those two traces (sha256), named in the header; each has a train-tail history;
* the spec and the driver contain no NFS path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

RUNNER = Path(__file__).resolve().parents[1] / "runner"
TRE = RUNNER.parents[1]
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

NAME = "campaign-bl-preserve-muold-20261104"
SPEC = RUNNER / "campaigns" / f"{NAME}.yaml"
DRIVER_DIR = RUNNER / "drivers" / "bl-preserve-muold-20261104"
PAPER = "campaign-bl-preserve-paper-20261104"
HISTORIES = TRE / "baselines" / "histories" / "preserve_history_w10_train-tail-20261104.sha256"
ROOT = "/fake/data"
TRACES = ["Alternating_hot_model_periodic_A", "Sinusoidal_demand"]


def _load(tmp_path, name=NAME):
    return C.load_campaign(RUNNER / "campaigns" / f"{name}.yaml",
                           environ={"CAMPAIGN_DATA_ROOT": ROOT, "TRE_REPO": str(tmp_path)})


def _sha_list(path: Path) -> dict:
    return {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in path.read_text().splitlines() if l.strip()}


def _cm_mu() -> dict:
    docs = [d for d in yaml.safe_load_all((TRE / "deploy" / "baselines" / "tre" / "policy-configmaps.yaml").read_text()) if d]
    cm = [d for d in docs if d["metadata"]["name"] == "tre-v2-baseline-preserve"][0]
    return yaml.safe_load(cm["data"]["preserve.yaml"])["mu"]


def test_plan_is_the_muold_arm_on_alternating_then_sinusoidal(tmp_path):
    c = _load(tmp_path)
    plan = C.build_plan(c)
    assert [(i.trace, i.arm) for i in plan] == [(t, "preserve-v1-muold") for t in TRACES]
    assert {i.base_arm for i in plan} == {"preserve"} and {i.label for i in plan} == {"PreServe-V1-muold"}
    assert (c.get("report") or {}).get("enabled") is False
    assert c["registry_expect"]["registry_id"] == _load(tmp_path, PAPER)["registry_expect"]["registry_id"]
    assert c["results_root"] == f"{ROOT}/{NAME}-run" and c["seed"] == 1


def test_arm_is_the_paper_v1_arm_except_mu(tmp_path):
    c, paper = _load(tmp_path), _load(tmp_path, PAPER)
    mine, ref = C.arm_def(c, "preserve-v1-muold"), C.arm_def(paper, "preserve-v1")
    assert mine.base == ref.base and set(mine.policy) <= C.PRESERVE_KEYS
    assert {k for k in set(mine.policy) | set(ref.policy) if mine.policy.get(k) != ref.policy.get(k)} == {"mu"}
    mu = mine.policy["mu"]
    assert set(mu) == set(_cm_mu()) and all(mu[m] != _cm_mu()[m] for m in mu)
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for m in mu:
        assert " / ".join(f"{mu[m][k]}" for k in ("p", "d", "t")) in header, m
    assert {p: dict(paper["preserve"]).get(p) for p in ("match_parts", "history_root")} == {
        p: dict(c["preserve"]).get(p) for p in ("match_parts", "history_root")}


def test_traces_are_the_paper_campaigns_files_and_have_histories(tmp_path):
    c = _load(tmp_path)
    listed = _sha_list(DRIVER_DIR / f"{NAME}.traces.sha256")
    paper = _sha_list(RUNNER / "drivers" / "bl-preserve-paper-20261104" / f"{PAPER}.traces.sha256")
    assert listed.items() <= paper.items() and len(listed) == 2
    wanted = {t["dir"][len(ROOT) + 1:]: t["trace"] for t in c["traces"]}
    assert {str(Path(r).parent) for r in listed} == set(wanted)
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for rel, sha in listed.items():
        assert f"{wanted[str(Path(rel).parent)]} " in header and sha in header, rel
    hist = {str(Path(r).parent)[len("history/"):] for r in _sha_list(HISTORIES)}
    assert set(wanted) <= hist


def test_no_site_path_in_spec_or_driver():
    for p in (SPEC, *DRIVER_DIR.glob("*")):
        text = p.read_text()
        assert "nfs_shared_data" not in text and "/data/nfs" not in text, p
