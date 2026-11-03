"""campaign-bl-preserve-paper-20261104 (PreServe as in the paper: V2 x 7, V1 x 7, 4 V1 sensitivity arms): the spec, its driver,
the trace list, the history list and the frozen ConfigMap agree with each other and carry no site path.

No cluster, no data root: the spec is loaded with a fake CAMPAIGN_DATA_ROOT. The invariants:
* 18 arms in the user's order: V2 on the 7 traces, V1 on the 7 traces (file order), then D_hat-completed D and TTFT+TPOT mu,
  each on Decode_heavy and Real_code; reports off; the registry pin of the 11-03 PreServe batch;
* V2 = the ConfigMap's default rules, V1 = freeze_params.predictor_rules("v1") (the sets the ConfigMap holds); the sensitivity
  arms differ from V1 in one key each; every arm override is a PreServe key the runner accepts;
* the traces are the 11-03 PreServe batch's files; the history list holds one train-tail history per trace directory, under
  the name the runner and the policy use for history_variant train-tail-20261104;
* the spec and the driver contain no NFS path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

RUNNER = Path(__file__).resolve().parents[1] / "runner"
TRE = RUNNER.parents[1]
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

NAME = "campaign-bl-preserve-paper-20261104"
SPEC = RUNNER / "campaigns" / f"{NAME}.yaml"
DRIVER_DIR = RUNNER / "drivers" / "bl-preserve-paper-20261104"
PREV = "campaign-bl-w10-preserve-20261103"
HISTORIES = TRE / "baselines" / "histories" / "preserve_history_w10_train-tail-20261104.sha256"
ROOT = "/fake/data"
TRACES = ["Alternating_hot_model_periodic_A", "Simultaneous_spike_ramp_twice_tps1o2", "Prefill_mixed_corner_decode_mix",
          "Decode_heavy_burst", "Sinusoidal_demand", "Real_conv_2023_slice_a_tok70", "Real_code_2024_slice_a_tok70"]


def _load(tmp_path):
    return C.load_campaign(SPEC, environ={"CAMPAIGN_DATA_ROOT": ROOT, "TRE_REPO": str(tmp_path)})


def _sha_list(path: Path) -> dict:
    return {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in path.read_text().splitlines() if l.strip()}


def _cm_params() -> dict:
    docs = [d for d in yaml.safe_load_all((TRE / "deploy" / "baselines" / "tre" / "policy-configmaps.yaml").read_text()) if d]
    cm = [d for d in docs if d["metadata"]["name"] == "tre-v2-baseline-preserve"][0]
    return yaml.safe_load(cm["data"]["preserve.yaml"])


def test_plan_is_v2_then_v1_then_the_four_sensitivity_arms(tmp_path):
    c = _load(tmp_path)
    plan = C.build_plan(c)
    assert len(plan) == 18 and [i.n for i in plan] == list(range(1, 19))
    want = ([(t, "preserve-v2") for t in TRACES] + [(t, "preserve-v1") for t in TRACES]
            + [(t, a) for a in ("preserve-v1-dpred", "preserve-v1-mutt") for t in ("Decode_heavy_burst",
                                                                                 "Real_code_2024_slice_a_tok70")])
    assert [(i.trace, i.arm) for i in plan] == want
    assert {i.base_arm for i in plan} == {"preserve"} and len({i.key for i in plan}) == 18
    assert len({i.label for i in plan}) == 4 and "PreServe" not in {i.label for i in plan}
    assert (c.get("report") or {}).get("enabled") is False
    assert c["registry_expect"]["registry_id"] == yaml.safe_load(
        (RUNNER / "campaigns" / f"{PREV}.yaml").read_text())["registry_expect"]["registry_id"]
    assert c["results_root"] == f"{ROOT}/{NAME}-run" and c["preserve"]["history_root"].startswith(ROOT + "/datasets/")


def test_arms_select_v1_v2_and_one_sensitivity_key_each(tmp_path):
    pytest.importorskip("numpy")
    sys.path.insert(0, str(TRE / "baselines"))
    from tre_baselines.tools import freeze_params as fp

    c = _load(tmp_path)
    pol = {a: C.arm_def(c, a).policy for a in ("preserve-v2", "preserve-v1", "preserve-v1-dpred", "preserve-v1-mutt")}
    assert all(set(p) <= C.PRESERVE_KEYS for p in pol.values())
    cm = _cm_params()
    assert "predictor_rules" not in pol["preserve-v2"] and cm["predictor_rules"] == fp.predictor_rules("v2")
    assert [t for t, _s in cm["predictor_rules"]] == TRACES
    assert pol["preserve-v1"]["predictor_rules"] == fp.predictor_rules("v1")
    assert {s for _t, s in fp.predictor_rules("v1")} | {"v2"} == set(cm["predictors"])
    assert {k: v for k, v in pol["preserve-v1-dpred"].items() if pol["preserve-v1"].get(k) != v} == {
        "tier1_d_source": "done_or_predicted"}
    diff = {k for k, v in pol["preserve-v1-mutt"].items() if pol["preserve-v1"].get(k) != v}
    assert diff == {"mu"} and set(pol["preserve-v1-mutt"]["mu"]) == set(cm["mu"])
    assert pol["preserve-v1-mutt"]["mu"]["dsqwen-14b"] == cm["mu"]["dsqwen-14b"]   # 14b: the same mu under both rules
    assert pol["preserve-v1-mutt"]["mu"]["dsqwen-7b"] != cm["mu"]["dsqwen-7b"]
    assert cm["history_variant"] == "train-tail-20261104" and cm["tier1_d_source"] == "done_tokens"


def test_traces_are_the_previous_batchs_files_and_named_in_the_header(tmp_path):
    c = _load(tmp_path)
    listed = _sha_list(DRIVER_DIR / f"{NAME}.traces.sha256")
    assert listed == _sha_list(RUNNER / "drivers" / "bl-w10-preserve-20261103" / f"{PREV}.traces.sha256")
    wanted = {t["dir"][len(ROOT) + 1:]: t["trace"] for t in c["traces"]}
    assert {str(Path(r).parent) for r in listed} == set(wanted) and len(wanted) == 7
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for rel, sha in listed.items():
        assert f"{wanted[str(Path(rel).parent)]} " in header and sha in header, rel


def test_one_train_tail_history_per_trace_under_the_variant_name(tmp_path):
    c = _load(tmp_path)
    hist = _sha_list(HISTORIES)
    assert {str(Path(r).parent)[len("history/"):] for r in hist} == {t["dir"][len(ROOT) + 1:] for t in c["traces"]}
    assert {Path(r).name for r in hist} == {C.PH.history_names(10.0, _cm_params()["history_variant"])[-1]}


def test_no_site_path_in_spec_or_driver():
    for p in (SPEC, *DRIVER_DIR.glob("*")):
        text = p.read_text()
        assert "nfs_shared_data" not in text and "/data/nfs" not in text, p
