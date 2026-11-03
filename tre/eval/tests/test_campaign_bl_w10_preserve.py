"""campaign-bl-w10-preserve-20261103 (PreServe with a 10 s window on the 7 ICSE-suite traces, then one APA rerun): the spec, its
driver, the trace list and the history list agree with each other and carry no site path.

No cluster, no data root: the spec is loaded with a fake CAMPAIGN_DATA_ROOT. The invariants:
* 8 arms: PreServe on the 7 traces in file order first, then one APA arm (own id and label) on Alternating, last;
* reports off (project rule 2026-11-02); the registry is pinned to the id of campaign-bl-w10-apa-ts-20261103;
* the trace files are the files of campaign-bl-w10-apa-ts-20261103 (same sha256 list), named in the spec header;
* the history list holds one 10 s history per trace directory, under the name the runner and the policy look for;
* the spec and the driver contain no NFS path: everything site-specific comes from CAMPAIGN_DATA_ROOT.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

RUNNER = Path(__file__).resolve().parents[1] / "runner"
TRE = RUNNER.parents[1]
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

NAME = "campaign-bl-w10-preserve-20261103"
SPEC = RUNNER / "campaigns" / f"{NAME}.yaml"
DRIVER_DIR = RUNNER / "drivers" / "bl-w10-preserve-20261103"
APA_TS = "campaign-bl-w10-apa-ts-20261103"
HISTORIES = TRE / "baselines" / "histories" / "preserve_history_w10-20261103.sha256"
ROOT = "/fake/data"


def _load(tmp_path):
    return C.load_campaign(SPEC, environ={"CAMPAIGN_DATA_ROOT": ROOT, "TRE_REPO": str(tmp_path)})


def _sha_list(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            sha, rel = line.split(None, 1)
            out[rel.strip()] = sha
    return out


def test_plan_is_preserve_on_seven_traces_then_one_apa_rerun(tmp_path):
    c = _load(tmp_path)
    plan = C.build_plan(c)
    assert len(plan) == 8 and [i.n for i in plan] == list(range(1, 9))
    pre, last = plan[:7], plan[7]
    assert {(i.arm, i.base_arm) for i in pre} == {("preserve", "preserve")}
    assert [i.trace for i in pre] == [t["trace"] for t in c["traces"][:7]] and len({i.trace for i in pre}) == 7
    assert last.base_arm == "apa" and last.arm != "apa" and last.label != C.DEFAULT_LABELS["apa"]
    assert last.trace == "Alternating_hot_model_periodic_A" and last.out_name == pre[0].out_name
    assert len({i.key for i in plan}) == 8                                 # the rerun has its own arm directory
    assert (c.get("report") or {}).get("enabled") is False
    assert c["registry_expect"]["registry_id"] == yaml.safe_load(
        (RUNNER / "campaigns" / f"{APA_TS}.yaml").read_text())["registry_expect"]["registry_id"]
    assert c["results_root"] == f"{ROOT}/{NAME}-run"
    assert c["preserve"]["trace_host_root"].startswith(ROOT + "/")


def test_traces_are_the_apa_ts_files_and_named_in_the_header(tmp_path):
    c = _load(tmp_path)
    listed = _sha_list(DRIVER_DIR / f"{NAME}.traces.sha256")
    assert listed == _sha_list(RUNNER / "drivers" / "bl-w10-apa-ts-20261103" / f"{APA_TS}.traces.sha256")
    wanted = {t["dir"][len(ROOT) + 1:]: t["trace"] for t in c["traces"]}
    assert {str(Path(r).parent) for r in listed} == set(wanted) and len(wanted) == 7
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for rel, sha in listed.items():
        assert f"{wanted[str(Path(rel).parent)]} " in header and sha in header, rel


def test_one_ten_second_history_per_trace(tmp_path):
    c = _load(tmp_path)
    hist = _sha_list(HISTORIES)
    assert {str(Path(r).parent) for r in hist} == {t["dir"][len(ROOT) + 1:] for t in c["traces"]}
    assert {Path(r).name for r in hist} == {C.PH.history_names(10.0)[-1]}
    assert all(re.fullmatch(r"[0-9a-f]{64}", s) for s in hist.values())


def test_no_site_path_in_spec_or_driver():
    for p in (SPEC, *DRIVER_DIR.glob("*")):
        text = p.read_text()
        assert "nfs_shared_data" not in text and "/data/nfs" not in text, p
