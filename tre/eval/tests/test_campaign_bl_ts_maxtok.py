"""campaign-bl-ts-maxtok-20261104 (TokenScale with the 2026-11-04 lambda' output rule on the 7 ICSE-suite traces): the spec, its
driver and the trace list agree with the 11-03 campaign it repeats and carry no site path.

No cluster, no data root: the spec is loaded with a fake CAMPAIGN_DATA_ROOT. The invariants:
* 7 arms = 7 traces x {tokenscale}, traces in the 11-03 file order; no other arm; reports off (project rule 2026-11-02);
* the registry pin, seed, env and validity of campaign-bl-w10-apa-ts-20261103 (the only method difference is in the image);
* the trace list = the 11-03 list (same files, same sha256), and the sha256 values are in the spec header;
* the spec and the driver contain no NFS path: everything site-specific comes from CAMPAIGN_DATA_ROOT.
"""
from __future__ import annotations

import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

NAME = "campaign-bl-ts-maxtok-20261104"
PREV = "campaign-bl-w10-apa-ts-20261103"
SPEC = RUNNER / "campaigns" / f"{NAME}.yaml"
DRIVER_DIR = RUNNER / "drivers" / "bl-ts-maxtok-20261104"
PREV_DIR = RUNNER / "drivers" / "bl-w10-apa-ts-20261103"
ROOT = "/fake/data"


def _load(tmp_path, name=NAME):
    return C.load_campaign(RUNNER / "campaigns" / f"{name}.yaml", environ={"CAMPAIGN_DATA_ROOT": ROOT, "TRE_REPO": str(tmp_path)})


def _sha_list(path: Path) -> dict:
    return {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in path.read_text().splitlines() if l.strip()}


def test_plan_is_tokenscale_on_the_seven_traces_of_1103(tmp_path):
    c, prev = _load(tmp_path), _load(tmp_path, PREV)
    plan = C.build_plan(c)
    assert [(i.trace, i.arm) for i in plan] == [(t["trace"], "tokenscale") for t in prev["traces"]]
    assert [t["dir"] for t in c["traces"]] == [t["dir"] for t in prev["traces"]]
    assert (c.get("report") or {}).get("enabled") is False
    assert c["results_root"] == f"{ROOT}/{NAME}-run" and c["seed"] == prev["seed"] == 1
    for key in ("registry_expect", "env", "validity", "precheck"):
        assert c[key] == prev[key], key


def test_trace_list_is_the_1103_list_and_in_header():
    listed = _sha_list(DRIVER_DIR / f"{NAME}.traces.sha256")
    assert listed == _sha_list(PREV_DIR / f"{PREV}.traces.sha256") and len(listed) == 7
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for rel, sha in listed.items():
        assert f"{Path(rel).parts[1]} " in header and sha in header, rel


def test_no_site_path_in_spec_or_driver():
    for p in (SPEC, *DRIVER_DIR.glob("*")):
        text = p.read_text()
        assert "nfs_shared_data" not in text and "/data/nfs" not in text, p
