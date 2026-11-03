"""campaign-bl-w10-apa-ts-20261103 (APA and TokenScale with a 10 s window on the 7 ICSE-suite traces): the spec, its driver and the
trace list agree with each other and carry no site path.

No cluster, no data root: the spec is loaded with a fake CAMPAIGN_DATA_ROOT. The invariants:
* 14 arms = 7 traces x {apa, tokenscale}, APA first; no PreServe / TRE arm; reports off (project rule 2026-11-02);
* the registry is pinned (a 16-hex id) and the results root is under the data root;
* the trace list names exactly the 7 trace files of the spec, and the sha256 values in the spec header are the ones of the list;
* the spec and the driver contain no NFS path: everything site-specific comes from CAMPAIGN_DATA_ROOT.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import campaign as C  # noqa: E402

NAME = "campaign-bl-w10-apa-ts-20261103"
SPEC = RUNNER / "campaigns" / f"{NAME}.yaml"
DRIVER_DIR = RUNNER / "drivers" / "bl-w10-apa-ts-20261103"
ROOT = "/fake/data"


def _load(tmp_path):
    return C.load_campaign(SPEC, environ={"CAMPAIGN_DATA_ROOT": ROOT, "TRE_REPO": str(tmp_path)})


def test_plan_is_apa_and_tokenscale_on_seven_traces(tmp_path):
    c = _load(tmp_path)
    plan = C.build_plan(c)
    assert len(plan) == 14 and {i.arm for i in plan} == {"apa", "tokenscale"}
    assert [i.arm for i in plan if i.position == 1][0] == "apa"          # user 2026-11-03: APA first
    assert len(C.resolve_traces(c)) == 7
    assert (c.get("report") or {}).get("enabled") is False
    assert re.fullmatch(r"[0-9a-f]{16}", str(c["registry_expect"]["registry_id"]))
    assert c["results_root"] == f"{ROOT}/{NAME}-run"


def test_trace_list_matches_spec_and_header(tmp_path):
    c = _load(tmp_path)
    listed = {}
    for line in (DRIVER_DIR / f"{NAME}.traces.sha256").read_text().splitlines():
        sha, rel = line.split(None, 1)
        listed[str(Path(rel.strip()).parent)] = sha
    wanted = {t["dir"][len(ROOT) + 1:]: t["trace"] for t in c["traces"]}
    assert set(listed) == set(wanted) and len(wanted) == 7
    header = "\n".join(l for l in SPEC.read_text().splitlines() if l.startswith("#"))
    for d, sha in listed.items():
        assert f"{wanted[d]} " in header and sha in header, (wanted[d], sha)


def test_no_site_path_in_spec_or_driver():
    for p in (SPEC, *DRIVER_DIR.glob("*")):
        text = p.read_text()
        assert "nfs_shared_data" not in text and "/data/nfs" not in text, p


def test_plan_is_trace_major(tmp_path):
    """User 2026-11-03: both arms of a trace run back to back, traces in file order (so each trace can be compared at once)."""
    c = _load(tmp_path)
    plan = C.build_plan(c)
    assert [i.n for i in plan] == list(range(1, 15))
    assert [i.trace for i in plan][::2] == [t["trace"] for t in c["traces"]]
    assert all(plan[2 * k].trace == plan[2 * k + 1].trace and plan[2 * k].arm != plan[2 * k + 1].arm for k in range(7))
