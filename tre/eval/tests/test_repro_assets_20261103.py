"""Reproducibility assets of the 2026-10-31 .. 2026-11-03 runs (registries, controller env, specs, drivers, trace-set configs).

Invariants, not implementation:

* every registry snapshot is named by its content id (arm_registry.registry_id) and listed with its file sha256;
* every registry id a repo spec expects, and every registry a driver uses as reference, has a snapshot, and every snapshot stamps the
  window of the controller env of the runs that carried it (the controller refuses to start otherwise);
* every repo copy of a campaign file is the file recorded in originals.tsv, and a spec can be turned back into the original
  byte for byte (so the sha256 the runner recorded in the arm manifests is the sha256 of the original);
* the new files hold no site path;
* the trace-set configs of the repo match the checksums of TRACE_FILES.sha256 / traces.tsv and the loadgen configs of main;
* controller_env.py makes the live env equal a file in one `kubectl patch` (with the resourceVersion it planned on), never while a runner,
  a driver or an exclusive window is active and never with an effective window the registry does not stamp.
  Historical records are compared with their own records (runs.tsv, env files), not with today's overlay.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

TRE = Path(__file__).resolve().parents[2]
RUNNER = TRE / "eval" / "runner"
CE_DIR = TRE / "eval" / "controller_env"
REG_DIR = TRE / "eval" / "registries"
DRV_DIR = RUNNER / "drivers"
SPEC_DIR = RUNNER / "campaigns"
TS_DIR = TRE / "loadgen_v1" / "configs" / "trace_sets"
for p in (str(RUNNER), str(CE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arm_registry as AR  # noqa: E402
import campaign as C  # noqa: E402
import controller_env as CE  # noqa: E402

yaml = pytest.importorskip("yaml")

SITE = "/data/nfs_shared_data"
WINDOW_REGISTRIES = {"f75155148eba1e61", "0ab5352747bdd70e", "ee3712855753d0e8", "5028a1209a5fe38d", "f3ac339a6e5de9be"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tsv(path: Path) -> list[dict]:
    rows = [l.rstrip("\n").split("\t") for l in path.read_text().splitlines() if l and not l.startswith("#")]
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def originals() -> tuple[str, list[dict]]:
    text = (DRV_DIR / "originals.tsv").read_text()
    root = re.search(r"^# data root of the originals[^:]*: (\S+)$", text, flags=re.M).group(1)
    return root, tsv(DRV_DIR / "originals.tsv")


def repo_specs() -> list[Path]:
    root, rows = originals()
    return [TRE.parent / r["repo_path"] for r in rows if r["relation"] in ("spec-param", "regenerated")]


# ------------------------------------------------------------------ registries
def test_every_registry_snapshot_is_named_by_its_content_id_and_indexed():
    files = sorted(REG_DIR.glob("*.yaml"))
    assert {f.stem for f in files} == WINDOW_REGISTRIES
    index = {r["registry_id"]: r for r in tsv(REG_DIR / "index.tsv")}
    assert set(index) == WINDOW_REGISTRIES
    for f in files:
        assert AR.registry_id(f.read_text()) == f.stem
        assert index[f.stem]["file"] == f.name and index[f.stem]["file_sha256"] == sha(f.read_bytes())


def test_each_snapshot_stamps_the_window_of_the_controller_env_of_the_runs_that_carried_it():
    """Historical records are checked against their own record (runs.tsv -> env file), not against today's overlay."""
    carried: set[str] = set()
    for run in tsv(CE_DIR / "runs.tsv"):
        env = CE.parse_env_file((CE_DIR / run["env_file"]).read_text())
        for rid in run["registry_id"].split("|"):
            carried.add(rid)
            text = (REG_DIR / f"{rid}.yaml").read_text()
            assert CE.registry_window_errors(text, env["TRE_METRICS_WINDOW_MS"]) == [], (run["run"], rid)
            for m in yaml.safe_load(text)["models"]:
                assert m["trs"]["numerator"] == "step_counters", (rid, m["name"])
    assert carried == WINDOW_REGISTRIES


def test_every_registry_a_spec_expects_or_a_driver_uses_has_a_snapshot():
    have = {f.stem for f in REG_DIR.glob("*.yaml")}
    for spec in repo_specs():
        rid = (yaml.safe_load(spec.read_text()).get("registry_expect") or {}).get("registry_id")
        if rid is not None:
            assert str(rid) in have, f"{spec.name}: registry_expect.registry_id {rid} has no snapshot"
    for drv in DRV_DIR.glob("*/*.sh"):
        for rid in re.findall(r"eval/registries/([0-9a-f]{16})\.yaml", drv.read_text()):
            assert rid in have, f"{drv.name} uses registry {rid}"
        for rid in re.findall(r"^ID_(?:EXPECT|STD|WIN10)=([0-9a-f]{16})", drv.read_text(), flags=re.M):
            assert rid in have or rid == "af9aef4701804743", f"{drv.name} names registry {rid}"   # af9a.. = the standard W 20 s registry run_all3.sh restores


# ------------------------------------------------------------------ originals, specs, drivers
def test_every_repo_copy_is_the_file_recorded_in_originals_tsv():
    root, rows = originals()
    assert len(rows) >= 50
    for r in rows:
        f = TRE.parent / r["repo_path"]
        assert sha(f.read_bytes()) == r["repo_sha256"], r["repo_path"]
        if r["relation"] in ("identical", "record"):
            assert r["original_sha256"] == r["repo_sha256"], r["repo_path"]
        if r["relation"] == "spec-param":
            text = "".join(l for l in f.read_text().splitlines(True) if not l.startswith("# [repo]"))
            assert sha(text.replace("${CAMPAIGN_DATA_ROOT}", root).encode()) == r["original_sha256"], \
                f"{r['repo_path']}: the original is not recoverable from the repo copy"
    assert {r["relation"] for r in rows} <= {"identical", "spec-param", "script-param", "doc-param", "data-relative-paths", "regenerated", "record"}


def test_state_records_and_trace_set_docs_are_listed_verbatim():
    root, rows = originals()
    listed = {r["repo_path"]: r for r in rows}
    records = sorted(DRV_DIR.glob("*/state-records/*/*"))
    assert len(records) >= 15
    for f in records:
        r = listed[str(f.relative_to(TRE.parent))]
        assert r["relation"] == "record" and r["original_sha256"] == sha(f.read_bytes())
    assert not [r for r in rows if r["relation"] == "record" and not (TRE.parent / r["repo_path"]).is_file()]
    for name in ("SOURCES.md", "audit.md"):
        for ts in ("traces-icse-r3c-20261101", "traces-icse-ktrace-20261031", "traces-icse-r3e-20261101"):
            assert listed[f"tre/loadgen_v1/configs/trace_sets/{ts}/{name}"]["relation"] == "identical"


def test_the_new_code_holds_no_site_path():
    root, rows = originals()
    assert root.startswith(SITE)
    code = [TRE.parent / r["repo_path"] for r in rows if r["relation"] in ("spec-param", "script-param", "regenerated", "identical")
            and "trace_sets" not in r["repo_path"]]
    code += [CE_DIR / "controller_env.py", Path(__file__)]
    for f in code:
        if f == Path(__file__):
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            assert SITE not in line, f"{f.relative_to(TRE.parent)}:{n}: {line.strip()[:100]}"


def test_a_repo_spec_needs_the_data_root_and_follows_it():
    env = {"TRE_REPO": str(TRE)}
    for spec in repo_specs():
        with pytest.raises(C.CampaignError):
            C.load_campaign(spec, environ=env)
        c = C.load_campaign(spec, environ={**env, "CAMPAIGN_DATA_ROOT": "/dataroot"})
        assert c["results_root"].startswith("/dataroot/") and c["results_root"].endswith("-run")
        for t in c["traces"]:
            assert str(t.get("dir") or c.get("trace_root")).startswith("/dataroot/")
        assert c["report"]["enabled"] is False or spec.name.startswith(("campaign-icse-r3c-full-D", "campaign-icse-ktrace-2", "campaign-icse-r3e-full",
                                                                          "campaign-icse-r3c-full-F", "campaign-icse-ktrace-full-F"))


def test_rep2_of_the_tre_rerun_differs_from_rep1_only_in_name_round_results_root_and_the_rep_line():
    keep = lambda p: [l for l in p.read_text().splitlines() if not re.match(r"(name|round|results_root):|# REP:", l)]
    r1 = SPEC_DIR / "campaign-tre-rerun-r3e-20261103.yaml"
    r2 = SPEC_DIR / "campaign-tre-rerun-r3e-rep2-20261103.yaml"
    assert keep(r1) == keep(r2)
    assert yaml.safe_load(r1.read_text())["registry_expect"]["registry_id"] == "f75155148eba1e61"


# ------------------------------------------------------------------ trace sets
def test_trace_set_files_match_their_checksums_and_the_loadgen_configs_of_main():
    rows = tsv(TS_DIR / "traces.tsv")
    assert len(rows) == 7
    listed = {}
    for line in (TS_DIR / "TRACE_FILES.sha256").read_text().splitlines():
        h, p = line.split("  ", 1)
        listed[p] = h
    for r in rows:
        base = f"{r['set']}/{r['trace']}"
        d = TS_DIR / r["set"] / r["trace"]
        assert sha((d / "manifest.json").read_bytes()) == r["manifest_sha256"] == listed[f"{base}/seed1/manifest.json"]
        assert sha((d / "preserve_history.json").read_bytes()) == r["preserve_history_sha256"] == listed[f"{base}/seed1/preserve_history.json"]
        assert listed[r["trace_file"]] == r["trace_sha256"]
        assert sha((d / "gen" / "config.yaml").read_bytes()) == r["gen_config_sha256"] == listed[f"{r['set']}/_gen/{r['trace']}/seed1/config.yaml"]
        assert sha((TRE.parent / r["loadgen_config"]).read_bytes()) == r["loadgen_config_sha256"]
    # the trace files the baselines and the TRE rerun verified at launch are these
    for lst in (DRV_DIR / "bl-rep2-20261103" / "campaign-bl-rep2-r3e-20261103.traces.sha256",
                DRV_DIR / "tre-rerun-20261103" / "campaign-tre-rerun-r3e-20261103.traces.sha256"):
        got = dict(l.split("  ", 1)[::-1] for l in lst.read_text().splitlines())
        assert {r["trace_file"]: r["trace_sha256"] for r in rows} == got, lst.name


# ------------------------------------------------------------------ controller env
def test_the_two_env_records_differ_in_one_value_and_name_the_known_no_ops():
    a = CE.parse_env_file((CE_DIR / "controller-env-20261031-w10.env").read_text())
    b = CE.parse_env_file((CE_DIR / "controller-env-20261103-live.env").read_text())
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"}
    assert (a["TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"], b["TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"]) == ("0", "false")
    for rec in (a, b):  # the entries the README table explains (keys it lists, not the whole set)
        assert rec["TRE_METRICS_WINDOW_MS"] == "10000" and rec["TRE_DISABLE_ETA_GATE"] == "false"
        assert "TRE_SIGNAL_SOURCE" not in rec and rec["TRE_ACTION_COOLDOWN"] == "1"
    cfg = (TRE / "controller" / "tre_controller" / "config.py").read_text()
    removed = re.search(r"REMOVED_TIMER_ENV = \((.*?)\)", cfg, flags=re.S).group(1)
    assert "TRE_ACTION_COOLDOWN" in removed       # read by no image: a no-op on the live Deployment


def test_runs_tsv_names_existing_specs_and_env_files():
    for r in tsv(CE_DIR / "runs.tsv"):
        assert (SPEC_DIR / r["spec"]).is_file() and (CE_DIR / r["env_file"]).is_file()


class FakeCluster:
    """kubectl for controller_env: one Deployment, one registry ConfigMap; records every call."""

    def __init__(self, env: dict, registry_window: int = 10000):
        self.env = dict(env)
        self.rv = 100
        self.race = False  # a concurrent edit lands right after the Deployment was read
        self.calls: list[list[str]] = []
        models = [{"name": m, "trs": {"theta_window_ms": registry_window, "numerator": "step_counters"}} for m in ("m1", "m2")]
        self.registry = yaml.safe_dump({"models": models})

    def deploy(self) -> dict:
        env = [{"name": k, "value": v} for k, v in self.env.items()] + [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}]
        return {"metadata": {"resourceVersion": str(self.rv)},
                "spec": {"template": {"spec": {"containers": [{"name": "controller", "env": env}]}}}}

    def __call__(self, args: list[str], timeout: int = 300) -> str:
        self.calls.append(list(args))
        if args[:2] == ["get", "deploy"]:
            out = json.dumps(self.deploy())
            if self.race:
                self.rv += 1
            return out
        if args[:2] == ["get", "cm"]:
            return json.dumps({"data": {"registry.yaml": self.registry}})
        if args[0] == "patch":
            patch = json.loads(args[args.index("-p") + 1])
            if patch["metadata"]["resourceVersion"] != str(self.rv):
                raise CE.EnvError("kubectl patch: Operation cannot be fulfilled: the object has been modified")
            (container,) = patch["spec"]["template"]["spec"]["containers"]
            assert container["name"] == "controller"
            for item in container["env"]:
                if item.get("$patch") == "delete":
                    self.env.pop(item["name"], None)
                else:
                    self.env[item["name"]] = item["value"]
            self.rv += 1
            return ""
        if args[:2] == ["rollout", "status"]:
            return ""
        raise AssertionError(args)

    def writes(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] in ("patch", "set", "rollout")]


@pytest.fixture
def cluster(monkeypatch):
    fc = FakeCluster(CE.parse_env_file((CE_DIR / "controller-env-20261031-w10.env").read_text()))
    monkeypatch.setattr(CE, "kubectl", fc)
    monkeypatch.setattr(CE, "runner_active", lambda: [])
    monkeypatch.delenv("TRE_EXCLUSIVE_WINDOW_FILE", raising=False)
    monkeypatch.delenv("CAMPAIGN_DATA_ROOT", raising=False)
    return fc


def _want(tmp_path, env: dict):
    f = tmp_path / "want.env"
    f.write_text(CE.render_env_file(env))
    return str(f)


def test_apply_makes_the_live_env_equal_the_file_in_one_patch_and_leaves_a_rollback(cluster, tmp_path):
    want = CE.parse_env_file((CE_DIR / "controller-env-20261103-live.env").read_text())
    want["EXTRA_NEW"] = 'a "quoted" value'
    want.pop("TRE_ACTION_COOLDOWN")
    before = dict(cluster.env)
    assert CE.main(["apply", _want(tmp_path, want), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert cluster.env == want
    assert len([c for c in cluster.calls if c[0] == "patch"]) == 1                      # one rollout
    assert CE.parse_env_file((tmp_path / "bk" / "controller-env.before.env").read_text()) == before
    assert json.loads((tmp_path / "bk" / "deploy.before.json").read_text())["metadata"]["resourceVersion"] == "100"
    assert "POD_IP" not in cluster.env                                                  # an entry fed by valueFrom is never touched
    assert CE.main(["apply", str(tmp_path / "bk" / "controller-env.before.env"), "--backup-dir", str(tmp_path / "bk2")]) == 0
    assert cluster.env == before                                                        # the rollback file restores it


def test_apply_with_nothing_to_change_does_not_touch_the_cluster(cluster, tmp_path):
    cluster.calls.clear()
    assert CE.main(["apply", _want(tmp_path, dict(cluster.env)), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert not cluster.writes() and not (tmp_path / "bk").exists()


def test_apply_is_refused_when_the_registry_stamps_another_window(cluster, tmp_path):
    cluster.registry = FakeCluster({}, registry_window=20000).registry
    before = dict(cluster.env)
    assert CE.main(["apply", _want(tmp_path, dict(cluster.env, TRE_ACTION_COOLDOWN="2")), "--backup-dir", str(tmp_path / "bk")]) == 2
    assert cluster.env == before and not cluster.writes()
    assert CE.main(["apply", _want(tmp_path, dict(cluster.env, TRE_METRICS_WINDOW_MS="20000")), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert cluster.env["TRE_METRICS_WINDOW_MS"] == "20000"


def test_a_file_without_the_window_key_is_judged_by_the_default_the_controller_falls_back_to(cluster, tmp_path, monkeypatch):
    """Applying such a file UNSETS the variable; the controller then uses its code default, which must match the registry."""
    from tre_controller.config import ControllerConfig

    assert CE.controller_default_window() == str(ControllerConfig.from_env({}).metrics_window_ms)   # read from the code
    want = {k: v for k, v in cluster.env.items() if k != "TRE_METRICS_WINDOW_MS"}
    path = _want(tmp_path, want)
    monkeypatch.setattr(CE, "controller_default_window", lambda: "30000")                 # registry stamps 10000
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 2
    assert not cluster.writes() and "TRE_METRICS_WINDOW_MS" in cluster.env
    monkeypatch.setattr(CE, "controller_default_window", lambda: "10000")
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 0
    assert "TRE_METRICS_WINDOW_MS" not in cluster.env
    cluster.env["TRE_METRICS_WINDOW_MS"] = "10000"
    monkeypatch.setattr(CE, "controller_default_window", lambda: (_ for _ in ()).throw(RuntimeError("no code")))
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk3")]) == 2          # unreadable default: refused


def test_apply_is_refused_while_a_runner_or_a_driver_is_active(cluster, tmp_path, monkeypatch):
    path = _want(tmp_path, dict(cluster.env, TRE_ACTION_COOLDOWN="2"))
    monkeypatch.setattr(CE, "runner_active", lambda: ["4242"])
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 2 and not cluster.writes()
    monkeypatch.setattr(CE, "runner_active", lambda: [])
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 0 and cluster.env["TRE_ACTION_COOLDOWN"] == "2"


def test_the_active_patterns_match_runners_and_drivers_but_not_this_tool():
    hit = ["python3 /x/tre/eval/runner/campaign.py run spec.yaml", "bash /x/tre/eval/runner/run_arm_pilot.sh a b",
           "bash /x/tre/eval/runner/drivers/icse-r3c-20261101/run_r3c_full_D.sh", "bash /x/run_all3.sh",
           "bash tre/eval/runner/run_campaign.sh spec.yaml"]
    miss = ["python3 tre/eval/controller_env/controller_env.py apply x.env --backup-dir /tmp/b", "bash -c ls", "python3 -m pytest"]
    for line in hit:
        assert any(re.search(p, line) for p in CE._ACTIVE_PATTERNS), line
    for line in miss:
        assert not any(re.search(p, line) for p in CE._ACTIVE_PATTERNS), line


def test_apply_is_refused_while_an_exclusive_window_marker_exists(cluster, tmp_path, monkeypatch):
    path = _want(tmp_path, dict(cluster.env, TRE_ACTION_COOLDOWN="2"))
    marker = tmp_path / "root" / "TRE_EXCLUSIVE_WINDOW"
    marker.parent.mkdir()
    marker.write_text("validation 2026-11-03T16:18:02+08:00 2026-11-03T21:18:02+08:00 owner=x\n")
    monkeypatch.setenv("CAMPAIGN_DATA_ROOT", str(marker.parent))                           # default location
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 2 and not cluster.writes()
    monkeypatch.delenv("CAMPAIGN_DATA_ROOT")
    monkeypatch.setenv("TRE_EXCLUSIVE_WINDOW_FILE", str(marker))                           # explicit location
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 2 and not cluster.writes()
    marker.unlink()
    assert CE.main(["apply", path, "--backup-dir", str(tmp_path / "bk")]) == 0


def test_apply_fails_instead_of_overwriting_a_concurrent_edit_of_the_deployment(cluster, tmp_path):
    cluster.race = True                                                                    # the Deployment changes after it was read
    before = dict(cluster.env)
    assert CE.main(["apply", _want(tmp_path, dict(cluster.env, TRE_ACTION_COOLDOWN="2")), "--backup-dir", str(tmp_path / "bk")]) == 2
    assert cluster.env == before


def test_env_file_parser_rejects_what_is_not_the_whole_state():
    assert CE.parse_env_file("# c\n\nA=1\nB=x=y\nC=\n") == {"A": "1", "B": "x=y", "C": ""}
    for bad in ("A\n", "1A=2\n", "A=1\nA=2\n"):
        with pytest.raises(CE.EnvError):
            CE.parse_env_file(bad)
