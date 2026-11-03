"""Reproducibility assets of the 2026-10-31 .. 2026-11-03 runs (registries, controller env, specs, drivers, trace-set configs).

Invariants, not implementation:

* every registry snapshot is named by its content id (arm_registry.registry_id) and listed with its file sha256;
* every registry id a repo spec expects, and every registry a driver uses as reference, has a snapshot, and every snapshot stamps the
  window the overlay gives the controller (the controller refuses to start otherwise);
* every repo copy of a campaign file is the file recorded in originals.tsv, and a spec can be turned back into the original
  byte for byte (so the sha256 the runner recorded in the arm manifests is the sha256 of the original);
* the new files hold no site path;
* the trace-set configs of the repo match the checksums of TRACE_FILES.sha256 / traces.tsv and the loadgen configs of main;
* controller_env.py makes the live env equal a file in one `kubectl set env`, never while a runner is active and never with a
  window the registry does not stamp; the live-env records differ from the overlay exactly where eval/controller_env/README.md says.
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


def overlay_env() -> dict:
    d = yaml.safe_load((TRE / "deploy" / "overlays" / "tre-v2" / "controller.yaml").read_text())
    return {e["name"]: e.get("value") for e in d["spec"]["template"]["spec"]["containers"][0]["env"]}


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


def test_the_snapshots_stamp_the_window_the_overlay_gives_the_controller():
    window = int(overlay_env()["TRE_METRICS_WINDOW_MS"])
    for f in sorted(REG_DIR.glob("*.yaml")):
        assert CE.registry_window_errors(f.read_text(), str(window)) == [], f.name
        for m in yaml.safe_load(f.read_text())["models"]:
            assert m["trs"]["numerator"] == "step_counters", (f.name, m["name"])


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
        if r["relation"] == "identical":
            assert r["original_sha256"] == r["repo_sha256"], r["repo_path"]
        if r["relation"] == "spec-param":
            text = "".join(l for l in f.read_text().splitlines(True) if not l.startswith("# [repo]"))
            assert sha(text.replace("${CAMPAIGN_DATA_ROOT}", root).encode()) == r["original_sha256"], \
                f"{r['repo_path']}: the original is not recoverable from the repo copy"
    assert {r["relation"] for r in rows} <= {"identical", "spec-param", "script-param", "doc-param", "data-relative-paths", "regenerated"}


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
def test_the_live_env_records_differ_from_the_overlay_where_the_readme_says():
    ov = {k: v for k, v in overlay_env().items()}
    a = CE.parse_env_file((CE_DIR / "controller-env-20261031-w10.env").read_text())
    b = CE.parse_env_file((CE_DIR / "controller-env-20261103-live.env").read_text())
    assert {k for k in a if a[k] != b.get(k)} | (set(a) ^ set(b)) == {"TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"}
    for live in (a, b):
        assert live["TRE_METRICS_WINDOW_MS"] == ov["TRE_METRICS_WINDOW_MS"] == "10000"
        only_live, only_overlay = set(live) - set(ov), set(ov) - set(live)
        assert only_live == {"TRE_ACTION_COOLDOWN", "TRE_DISABLE_ETA_GATE"} and only_overlay == {"TRE_SIGNAL_SOURCE"}
        assert ov["TRE_SIGNAL_SOURCE"] == "zm"
        assert live["TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"] in ("0", "false") and ov["TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"] == "0"
        # every other shared variable has the overlay value (the runner writes "false" for "0": same meaning, _get_bool)
        assert {k for k in set(live) & set(ov) if live[k] != ov[k]} <= {"TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"}
    cfg = (TRE / "controller" / "tre_controller" / "config.py").read_text()
    removed = re.search(r"REMOVED_TIMER_ENV = \((.*?)\)", cfg, flags=re.S).group(1)
    assert "TRE_ACTION_COOLDOWN" in removed       # read by no image: a no-op on the live Deployment


def test_runs_tsv_names_existing_specs_and_env_files():
    for r in tsv(CE_DIR / "runs.tsv"):
        assert (SPEC_DIR / r["spec"]).is_file() and (CE_DIR / r["env_file"]).is_file()
        assert r["registry_id"] in WINDOW_REGISTRIES


class FakeCluster:
    """kubectl for controller_env: one Deployment, one registry ConfigMap; records every call."""

    def __init__(self, env: dict, registry_window: int = 10000):
        self.env = dict(env)
        self.calls: list[list[str]] = []
        models = [{"name": m, "trs": {"theta_window_ms": registry_window, "numerator": "step_counters"}} for m in ("m1", "m2")]
        self.registry = yaml.safe_dump({"models": models})

    def deploy(self) -> dict:
        env = [{"name": k, "value": v} for k, v in self.env.items()] + [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}]
        return {"spec": {"template": {"spec": {"containers": [{"name": "controller", "env": env}]}}}}

    def __call__(self, args: list[str], timeout: int = 300) -> str:
        self.calls.append(list(args))
        if args[:2] == ["get", "deploy"]:
            return json.dumps(self.deploy()) if "json" in args else "kind: Deployment\n"
        if args[:2] == ["get", "cm"]:
            return json.dumps({"data": {"registry.yaml": self.registry}})
        if args[:2] == ["set", "env"]:
            for item in args[5:]:
                if item.endswith("-"):
                    self.env.pop(item[:-1], None)
                else:
                    k, _, v = item.partition("=")
                    self.env[k] = v
            return ""
        if args[:2] == ["rollout", "status"]:
            return ""
        raise AssertionError(args)


@pytest.fixture
def cluster(monkeypatch):
    fc = FakeCluster(CE.parse_env_file((CE_DIR / "controller-env-20261031-w10.env").read_text()))
    monkeypatch.setattr(CE, "kubectl", fc)
    monkeypatch.setattr(CE, "runner_active", lambda: [])
    return fc


def test_apply_makes_the_live_env_equal_the_file_in_one_set_env_and_leaves_a_rollback(cluster, tmp_path):
    want = CE.parse_env_file((CE_DIR / "controller-env-20261103-live.env").read_text())
    want["EXTRA_NEW"] = "1"
    want.pop("TRE_ACTION_COOLDOWN")
    f = tmp_path / "want.env"
    f.write_text(CE.render_env_file(want))
    before = dict(cluster.env)
    assert CE.main(["apply", str(f), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert cluster.env == want
    assert [c for c in cluster.calls if c[:2] == ["set", "env"]].__len__() == 1       # one rollout
    assert CE.parse_env_file((tmp_path / "bk" / "controller-env.before.env").read_text()) == before
    assert "valueFrom" not in json.dumps(cluster.env) and "POD_IP" not in cluster.env      # an entry fed by valueFrom is never touched
    assert CE.main(["apply", str(tmp_path / "bk" / "controller-env.before.env"), "--backup-dir", str(tmp_path / "bk2")]) == 0
    assert cluster.env == before                                                           # the rollback file restores it


def test_apply_with_nothing_to_change_does_not_touch_the_cluster(cluster, tmp_path):
    f = tmp_path / "same.env"
    f.write_text(CE.render_env_file(dict(cluster.env)))
    cluster.calls.clear()
    assert CE.main(["apply", str(f), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert not [c for c in cluster.calls if c[0] in ("set", "rollout")] and not (tmp_path / "bk").exists()


def test_apply_is_refused_while_a_runner_is_active_or_when_the_registry_stamps_another_window(cluster, tmp_path, monkeypatch):
    want = dict(cluster.env, TRE_METRICS_WINDOW_MS="20000", TRE_ACTION_COOLDOWN="2")
    f = tmp_path / "want.env"
    f.write_text(CE.render_env_file(want))
    before = dict(cluster.env)
    assert CE.main(["apply", str(f), "--backup-dir", str(tmp_path / "bk")]) == 2          # registry says 10000
    cluster.registry = FakeCluster({}, registry_window=20000).registry
    monkeypatch.setattr(CE, "runner_active", lambda: ["4242"])
    assert CE.main(["apply", str(f), "--backup-dir", str(tmp_path / "bk")]) == 2          # a runner is active
    assert cluster.env == before and not [c for c in cluster.calls if c[:2] == ["set", "env"]]
    monkeypatch.setattr(CE, "runner_active", lambda: [])
    assert CE.main(["apply", str(f), "--backup-dir", str(tmp_path / "bk")]) == 0
    assert cluster.env["TRE_METRICS_WINDOW_MS"] == "20000"


def test_env_file_parser_rejects_what_is_not_the_whole_state():
    assert CE.parse_env_file("# c\n\nA=1\nB=x=y\nC=\n") == {"A": "1", "B": "x=y", "C": ""}
    for bad in ("A\n", "1A=2\n", "A=1\nA=2\n"):
        with pytest.raises(CE.EnvError):
            CE.parse_env_file(bad)
