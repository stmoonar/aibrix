"""tools/preserve_mu on a tiny synthetic calibration capture (tests/fixtures/preserve_calib).

Fixture (model dsqwen-7b, two 30 s windows per cell, 1 replica):
  c100 hold: window 1 clean, 3 x (in 1000, out 100)  -> p 100, d 10, t 110 tok/s
             window 2 has a request with TPOT 100 ms > 75 ms -> excluded
  c101 hold: window 1 clean, 3 x (in 500, out 300)   -> p 50, d 30, t 80 tok/s
             window 2 has a model_error sent in it   -> excluded
  c102 hold but void, c103 static: huge loads        -> ignored
So mu = {p: 100, d: 30, t: 110}: each rate is its own maximum over clean windows.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from tre_baselines.tools import preserve_mu as pm

FIX = Path(__file__).resolve().parent / "fixtures" / "preserve_calib"
REGISTRY = str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml")


@pytest.fixture(scope="module")
def label():
    from tre_common.registry import load_registry

    return pm.label_for("dsqwen-7b", load_registry(REGISTRY))


def test_request_criterion(label) -> None:
    prof = pm.profile_model(FIX / "dsqwen-7b", label)
    assert prof.model == "dsqwen-7b"
    assert prof.cells == 2 and prof.windows == 4 and prof.clean == 2
    assert prof.skipped_cells == {"void": 1, "primitive": 1}
    assert prof.as_mu() == {"p": 100.0, "d": 30.0, "t": 110.0}


def test_replicas_and_contamination(label) -> None:
    prof = pm.profile_model(FIX / "dsqwen-7b", label, replicas=2)
    assert prof.as_mu() == {"p": 50.0, "d": 15.0, "t": 55.0}
    strict = dataclasses.replace(label, tpot_p95_ms=15.0)  # c101's TPOT 10 ms passes, c100's 20 ms not
    assert pm.profile_model(FIX / "dsqwen-7b", strict).as_mu() == {"p": 50.0, "d": 30.0, "t": 80.0}


def test_label_criterion_agrees_on_the_fixture(label) -> None:
    lab = dataclasses.replace(label, min_completed_requests=0)
    prof = pm.profile_model(FIX / "dsqwen-7b", lab, criterion="label", min_latency_samples=1)
    assert prof.clean == 2 and prof.as_mu() == {"p": 100.0, "d": 30.0, "t": 110.0}
    # with the calibration's min-n guard (20) these tiny windows are unlabeled: no mu
    assert pm.profile_model(FIX / "dsqwen-7b", label, criterion="label").clean == 0


def test_cli_prints_the_policy_mu_format(capsys) -> None:
    assert pm.main([str(FIX), "--registry", REGISTRY]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out == {"mu": {"dsqwen-7b": {"p": 100.0, "d": 30.0, "t": 110.0}}}
    # the output plugs straight into the policy's fail-closed parser
    from tre_baselines.policies.preserve_tier1 import parse_mu

    assert parse_mu(out["mu"], ["dsqwen-7b"])["dsqwen-7b"].d == 30.0
    # a model without any clean window -> exit 1, no entry
    assert pm.main([str(FIX / "dsqwen-7b"), "--registry", REGISTRY, "--window-s", "300"]) == 1
    assert yaml.safe_load(capsys.readouterr().out) == {"mu": {}}
    with pytest.raises(SystemExit):
        pm.main([str(FIX / "nowhere"), "--registry", REGISTRY])


# ------------------------------------------------------------------ alg1 (PreServe Alg. 1 on a dataset dir)

T0 = 1_000_000
WIN_COLS = ["model", "shape", "primitive", "rho", "cell_id", "attempt", "split", "cell_status",
            "possibly_contaminated", "in_warmup", "window_start_ms", "window_end_ms"]
REQ_COLS = ["model", "cell_id", "attempt", "request_id", "send_ts_ms", "first_token_ts_ms", "done_ts_ms",
            "input_tokens", "output_tokens", "expected_prompt_tokens", "outcome", "target_pod"]


def _alg1_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """One cell, five 10 s windows, NL SLO 50 ms:
    w0 clean: 2 x (1000 in, 100 out, NL 10); the 2nd is sent at 9.9 s and done in w1 -> p 200 d 20 t 220
    w1 clean: 1 x (5000 in, 10 out), P share 0.998                                -> p 500 d 1  t 501
    w2 dirty: 1 x (8000 in, 100 out) with NL 60 ms                                -> p 800 d 10 t 810
    w3 dirty: a client timeout (expected 3000 in) + 1 clean (1000 in, 100 out)    -> p 400 d 10 t 410
    w4 in warm-up: a huge clean load                                              -> dropped
    so mu = {p: 500 (w1, no P-share filter), d: 20 (w0, by arrival), t: 501}; p and t bracketed by w2."""
    import csv

    ddir = tmp_path / "data" / "src1" / "dataset_hybrid"
    ddir.mkdir(parents=True)
    with (ddir / "windows.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(WIN_COLS)
        for i in range(5):
            wr.writerow(["dsqwen-7b", "S1", "hold", "0.5", "c1", "1", "train", "valid", "False",
                         "True" if i == 4 else "False", T0 + i * 10000, T0 + (i + 1) * 10000])
    (ddir / "cells.csv").write_text(f"cell_id,attempt,end_ms\nc1,1,{T0 + 50000}\n")
    reqs = [  # (send offset ms, e2e ms, in, out, outcome)
        (100, 1000, 1000, 100, "ok"), (9900, 1000, 1000, 100, "ok"),
        (10100, 200, 5000, 10, "ok"),
        (20100, 6000, 8000, 100, "ok"),
        (30100, 30000, None, None, "client_timeout"), (30200, 1000, 1000, 100, "ok"),
        (40100, 1000, 90000, 9000, "ok")]
    with (ddir / "requests.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(REQ_COLS)
        for k, (s, e2e, L, out, oc) in enumerate(reqs):
            wr.writerow(["dsqwen-7b", "c1", "1", f"r{k}", T0 + s, "" if oc != "ok" else T0 + s + 50, T0 + s + e2e,
                         "" if L is None else L, "" if out is None else out, 3000, oc, "pod-a"])
    cfg = {"window_s": 10, "route_timeout_s": 150,
           "filters": {"splits": ["train", "auxiliary"], "cell_status": ["valid"], "held_out_shapes": ["M"],
                       "exclude_in_warmup": True, "exclude_possibly_contaminated": True,
                       "window_inside_cell": True},
           "sources": {"dsqwen-7b": [{"name": "src1", "path": "src1/dataset_hybrid"}]}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _nl(tmp_path, 50.0)
    return path, tmp_path / "data"


def _nl(tmp_path: Path, v: float) -> str:
    """The --nl-slo file (preserve_nl_slo output format)."""
    p = tmp_path / "nl_slo.yaml"
    p.write_text(yaml.safe_dump({"nl_slo_ms": {"dsqwen-7b": v}}))
    return str(p)


def _alg1(tmp_path: Path, cfg: Path, root: Path, out: Path, *extra: str) -> int:
    return pm.main(["alg1", "--config", str(cfg), "--data-root", str(root), "--nl-slo", str(tmp_path / "nl_slo.yaml"),
                    "--out", str(out), *extra])


def test_alg1_max_over_clean_windows_by_arrival(tmp_path) -> None:
    import json

    cfg, root = _alg1_dataset(tmp_path)
    out = tmp_path / "out"
    assert _alg1(tmp_path, cfg, root, out) == 0
    res = json.loads((out / "result.json").read_text())
    m = res["models"]["dsqwen-7b"]
    assert res["mu"] == {"dsqwen-7b": {"p": 500.0, "d": 20.0, "t": 501.0}}
    assert yaml.safe_load((out / "mu.yaml").read_text()) == {"mu": res["mu"]}
    assert (m["windows_kept"], m["clean"], m["dirty"]) == (4, 2, 2)
    assert res["sources"][0]["dropped"] == {"in_warmup": 1}
    T = m["terms"]
    assert T["p"]["window"]["window_start_ms"] == T0 + 10000             # the P-share-0.998 window counts
    assert T["p"]["bracketed"] and T["p"]["max_dirty"] == 800.0 and not T["d"]["bracketed"]
    assert [x["value"] for x in T["t"]["top"]] == [501.0, 220.0]
    # provenance: every file read with its sha256, the NL SLO used
    assert sorted(Path(f["path"]).name for f in res["provenance"]["files"]) == ["cells.csv", "requests.csv",
                                                                                "windows.csv"]
    assert all(len(f["sha256"]) == 64 for f in res["provenance"]["files"]) and m["nl_slo_ms"] == 50.0
    with pytest.raises(SystemExit):                                     # never overwrites a result
        _alg1(tmp_path, cfg, root, out)


def test_alg1_a_stricter_nl_slo_dirties_the_window_holding_mu(tmp_path) -> None:
    cfg, root = _alg1_dataset(tmp_path)
    _nl(tmp_path, 15.0)                                                 # w1's NL is 20 ms
    assert _alg1(tmp_path, cfg, root, tmp_path / "o") == 0
    assert yaml.safe_load((tmp_path / "o" / "mu.yaml").read_text())["mu"]["dsqwen-7b"] == {
        "p": 200.0, "d": 20.0, "t": 220.0}


def test_alg1_envelope_drops_windows_outside_the_shape_bounds(tmp_path) -> None:
    import json

    cfg, root = _alg1_dataset(tmp_path)
    env = tmp_path / "env.yaml"                       # P share: w0 .91 in; w1 .998, w2 .988, w3 .976 out
    env.write_text(yaml.safe_dump({"name": "t", "bounds": {"dsqwen-7b": {"share": [0.5, 0.95]}}}))
    out = tmp_path / "out"
    assert _alg1(tmp_path, cfg, root, out, "--envelope", str(env)) == 0
    res = json.loads((out / "result.json").read_text())
    m = res["models"]["dsqwen-7b"]
    assert res["mu"] == {"dsqwen-7b": {"p": 200.0, "d": 20.0, "t": 220.0}}       # only w0 is inside
    assert (m["clean"], m["outside_envelope"]) == (1, 3) and res["config"]["envelope"]["name"] == "t"
    # a mean-length bound counts per request: w0 has 1000 in / 100 out on average
    env.write_text(yaml.safe_dump({"name": "l", "bounds": {"dsqwen-7b": {"mean_in": [900, 1100], "mean_out": [90, 110]}}}))
    assert _alg1(tmp_path, cfg, root, tmp_path / "o2", "--envelope", str(env)) == 0
    assert yaml.safe_load((tmp_path / "o2" / "mu.yaml").read_text())["mu"]["dsqwen-7b"]["p"] == 200.0


def test_alg1_takes_the_nl_slo_file_and_refuses_one_off_its_pin(tmp_path) -> None:
    cfg, root = _alg1_dataset(tmp_path)
    c = yaml.safe_load(cfg.read_text())
    c["nl_slo_expect_ms"] = {"dsqwen-7b": 50.0}
    cfg.write_text(yaml.safe_dump(c))
    assert _alg1(tmp_path, cfg, root, tmp_path / "o") == 0             # file 50.0 == pin
    _nl(tmp_path, 50.01)
    with pytest.raises(SystemExit, match="pins"):
        _alg1(tmp_path, cfg, root, tmp_path / "o2")
    c["nl_slo_ms"] = {"dsqwen-7b": 50.0}                                # a hand-typed value is refused
    cfg.write_text(yaml.safe_dump(c))
    with pytest.raises(SystemExit, match="--nl-slo"):
        _alg1(tmp_path, cfg, root, tmp_path / "o3")


def test_alg1_default_envelope_must_be_named_and_match_its_traces(tmp_path) -> None:
    cfg, root = _alg1_dataset(tmp_path)
    c = yaml.safe_load(cfg.read_text())
    c["envelope_default"] = {"name": "lengths-minmax", "config": "env-cfg.yaml"}
    cfg.write_text(yaml.safe_dump(c))
    (tmp_path / "env-cfg.yaml").write_text(yaml.safe_dump({"window_s": 10, "traces": [{"trace": "T", "sha256": "ab"}]}))
    env = tmp_path / "env.yaml"
    body = {"name": "lengths-minmax", "source": {"window_s": 10.0, "trace_sha256": {"T": "ab"}},
            "bounds": {"dsqwen-7b": {"mean_in": [900, 1100], "mean_out": [90, 110]}}}
    env.write_text(yaml.safe_dump(body))
    with pytest.raises(SystemExit, match="--no-envelope"):              # the default needs a file (or an opt-out)
        _alg1(tmp_path, cfg, root, tmp_path / "o")
    assert _alg1(tmp_path, cfg, root, tmp_path / "o", "--envelope", str(env)) == 0
    assert _alg1(tmp_path, cfg, root, tmp_path / "o1", "--no-envelope") == 0
    env.write_text(yaml.safe_dump({**body, "source": {"window_s": 10.0, "trace_sha256": {"T": "cd"}}}))
    with pytest.raises(SystemExit, match="differs"):                    # made from other trace files
        _alg1(tmp_path, cfg, root, tmp_path / "o2", "--envelope", str(env))
    env.write_text(yaml.safe_dump({**body, "name": "share-minmax"}))
    with pytest.raises(SystemExit, match="not the default"):
        _alg1(tmp_path, cfg, root, tmp_path / "o3", "--envelope", str(env))
    assert _alg1(tmp_path, cfg, root, tmp_path / "o3", "--envelope", str(env), "--envelope-variant") == 0
