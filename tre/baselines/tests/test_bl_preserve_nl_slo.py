"""tools/preserve_nl_slo reproduces the 2026-11-04 NL SLO from raw-data-shaped inputs.

Tiny fixture with the real inputs' decisive values: registry c / b (7b 41.1 / 0.0694, 14b 45.3 / 0.0879), the
idle TPOT medians of the 10-03 capture (7b 12.188190470139187, 14b 13.982359816630682 ms; a failed record and
the spread around the median must not count), the mix shapes of the 10-26 isolated run (7b 459 x 271, 14b
463 x 294), k 3 / 5 (7b) and 5 / 3 (14b), floor 450 ms. 14b's k_ttft (c + b L) = 430 ms, so the floor binds.
Expected: NL SLO 62.38 / 43.34 ms (docs/slo-idle-multiples-20261104.md section 4.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tre_baselines.tools import preserve_nl_slo as nl

TPOT = {"dsqwen-7b": 12.188190470139187, "dsqwen-14b": 13.982359816630682}
MIX = {"dsqwen-7b": (459, 271), "dsqwen-14b": (463, 294)}


def _fixture(tmp_path: Path) -> tuple:
    root = tmp_path / "data"
    for m, t in TPOT.items():
        d = root / "cb" / m
        d.mkdir(parents=True)
        recs = [{"outcome": "ok", "tpot_ms": v} for v in (t - 1.0, t, t + 2.0)]
        recs.append({"outcome": "client_timeout", "tpot_ms": 999.0})
        (d / "idle.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (root / "fit.json").write_text(json.dumps({"models": {"dsqwen-7b": {"c_ms": 41.092, "b_ms_per_token": 0.06944},
                                                          "dsqwen-14b": {"c_ms": 45.28, "b_ms_per_token": 0.08792}}}))
    iso = root / "iso"
    iso.mkdir()
    (iso / "isolated-meta.json").write_text(json.dumps({"shapes": {m: [["mix", L, o], ["b00", 100, 50]]
                                                                   for m, (L, o) in MIX.items()}}))
    for m, (L, o) in MIX.items():
        r = {"shape": "mix", "run": 0, "ok": True, "in": L, "out": o, "prompt_tokens": L, "completion_tokens": o,
             "queue_before": 0, "e2e_s": 0.0125 * o}
        (iso / f"{m}.jsonl").write_text(json.dumps(r) + "\n")
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "dsqwen-7b", "slo": {"ttft_idle_c_ms": 41.1, "ttft_idle_b_ms_per_token": 0.0694}},
        {"name": "dsqwen-14b", "slo": {"ttft_idle_c_ms": 45.3, "ttft_idle_b_ms_per_token": 0.0879}}]}))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "round_decimals": 2,
        "inputs": {"idle_raw": "cb/{model}/idle.jsonl", "idle_fit": "fit.json", "isolated_dir": "iso", "mix_shape": "mix"},
        "models": {"dsqwen-7b": {"k_ttft": 3, "k_tpot": 5, "ttft_floor_ms": 450},
                   "dsqwen-14b": {"k_ttft": 5, "k_tpot": 3, "ttft_floor_ms": 450}}}))
    return cfg, root, reg


def test_nl_slo_reproduces_the_fixed_numbers(tmp_path) -> None:
    cfg, root, reg = _fixture(tmp_path)
    out = tmp_path / "out"
    assert nl.main(["--config", str(cfg), "--data-root", str(root), "--registry", str(reg), "--out", str(out)]) == 0
    doc = yaml.safe_load((out / "nl_slo.yaml").read_text())
    assert doc["nl_slo_ms"] == {"dsqwen-7b": 62.38, "dsqwen-14b": 43.34}
    m7, m14 = doc["models"]["dsqwen-7b"], doc["models"]["dsqwen-14b"]
    assert m7["tpot_idle_ms"] == TPOT["dsqwen-7b"] and (m7["tpot_idle_n"], m7["tpot_idle_left_out"]) == (3, 1)
    assert m14["ttft_slo_mix_ms"] == 450.0 and m14["tpot_slo_ms"] == pytest.approx(3 * TPOT["dsqwen-14b"])
    assert m7["isolated_nl_mix_ms"] == pytest.approx(12.5)
    assert sorted(f["path"] for f in doc["provenance"]["inputs"]) == [
        "cb/dsqwen-14b/idle.jsonl", "cb/dsqwen-7b/idle.jsonl", "fit.json", "iso/dsqwen-14b.jsonl",
        "iso/dsqwen-7b.jsonl", "iso/isolated-meta.json"]
    with pytest.raises(SystemExit):                                     # never overwrites
        nl.main(["--config", str(cfg), "--data-root", str(root), "--registry", str(reg), "--out", str(out)])
