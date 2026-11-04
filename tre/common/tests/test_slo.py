"""tre_common.slo: the one SLO definition (2026-11-04) - values, the verdict, the switch."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tre_common import slo
from tre_common.registry import _parse_registry, load_registry

REGISTRY = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def _raw(mode: str | None = None) -> dict:
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    if mode is not None:
        raw["slo_mode"] = mode
    return raw


def test_idle_multiple_thresholds_of_the_repo_registry() -> None:
    reg = _parse_registry(_raw("idle_multiple"))
    t7, t8, t14 = (reg.slo_thresholds(m) for m in ("dsqwen-7b", "dsllama-8b", "dsqwen-14b"))
    assert round(t7.ttft_ms(2048), 1) == 549.7          # 3 * (41.1 + 0.0694 * 2048)
    assert round(t14.ttft_ms(1024), 1) == 676.5         # 5 * (45.3 + 0.0879 * 1024)
    assert t7.ttft_ms(256) == t8.ttft_ms(1024) == t14.ttft_ms(256) == 450.0   # the floor
    assert t14.ttft_ms(None) == 450.0                    # no length: the floor
    assert [round(t.tpot_ms, 2) for t in (t7, t8, t14)] == [60.94, 63.37, 41.95]


def test_fixed_mode_reproduces_500_75_for_every_length() -> None:
    reg = load_registry(str(REGISTRY))
    assert reg.slo_mode() == "fixed"                     # the live definition (release: idle_multiple)
    for model in ("dsqwen-7b", "dsllama-8b", "dsqwen-14b"):
        t = reg.slo_thresholds(model)
        assert {t.ttft_ms(L) for L in (None, 1, 2048, 8192)} == {500.0} and t.tpot_ms == 75.0


def test_the_verdict() -> None:
    t = slo.slo_thresholds({"ttft_p95_ms": 500, "tpot_p95_ms": 75}, "fixed")
    ok = slo.judge_request(t, completed=True, prompt_tokens=10, ttft_ms=400, e2e_ms=400 + 9 * 70, output_tokens=10)
    assert not ok.violated and ok.tpot_ms == pytest.approx(70.0)
    slow = slo.judge_request(t, completed=True, prompt_tokens=10, ttft_ms=400, e2e_ms=400 + 9 * 80, output_tokens=10)
    assert (slow.cause, slow.v_tpot) == ("tpot", True)
    # one output token: TTFT only (no TPOT)
    one = slo.judge_request(t, completed=True, prompt_tokens=10, ttft_ms=400, e2e_ms=9_000, output_tokens=1)
    assert (one.violated, one.tpot_ms) == (False, None)
    failed = slo.judge_request(t, completed=False, prompt_tokens=10, ttft_ms=600, e2e_ms=None, output_tokens=0)
    assert (failed.cause, failed.violated, failed.v_ttft) == ("failed", True, True)


def test_one_switch() -> None:
    raw = _raw()
    raw["safescale"]["slo_mode"] = "fixed"               # a retired key next to the switch
    with pytest.raises(ValueError, match="one switch"):
        slo.resolve_slo_mode(raw)
    legacy = _raw()
    legacy.pop("slo_mode")
    legacy["safescale"]["slo_mode"] = "fixed"
    for model in legacy["models"]:
        model["slo"]["ttft_slo_mode"] = "slowdown"
    res = slo.resolve_slo_mode(legacy)                   # the pre-11-04 live registry: reproduced, flagged
    assert res.mode == "slowdown" and res.legacy_conflict and res.source.startswith("legacy")
    with pytest.raises(ValueError, match="slo_mode must be one of"):
        slo.resolve_slo_mode({**_raw(), "slo_mode": "median"})


def test_idle_multiple_is_refused_at_load_without_its_parameters() -> None:
    raw = _raw("idle_multiple")
    del raw["models"][2]["slo"]["idle_multiple"]["tpot_idle_ms"]
    with pytest.raises(ValueError, match="tpot_idle_ms"):
        _parse_registry(raw)
    raw = _raw("fixed")                                  # the block is parsed in every mode
    raw["models"][0]["slo"]["idle_multiple"]["ttft_k"] = -1
    with pytest.raises(ValueError, match="ttft_k"):
        _parse_registry(raw)


def test_no_built_in_parameter_values() -> None:
    # 2026-11-04: every threshold is the registry's - a block without the mode's parameter is refused
    with pytest.raises(slo.SloConfigError, match="tpot_p95_ms"):
        slo.slo_thresholds({"ttft_p95_ms": 500}, "fixed")
    block = {"ttft_p95_ms": 500, "tpot_p95_ms": 75, "ttft_idle_c_ms": 40.0, "ttft_idle_b_ms_per_token": 0.05,
             "ttft_floor_ms": 500}
    with pytest.raises(slo.SloConfigError, match="ttft_slowdown_k"):
        slo.slo_thresholds(block, "slowdown")
    raw = _raw("slowdown")
    del raw["models"][1]["slo"]["ttft_floor_ms"]
    with pytest.raises(ValueError, match="ttft_floor_ms"):           # refused at load in every mode
        _parse_registry(raw)


@pytest.mark.parametrize("mode", slo.SLO_MODES)
def test_the_vectorised_verdict_is_judge_request(mode) -> None:
    np = pytest.importorskip("numpy")
    t = _parse_registry(_raw(mode)).slo_thresholds("dsqwen-14b")
    rows = [(done, L, ttft, tpot) for done in (True, False) for L in (None, 128, 1024, 6000)
            for ttft in (None, 300.0, 460.0, 700.0, 2100.0) for tpot in (None, 30.0, 45.0, 80.0)]
    want = [slo.judge_request(t, completed=d, prompt_tokens=L, ttft_ms=tt, tpot_ms=tp).violated for d, L, tt, tp in rows]
    nan = float("nan")
    got = slo.violated_array(
        t, completed=[r[0] for r in rows], prompt_tokens=[nan if r[1] is None else r[1] for r in rows],
        ttft_ms=[nan if r[2] is None else r[2] for r in rows], tpot_ms=[nan if r[3] is None else r[3] for r in rows])
    assert got.tolist() == want


def test_load_registry_slo(tmp_path) -> None:
    rs = slo.load_registry_slo(str(REGISTRY))
    assert (rs.mode, rs.source) == ("fixed", "slo_mode") and rs.of("dsqwen-7b").tpot_ms == 75.0
    assert slo.load_registry_slo(str(REGISTRY), mode="idle_multiple").of("dsqwen-14b").ttft_ms(None) == 450.0
    legacy = _raw()
    legacy.pop("slo_mode")
    legacy["safescale"]["slo_mode"] = "fixed"
    for model in legacy["models"]:
        model["slo"]["ttft_slo_mode"] = "slowdown"
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    with pytest.raises(slo.SloConfigError, match="name the mode"):  # the two retired keys disagree
        slo.load_registry_slo(str(path))
    assert slo.load_registry_slo(str(path), mode="fixed").source == "explicit"
