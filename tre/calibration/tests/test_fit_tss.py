"""The TSS-parameter fit tools (calibration/scripts/fit_tss/): the invariants they rest on,
hermetic (synthetic rows, no data root). The fit itself is checked against the published
2026-11-03 numbers on the real data (fit_tss/README.md, "Equivalence check"), not here.

1. The fit's vectorised TSS is the controller's formula (``recompute_tss_rows``, raw).
2. Sealed held-out sets are refused from the path string alone, before any file is opened.
3. The tools name no site path: every location is configuration.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

FIT = Path(__file__).resolve().parents[1] / "scripts" / "fit_tss"


def _supp_lib(monkeypatch, data_root):
    if data_root is None:
        monkeypatch.delenv("TRE_FIT_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("TRE_FIT_DATA_ROOT", str(data_root))
    spec = importlib.util.spec_from_file_location("tre_fit_tss_supp_lib", FIT / "supp_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_tss_equals_the_controller_formula(monkeypatch):
    L = _supp_lib(monkeypatch, None)
    nan = float("nan")
    W = pd.DataFrame({
        "window_start_ms": [0, 10_000, 20_000, 30_000, 40_000, 50_000],
        "window_end_ms": [10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
        "prompt_tokens_total": [5000.0, 0.0, 800.0, nan, 1200.0, 300.0],
        "generation_tokens_total": [2000.0, 900.0, 0.0, 400.0, 3000.0, 50.0],
        "avg_running": [12.0, 3.0, 0.0, 4.0, 0.0, 0.4],
        "avg_waiting": [2.5, 0.0, 0.0, 1.0, 6.0, 0.0],
        "kv_cache_hit_rate": [0.2, nan, 0.0, 0.0, 0.5, 0.0],
        "assigned_replicas": [2.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        "routable_pods": [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    })
    for lam, wp in ((20.0, 0.01), (32.91, 0.009472), (2.0, 0.0)):
        tss = L.tss_vec(W, lam, wp)
        L.assert_tss(W, lam, wp, tss)  # raises if any row differs from recompute_tss_rows
        assert np.isnan(tss[2]) and np.isnan(tss[3])  # nothing in flight; tokens unknown
        assert tss[1] == pytest.approx(900.0 / 3.0)  # no prefill, no queue: decode rate per running request
        assert tss[5] == pytest.approx((wp * 300.0 + 50.0) / 1.0)  # qmin floor on R + lambda Wq


def test_guard_refuses_m_and_e_from_the_path_alone(monkeypatch):
    X = Path("/nonexistent-tre-fit-data")  # never created: the guard must not touch the file system
    L = _supp_lib(monkeypatch, X)
    sealed = [X / "calib_20261003/M", X / "calib_20261003/M/dsqwen-7b/dataset_hybrid",
              X / "calib_20261003/next-20261005/M2/dsqwen-7b", X / "calib_supp_20261103/E",
              X / "calib_supp_20261103/datasets/w10000/E/dsqwen-7b/dataset_hybrid", X / "calib_supp_20261103/Emix",
              X / "calib_supp_20261103/datasets/E_sealed", X / "somewhere/holdout_set",
              X / "datasets-w10000/validation/dsqwen-7b"]
    open_ = [X / "analysis-tss-params-20261030/win10/alphafit-w10000-s10000/datasets/run2/dsqwen-7b/dataset_hybrid",
             X / "calib_supp_20261103/datasets/w10000/A/dsqwen-7b/dataset_hybrid", X / "calib_supp_20261103/B",
             X / "calib_20261003/run2/dsqwen-7b"]
    for p in sealed:
        assert L.sealed_reason(p) is not None, p
        with pytest.raises(SystemExit, match="REFUSED"):
            L.guard_path(p)
    for p in open_:
        assert L.sealed_reason(p) is None, (p, L.sealed_reason(p))
    assert not X.exists()


def test_old_roots_need_the_data_root(monkeypatch):
    L = _supp_lib(monkeypatch, None)
    with pytest.raises(SystemExit, match="TRE_FIT_DATA_ROOT"):
        L.resolve_specs(["old"], "dsqwen-7b", 10000)


@pytest.mark.parametrize("path", sorted(p.name for p in FIT.iterdir() if p.suffix in (".py", ".sh")))
def test_fit_tools_parse_and_hard_code_no_site_path(path):
    text = (FIT / path).read_text(encoding="utf-8")
    for literal in ("/data/nfs_shared_data", "192.168.", "nscc-ds-", "node9", "node10"):
        assert literal not in text, f"{path} hard-codes {literal}"
    if path.endswith(".sh"):
        if shutil.which("bash") is None:
            pytest.skip("bash not available")
        subprocess.run(["bash", "-n", str(FIT / path)], check=True)
