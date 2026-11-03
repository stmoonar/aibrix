"""PreServe Tier-1 min-max weights (2026-11-04, preserve_train/trace_train.py): the constants travel with the file,
and the policy refuses a file whose normalization or window differs from its params."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preserve_train.export import export_npz
from tre_baselines.config import Config, ModelLimits
from tre_baselines.policies import preserve_predictor as pp
from tre_baselines.policies.preserve import PreServePolicy

FIX = Path(__file__).resolve().parent / "fixtures"
H = 4
MODELS = ("dsqwen-7b", "dsllama-8b")


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _ConstModel:
    """An mLSTMModel state_dict whose output is fc.bias whatever the input (fc.weight = 0)."""

    def __init__(self, bias: float):
        rng = np.random.default_rng(0)
        self.sd = {"mLSTM.cells.0.W_ih.weight": rng.normal(size=(4 * H, 1)), "mLSTM.cells.0.W_ih.bias": np.zeros(4 * H),
                   "mLSTM.cells.0.W_hh.weight": rng.normal(size=(4 * H, H)), "mLSTM.cells.0.W_hh.bias": np.zeros(4 * H),
                   "mLSTM.cells.0.W_m.weight": np.zeros((H, 1)), "mLSTM.cells.0.W_m.bias": np.zeros(H),
                   "mLSTM.cells.0.U_m.weight": np.zeros((H, H)), "mLSTM.cells.0.U_m.bias": np.zeros(H),
                   "fc.weight": np.zeros((1, H)), "fc.bias": np.array([bias])}

    def state_dict(self):
        return {k: _T(v) for k, v in self.sd.items()}


def _write(path: Path, *, window_s=10.0, model="dsqwen-7b", bias=(0.5, -2.0)) -> str:
    export_npz({"prompt": _ConstModel(bias[0]), "response": _ConstModel(bias[1])}, str(path), seq_len=3, hidden=H,
               eps=1.0, window_s=window_s, normalization=pp.MINMAX,
               minmax={"prompt": (100.0, 300.0), "response": (10.0, 50.0)}, meta={"model": model})
    return pp.sha256_of(str(path))


def test_minmax_constants_round_trip_through_the_file(tmp_path) -> None:
    sha = _write(tmp_path / "w.npz")
    preds = pp.load_predictors(str(tmp_path / "w.npz"), sha)
    p, d = preds["prompt"], preds["response"]
    assert (p.normalization, p.norm_min, p.norm_max, p.model, p.window_s) == (pp.MINMAX, 100.0, 300.0, "dsqwen-7b", 10.0)
    # out' = 0.5 -> 100 + 0.5 x 200 = 200 for any history; a negative de-normalized value clamps at 0
    assert p.predict([1e6, 0.0, 5.0]) == pytest.approx(200.0)
    assert d.predict([10.0, 20.0, 30.0]) == 0.0  # 10 + (-2) x 40 < 0
    with pytest.raises(ValueError, match="minmax"):  # constants go with the normalization
        export_npz({"prompt": _ConstModel(0), "response": _ConstModel(0)}, str(tmp_path / "x.npz"), seq_len=3,
                   hidden=H, eps=1.0, window_s=10.0, normalization=pp.MINMAX)


def _policy(params: dict) -> PreServePolicy:
    lim = {m: ModelLimits(name=m, min_replicas=1, max_replicas=8, gpus_per_replica=1, ttft_slo_ms=500.0,
                          tpot_slo_ms=75.0, max_num_seqs=256) for m in MODELS}
    base = {"mu": {m: {"p": 100.0, "d": 100.0, "t": 150.0} for m in MODELS}, "seq_len": 3,
            "trace_path": str(FIX / "preserve_trace_requests.json"), "window_s": 10.0,
            "predictor_rules": [["", "trace"]]}
    cfg = Config(sm_url="http://sm.test", redis_url="redis://r.test", policy="preserve",
                 policy_params={**base, **params}, models=lim, seed=0)
    return PreServePolicy(cfg, history={m: [[1.0, 1.0]] * 3 for m in MODELS})


def test_policy_refuses_normalization_window_and_model_mismatch(tmp_path) -> None:
    files = {m: {"path": str(tmp_path / f"{m}.npz"), "sha256": _write(tmp_path / f"{m}.npz", model=m)} for m in MODELS}
    per_model = {"trace": {"models": files}}
    pol = _policy({"predictors": per_model, "predictor_normalization": "minmax"})
    assert {m: pol._predictors[m]["prompt"].model for m in MODELS} == {m: m for m in MODELS}
    with pytest.raises(ValueError, match="normalization"):  # default per_sequence_mean
        _policy({"predictors": per_model})
    w30 = {"path": str(tmp_path / "w30.npz"), "sha256": _write(tmp_path / "w30.npz", window_s=30.0)}
    with pytest.raises(ValueError, match="30.0 s windows"):
        _policy({"predictors": {"trace": w30}, "predictor_normalization": "minmax"})
    swapped = {"trace": {"models": {MODELS[0]: files[MODELS[1]], MODELS[1]: files[MODELS[0]]}}}
    with pytest.raises(ValueError, match="trained for model"):
        _policy({"predictors": swapped, "predictor_normalization": "minmax"})
