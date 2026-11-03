"""tools/preserve_history_azure.py: the window changes only the window. Same Azure segment, every request
keeps its mapped length, the level is the planned mean per window, and the policy takes each file only at
its own window. Synthetic Azure week files (one request every 10 s) and a given plan."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from tre_baselines.policies import preserve_history as ph
from tre_baselines.tools import preserve_history_azure as pha

TRACE = "Real_code_2024_slice_a_tok70"


@pytest.fixture(scope="module")
def azure(tmp_path_factory):
    root = tmp_path_factory.mktemp("azure")
    (root / "2024").mkdir()
    rng = np.random.default_rng(3)
    for fam, f in pha.FILES.items():
        n = int((f["end"] - f["start"]).total_seconds()) // 10
        ins, outs = rng.integers(20, 8000, n), rng.integers(1, 2000, n)
        t0 = f["start"]
        lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
        lines += [f"{(t0 + timedelta(seconds=10 * i)).isoformat(sep=' ')},{ins[i]},{outs[i]}" for i in range(n)]
        (root / f["rel"]).write_text("\n".join(lines) + "\n")
    return root


def _plan(window_s):
    return {m: {"P_per_window": 9000.0 * window_s / 30, "D_per_window": 2500.0 * window_s / 30,
                "max_in": 650, "max_out": 380} for m in pha.MODELS}


def test_another_window_keeps_the_segment_lengths_and_planned_level(azure):
    d30 = pha.build("r3e", TRACE, 1, 1.0, plan_doc=_plan(30), azure_root=azure, window_s=30)
    d10 = pha.build("r3e", TRACE, 1, 1.0, plan_doc=_plan(10), azure_root=azure, window_s=10)
    for m in pha.MODELS:
        p30, p10 = d30["source"]["per_model"][m], d10["source"]["per_model"][m]
        assert p10["segment_start_utc"] == p30["segment_start_utc"]
        end = datetime.fromisoformat(p30["segment_end_utc_exclusive"])
        assert datetime.fromisoformat(p10["history_start_utc"]) == end - timedelta(seconds=250)
        r30, r10 = np.array(d30["models"][m]), np.array(d10["models"][m])
        assert r30.shape == r10.shape == (25, 2)
        for c, key in ((0, "scale_P"), (1, "scale_D")):
            # unscaled token sums: 30 s windows 17..24 = 10 s windows 1..24 in threes (same mapped lengths)
            w30 = r30[17:, c] / p30[key]
            w10 = r10[1:, c].reshape(8, 3).sum(axis=1) / p10[key]
            assert np.allclose(w30, w10, rtol=1e-4)
            assert r10[:, c].mean() == pytest.approx(_plan(10)[m][("P_per_window", "D_per_window")[c]], rel=1e-3)
    ph.parse(d30, window_s=30.0, models=pha.MODELS, seq_len=25, trace=TRACE, seed=1)
    ph.parse(d10, window_s=10.0, models=pha.MODELS, seq_len=25, trace=TRACE, seed=1)
    with pytest.raises(ph.HistoryError):
        ph.parse(d10, window_s=30.0, models=pha.MODELS, seq_len=25, trace=TRACE, seed=1)
    assert pha.out_name(30, 25) == "preserve_history.json" and pha.out_name(10, 25) == "preserve_history_w10.json"
