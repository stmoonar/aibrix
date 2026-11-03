"""preserve_train (the offline PreServe mLSTM trainer, a copy of the v1 trainer): what it ships is what the
policy loads. Synthetic two-day CSVs at the Azure file names; W = 600 s keeps the windows few."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

W = 600


def _azure(root):
    """Two days of one request every 30 s per stream, from each week file's first day (code 05-10, conv
    05-12), so the trainer's exclusions and paper_past cut-offs fall inside."""
    rng = np.random.default_rng(7)
    (root / "2024").mkdir(parents=True)
    for name, day in (("code", datetime(2024, 5, 10, tzinfo=timezone.utc)),
                      ("conv", datetime(2024, 5, 12, tzinfo=timezone.utc))):
        lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
        for i in range(2 * 86400 // 30):
            t = day + timedelta(seconds=30 * i, microseconds=int(rng.integers(0, 10**6)))
            lines.append(f"{t.strftime('%Y-%m-%d %H:%M:%S.%f')}+00:00,{int(rng.integers(50, 4000))},"
                         f"{int(rng.integers(5, 600))}")
        (root / "2024" / f"AzureLLMInferenceTrace_{name}_1week.csv").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def azure(tmp_path):
    return _azure(tmp_path / "azure")


def test_pandas_free_window_caches_equal_the_trainers_own_windowing(azure, tmp_path):
    pytest.importorskip("pandas")
    from preserve_train import data, pipeline

    out = tmp_path / "train"
    assert pipeline.main(["windows", "--train-dir", str(out), "--window-s", str(W), "--azure-root", str(azure)]) == 0
    for name, rel in pipeline.CSV_REL.items():
        ref = data.window_csv(str(azure / rel), W)
        with np.load(out / f"windows_{name}_{W}s.npz", allow_pickle=False) as z:
            assert str(z["sha256"]) == data.sha256_file(str(azure / rel))
            for k in pipeline.KEYS:
                assert np.array_equal(z[k], np.asarray(ref[k])), (name, k)


def test_shipped_weights_declare_the_training_window(azure, tmp_path):
    pytest.importorskip("torch")
    from preserve_train import pipeline, train_mlstm
    from tre_baselines.policies import preserve_predictor as pp

    out, wdir = tmp_path / "train", tmp_path / "weights"
    assert pipeline.main(["windows", "--train-dir", str(out), "--window-s", str(W), "--azure-root", str(azure)]) == 0
    argv = [f"--csv=code={azure / pipeline.CSV_REL['code']}", f"--csv=conv={azure / pipeline.CSV_REL['conv']}",
            "--out", str(out), "--window-s", str(W), "--protocol", "paper_past", "--no-verify-sha",
            "--epochs", "1", "--jobs", "1", "--norms", "per_sequence_mean", "--file-suffix=-t"]
    assert train_mlstm.main(argv) == 0
    assert pipeline.main(["finish", "--train-dir", str(out), "--window-s", str(W), "--weights-out", str(wdir)]) == 0
    man = json.loads((wdir / "manifest.json").read_text())
    assert man["window_s"] == W and set(man["files"]) == {"code", "conv", "pooled"}
    for rec in man["files"].values():
        prs = pp.load_predictors(str(wdir / rec["file"]), rec["sha256"])
        assert {p.window_s for p in prs.values()} == {float(W)}
    assert man["held_out_mape_naive_last_window"][str(W)]["pooled"].keys() == {"code", "conv"}
